"""Real ZETRABOT telemetry: parsing, normalisation, replay, HARVEST wiring, FIWARE.

Fixture: ``tests/fixtures/zetrabot_mission_fixture.csv`` -- 432 rows copied
verbatim from ``telemetry/telemetria_mision_63.csv`` (the first rows of every
message/signal-set combination, the rows around a session-counter reset and
around an SOC step), written in a deliberately scrambled order, plus ten
synthetic edge rows (ids 9000001..9000010: an unknown signal, an unknown
message type, empty / invalid / null signals, a non-numeric value, a row
without a timestamp, and a second tractor).

The tests marked *real mission* run only when the full export is present
(it is git-ignored) and check the calibration figures against the Zetrack V2
report where the two can be compared.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import threading
import time
import unittest
import urllib.request
from http.server import HTTPServer
from pathlib import Path

from harvest_integrations import diagnostics as diag
from harvest_integrations.fiware.entities import telemetry_entities
from harvest_integrations.fiware.sync import ContextSync
from harvest_integrations.runtime import FleetRuntime
from harvest_integrations.telemetry import (
    CsvTelemetrySource, ReplayOptions, ReplayPlayer, TelemetryMessage, TelemetryNormalizer,
    TelemetryRegistry, ZetrabotTelemetry, parse_timestamp, read_csv, sort_messages,
)
from harvest_integrations.telemetry import zetrack
from harvest_integrations.telemetry.analysis import ESTIMATE_LABEL, MissionAnalysis
from harvest_integrations.telemetry.service import TelemetryService, build_telemetry_service
from harvest_integrations.telemetry.zetrack import CsvParseStats

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "zetrabot_mission_fixture.csv"
REAL = REPO / "telemetry" / "telemetria_mision_63.csv"
DEAD_BROKER = "http://127.0.0.1:59999"

UTC = dt.timezone.utc


def _msg(ts: str, message: str, signals: dict, tractor="1", seq=None, rid=None) -> TelemetryMessage:
    return TelemetryMessage(tractor_id=tractor, timestamp=parse_timestamp(ts), source_message=message,
                            signals=signals, mission_id="63", sequence=seq, record_id=rid,
                            source="unit-test")


def _telemetry_cfg(speed=0.0, **extra):
    cfg = {"tractors": {"model": {"battery_capacity_kwh": 44.8}},
           "integrations": {"telemetry": {"source": "csv",
                                          "csv": {"file": str(FIXTURE), "speed": speed,
                                                  "max_gap_s": 0.05},
                                          "tractor_ids": {"1": "zetrabot_1"}}}}
    cfg["integrations"]["telemetry"].update(extra)
    return cfg


class _NoTelemetryEnv(unittest.TestCase):
    """Guards against a developer's HARVEST_TELEMETRY_* leaking into tests."""
    _KEYS = ("HARVEST_TELEMETRY_SOURCE", "HARVEST_TELEMETRY_FILE", "HARVEST_TELEMETRY_SPEED",
             "HARVEST_TELEMETRY_MAX_GAP_S", "HARVEST_TELEMETRY_LOOP", "ORION_URL",
             "HARVEST_MONGO_HOST")

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in self._KEYS}
        for k in self._KEYS:
            os.environ.pop(k, None)
        os.environ["ORION_URL"] = DEAD_BROKER
        os.environ["HARVEST_MONGO_HOST"] = "127.0.0.1:59998"
        diag._cache._data.clear()

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        diag._cache._data.clear()


# --------------------------------------------------------------------------- #
#  CSV parsing + JSON signals
# --------------------------------------------------------------------------- #
class TestCsvParsing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stats = CsvParseStats()
        cls.messages = read_csv(FIXTURE, cls.stats)

    def test_fixture_present_and_parsed(self):
        self.assertTrue(FIXTURE.exists())
        self.assertEqual(self.stats.rows, 442)
        # one synthetic row has no timestamp and cannot be placed on a timeline
        self.assertEqual(self.stats.skipped_no_timestamp, 1)
        self.assertEqual(len(self.messages), 441)

    def test_columns_and_extras_preserved(self):
        m = self.messages[0]
        self.assertEqual(m.tractor_id, "1")
        self.assertEqual(m.mission_id, "63")
        self.assertEqual(m.schema_version, "1.0")
        self.assertEqual(m.source, "csv-replay")
        self.assertIsNotNone(m.received_at)          # created_at column
        self.assertEqual(m.extra.get("user_id"), "f24f5a11-d40a-45cb-bb57-ab034795444b")
        self.assertEqual(m.timestamp.tzinfo, UTC)

    def test_json_signals_decoded(self):
        multi = [m for m in self.messages if m.source_message == "PmsMotorSpeed" and len(m.signals) == 4]
        self.assertTrue(multi, "fixture must carry a 4-key PmsMotorSpeed row")
        self.assertEqual(set(multi[0].signals), {"T1SpeedAbs", "T2SpeedAbs", "T3SpeedAbs", "T4SpeedAbs"})
        self.assertTrue(all(isinstance(v, (int, float)) for v in multi[0].signals.values()))

    def test_invalid_json_is_kept_as_a_message_with_error(self):
        self.assertEqual(self.stats.invalid_signals_json, 1)
        broken = next(m for m in self.messages if m.record_id == "9000005")
        self.assertEqual(broken.signals, {})
        self.assertIn("invalid JSON", broken.extra["signals_error"])
        self.assertIn("{broken json", broken.extra["signals_text"])

    def test_empty_and_null_signals(self):
        empty = next(m for m in self.messages if m.record_id == "9000006")
        self.assertEqual(empty.signals, {})
        null = next(m for m in self.messages if m.record_id == "9000007")
        self.assertIn("MainBatteryVoltage", null.signals)
        self.assertIsNone(null.signals["MainBatteryVoltage"])

    def test_missing_required_column_raises(self):
        with self.assertRaises(ValueError):
            read_csv("id,tractor_id,timestamp\n1,1,2026-05-12T09:04:32.184+00:00\n")

    def test_parse_timestamp_variants(self):
        z = parse_timestamp("2026-05-14T08:29:07Z")
        plus = parse_timestamp("2026-05-14T08:29:07+00:00")
        naive = parse_timestamp("2026-05-14T08:29:07")
        self.assertEqual(z, plus)
        self.assertEqual(naive, plus)
        self.assertIsNone(parse_timestamp("not a date"))
        self.assertIsNone(parse_timestamp(""))

    def test_signal_dictionary_matches_model(self):
        model_fields = set(ZetrabotTelemetry.__dataclass_fields__)
        for spec in zetrack.SIGNALS:
            self.assertIn(spec.field, model_fields, spec.signal)
        self.assertGreaterEqual(len(zetrack.supported_fields()), 40)
        # every message type observed in the sample is known to the dictionary
        for message in ("BatteryStatus1", "BatteryStatus2", "BatteryStatus3", "PmsMotorSpeed",
                        "PmsMotorCurrent", "MotorTemp", "IpmMotorSpeed", "IpmMotorCurrent",
                        "MiscInfo", "FunctionStatus", "SensorsAnalogValues1",
                        "SensorsAnalogValues3", "LimitsStatus", "MemoryData2"):
            self.assertIn(message, zetrack.KNOWN_MESSAGES)


# --------------------------------------------------------------------------- #
#  Ordering
# --------------------------------------------------------------------------- #
class TestOrdering(unittest.TestCase):
    def test_file_order_is_scrambled_but_history_is_sorted(self):
        import csv
        with open(FIXTURE, newline="") as fh:
            raw_ts = [r["timestamp"] for r in csv.DictReader(fh) if r["timestamp"]]
        self.assertNotEqual(raw_ts, sorted(raw_ts), "fixture is meant to be out of order")
        messages = read_csv(FIXTURE)
        ts = [m.timestamp for m in messages]
        self.assertEqual(ts, sorted(ts))

    def test_ties_break_on_sequence_then_id(self):
        a = _msg("2026-05-12T09:04:32.184+00:00", "MotorTemp", {"TempT1": 1}, seq=5, rid="20")
        b = _msg("2026-05-12T09:04:32.184+00:00", "MotorTemp", {"TempT1": 2}, seq=3, rid="30")
        c = _msg("2026-05-12T09:04:32.184+00:00", "MotorTemp", {"TempT1": 3}, seq=3, rid="10")
        self.assertEqual([m.signals["TempT1"] for m in sort_messages([a, b, c])], [3, 2, 1])

    def test_sequence_reset_does_not_reorder_days(self):
        # sequence restarts at 1 on the second day: timestamp must still win
        day1 = _msg("2026-05-12T10:40:00+00:00", "MotorTemp", {"TempT1": 1}, seq=9000)
        day2 = _msg("2026-05-13T11:25:00+00:00", "MotorTemp", {"TempT1": 2}, seq=1)
        self.assertEqual([m.sequence for m in sort_messages([day2, day1])], [9000, 1])


# --------------------------------------------------------------------------- #
#  Normalisation
# --------------------------------------------------------------------------- #
class TestNormalizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.messages = read_csv(FIXTURE)
        cls.norm = TelemetryNormalizer()
        cls.norm.ingest_all(cls.messages)
        cls.state = cls.norm.state("1")

    def test_canonical_fields_populated(self):
        s = self.state
        self.assertEqual(s.soc_pct, 18.0)                     # last SOC row (synthetic 9000001)
        self.assertEqual(s.mission_id, "63")
        self.assertEqual(s.source, "csv-replay")
        self.assertIsNotNone(s.battery_voltage_v)
        self.assertIsNotNone(s.battery_current_a)
        self.assertIsNotNone(s.discharged_energy_session_kwh)
        self.assertGreater(s.discharged_energy_kwh, 0.0)
        self.assertEqual(set(s.wheel_speed_rpm), {"T1", "T2", "T3", "T4"})
        self.assertTrue(set(s.motor_temp_c) >= {"T1", "T2"})
        self.assertIsNotNone(s.pto_speed_rpm)
        self.assertIsNotNone(s.hydraulic_pressure_bar)
        self.assertIsNotNone(s.steering_angle_deg)
        self.assertIsNotNone(s.brake_pedal_pct)
        self.assertIn("BHD", s.implement_speed_rpm)
        self.assertIsInstance(s.pto_active, bool)
        self.assertIsInstance(s.drive_active, bool)
        self.assertIsInstance(s.drive_mode, int)
        self.assertIn("PtoLed", s.function_leds)

    def test_unknown_fields_and_messages_preserved(self):
        s = self.state
        self.assertEqual(s.unknown_signals.get("BatteryStatus1.FutureSignalX"), 42)
        self.assertEqual(s.unknown_signals.get("NewMessageType.Foo"), 1)
        self.assertEqual(s.unknown_signals.get("NewMessageType.Bar"), "abc")
        self.assertEqual(s.unknown_messages.get("NewMessageType"), 1)
        self.assertEqual(s.signals_raw.get("FutureSignalX"), 42)
        self.assertEqual(s.signals_raw.get("CursorPositionOffset"), 0)   # MemoryData2, raw-only
        self.assertIn("LimitsStatus", s.message_counts)

    def test_non_numeric_and_null_values_do_not_clobber(self):
        s = self.state
        self.assertGreaterEqual(s.invalid_values, 2)          # "not-a-number" + null voltage
        self.assertIsInstance(s.battery_current_a, float)     # the string never landed
        self.assertIsInstance(s.battery_voltage_v, float)     # null never landed
        self.assertEqual(s.signals_raw.get("BatteryCurrent"), "not-a-number")  # but is preserved raw

    def test_session_counter_accumulates_across_resets(self):
        s = self.state
        self.assertGreaterEqual(s.session_resets, 1)
        # cumulative >= the last session value, and >= the largest single reading
        readings = [m.signals["DischEnrgActualSesion"] for m in self.messages
                    if m.tractor_id == "1" and isinstance(m.signals.get("DischEnrgActualSesion"), (int, float))]
        self.assertGreaterEqual(s.discharged_energy_kwh, max(readings))
        self.assertGreaterEqual(s.discharged_energy_kwh, s.discharged_energy_session_kwh)

    def test_derived_power(self):
        n = TelemetryNormalizer()
        n.ingest(_msg("2026-05-13T15:00:00+00:00", "BatteryStatus1", {"MainBatteryVoltage": 100.0}))
        s = n.ingest(_msg("2026-05-13T15:00:01+00:00", "BatteryStatus3", {"BatteryCurrent": 50.0}))
        self.assertAlmostEqual(s.battery_power_kw, 5.0)
        # a voltage older than the pairing window makes the product meaningless
        s = n.ingest(_msg("2026-05-13T15:05:00+00:00", "BatteryStatus3", {"BatteryCurrent": 60.0}))
        self.assertIsNone(s.battery_power_kw)

    def test_asynchronous_partial_updates_keep_other_fields(self):
        n = TelemetryNormalizer()
        n.ingest(_msg("2026-05-13T15:00:00+00:00", "BatteryStatus1",
                      {"MainBatterySOC": 47, "MainBatteryVoltage": 102.5}))
        s = n.ingest(_msg("2026-05-13T15:00:03+00:00", "BatteryStatus1", {"MainBatteryVoltage": 101.0}))
        self.assertEqual(s.soc_pct, 47.0)                     # untouched by the partial update
        self.assertEqual(s.battery_voltage_v, 101.0)
        self.assertEqual(s.age_of("soc_pct"), 3.0)            # SOC is 3 s older than the latest
        self.assertEqual(s.age_of("battery_voltage_v"), 0.0)
        self.assertEqual(s.message_counts["BatteryStatus1"], 2)

    def test_missing_values_stay_none(self):
        second = self.norm.state("2")                         # synthetic tractor 2: SOC/V/I only
        self.assertIsNotNone(second)
        self.assertEqual(second.mission_id, "64")
        self.assertEqual(second.soc_pct, 80.0)
        self.assertIsNone(second.battery_temp_c)
        self.assertIsNone(second.pto_active)
        self.assertIsNone(second.speed_kmh)
        self.assertIsNone(second.position)
        self.assertEqual(second.wheel_speed_rpm, {})
        self.assertIsNone(self.state.position)                # never fabricated for anyone

    def test_out_of_order_message_is_counted_and_never_moves_time_backwards(self):
        n = TelemetryNormalizer()
        n.ingest(_msg("2026-05-13T15:00:10+00:00", "BatteryStatus1", {"MainBatterySOC": 40}))
        s = n.ingest(_msg("2026-05-13T15:00:05+00:00", "BatteryStatus1",
                          {"MainBatterySOC": 41, "MainBatteryTemp": 30}))
        self.assertEqual(s.out_of_order, 1)
        self.assertEqual(s.timestamp, parse_timestamp("2026-05-13T15:00:10+00:00"))
        self.assertEqual(s.soc_pct, 40.0)                     # the fresher value stays
        self.assertEqual(s.battery_temp_c, 30.0)              # a never-seen field is still learnt

    def test_to_dict_is_json_serialisable(self):
        doc = self.state.to_dict()
        json.dumps(doc)
        self.assertEqual(doc["tractor_id"], "1")
        self.assertTrue(doc["timestamp"].endswith("Z"))
        self.assertIn("soc_pct", doc["observed_at"])


# --------------------------------------------------------------------------- #
#  Replay
# --------------------------------------------------------------------------- #
class TestReplay(unittest.TestCase):
    def setUp(self):
        self.messages = read_csv(FIXTURE)

    def test_burst_replay_preserves_order(self):
        seen = []
        player = ReplayPlayer(self.messages, seen.append, ReplayOptions(speed=0))
        player.start()
        player.join(timeout=10)
        self.assertEqual(player.status()["state"], "finished")
        self.assertEqual([m.record_id for m in seen], [m.record_id for m in self.messages])
        self.assertEqual(player.status()["progress_pct"], 100.0)

    def test_gap_compression_and_timing(self):
        msgs = [_msg("2026-05-13T15:00:00+00:00", "MotorTemp", {"TempT1": 1}),
                _msg("2026-05-13T15:00:01+00:00", "MotorTemp", {"TempT1": 2}),   # 1 s
                _msg("2026-05-13T17:00:01+00:00", "MotorTemp", {"TempT1": 3})]   # 2 h gap
        seen = []
        started = time.time()
        player = ReplayPlayer(msgs, seen.append, ReplayOptions(speed=10, max_gap_s=0.2))
        player.start()
        player.join(timeout=10)
        elapsed = time.time() - started
        self.assertEqual(len(seen), 3)
        self.assertLess(elapsed, 3.0)                          # 0.1 s + 0.2 s cap, not 12 min
        status = player.status()
        self.assertEqual(status["gaps_compressed"], 1)
        self.assertGreater(status["gap_time_skipped_s"], 7000)
        self.assertEqual(status["position"], "2026-05-13T17:00:01.000Z")

    def test_stop_is_responsive(self):
        msgs = [_msg("2026-05-13T15:00:00+00:00", "MotorTemp", {"TempT1": 1}),
                _msg("2026-05-13T16:00:00+00:00", "MotorTemp", {"TempT1": 2})]
        player = ReplayPlayer(msgs, lambda m: None, ReplayOptions(speed=1, max_gap_s=None))
        player.start()
        time.sleep(0.1)
        player.stop()
        player.join(timeout=5)
        self.assertFalse(player.is_alive())
        self.assertEqual(player.status()["state"], "stopped")

    def test_csv_source_history_window_and_subscribe(self):
        src = CsvTelemetrySource(FIXTURE, ReplayOptions(speed=0))
        window = list(src.history(start=parse_timestamp("2026-05-13T00:00:00Z"),
                                  end=parse_timestamp("2026-05-13T23:59:59Z"), tractor_id="1"))
        self.assertTrue(window)
        self.assertTrue(all(m.timestamp.date() == dt.date(2026, 5, 13) for m in window))
        seen = []
        sub = src.subscribe(seen.append)
        sub_thread_done = False
        for _ in range(100):
            if not sub.is_alive():
                sub_thread_done = True
                break
            time.sleep(0.05)
        self.assertTrue(sub_thread_done)
        self.assertEqual(len(seen), 441)
        info = src.describe()
        self.assertEqual(info["kind"], "csv-replay")
        self.assertEqual(info["tractors"], ["1", "2"])
        self.assertEqual(info["missions"], ["63", "64"])
        src.close()

    def test_service_end_to_end(self):
        src = CsvTelemetrySource(FIXTURE, ReplayOptions(speed=0))
        service = TelemetryService(src, tractor_ids={"1": "zetrabot_1"}, nominal_capacity_kwh=44.8,
                                   mode="replay").start()
        for _ in range(100):
            if (service.replay_status() or {}).get("state") == "finished":
                break
            time.sleep(0.05)
        try:
            self.assertEqual(service.replay_status()["state"], "finished")
            states = {s.tractor_id: s for s in service.states()}
            self.assertEqual(set(states), {"1", "2"})
            rows = {t.id: t for t in service.tractor_states()}
            self.assertEqual(set(rows), {"zetrabot_1", "zetrabot_2"})
            self.assertEqual(rows["zetrabot_1"].soc_pct, 18.0)
            self.assertAlmostEqual(rows["zetrabot_1"].energy_kwh, 0.18 * 44.8, places=6)
            self.assertIsNone(rows["zetrabot_1"].position)
            self.assertFalse(rows["zetrabot_1"].charging)
            self.assertFalse(rows["zetrabot_1"].available)    # recording ended -> not present
            dev = service.device_rows()
            self.assertEqual({d["protocol"] for d in dev}, {"csv-replay"})
            self.assertEqual(dev[0]["endpoint"], FIXTURE.name)
            doc = service.document()
            self.assertEqual(doc["state"], "active")
            self.assertEqual(doc["mode"], "replay")
            self.assertEqual(doc["replay"]["state"], "finished")
            self.assertEqual(doc["source"]["kind"], "csv-replay")
            self.assertEqual(doc["tractors"][0]["harvest_id"], "zetrabot_1")
            json.dumps(doc)
        finally:
            service.close()


# --------------------------------------------------------------------------- #
#  HARVEST wiring: FleetRuntime, Diagnostics, HTTP
# --------------------------------------------------------------------------- #
class TestRuntimeIntegration(_NoTelemetryEnv):
    def _wait_finished(self, runtime):
        for _ in range(200):
            status = runtime.telemetry.replay_status() or {}
            if status.get("state") in ("finished", "failed"):
                return status
            time.sleep(0.05)
        return runtime.telemetry.replay_status()

    def test_snapshot_merges_real_tractor_and_diagnostics_report_it(self):
        runtime = FleetRuntime(_telemetry_cfg(speed=0))
        try:
            self.assertIsNotNone(runtime.telemetry)
            self.assertEqual(self._wait_finished(runtime)["state"], "finished")
            snap = runtime.snapshot()
            ids = [t.id for t in snap.tractors]
            self.assertIn("T1", ids)                          # the sim fleet is untouched
            self.assertIn("zetrabot_1", ids)
            self.assertIn("zetrabot_2", ids)
            real = snap.tractor("zetrabot_1")
            self.assertEqual(real.soc_pct, 18.0)
            self.assertIsNone(real.position)
            self.assertEqual(len(snap.chargers), 2)           # nothing else changed
            self.assertEqual(runtime.status()["telemetry"]["source"], "csv-replay")

            rows = runtime.device_diagnostics()
            real_rows = [r for r in rows if r["protocol"] == "csv-replay"]
            self.assertEqual(len(real_rows), 2)
            self.assertTrue(all(r["protocol"] != "sim" for r in real_rows))

            doc = diag.collect(runtime, diag.ClientRegistry())
            services = {s["name"]: s for s in doc["services"]}
            row = services["Real telemetry"]
            self.assertEqual(row["state"], diag.HEALTHY)
            self.assertIn("csv-replay", row["detail"])
            self.assertIn("finished", row["detail"])
            self.assertTrue(any("mission 63" in f for f in row["facts"]))
            self.assertEqual(doc["telemetry"]["state"], "active")
            self.assertEqual(doc["telemetry"]["tractors"][0]["source"], "csv-replay")
            # the existing rows are still there and unchanged in spirit
            self.assertEqual(services["Fleet backend"]["state"], diag.SIMULATED)
        finally:
            runtime.close()

    def test_id_mapping_overlays_an_existing_fleet_tractor(self):
        runtime = FleetRuntime(_telemetry_cfg(speed=0, tractor_ids={"1": "T1"}))
        try:
            self._wait_finished(runtime)
            snap = runtime.snapshot()
            ids = [t.id for t in snap.tractors]
            self.assertEqual(ids.count("T1"), 1)              # replaced, not duplicated
            self.assertEqual(snap.tractor("T1").soc_pct, 18.0)
        finally:
            runtime.close()

    def test_no_source_is_inactive_not_failed(self):
        runtime = FleetRuntime({})
        try:
            self.assertIsNone(runtime.telemetry)
            self.assertEqual(runtime.telemetry_document()["state"], "inactive")
            services = {s["name"]: s for s in diag.collect(runtime, diag.ClientRegistry())["services"]}
            self.assertEqual(services["Real telemetry"]["state"], diag.INACTIVE)
            self.assertIn("run_harvest_dashboard.sh replay", services["Real telemetry"]["detail"])
        finally:
            runtime.close()

    def test_env_overrides_config(self):
        os.environ["HARVEST_TELEMETRY_SOURCE"] = "csv"
        os.environ["HARVEST_TELEMETRY_FILE"] = str(FIXTURE)
        os.environ["HARVEST_TELEMETRY_SPEED"] = "0"
        runtime = FleetRuntime({})
        try:
            self.assertIsNotNone(runtime.telemetry)
            self.assertEqual(runtime.telemetry.source.kind, "csv-replay")
            self.assertEqual(runtime.telemetry.source.replay.speed, 0.0)
        finally:
            runtime.close()

    def test_missing_file_is_reported_as_failed(self):
        cfg = _telemetry_cfg(speed=0)
        cfg["integrations"]["telemetry"]["csv"]["file"] = "/nonexistent/mission.csv"
        runtime = FleetRuntime(cfg)
        try:
            # the service exists but its subscription failed -> failed state, fleet still fine
            doc = runtime.telemetry_document()
            self.assertEqual(doc["state"], "failed")
            self.assertIn("not found", doc["error"])
            self.assertEqual(len(runtime.snapshot().tractors), 3)
            services = {s["name"]: s for s in diag.collect(runtime, diag.ClientRegistry())["services"]}
            self.assertEqual(services["Real telemetry"]["state"], diag.FAILED)
        finally:
            runtime.close()


class TestAwsScaffold(_NoTelemetryEnv):
    def test_no_sdk_imported_and_adapters_are_explicit_about_being_pending(self):
        import harvest_integrations.aws as aws
        self.assertNotIn("boto3", sys.modules)
        self.assertNotIn("awsiot", sys.modules)
        src = aws.build_aws_source({"adapter": "iot-core", "region": "eu-west-1", "secret": "x"})
        self.assertEqual(src.kind, "aws-iot-core")
        self.assertNotIn("secret", src.describe())
        with self.assertRaises(NotImplementedError):
            src.subscribe(lambda m: None)
        with self.assertRaises(NotImplementedError):
            list(src.history())
        with self.assertRaises(ValueError):
            aws.build_aws_source({"adapter": "kinesis"})

    def test_runtime_reports_aws_scaffold_as_failed_source(self):
        cfg = {"integrations": {"telemetry": {"source": "aws", "aws": {"adapter": "s3"}}}}
        runtime = FleetRuntime(cfg)
        try:
            self.assertIsNotNone(runtime.telemetry)
            doc = runtime.telemetry_document()
            self.assertEqual(doc["state"], "failed")
            self.assertIn("not implemented yet", doc["error"])
            self.assertEqual(len(runtime.snapshot().tractors), 3)   # control path unaffected
        finally:
            runtime.close()

    def test_unknown_source_kind_is_reported_not_raised(self):
        runtime = FleetRuntime({"integrations": {"telemetry": {"source": "kafka"}}})
        try:
            self.assertIsNone(runtime.telemetry)
            self.assertIn("Unknown", runtime.telemetry_error)
            self.assertEqual(runtime.telemetry_document()["state"], "failed")
        finally:
            runtime.close()


class TestHttpEndpoints(_NoTelemetryEnv):
    def setUp(self):
        super().setUp()
        import server
        self.server_mod = server
        os.environ["HARVEST_TELEMETRY_SOURCE"] = "csv"
        os.environ["HARVEST_TELEMETRY_FILE"] = str(FIXTURE)
        os.environ["HARVEST_TELEMETRY_SPEED"] = "0"
        if server._FLEET_RUNTIME is not None:
            server._FLEET_RUNTIME.close()
            server._FLEET_RUNTIME = None
        server._TELEMETRY_ANALYSIS.clear()
        self.httpd = HTTPServer(("127.0.0.1", 0), server.Handler)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
        if self.server_mod._FLEET_RUNTIME is not None:
            self.server_mod._FLEET_RUNTIME.close()
            self.server_mod._FLEET_RUNTIME = None
        self.server_mod._TELEMETRY_ANALYSIS.clear()
        super().tearDown()

    def _get(self, path):
        with urllib.request.urlopen(f"{self.base}{path}", timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())

    def test_telemetry_endpoints(self):
        status, doc = self._get("/api/telemetry")
        self.assertEqual(status, 200)
        self.assertEqual(doc["source"]["kind"], "csv-replay")
        for _ in range(200):
            if (doc.get("replay") or {}).get("state") == "finished":
                break
            time.sleep(0.05)
            _, doc = self._get("/api/telemetry")
        self.assertEqual(doc["replay"]["state"], "finished")
        ids = {t["harvest_id"] for t in doc["tractors"]}
        self.assertIn("zetrabot_1", ids)
        _, snap = self._get("/api/fleet/snapshot")
        self.assertIn("zetrabot_1", [t["id"] for t in snap["tractors"]])
        _, analysis = self._get("/api/telemetry/analysis")
        self.assertEqual(analysis["state"], "ok")
        self.assertIn("capacity_estimate", analysis)
        self.assertEqual(analysis["capacity_estimate"]["status"], ESTIMATE_LABEL)
        _, d = self._get("/api/diagnostics")
        self.assertEqual(d["telemetry"]["state"], "active")


# --------------------------------------------------------------------------- #
#  FIWARE mapping
# --------------------------------------------------------------------------- #
class TestFiwareMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src = CsvTelemetrySource(FIXTURE, ReplayOptions(speed=0))
        cls.service = TelemetryService(src, tractor_ids={"1": "zetrabot_1"}, nominal_capacity_kwh=44.8)
        cls.service.start()
        for _ in range(200):
            if (cls.service.replay_status() or {}).get("state") == "finished":
                break
            time.sleep(0.05)
        cls.doc = cls.service.document()
        cls.entities = telemetry_entities(cls.doc)

    @classmethod
    def tearDownClass(cls):
        cls.service.close()

    def _entity(self, hid):
        return next(e for e in self.entities if e["id"] == f"urn:ngsi-ld:TractorTelemetry:{hid}")

    def test_entity_shape(self):
        self.assertEqual(len(self.entities), 2)
        e = self._entity("zetrabot_1")
        self.assertEqual(e["type"], "TractorTelemetry")
        self.assertEqual(e["source"]["value"], "csv-replay")
        self.assertEqual(e["mode"]["value"], "replay")
        self.assertEqual(e["missionId"]["value"], "63")
        self.assertEqual(e["robotTractorId"]["value"], "1")
        self.assertEqual(e["replayState"]["value"], "finished")
        self.assertEqual(e["refElectricTractor"]["object"], "urn:ngsi-ld:ElectricTractor:zetrabot_1")
        self.assertEqual(e["socPct"]["value"], 18.0)
        self.assertEqual(e["socPct"]["unitCode"], "P1")
        self.assertEqual(e["batteryVoltageV"]["unitCode"], "VLT")
        self.assertIn("dischargedEnergyKwh", e)
        self.assertIn("motorTempC", e)
        self.assertIsInstance(e["ptoActive"]["value"], bool)
        json.dumps(self.entities)

    def test_observed_at_is_the_robots_time_per_field(self):
        e = self._entity("zetrabot_1")
        t = self.doc["tractors"][0]
        # each attribute carries the time ITS signal was last observed
        self.assertEqual(e["socPct"]["observedAt"], t["observed_at"]["soc_pct"])
        self.assertEqual(e["batteryCurrentA"]["observedAt"], t["observed_at"]["battery_current_a"])
        self.assertNotEqual(e["socPct"]["observedAt"], e["hydraulicPressureBar"]["observedAt"])
        self.assertTrue(e["socPct"]["observedAt"].startswith("2026-05-14"))
        # and the entity-level facts carry the latest message time
        self.assertEqual(e["messagesIngested"]["observedAt"], t["timestamp"])

    def test_known_raw_only_signals_are_not_flagged_unknown(self):
        e = self._entity("zetrabot_1")
        unknown = e["unknownSignals"]["value"]
        self.assertFalse(any(k.startswith("FunctionStatus.") for k in unknown), unknown)
        self.assertNotIn("MemoryData2.CursorPositionOffset", unknown)
        self.assertIn("functionLeds", e)
        self.assertIn("BhdLed", e["functionLeds"]["value"])

    def test_absent_fields_are_absent_and_no_position(self):
        e2 = self._entity("zetrabot_2")                       # only SOC / V / I were ever seen
        self.assertNotIn("batteryTempC", e2)
        self.assertNotIn("ptoActive", e2)
        self.assertNotIn("speedKmh", e2)
        for e in self.entities:
            self.assertNotIn("position", e)
            self.assertNotIn("location", e)
            self.assertFalse(any(v.get("type") == "GeoProperty" for v in e.values() if isinstance(v, dict)))

    def test_unknown_signals_travel_to_the_broker(self):
        e = self._entity("zetrabot_1")
        self.assertEqual(e["unknownSignals"]["value"]["BatteryStatus1.FutureSignalX"], 42)

    def test_sync_mirrors_nothing_when_inactive(self):
        class _Harvest:
            def telemetry(self):
                return {"state": "inactive", "tractors": []}
        sync = ContextSync(_Harvest(), broker=None)
        self.assertEqual(sync._telemetry_entities(), [])

        class _Harvest2:
            def telemetry(self):
                return TestFiwareMapping.doc
        self.assertEqual(len(ContextSync(_Harvest2(), broker=None)._telemetry_entities()), 2)

        class _Down:
            def telemetry(self):
                raise OSError("connection refused")
        self.assertEqual(ContextSync(_Down(), broker=None)._telemetry_entities(), [])


# --------------------------------------------------------------------------- #
#  Analysis
# --------------------------------------------------------------------------- #
class TestAnalysisFixture(unittest.TestCase):
    def test_fixture_analysis_is_defensible(self):
        analysis = MissionAnalysis(read_csv(FIXTURE), harvest_model={"battery_capacity_kwh": 44.8,
                                                                     "pto_power_kw": 10.0},
                                   tractor_id="1")
        r = analysis.run()
        self.assertEqual(r["mission"]["mission_ids"], ["63"])
        self.assertEqual(r["soc"]["first_pct"], 64.0)
        self.assertEqual(r["soc"]["last_pct"], 18.0)
        self.assertEqual(r["soc"]["delta_pct"], 46.0)
        d = r["discharged_energy"]
        self.assertGreaterEqual(d["sum_of_sessions_kwh"], d["largest_session_kwh"])
        self.assertGreaterEqual(r["mission"]["power_sessions"], 2)
        self.assertEqual(r["capacity_estimate"]["status"], ESTIMATE_LABEL)
        self.assertEqual(r["capacity_estimate"]["nominal_config_kwh"], 44.8)
        params = [row["parameter"] for row in r["harvest_model_comparison"]]
        self.assertIn("battery_capacity_kwh", params)
        self.assertIn("pto_power_kw", params)
        self.assertTrue(any("GPS" in n for n in r["not_available"]))
        text = analysis.text()
        self.assertIn("ESTIMATE", text)
        json.dumps(r)

    def test_empty_input(self):
        self.assertIn("error", MissionAnalysis([]).run())


@unittest.skipUnless(REAL.exists(), "full mission export not present (git-ignored)")
class TestRealMission(unittest.TestCase):
    """Cross-checks against the Zetrack V2 report (2026-09-21) for mission 63."""

    @classmethod
    def setUpClass(cls):
        cls.stats = CsvParseStats()
        cls.messages = read_csv(REAL, cls.stats)
        cls.result = MissionAnalysis(cls.messages, harvest_model={"battery_capacity_kwh": 44.8}).run()

    def test_parse_whole_export(self):
        self.assertEqual(self.stats.rows, 56428)
        self.assertEqual(self.stats.invalid_signals_json, 0)
        self.assertEqual(len(self.messages), 56428)
        ts = [m.timestamp for m in self.messages]
        self.assertEqual(ts, sorted(ts))
        self.assertEqual(len({m.source_message for m in self.messages}), 14)

    def test_figures_match_the_zetrack_report_where_comparable(self):
        r = self.result
        self.assertEqual((r["soc"]["first_pct"], r["soc"]["last_pct"]), (64.0, 18.0))     # report: 64 / 18
        self.assertAlmostEqual(r["discharged_energy"]["largest_session_kwh"], 6.43, places=2)  # report: 6.43
        self.assertAlmostEqual(r["distance"]["delta_km"], 3.70, places=1)                # report: 3.70 km
        self.assertAlmostEqual(r["speed_kmh"]["mean"], 0.753, places=2)                  # report: 0.753
        self.assertEqual(r["speed_kmh"]["max"], 8.0)                                     # report: 8
        self.assertAlmostEqual(r["regimes"]["pto_active"]["hours"], 2.50, delta=0.05)    # report: 2.50 h
        self.assertAlmostEqual(r["regimes"]["moving_speed_gt_0"]["hours"], 2.55, delta=0.05)  # report: 2.55 h
        self.assertEqual(r["power"]["peak_current_a"], 295.094)                          # report max 295 A

    def test_energy_reconstruction_is_self_consistent(self):
        r = self.result
        total = r["discharged_energy"]["sum_of_sessions_kwh"]
        self.assertAlmostEqual(total, 20.06, places=1)
        # independent V x I integral agrees with the counter within 5 %
        self.assertAlmostEqual(r["power"]["integrated_kwh"] / total, 1.0, delta=0.05)
        est = r["capacity_estimate"]["effective_capacity_kwh"]
        self.assertAlmostEqual(est, 43.6, delta=0.3)
        self.assertEqual(r["capacity_estimate"]["status"], ESTIMATE_LABEL)

    def test_normaliser_over_full_mission(self):
        norm = TelemetryNormalizer()
        norm.ingest_all(self.messages)
        s = norm.state("1")
        self.assertEqual(s.messages, 56428)
        self.assertEqual(s.out_of_order, 0)
        self.assertEqual(s.unknown_messages, {})
        self.assertEqual(s.session_resets, 7)
        self.assertAlmostEqual(s.discharged_energy_kwh, 20.06, places=1)
        self.assertEqual(s.soc_pct, 18.0)
        self.assertIsNone(s.position)


if __name__ == "__main__":
    unittest.main()
