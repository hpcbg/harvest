"""Mission KPIs, provenance and real-vs-model validation (telemetry/kpi.py).

The synthetic mission below is small enough to compute by hand::

    t [s]      state (V = 100 V throughout)                  regime
    0..600     PTO off, speed 0, I 10 A  -> 1 kW             stationary_no_pto
    600..1800  PTO on,  speed 6, I 50 A  -> 5 kW             pto_moving   (2 km)
    1800..2700 PTO off, speed 4, I 20 A  -> 2 kW             moving_no_pto (1 km)
    2700..3600 PTO off, speed 0, I 10 A  -> 1 kW             stationary_no_pto
    3600 ..+3h power-off gap (SOC 70 -> 69)

    energy  = 1/6 + 5/3 + 1/2 + 1/4          = 2.58333 kWh
    LifetimeKm 100.0 -> 103.0                = 3.0 km (= speed integral)
    SOC 80 -> 69                             = 11 pts
    model   = 10 kW x 1/3 h + 0.5 kWh/km x 3 km + 0.3 kW x 5/12 h = 4.95833 kWh

A heartbeat record every 60 s keeps every interval under the 120 s gap cap.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import HTTPServer
from pathlib import Path

from harvest_integrations.telemetry import TelemetryMessage, read_csv
from harvest_integrations.telemetry.analysis import MissionAnalysis
from harvest_integrations.telemetry.kpi import (
    CONFIGURED, DERIVED, ESTIMATED, MEASURED, MISSING, PROVENANCE, build_mission_kpis,
    export_csv, export_json, kpis_from_csv, mission_kpis,
)
from harvest_integrations.telemetry.model import FIELD_PROVENANCE
from harvest_integrations.telemetry.service import TelemetryService
from harvest_integrations.telemetry.zetrack import CsvTelemetrySource

from tests.test_telemetry import FIXTURE, REAL, REPO, _NoTelemetryEnv

T0 = dt.datetime(2026, 5, 12, 10, 0, 0, tzinfo=dt.timezone.utc)
MODEL = {"battery_capacity_kwh": 20.0, "pto_power_kw": 10.0, "driving_kwh_per_km": 0.5,
         "idle_kwh_per_h": 0.3, "max_power_kw": 45.2, "eco_speed_kmh": 10}
ENERGY = 1 / 6 + 5 / 3 + 1 / 2 + 1 / 4
MODEL_ENERGY = 10 * (1 / 3) + 0.5 * 3.0 + 0.3 * (5 / 12)


class _Seq:
    def __init__(self):
        self.n = 0

    def __call__(self, t_s, message, signals, tractor="7", mission="m1"):
        self.n += 1
        return TelemetryMessage(tractor_id=tractor, timestamp=T0 + dt.timedelta(seconds=t_s),
                                source_message=message, signals=signals, mission_id=mission,
                                sequence=self.n, record_id=str(self.n), source="unit-test")


def synthetic_mission(dropout=True):
    m = _Seq()
    msgs = [
        m(0, "FunctionStatus", {"PtoLed": 0, "GoLed": 1}),
        m(0, "MiscInfo", {"SpeedDisplay": 0, "LifetimeKm": 100.0}),
        m(0, "BatteryStatus1", {"MainBatterySOC": 80, "MainBatteryVoltage": 100, "MainBatteryTemp": 30}),
        m(0, "BatteryStatus3", {"BatteryCurrent": 10, "DischEnrgActualSesion": 0.0}),
        m(0, "MotorTemp", {"TempT1": 40, "TempT3": 55, "TempPTO": 33}),
        m(600, "FunctionStatus", {"PtoLed": 1}),
        m(600, "MiscInfo", {"SpeedDisplay": 6}),
        m(600, "BatteryStatus3", {"BatteryCurrent": 50}),
        m(1800, "FunctionStatus", {"PtoLed": 0}),
        m(1800, "MiscInfo", {"SpeedDisplay": 4}),
        m(1800, "BatteryStatus3", {"BatteryCurrent": 20}),
        m(2700, "MiscInfo", {"SpeedDisplay": 0}),
        m(2700, "BatteryStatus3", {"BatteryCurrent": 10}),
        m(3000, "BatteryStatus1", {"MainBatteryTemp": 35}),
        m(3540, "BatteryStatus3", {"DischEnrgActualSesion": 2.5}),
        m(3600, "MiscInfo", {"LifetimeKm": 103.0}),
        m(3600, "BatteryStatus1", {"MainBatterySOC": 70}),
        m(3600 + 3 * 3600, "BatteryStatus1", {"MainBatterySOC": 69}),
    ]
    if dropout:
        msgs.append(m(1200, "BatteryStatus1", {"MainBatteryVoltage": 0, "MainBatteryTemp": 0}))
    for t in range(60, 3600, 60):
        msgs.append(m(t, "LimitsStatus", {}))
    return msgs


def kpis(messages, model=MODEL):
    return mission_kpis(messages, model, source={"kind": "unit-test"})


# --------------------------------------------------------------------------- #
#  KPI calculations
# --------------------------------------------------------------------------- #
class TestKpiCalculations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = kpis(synthetic_mission())
        cls.k = {key: item["value"] for key, item in cls.doc["kpis"].items()}

    def test_state_and_identity(self):
        self.assertEqual(self.doc["state"], "ok")
        self.assertEqual(self.k["mission_id"], "m1")
        self.assertEqual(self.k["tractor_id"], "7")

    def test_time_and_distance(self):
        self.assertAlmostEqual(self.k["duration_h"], 1.0, places=2)
        self.assertAlmostEqual(self.k["wall_span_h"], 4.0, places=2)
        self.assertAlmostEqual(self.k["distance_km"], 3.0, places=2)
        self.assertAlmostEqual(self.k["moving_h"], 7 / 12, places=2)
        self.assertAlmostEqual(self.k["pto_active_h"], 1 / 3, places=2)
        self.assertAlmostEqual(self.k["pto_utilisation_pct"], 100 / 3, places=1)

    def test_energy(self):
        self.assertAlmostEqual(self.k["energy_kwh"], ENERGY, places=3)
        self.assertAlmostEqual(self.k["energy_counter_kwh"], 2.5, places=3)
        self.assertAlmostEqual(self.k["mean_power_kw"], ENERGY, places=2)
        self.assertAlmostEqual(self.k["peak_power_kw"], 5.0, places=2)
        self.assertAlmostEqual(self.k["energy_per_km"], ENERGY / 3.0, places=2)
        self.assertAlmostEqual(self.k["energy_per_moving_h"], ENERGY / (7 / 12), places=2)
        self.assertAlmostEqual(self.k["energy_pto_kwh"], 5 / 3, places=2)
        self.assertAlmostEqual(self.k["energy_moving_kwh"], 5 / 3 + 1 / 2, places=2)
        self.assertAlmostEqual(self.k["energy_idle_kwh"], 1 / 6 + 1 / 4, places=3)
        self.assertEqual(self.k["soc_delta_pct"], 11.0)
        self.assertAlmostEqual(self.k["estimated_remaining_energy_kwh"], 0.69 * 20.0, places=2)
        self.assertEqual(self.k["peak_current_a"], 50.0)

    def test_dropout_is_rejected_not_integrated(self):
        # the 0 V reading at t=1200 would otherwise zero the power for 60 s
        with_dropout = self.k["energy_kwh"]
        without = kpis(synthetic_mission(dropout=False))["kpis"]["energy_kwh"]["value"]
        self.assertAlmostEqual(with_dropout, without, places=6)
        flags = self.doc["kpis"]["energy_kwh"]["flags"]
        self.assertTrue(any("dropout" in f for f in flags))
        self.assertEqual(self.k["battery_temp_max_c"], 35.0)
        self.assertEqual(self.k["battery_temp_mean_c"], 32.5)     # the 0 degC reading is excluded

    def test_model_validation(self):
        self.assertAlmostEqual(self.k["configured_capacity_kwh"], 20.0)
        self.assertAlmostEqual(self.k["estimated_capacity_kwh"], ENERGY / 0.11, places=1)
        self.assertAlmostEqual(self.k["model_energy_kwh"], MODEL_ENERGY, places=2)
        self.assertAlmostEqual(self.k["energy_error_kwh"], MODEL_ENERGY - ENERGY, places=2)
        self.assertAlmostEqual(self.k["energy_error_pct"], (MODEL_ENERGY - ENERGY) / ENERGY * 100, places=0)
        self.assertAlmostEqual(self.k["model_soc_delta_pct"], MODEL_ENERGY / 20 * 100, places=1)
        self.assertAlmostEqual(self.k["soc_error_pct_points"], MODEL_ENERGY / 20 * 100 - 11, places=1)
        self.assertAlmostEqual(self.k["counter_vs_vxi_pct"], (2.5 - ENERGY) / ENERGY * 100, places=1)

    def test_operating_state_shares_sum_to_100(self):
        shares = self.k["operating_state_shares_pct"]
        self.assertAlmostEqual(sum(v for v in shares.values() if v), 100.0, delta=0.2)
        self.assertEqual(shares["unclassified"], 0.0)


# --------------------------------------------------------------------------- #
#  Provenance
# --------------------------------------------------------------------------- #
class TestProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = kpis(synthetic_mission())
        cls.p = {key: item["provenance"] for key, item in cls.doc["kpis"].items()}

    def test_every_kpi_is_classified_with_a_method(self):
        self.assertEqual(set(self.doc["provenance_classes"]), set(PROVENANCE))
        for key, item in self.doc["kpis"].items():
            self.assertIn(item["provenance"], PROVENANCE, key)
            self.assertTrue(item["method"], key)
            # value None <=> MISSING: a missing value is never shown as a number
            self.assertEqual(item["value"] is None, item["provenance"] == MISSING, key)

    def test_expected_classes(self):
        expect = {
            "soc_initial_pct": MEASURED, "soc_final_pct": MEASURED, "peak_current_a": MEASURED,
            "battery_temp_max_c": MEASURED,
            "energy_kwh": DERIVED, "soc_delta_pct": DERIVED, "mean_power_kw": DERIVED,
            "distance_km": DERIVED, "energy_per_km": DERIVED,
            "configured_capacity_kwh": CONFIGURED,
            "estimated_capacity_kwh": ESTIMATED, "estimated_remaining_energy_kwh": ESTIMATED,
            "model_energy_kwh": ESTIMATED, "energy_counter_kwh": ESTIMATED,
            "position": MISSING, "charging_state": MISSING,
        }
        for key, cls in expect.items():
            self.assertEqual(self.p[key], cls, key)

    def test_remaining_energy_is_labelled_as_configured_dependent(self):
        item = self.doc["kpis"]["estimated_remaining_energy_kwh"]
        self.assertEqual(item["label"], "Estimated remaining energy")
        self.assertIn("configured capacity", item["method"])
        self.assertIn("config.yaml tractors.model.battery_capacity_kwh", item["sources"])

    def test_session_counter_is_flagged_unconfirmed(self):
        item = self.doc["kpis"]["energy_counter_kwh"]
        self.assertTrue(any("NOT confirmed" in f for f in item["flags"]))
        ids = {lim["id"] for lim in self.doc["limitations"]}
        self.assertIn("disch_counter_semantics_unconfirmed", ids)

    def test_summary_carries_provenance(self):
        self.assertEqual(set(self.doc["summary_provenance"]) - set(self.doc["summary"]), set())
        self.assertEqual(self.doc["summary_provenance"]["model_energy_kwh"], ESTIMATED)

    def test_live_document_field_provenance(self):
        self.assertEqual(FIELD_PROVENANCE["soc_pct"], MEASURED)
        self.assertEqual(FIELD_PROVENANCE["battery_power_kw"], DERIVED)
        self.assertEqual(FIELD_PROVENANCE["estimated_remaining_energy_kwh"], ESTIMATED)
        self.assertEqual(FIELD_PROVENANCE["position"], MISSING)
        service = TelemetryService(CsvTelemetrySource(FIXTURE), nominal_capacity_kwh=44.8)
        service.normalizer.ingest_all(read_csv(FIXTURE))
        doc = service.document()
        self.assertEqual(doc["provenance"]["discharged_energy_kwh"], ESTIMATED)
        t = next(x for x in doc["tractors"] if x["tractor_id"] == "1")
        self.assertNotIn("energy_kwh_nominal", t)
        self.assertAlmostEqual(t["estimated_remaining_energy_kwh"], t["soc_pct"] / 100 * 44.8, places=2)


# --------------------------------------------------------------------------- #
#  Missing data, division by zero, incomplete missions, async timestamps
# --------------------------------------------------------------------------- #
class TestMissingAndEdgeCases(unittest.TestCase):
    def test_empty_input(self):
        doc = kpis([])
        self.assertEqual(doc["state"], "empty")
        self.assertEqual(doc["kpis"], {})
        ids = {lim["id"] for lim in doc["limitations"]}
        self.assertIn("aws_not_implemented", ids)
        json.dumps(doc)

    def test_single_message(self):
        m = _Seq()
        doc = kpis([m(0, "BatteryStatus1", {"MainBatterySOC": 50})])
        k = doc["kpis"]
        self.assertEqual(k["soc_initial_pct"]["value"], 50.0)
        self.assertEqual(k["soc_delta_pct"]["value"], 0.0)
        for key in ("energy_kwh", "mean_power_kw", "energy_per_km", "energy_per_moving_h",
                    "estimated_capacity_kwh", "distance_km", "moving_h", "pto_active_h",
                    "model_energy_kwh", "energy_error_pct"):
            self.assertEqual(k[key]["provenance"], MISSING, key)
            self.assertIsNone(k[key]["value"], key)
        self.assertIn("SOC did not decrease", " ".join(k["estimated_capacity_kwh"]["flags"]))
        json.dumps(doc)

    def test_no_soc_no_distance_signals(self):
        import dataclasses
        msgs = [dataclasses.replace(m, signals={k: v for k, v in m.signals.items()
                                                if k not in ("MainBatterySOC", "LifetimeKm")})
                for m in synthetic_mission()]
        doc = kpis(msgs)
        k = doc["kpis"]
        self.assertEqual(k["soc_delta_pct"]["provenance"], MISSING)
        self.assertEqual(k["estimated_capacity_kwh"]["provenance"], MISSING)
        self.assertEqual(k["estimated_remaining_energy_kwh"]["provenance"], MISSING)
        self.assertEqual(k["soc_error_pct_points"]["provenance"], MISSING)
        # distance falls back to the speed integral -- and says so
        self.assertEqual(k["distance_km"]["provenance"], ESTIMATED)
        self.assertAlmostEqual(k["distance_km"]["value"], 3.0, places=2)
        self.assertEqual(k["energy_kwh"]["provenance"], DERIVED)

    def test_zero_moving_time_and_zero_distance(self):
        m = _Seq()
        msgs = [m(0, "FunctionStatus", {"PtoLed": 0}), m(0, "MiscInfo", {"SpeedDisplay": 0, "LifetimeKm": 5.0}),
                m(0, "BatteryStatus1", {"MainBatteryVoltage": 100, "MainBatterySOC": 60}),
                m(0, "BatteryStatus3", {"BatteryCurrent": 5})]
        msgs += [m(t, "LimitsStatus", {}) for t in range(60, 1860, 60)]
        msgs.append(m(1800, "MiscInfo", {"LifetimeKm": 5.0}))
        doc = kpis(msgs)
        k = doc["kpis"]
        self.assertEqual(k["moving_h"]["value"], 0.0)
        self.assertEqual(k["distance_km"]["value"], 0.0)
        self.assertEqual(k["energy_per_km"]["provenance"], MISSING)
        self.assertEqual(k["energy_per_moving_h"]["provenance"], MISSING)
        rows = {r["metric"]: r for r in doc["comparison"]["rows"]}
        self.assertEqual(rows["Energy per km"]["status"], "unavailable")
        self.assertAlmostEqual(k["model_energy_kwh"]["value"], 0.3 * 0.5, places=3)   # idle only
        json.dumps(doc)

    def test_asynchronous_start_is_unclassified_and_power_unknown(self):
        m = _Seq()
        msgs = [m(0, "BatteryStatus1", {"MainBatteryVoltage": 100}),
                m(30, "BatteryStatus3", {"BatteryCurrent": 10}),         # I arrives 30 s after V
                m(60, "FunctionStatus", {"PtoLed": 0}),                   # PTO state at 60 s
                m(90, "MiscInfo", {"SpeedDisplay": 0}),                   # speed at 90 s
                m(150, "LimitsStatus", {})]
        r = MissionAnalysis(msgs).run()
        self.assertAlmostEqual(r["quality"]["power_unknown_h"], 30 / 3600, places=3)
        ex = r["exclusive_regimes"]
        self.assertAlmostEqual(ex["unclassified"]["hours"], 90 / 3600, places=3)
        self.assertAlmostEqual(ex["stationary_no_pto"]["hours"], 60 / 3600, places=3)
        # energy: 1 kW from 30 s to 150 s only (no invented power before I arrived)
        self.assertAlmostEqual(r["power"]["integrated_kwh"], 120 / 3600, places=3)
        doc = build_mission_kpis(r, MODEL)
        self.assertTrue(any("unclassified" in f for f in doc["kpis"]["model_energy_kwh"]["flags"]))

    def test_unsorted_input_gives_same_result(self):
        msgs = synthetic_mission()
        a = kpis(msgs)["summary"]
        b = kpis(list(reversed(msgs)))["summary"]
        self.assertEqual(a, b)

    def test_missing_model_parameters(self):
        doc = kpis(synthetic_mission(), model={"battery_capacity_kwh": 20.0})
        k = doc["kpis"]
        self.assertEqual(k["model_energy_kwh"]["provenance"], MISSING)
        self.assertIn("not configured", " ".join(k["model_energy_kwh"]["flags"]))
        self.assertEqual(k["configured_capacity_kwh"]["value"], 20.0)
        doc = kpis(synthetic_mission(), model={})
        self.assertEqual(doc["kpis"]["configured_capacity_kwh"]["provenance"], MISSING)
        self.assertEqual(doc["calibration_proposals"], [])


# --------------------------------------------------------------------------- #
#  Real vs model, findings, proposals, limitations
# --------------------------------------------------------------------------- #
class TestComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = kpis(synthetic_mission())
        cls.rows = {r["metric"]: r for r in cls.doc["comparison"]["rows"]}

    def test_sign_convention_and_values(self):
        total = self.rows["Total energy"]
        self.assertAlmostEqual(total["real"], ENERGY, places=2)
        self.assertAlmostEqual(total["model"], MODEL_ENERGY, places=2)
        self.assertAlmostEqual(total["difference"], MODEL_ENERGY - ENERGY, places=2)
        self.assertGreater(total["error_pct"], 0)          # model predicts more
        pto = self.rows["PTO-active consumption"]
        self.assertAlmostEqual(pto["model"], 10 / 3 + 0.5 * 2.0, places=2)
        self.assertAlmostEqual(pto["real"], 5 / 3, places=2)
        drive = self.rows["Driving without PTO"]
        self.assertAlmostEqual(drive["model"], 0.5 * 1.0, places=2)
        idle = self.rows["Idle (stationary, PTO off)"]
        self.assertAlmostEqual(idle["model"], 0.3 * 5 / 12, places=3)

    def test_unavailable_metrics_are_not_forced(self):
        for metric in ("Peak power", "Charging behaviour"):
            row = self.rows[metric]
            self.assertEqual(row["status"], "unavailable", metric)
            self.assertIsNone(row["model"], metric)
            self.assertIsNone(row["error_pct"], metric)
        self.assertEqual(self.rows["Battery capacity"]["model_provenance"], CONFIGURED)

    def test_powered_off_gap(self):
        row = self.rows["SOC change while powered off"]
        self.assertEqual(row["real"], 1.0)
        self.assertAlmostEqual(row["model"], 0.3 * 3 / 20 * 100, places=1)
        self.assertIn("upper bound", row["flags"])

    def test_findings_are_rule_based(self):
        levels = {"ok", "info", "warning", "limitation"}
        ids = {f["id"]: f for f in self.doc["findings"]}
        for f in self.doc["findings"]:
            self.assertIn(f["level"], levels)
        self.assertEqual(ids["model_energy"]["level"], "warning")         # +92 % > 25 %
        self.assertIn("higher than the measured", ids["model_energy"]["text"])
        self.assertIn("single_mission", ids)
        self.assertIn("gps", ids)
        self.assertIn("charging", ids)
        # deterministic: same input, same statements
        again = [f["text"] for f in kpis(synthetic_mission())["findings"]]
        self.assertEqual(again, [f["text"] for f in self.doc["findings"]])

    def test_proposals_are_never_applied(self):
        props = self.doc["calibration_proposals"]
        self.assertTrue(props)
        for p in props:
            self.assertIn("NOT applied", p["status"])
            self.assertTrue(p["parameter"].startswith("tractors.model."))
        self.assertIn("in-sample", self.doc["calibration_note"])
        # capacity estimate 23.5 vs configured 20 -> >10 % -> proposed
        self.assertIn("tractors.model.battery_capacity_kwh", {p["parameter"] for p in props})

    def test_capacity_consistent_within_soc_band_is_not_proposed(self):
        model = dict(MODEL, battery_capacity_kwh=round(ENERGY / 0.11, 2))
        doc = kpis(synthetic_mission(), model=model)
        ids = {f["id"]: f for f in doc["findings"]}
        self.assertEqual(ids["capacity"]["level"], "ok")
        self.assertIn("consistent", ids["capacity"]["text"])
        self.assertNotIn("tractors.model.battery_capacity_kwh",
                         {p["parameter"] for p in doc["calibration_proposals"]})

    def test_explicit_limitation_flags(self):
        ids = {lim["id"]: lim["scope"] for lim in self.doc["limitations"]}
        for lid in ("single_mission", "no_gps", "no_charging_state", "no_charging_cycle"):
            self.assertEqual(ids.get(lid), "data", lid)
        for lid in ("aws_not_implemented", "no_direct_control", "deviceio_optional", "isaac_proxy",
                    "capacity_semantics_unconfirmed", "disch_counter_semantics_unconfirmed"):
            self.assertEqual(ids.get(lid), "deployment", lid)

    def test_charging_cycle_detected_removes_the_flag(self):
        msgs = synthetic_mission()
        msgs.append(_Seq()(3600 + 4 * 3600, "BatteryStatus1", {"MainBatterySOC": 90}))
        ids = {lim["id"] for lim in kpis(msgs)["limitations"]}
        self.assertNotIn("no_charging_cycle", ids)
        self.assertIn("no_charging_state", ids)


# --------------------------------------------------------------------------- #
#  Export
# --------------------------------------------------------------------------- #
class TestExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = mission_kpis(synthetic_mission(), MODEL,
                               source={"kind": "unit-test", "sha256": "abc123"})

    def test_json_round_trip(self):
        back = json.loads(export_json(self.doc))
        self.assertEqual(back["schema"], "harvest-mission-kpis/1.0")
        self.assertEqual(back["summary"]["mission_id"], "m1")
        for key in ("mission_id", "tractor_id", "distance_km", "duration_h", "moving_h", "pto_active_h",
                    "soc_initial_pct", "soc_final_pct", "soc_delta_pct", "energy_kwh", "mean_power_kw",
                    "peak_power_kw", "energy_per_km", "energy_per_moving_h", "configured_capacity_kwh",
                    "estimated_capacity_kwh", "model_energy_kwh", "energy_error_pct"):
            self.assertIn(key, back["summary"])
        self.assertIn("analysis_version", back)
        self.assertEqual(back["input"]["sha256"], "abc123")

    def test_long_csv_is_traceable(self):
        rows = list(csv.DictReader(io.StringIO(export_csv(self.doc))))
        self.assertEqual(len(rows), len(self.doc["kpis"]))
        by_key = {r["key"]: r for r in rows}
        e = by_key["energy_kwh"]
        self.assertEqual(e["provenance"], DERIVED)
        self.assertIn("BatteryStatus3.BatteryCurrent", e["sources"])
        self.assertIn("V x I", e["method"])
        self.assertEqual(e["input_sha256"], "abc123")
        self.assertEqual(by_key["position"]["value"], "")
        self.assertEqual(by_key["position"]["provenance"], MISSING)

    def test_wide_csv_one_row_per_mission(self):
        rows = list(csv.DictReader(io.StringIO(export_csv(self.doc, wide=True))))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mission_id"], "m1")
        self.assertEqual(rows[0]["energy_kwh__provenance"], DERIVED)
        self.assertEqual(rows[0]["model_energy_kwh__provenance"], ESTIMATED)

    def test_cli_and_fixture(self):
        from harvest_integrations.telemetry import kpi
        with tempfile.TemporaryDirectory() as tmp:
            out_json, out_csv = Path(tmp) / "k.json", Path(tmp) / "k.csv"
            import contextlib
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                rc = kpi.main([str(FIXTURE), "--config", str(REPO / "config.yaml"),
                               "--json", str(out_json), "--csv-out", str(out_csv), "--tractor", "1"])
            self.assertEqual(rc, 0)
            self.assertIn("REAL vs HARVEST MODEL", buf.getvalue())
            doc = json.loads(out_json.read_text())
            self.assertEqual(doc["summary"]["mission_id"], "63")
            self.assertEqual(len(doc["input"]["sha256"]), 64)
            self.assertTrue(out_csv.read_text().startswith("mission_id,tractor_id,key"))


# --------------------------------------------------------------------------- #
#  API / UI
# --------------------------------------------------------------------------- #
class TestKpiHttp(_NoTelemetryEnv):
    def setUp(self):
        super().setUp()
        import server
        self.server = server
        self._saved_cfg = server.CONFIG_FILE
        if server._FLEET_RUNTIME is not None:
            server._FLEET_RUNTIME.close()
            server._FLEET_RUNTIME = None
        server._TELEMETRY_KPIS.clear()
        self.httpd = HTTPServer(("127.0.0.1", 0), server.Handler)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.server.CONFIG_FILE = self._saved_cfg
        if self.server._FLEET_RUNTIME is not None:
            self.server._FLEET_RUNTIME.close()
            self.server._FLEET_RUNTIME = None
        self.server._TELEMETRY_KPIS.clear()
        super().tearDown()

    def _get(self, path):
        with urllib.request.urlopen(f"{self.base}{path}", timeout=30) as resp:
            return resp.status, resp.headers, resp.read().decode()

    def test_replay_source_endpoints(self):
        os.environ["HARVEST_TELEMETRY_SOURCE"] = "csv"
        os.environ["HARVEST_TELEMETRY_FILE"] = str(FIXTURE)
        os.environ["HARVEST_TELEMETRY_SPEED"] = "0"
        status, _, body = self._get("/api/telemetry/kpis")
        self.assertEqual(status, 200)
        doc = json.loads(body)
        self.assertEqual(doc["state"], "ok")
        self.assertEqual(doc["mode"], "replay")
        self.assertEqual(doc["input"]["file"], FIXTURE.name)
        self.assertEqual(len(doc["input"]["sha256"]), 64)
        self.assertEqual(doc["summary"]["configured_capacity_kwh"], 44.8)
        _, headers, body = self._get("/api/telemetry/kpis.csv")
        self.assertIn("attachment", headers["Content-Disposition"])
        self.assertTrue(body.startswith("mission_id,tractor_id,key"))
        _, headers, body = self._get("/api/telemetry/kpis.csv?format=wide")
        self.assertIn("_wide.csv", headers["Content-Disposition"])
        _, headers, body = self._get("/api/telemetry/kpis.json")
        self.assertIn(".json", headers["Content-Disposition"])
        self.assertEqual(json.loads(body)["schema"], "harvest-mission-kpis/1.0")

    def test_offline_analysis_of_configured_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.yaml"
            cfg.write_text(json.dumps({
                "tractors": {"model": MODEL},
                "integrations": {"telemetry": {"source": "none", "csv": {"file": str(FIXTURE)}}}}))
            self.server.CONFIG_FILE = cfg
            doc = self.server._telemetry_kpis(None, refresh=True)
            self.assertEqual(doc["state"], "ok")
            self.assertEqual(doc["mode"], "offline")
            self.assertEqual(doc["config"]["tractors_model"]["battery_capacity_kwh"], 20.0)
            cfg.write_text(json.dumps({"integrations": {"telemetry": {
                "source": "none", "csv": {"file": str(Path(tmp) / "absent.csv")}}}}))
            doc = self.server._telemetry_kpis(None, refresh=True)
            self.assertEqual(doc["state"], "inactive")
            self.assertIn("not present", doc["error"])

    def test_dashboard_renders_kpi_view(self):
        html = (REPO / "dashboard.html").read_text(encoding="utf-8")
        self.assertIn("/api/telemetry/kpis", html)
        self.assertIn("Real vs HARVEST model", html)
        self.assertIn("Estimated remaining energy", html)
        self.assertNotIn("Energy at nominal capacity", html)
        for cls in PROVENANCE:
            self.assertIn(f".pv-{cls}", html)


# --------------------------------------------------------------------------- #
#  The real mission (git-ignored export)
# --------------------------------------------------------------------------- #
@unittest.skipUnless(REAL.exists(), "full mission export not present (git-ignored)")
class TestRealMissionKpis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = kpis_from_csv(REAL, REPO / "config.yaml")
        cls.s = cls.doc["summary"]

    def test_headline_figures(self):
        s = self.s
        self.assertEqual((s["mission_id"], s["tractor_id"]), ("63", "1"))
        self.assertAlmostEqual(s["energy_kwh"], 19.93, delta=0.05)
        self.assertAlmostEqual(s["energy_counter_kwh"], 20.06, delta=0.05)
        self.assertEqual(s["soc_delta_pct"], 46.0)
        self.assertAlmostEqual(s["distance_km"], 3.70, delta=0.05)
        self.assertAlmostEqual(s["pto_active_h"], 2.50, delta=0.05)
        self.assertAlmostEqual(s["moving_h"], 2.55, delta=0.05)
        self.assertAlmostEqual(s["estimated_capacity_kwh"], 43.3, delta=0.3)
        self.assertAlmostEqual(s["model_energy_kwh"], 27.37, delta=0.1)
        self.assertAlmostEqual(s["energy_error_pct"], 37.4, delta=0.5)

    def test_real_mission_limitations(self):
        ids = {lim["id"] for lim in self.doc["limitations"]}
        self.assertTrue({"single_mission", "no_gps", "no_charging_state", "no_charging_cycle"} <= ids)
        self.assertEqual(self.doc["kpis"]["position"]["provenance"], MISSING)


if __name__ == "__main__":
    unittest.main()
