"""
ZETRABOT / Zetrack V2 telemetry dictionary and CSV source.

THE EXPORT FORMAT (documented from ``telemetry/telemetria_mision_63.csv``)
--------------------------------------------------------------------------
One CSV row per *asynchronous message*, not one row per time step::

    id, user_id, tractor_id, mission_id, timestamp, sequence,
    schema_version, source_message, signals, created_at

* ``id``            -- database row id, unique, roughly monotonic in ingest order;
* ``user_id``       -- operator account (opaque; preserved in ``extra``, never displayed);
* ``tractor_id``    -- the robot (``1`` in the supplied mission);
* ``mission_id``    -- the mission (``63``);
* ``timestamp``     -- the robot's own time, ISO-8601 with ``+00:00``, millisecond
                       precision.  The file is NOT sorted by it and thousands of
                       rows share one millisecond (message bursts);
* ``sequence``      -- the robot's message counter.  Restarts at 1 on every
                       power cycle (three per day in the sample), so it only
                       orders records *within* a burst;
* ``schema_version``-- ``1.0`` throughout;
* ``source_message``-- which message this row carries (14 types, below);
* ``signals``       -- a JSON object whose keys depend on ``source_message``.
                       Often a *subset* of the message's fields (the robot
                       transmits changed values), frequently ``{}``;
* ``created_at``    -- server ingest time (0.2 s .. 74 s after ``timestamp``).

The supplied mission spans three working days (2026-05-12/13/14 UTC) with
multi-hour gaps in between and shorter gaps within a day.

Message types and signals in the sample (units as labelled in the Zetrack V2
report; values are already engineering units)::

    BatteryStatus1   MainBatterySOC %, MainBatteryVoltage V, MainBatteryTemp degC
    BatteryStatus2   AuxBatterySOC %, AuxBatteryVoltage V
    BatteryStatus3   BatteryCurrent A (+ discharge), DischEnrgActualSesion kWh
                     (energy discharged in the CURRENT power-on session; resets)
    PmsMotorSpeed    T1SpeedAbs..T4SpeedAbs rpm         (wheel motors)
    PmsMotorCurrent  T1Current, T3Current, T4Current A  (T2 never reported)
    MotorTemp        TempT1..TempT4, TempPTO, TempBHD, TempBHG degC
    IpmMotorSpeed    PTOSpeed, BHDSpeed, BHGSpeed rpm    (PTO + implement motors)
    IpmMotorCurrent  PTOCurrent, BHDCurrent, BHGCurrent A
    MiscInfo         SpeedDisplay km/h, OilTemp degC, MotorTemp degC,
                     LifetimeHoursActive h, LifetimeKm km
    FunctionStatus   PtoLed, GoLed, PLed, LightsLed, PalaLed, BhdLed, BhgLed,
                     AuxBhdLed, AuxBhgLed (0/1), DriveModeLed 0..2, MoveModeLed 0..4
    SensorsAnalogValues1  AccumPressure bar (hydraulic accumulator)
    SensorsAnalogValues3  SteerLeftAngle deg, BrakePedal %
    LimitsStatus     always {} in the sample (kept raw)
    MemoryData2      CursorPositionOffset (kept raw)

Not present anywhere in the export: latitude/longitude (the PDF's route map
is drawn from a separate stream), charger state, ambient temperature.  The
decoder therefore never populates ``position``.
"""
from __future__ import annotations

import csv
import datetime as _dt
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .model import (
    SOURCE_CSV_REPLAY,
    MessageCallback,
    Subscription,
    TelemetryMessage,
    TelemetrySource,
    parse_timestamp,
    sort_messages,
)

SCHEMA_VERSION_SUPPORTED = "1.0"
CSV_COLUMNS = ("id", "user_id", "tractor_id", "mission_id", "timestamp", "sequence",
               "schema_version", "source_message", "signals", "created_at")
CSV_REQUIRED = ("tractor_id", "timestamp", "source_message", "signals")


# --------------------------------------------------------------------------- #
#  Signal dictionary
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SignalSpec:
    """How one wire signal maps onto the canonical model.

    ``field`` names a :class:`~.model.ZetrabotTelemetry` attribute; when
    ``key`` is set the attribute is a dict and the value lands under that key
    (per-motor signals).  ``kind`` is ``float`` / ``int`` / ``bool``.
    """
    message: str
    signal: str
    field: str
    unit: str
    description: str
    key: Optional[str] = None
    kind: str = "float"


def _s(message, signal, field, unit, description, key=None, kind="float") -> SignalSpec:
    return SignalSpec(message, signal, field, unit, description, key, kind)


SIGNALS: Tuple[SignalSpec, ...] = (
    # Main battery
    _s("BatteryStatus1", "MainBatterySOC", "soc_pct", "%", "traction battery state of charge"),
    _s("BatteryStatus1", "MainBatteryVoltage", "battery_voltage_v", "V", "traction battery voltage"),
    _s("BatteryStatus1", "MainBatteryTemp", "battery_temp_c", "degC", "traction battery temperature"),
    _s("BatteryStatus3", "BatteryCurrent", "battery_current_a", "A", "traction battery current (+ = discharge)"),
    _s("BatteryStatus3", "DischEnrgActualSesion", "discharged_energy_session_kwh", "kWh",
       "energy discharged in the current power-on session (resets each session)"),
    # Auxiliary battery
    _s("BatteryStatus2", "AuxBatterySOC", "aux_soc_pct", "%", "auxiliary battery state of charge"),
    _s("BatteryStatus2", "AuxBatteryVoltage", "aux_battery_voltage_v", "V", "auxiliary battery voltage"),
    # Wheel motors
    _s("PmsMotorSpeed", "T1SpeedAbs", "wheel_speed_rpm", "rpm", "wheel motor T1 speed (abs)", "T1"),
    _s("PmsMotorSpeed", "T2SpeedAbs", "wheel_speed_rpm", "rpm", "wheel motor T2 speed (abs)", "T2"),
    _s("PmsMotorSpeed", "T3SpeedAbs", "wheel_speed_rpm", "rpm", "wheel motor T3 speed (abs)", "T3"),
    _s("PmsMotorSpeed", "T4SpeedAbs", "wheel_speed_rpm", "rpm", "wheel motor T4 speed (abs)", "T4"),
    _s("PmsMotorCurrent", "T1Current", "motor_current_a", "A", "wheel motor T1 current", "T1"),
    _s("PmsMotorCurrent", "T2Current", "motor_current_a", "A", "wheel motor T2 current", "T2"),
    _s("PmsMotorCurrent", "T3Current", "motor_current_a", "A", "wheel motor T3 current", "T3"),
    _s("PmsMotorCurrent", "T4Current", "motor_current_a", "A", "wheel motor T4 current", "T4"),
    _s("MotorTemp", "TempT1", "motor_temp_c", "degC", "wheel motor T1 temperature", "T1"),
    _s("MotorTemp", "TempT2", "motor_temp_c", "degC", "wheel motor T2 temperature", "T2"),
    _s("MotorTemp", "TempT3", "motor_temp_c", "degC", "wheel motor T3 temperature", "T3"),
    _s("MotorTemp", "TempT4", "motor_temp_c", "degC", "wheel motor T4 temperature", "T4"),
    # PTO
    _s("IpmMotorSpeed", "PTOSpeed", "pto_speed_rpm", "rpm", "PTO motor speed"),
    _s("IpmMotorCurrent", "PTOCurrent", "pto_current_a", "A", "PTO motor current"),
    _s("MotorTemp", "TempPTO", "pto_temp_c", "degC", "PTO motor temperature"),
    _s("FunctionStatus", "PtoLed", "pto_active", "bool", "PTO engaged indicator", kind="bool"),
    # Implement motors (BHD / BHG as named by the export)
    _s("IpmMotorSpeed", "BHDSpeed", "implement_speed_rpm", "rpm", "implement motor BHD speed", "BHD"),
    _s("IpmMotorSpeed", "BHGSpeed", "implement_speed_rpm", "rpm", "implement motor BHG speed", "BHG"),
    _s("IpmMotorCurrent", "BHDCurrent", "implement_current_a", "A", "implement motor BHD current", "BHD"),
    _s("IpmMotorCurrent", "BHGCurrent", "implement_current_a", "A", "implement motor BHG current", "BHG"),
    _s("MotorTemp", "TempBHD", "implement_temp_c", "degC", "implement motor BHD temperature", "BHD"),
    _s("MotorTemp", "TempBHG", "implement_temp_c", "degC", "implement motor BHG temperature", "BHG"),
    # Misc
    _s("MiscInfo", "SpeedDisplay", "speed_kmh", "km/h", "displayed ground speed"),
    _s("MiscInfo", "OilTemp", "oil_temp_c", "degC", "hydraulic oil temperature"),
    _s("MiscInfo", "MotorTemp", "controller_motor_temp_c", "degC", "motor temperature reported by MiscInfo"),
    _s("MiscInfo", "LifetimeHoursActive", "lifetime_hours", "h", "lifetime active hours counter"),
    _s("MiscInfo", "LifetimeKm", "lifetime_km", "km", "lifetime distance counter"),
    # Function / drive state
    _s("FunctionStatus", "GoLed", "drive_active", "bool", "drive (Go) active indicator", kind="bool"),
    _s("FunctionStatus", "PLed", "parked", "bool", "park indicator", kind="bool"),
    _s("FunctionStatus", "DriveModeLed", "drive_mode", "enum", "drive mode 0..2", kind="int"),
    _s("FunctionStatus", "MoveModeLed", "move_mode", "enum", "move mode 0..4", kind="int"),
    _s("FunctionStatus", "LightsLed", "lights_on", "bool", "lights indicator", kind="bool"),
    _s("FunctionStatus", "PalaLed", "shovel_active", "bool", "shovel (pala) indicator", kind="bool"),
    # Steering / brake / hydraulics
    _s("SensorsAnalogValues3", "SteerLeftAngle", "steering_angle_deg", "deg", "steering angle (left positive)"),
    _s("SensorsAnalogValues3", "BrakePedal", "brake_pedal_pct", "%", "brake pedal position"),
    _s("SensorsAnalogValues1", "AccumPressure", "hydraulic_pressure_bar", "bar", "hydraulic accumulator pressure"),
)

# Every "*Led" of FunctionStatus is additionally kept raw in ``function_leds``.
FUNCTION_LED_MESSAGE = "FunctionStatus"

# Message types whose signals have no canonical field (kept raw only).
RAW_ONLY_MESSAGES = ("LimitsStatus", "MemoryData2")

SIGNAL_INDEX: Dict[Tuple[str, str], SignalSpec] = {(s.message, s.signal): s for s in SIGNALS}
SIGNAL_BY_NAME: Dict[str, SignalSpec] = {}
for _spec in SIGNALS:
    # Fallback lookup by signal name alone: a live feed may group signals
    # differently from the CSV export (e.g. one flattened document).  Names
    # are unique across messages in the Zetrack dictionary.
    SIGNAL_BY_NAME.setdefault(_spec.signal, _spec)
KNOWN_MESSAGES: Tuple[str, ...] = tuple(sorted({s.message for s in SIGNALS} | set(RAW_ONLY_MESSAGES)))


def lookup(source_message: str, signal: str) -> Optional[SignalSpec]:
    spec = SIGNAL_INDEX.get((source_message, signal))
    if spec is None:
        spec = SIGNAL_BY_NAME.get(signal)
    return spec


def supported_fields() -> List[Dict[str, str]]:
    """Table of supported wire signals (used by docs/tests)."""
    return [{"message": s.message, "signal": s.signal, "field": s.field + (f"[{s.key}]" if s.key else ""),
             "unit": s.unit, "description": s.description} for s in SIGNALS]


# --------------------------------------------------------------------------- #
#  CSV parsing
# --------------------------------------------------------------------------- #
@dataclass
class CsvParseStats:
    rows: int = 0
    messages: int = 0
    skipped_no_timestamp: int = 0
    skipped_no_tractor: int = 0
    invalid_signals_json: int = 0
    non_object_signals: int = 0
    unsupported_schema: int = 0

    def to_dict(self) -> Dict[str, int]:
        return dict(self.__dict__)


def parse_csv_row(row: Dict[str, Any], stats: Optional[CsvParseStats] = None,
                  source: str = SOURCE_CSV_REPLAY) -> Optional[TelemetryMessage]:
    """One CSV row -> :class:`TelemetryMessage` (``None`` when unusable).

    Robust by design: a row with unparsable ``signals`` still yields a message
    with empty signals and the error noted in ``extra['signals_error']`` (the
    record existed; its payload did not decode), while a row without a
    timestamp or tractor cannot be placed on any timeline and is skipped.
    """
    stats = stats if stats is not None else CsvParseStats()
    stats.rows += 1
    ts = parse_timestamp(row.get("timestamp"))
    if ts is None:
        stats.skipped_no_timestamp += 1
        return None
    tractor_id = str(row.get("tractor_id") or "").strip()
    if not tractor_id:
        stats.skipped_no_tractor += 1
        return None

    raw_signals = row.get("signals")
    signals: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    if raw_signals is None or str(raw_signals).strip() == "":
        signals = {}
    else:
        try:
            decoded = json.loads(raw_signals) if isinstance(raw_signals, str) else raw_signals
        except (TypeError, ValueError) as exc:
            stats.invalid_signals_json += 1
            extra["signals_error"] = f"invalid JSON: {exc}"
            extra["signals_text"] = str(raw_signals)[:200]
            decoded = {}
        if isinstance(decoded, dict):
            signals = decoded
        else:
            stats.non_object_signals += 1
            extra["signals_error"] = "signals is not a JSON object"
            extra["signals_value"] = decoded
            signals = {}

    schema = str(row.get("schema_version") or "").strip() or None
    if schema is not None and schema != SCHEMA_VERSION_SUPPORTED:
        stats.unsupported_schema += 1     # counted, still ingested (fields are preserved raw)

    seq_raw = row.get("sequence")
    try:
        sequence = int(seq_raw) if seq_raw not in (None, "") else None
    except (TypeError, ValueError):
        sequence = None
        extra["sequence_raw"] = seq_raw

    for column, value in row.items():
        if column in CSV_COLUMNS or column is None:
            continue
        extra[column] = value            # unknown transport columns survive too
    if row.get("user_id"):
        extra["user_id"] = row["user_id"]

    mission = row.get("mission_id")
    stats.messages += 1
    return TelemetryMessage(
        tractor_id=tractor_id,
        timestamp=ts,
        source_message=str(row.get("source_message") or "").strip() or "Unknown",
        signals=signals,
        mission_id=str(mission).strip() if mission not in (None, "") else None,
        sequence=sequence,
        schema_version=schema,
        record_id=str(row.get("id")).strip() if row.get("id") not in (None, "") else None,
        received_at=parse_timestamp(row.get("created_at")),
        source=source,
        extra=extra,
    )


def read_csv(path_or_text: Any, stats: Optional[CsvParseStats] = None,
             source: str = SOURCE_CSV_REPLAY) -> List[TelemetryMessage]:
    """Parse a whole Zetrack export (path, or CSV text) into SORTED messages."""
    stats = stats if stats is not None else CsvParseStats()
    if isinstance(path_or_text, (str, os.PathLike)) and os.path.exists(str(path_or_text)):
        handle = open(path_or_text, newline="", encoding="utf-8-sig")
    else:
        handle = io.StringIO(str(path_or_text))
    with handle:
        reader = csv.DictReader(handle)
        missing = [c for c in CSV_REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Zetrack CSV is missing required columns: {missing} "
                             f"(found {reader.fieldnames})")
        messages = [m for m in (parse_csv_row(r, stats, source) for r in reader) if m is not None]
    return sort_messages(messages)


# --------------------------------------------------------------------------- #
#  The CSV telemetry source
# --------------------------------------------------------------------------- #
class CsvTelemetrySource(TelemetrySource):
    """A Zetrack CSV export behind the :class:`TelemetrySource` seam.

    ``history()`` is the parsed, timestamp-ordered file; ``subscribe()``
    *replays* it on the original timestamps (accelerated by ``replay.speed``)
    through :class:`~.replay.ReplayPlayer` -- the same code path a live source
    would use, so everything downstream (normaliser, HARVEST state, FIWARE
    mirror) is exercised exactly as it will be with AWS.
    """

    kind = SOURCE_CSV_REPLAY

    def __init__(self, path: Any, replay: Optional["ReplayOptions"] = None,
                 source_label: str = SOURCE_CSV_REPLAY):
        from .replay import ReplayOptions   # local: replay imports model only
        self.path = Path(path)
        self.replay = replay or ReplayOptions()
        self.kind = source_label
        self.stats = CsvParseStats()
        self._messages: Optional[List[TelemetryMessage]] = None
        self._subscriptions: List[Subscription] = []

    # -- loading -------------------------------------------------------------
    def load(self) -> List[TelemetryMessage]:
        if self._messages is None:
            if not self.path.exists():
                raise FileNotFoundError(f"telemetry CSV not found: {self.path}")
            self.stats = CsvParseStats()
            self._messages = read_csv(self.path, self.stats, self.kind)
        return self._messages

    # -- TelemetrySource -----------------------------------------------------
    def history(self, start: Optional[_dt.datetime] = None,
                end: Optional[_dt.datetime] = None,
                tractor_id: Optional[str] = None) -> Iterator[TelemetryMessage]:
        for msg in self.load():
            if start is not None and msg.timestamp < start:
                continue
            if end is not None and msg.timestamp > end:
                break
            if tractor_id is not None and msg.tractor_id != str(tractor_id):
                continue
            yield msg

    def subscribe(self, callback: MessageCallback, *,
                  start: Optional[_dt.datetime] = None) -> Subscription:
        from .replay import ReplayPlayer
        messages = list(self.history(start=start))
        player = ReplayPlayer(messages, callback, self.replay, label=self.path.name)
        player.start()
        self._subscriptions.append(player)
        return player

    def describe(self) -> Dict[str, Any]:
        msgs = self._messages or []
        info: Dict[str, Any] = {
            "kind": self.kind,
            "file": str(self.path),
            "loaded": self._messages is not None,
            "messages": len(msgs),
            "parse": self.stats.to_dict(),
            "replay": self.replay.to_dict(),
        }
        if msgs:
            info["first_timestamp"] = msgs[0].timestamp.isoformat()
            info["last_timestamp"] = msgs[-1].timestamp.isoformat()
            info["tractors"] = sorted({m.tractor_id for m in msgs})
            info["missions"] = sorted({m.mission_id for m in msgs if m.mission_id})
        return info

    def close(self) -> None:
        for sub in self._subscriptions:
            sub.stop()


__all__ = [
    "CSV_COLUMNS", "CSV_REQUIRED", "KNOWN_MESSAGES", "RAW_ONLY_MESSAGES", "SIGNALS",
    "SIGNAL_INDEX", "CsvParseStats", "CsvTelemetrySource", "SignalSpec", "lookup",
    "parse_csv_row", "read_csv", "supported_fields",
]
