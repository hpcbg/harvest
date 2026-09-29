"""
Canonical telemetry model and the telemetry-source seam.

This module is deliberately independent of AWS, CSV files and FIWARE.  It
defines three things every telemetry integration shares:

* :class:`TelemetryMessage` -- one *raw* asynchronous record as the robot's
  data pipeline emits it (a ``source_message`` name plus a small dict of
  signals stamped with the robot's own timestamp).  Nothing is interpreted
  here; the ZETRABOT export is a stream of such records, one message type at
  a time, and a future AWS feed is expected to look the same;
* :class:`ZetrabotTelemetry` -- the *canonical* per-tractor state HARVEST
  reads.  Every field is optional and stays ``None`` until the corresponding
  signal has actually been observed: no value is ever invented.  Fields the
  decoder does not know are kept verbatim in :attr:`ZetrabotTelemetry.unknown_signals`
  so a new signal on the wire is never silently dropped;
* :class:`TelemetrySource` -- the seam between HARVEST and the place the
  records come from (a CSV export today, AWS later).  ``history()`` answers
  "what happened between two instants", ``subscribe()`` delivers records as
  they arrive (or, for a replay, as they *originally* arrived).

Robot *control* is not in this module and never will be: telemetry flows in
through this seam, commands flow out through ``harvest_integrations.devices``
(Modbus / OPC-UA).  The two are separate on purpose.
"""
from __future__ import annotations

import datetime as _dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Tuple

# The label a source puts on the records it emits.  "csv-replay" is the only
# implemented one; the AWS labels are reserved so the rest of HARVEST (the
# Diagnostics view, the FIWARE mirror) can already tell them apart.
SOURCE_CSV_REPLAY = "csv-replay"
SOURCE_AWS_IOT = "aws-iot-core"
SOURCE_AWS_TIMESTREAM = "aws-timestream"
SOURCE_AWS_S3 = "aws-s3"
SOURCE_REST = "rest-api"

UTC = _dt.timezone.utc


def parse_timestamp(value: Any) -> Optional[_dt.datetime]:
    """Parse an ISO-8601 timestamp into an aware UTC datetime.

    Accepts the ``+00:00`` suffix the Zetrack export uses, a trailing ``Z``,
    and naive values (assumed UTC).  Returns ``None`` for anything unparsable
    instead of raising, so one bad row never stops a replay.
    """
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        ts = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z") or text.endswith("z"):
            text = text[:-1] + "+00:00"
        try:
            ts = _dt.datetime.fromisoformat(text)
        except ValueError:
            return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def iso_utc(ts: Optional[_dt.datetime]) -> Optional[str]:
    """Millisecond ISO-8601 with a ``Z`` suffix (NGSI-LD's ``observedAt`` form)."""
    if ts is None:
        return None
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# --------------------------------------------------------------------------- #
#  Raw record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TelemetryMessage:
    """One asynchronous telemetry record, exactly as the robot emitted it.

    ``signals`` is the decoded JSON object of the record; which keys it holds
    depends on ``source_message`` (see :mod:`.zetrack` for the ZETRABOT
    dictionary).  ``extra`` keeps every other column of the transport row
    (``user_id``, ``created_at`` ...) so nothing from the wire is lost.
    """
    tractor_id: str
    timestamp: _dt.datetime
    source_message: str
    signals: Mapping[str, Any] = field(default_factory=dict)
    mission_id: Optional[str] = None
    sequence: Optional[int] = None
    schema_version: Optional[str] = None
    record_id: Optional[str] = None
    received_at: Optional[_dt.datetime] = None
    source: str = "unknown"
    extra: Mapping[str, Any] = field(default_factory=dict)

    def sort_key(self) -> Tuple[_dt.datetime, int, int, str]:
        """Total order: timestamp, then the robot's sequence, then record id.

        The ZETRABOT export is not sorted, thousands of records share a
        millisecond timestamp (bursts), and ``sequence`` restarts at every
        power cycle -- so the timestamp is the primary key and the sequence
        only breaks ties inside one burst.
        """
        rid = self.record_id or ""
        try:
            rid_num = int(rid)
        except ValueError:
            rid_num = 0
        return (self.timestamp, self.sequence if self.sequence is not None else 0,
                rid_num, rid)


def sort_messages(messages: Iterable[TelemetryMessage]) -> list[TelemetryMessage]:
    return sorted(messages, key=TelemetryMessage.sort_key)


# --------------------------------------------------------------------------- #
#  Canonical state
# --------------------------------------------------------------------------- #
@dataclass
class ZetrabotTelemetry:
    """Canonical ZETRABOT telemetry state for one tractor.

    Units are in the field names.  Everything is ``Optional`` and starts as
    ``None``: the message types arrive asynchronously at different rates
    (SOC every ~2 min, battery current several times a second) so at any
    instant some fields are fresh, some stale and some not yet seen.
    :attr:`observed_at` records *when each canonical field was last updated*,
    which is how a consumer tells the three apart.

    Not in this export, and therefore deliberately absent from the model:
    GPS position (the Zetrack report's route map comes from a separate
    stream), charger/charging state, ambient conditions.
    """
    tractor_id: str
    mission_id: Optional[str] = None
    source: str = "unknown"
    schema_version: Optional[str] = None

    # Timing
    timestamp: Optional[_dt.datetime] = None          # latest record applied
    first_timestamp: Optional[_dt.datetime] = None

    # Main traction battery
    soc_pct: Optional[float] = None
    battery_voltage_v: Optional[float] = None
    battery_current_a: Optional[float] = None         # +ve = discharge
    battery_temp_c: Optional[float] = None
    battery_power_kw: Optional[float] = None          # DERIVED: V x I / 1000
    discharged_energy_session_kwh: Optional[float] = None   # as reported (resets per power cycle)
    discharged_energy_kwh: Optional[float] = None     # DERIVED: sessions accumulated

    # Auxiliary (low-voltage) battery
    aux_soc_pct: Optional[float] = None
    aux_battery_voltage_v: Optional[float] = None

    # Motion / traction
    speed_kmh: Optional[float] = None
    wheel_speed_rpm: Dict[str, float] = field(default_factory=dict)      # T1..T4
    motor_current_a: Dict[str, float] = field(default_factory=dict)      # T1..T4
    motor_temp_c: Dict[str, float] = field(default_factory=dict)         # T1..T4
    controller_motor_temp_c: Optional[float] = None   # MiscInfo.MotorTemp
    oil_temp_c: Optional[float] = None

    # PTO
    pto_active: Optional[bool] = None
    pto_speed_rpm: Optional[float] = None
    pto_current_a: Optional[float] = None
    pto_temp_c: Optional[float] = None

    # Auxiliary implement motors (named BHD / BHG in the Zetrack export)
    implement_speed_rpm: Dict[str, float] = field(default_factory=dict)
    implement_current_a: Dict[str, float] = field(default_factory=dict)
    implement_temp_c: Dict[str, float] = field(default_factory=dict)

    # Drive / function state
    drive_active: Optional[bool] = None               # GoLed
    parked: Optional[bool] = None                     # PLed
    drive_mode: Optional[int] = None                  # DriveModeLed 0..2
    move_mode: Optional[int] = None                   # MoveModeLed 0..4
    lights_on: Optional[bool] = None
    shovel_active: Optional[bool] = None              # PalaLed
    function_leds: Dict[str, int] = field(default_factory=dict)   # every *Led raw

    # Steering / brake / hydraulics
    steering_angle_deg: Optional[float] = None
    brake_pedal_pct: Optional[float] = None
    hydraulic_pressure_bar: Optional[float] = None

    # Odometry-style counters
    lifetime_km: Optional[float] = None
    lifetime_hours: Optional[float] = None

    # Position: NOT available in this export (documented, never fabricated).
    position: Optional[Tuple[float, float]] = None

    # Raw preservation
    signals_raw: Dict[str, Any] = field(default_factory=dict)      # latest value of every signal
    unknown_signals: Dict[str, Any] = field(default_factory=dict)  # signals with no canonical field
    unknown_messages: Dict[str, int] = field(default_factory=dict) # source_message types never decoded
    observed_at: Dict[str, _dt.datetime] = field(default_factory=dict)   # canonical field -> ts
    message_seen_at: Dict[str, _dt.datetime] = field(default_factory=dict)  # source_message -> ts
    message_counts: Dict[str, int] = field(default_factory=dict)

    # Ingestion counters
    messages: int = 0
    out_of_order: int = 0
    invalid_values: int = 0
    session_resets: int = 0

    # -- convenience --------------------------------------------------------
    def age_of(self, field_name: str, now: Optional[_dt.datetime] = None) -> Optional[float]:
        """Seconds since ``field_name`` was last observed (telemetry time)."""
        ts = self.observed_at.get(field_name)
        if ts is None:
            return None
        ref = now or self.timestamp
        if ref is None:
            return None
        return max(0.0, (ref - ts).total_seconds())

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready view (datetimes as ISO strings, tuples as lists)."""
        out: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, _dt.datetime):
                value = iso_utc(value)
            elif isinstance(value, dict):
                value = {k: (iso_utc(v) if isinstance(v, _dt.datetime) else v)
                         for k, v in value.items()}
            elif isinstance(value, tuple):
                value = list(value)
            out[f.name] = value
        return out


# Provenance of every canonical field (see telemetry/kpi.py for the classes).
# MEASURED = present on the wire; DERIVED = computed from measured values
# only; ESTIMATED = computed under an assumption that is not confirmed;
# MISSING = not in this source at all.  Shown next to the values so a
# derived or model-dependent number never looks like a measurement.
FIELD_PROVENANCE: Dict[str, str] = {
    **{name: "MEASURED" for name in (
        "soc_pct", "battery_voltage_v", "battery_current_a", "battery_temp_c",
        "discharged_energy_session_kwh", "aux_soc_pct", "aux_battery_voltage_v", "speed_kmh",
        "wheel_speed_rpm", "motor_current_a", "motor_temp_c", "controller_motor_temp_c",
        "oil_temp_c", "pto_active", "pto_speed_rpm", "pto_current_a", "pto_temp_c",
        "implement_speed_rpm", "implement_current_a", "implement_temp_c", "drive_active",
        "parked", "drive_mode", "move_mode", "lights_on", "shovel_active",
        "steering_angle_deg", "brake_pedal_pct", "hydraulic_pressure_bar", "lifetime_km",
        "lifetime_hours")},
    "battery_power_kw": "DERIVED",                    # V x I / 1000
    "discharged_energy_kwh": "ESTIMATED",             # session-counter semantics unconfirmed
    "estimated_remaining_energy_kwh": "ESTIMATED",    # SOC x CONFIGURED capacity
    "nominal_capacity_kwh": "CONFIGURED",
    "position": "MISSING",                            # no GPS in the export
    "charging": "MISSING",                            # no charging-state stream
}


# --------------------------------------------------------------------------- #
#  Source seam
# --------------------------------------------------------------------------- #
MessageCallback = Callable[[TelemetryMessage], None]


class Subscription(ABC):
    """Handle on a live (or replayed) delivery of messages."""

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def is_alive(self) -> bool: ...

    def status(self) -> Dict[str, Any]:
        return {"alive": self.is_alive()}


class TelemetrySource(ABC):
    """Where telemetry records come from.

    Implementations: :class:`~harvest_integrations.telemetry.zetrack.CsvTelemetrySource`
    (a Zetrack CSV export, replayed on the original timestamps) and the
    scaffolds in :mod:`harvest_integrations.aws` (IoT Core / Timestream / S3 /
    REST -- pending the ZETRABOT team's documentation).

    ``kind`` labels every record's ``source`` field so downstream consumers
    (Diagnostics, FIWARE) can state where a value came from.
    """

    kind: str = "unknown"

    @abstractmethod
    def history(self, start: Optional[_dt.datetime] = None,
                end: Optional[_dt.datetime] = None,
                tractor_id: Optional[str] = None) -> Iterator[TelemetryMessage]:
        """Records in ``[start, end]`` (both optional), in :meth:`TelemetryMessage.sort_key` order."""

    @abstractmethod
    def subscribe(self, callback: MessageCallback, *,
                  start: Optional[_dt.datetime] = None) -> Subscription:
        """Deliver records to ``callback`` as they arrive, in order, from a background thread."""

    def describe(self) -> Dict[str, Any]:
        """Read-only description for the Diagnostics view (no secrets)."""
        return {"kind": self.kind}

    def close(self) -> None:
        return None


class TelemetrySink(ABC):
    """Consumer of canonical states (HARVEST's fleet registry, a recorder ...)."""

    @abstractmethod
    def update(self, state: ZetrabotTelemetry, message: TelemetryMessage) -> None: ...


__all__ = [
    "FIELD_PROVENANCE", "SOURCE_AWS_IOT", "SOURCE_AWS_S3", "SOURCE_AWS_TIMESTREAM", "SOURCE_CSV_REPLAY",
    "SOURCE_REST", "MessageCallback", "Subscription", "TelemetryMessage", "TelemetrySink",
    "TelemetrySource", "ZetrabotTelemetry", "iso_utc", "parse_timestamp", "sort_messages",
]
