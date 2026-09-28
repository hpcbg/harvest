"""
Telemetry normaliser: raw asynchronous messages -> canonical state.

    TelemetryMessage (any source)  --ingest()-->  ZetrabotTelemetry (per tractor)
                                                       |
                                                  sinks: TelemetryRegistry
                                                         (HARVEST state), ...

Rules the normaliser enforces, in the order they bit during design:

* **partial updates** -- a message carries a subset of one message type's
  signals; only those fields change, everything else keeps its last value
  and its own ``observed_at`` stamp (asynchronous sampling is a property of
  the data, not something to paper over);
* **nothing invented** -- a field is ``None`` until observed; a value that
  is not a number stays out of the canonical field (and is counted) but is
  still kept raw;
* **unknown preserved** -- signals without a canonical field land in
  ``unknown_signals``; message types the dictionary does not know are
  counted in ``unknown_messages``; the latest raw value of *every* signal
  is in ``signals_raw``;
* **ordering** -- messages are expected in timestamp order.  A late message
  (older than the tractor's latest timestamp) is still applied -- its values
  are real -- but counted in ``out_of_order`` and never moves ``timestamp``
  backwards;
* **derived values are labelled** -- ``battery_power_kw`` (V x I) and
  ``discharged_energy_kwh`` (per-session counter accumulated across resets)
  are computed here, and only here, and their provenance is documented on
  the model.
"""
from __future__ import annotations

import datetime as _dt
import threading
from typing import Any, Dict, Iterable, List, Optional

from .model import TelemetryMessage, TelemetrySink, ZetrabotTelemetry
from . import zetrack

# V x I is only meaningful when both readings are recent relative to each
# other: the current is reported several times a second, the voltage every
# few seconds.  Beyond this the product is a stale pairing and stays None.
POWER_PAIRING_MAX_S = 120.0

# DischEnrgActualSesion resets to 0 on every power cycle.  A drop below half
# of the running session maximum (once the session has accumulated something)
# marks a reset; smaller decreases (the counter occasionally steps back a few
# hundred Wh right before a power-down) are not.
SESSION_RESET_FRACTION = 0.5
SESSION_RESET_MIN_KWH = 0.1


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


class _SessionAccumulator:
    """Accumulates the per-session discharge counter across resets."""

    def __init__(self) -> None:
        self.completed_kwh = 0.0
        self.session_max: Optional[float] = None
        self.resets = 0

    def push(self, value: float) -> float:
        if self.session_max is None:
            self.session_max = value
        elif (self.session_max > SESSION_RESET_MIN_KWH
              and value < SESSION_RESET_FRACTION * self.session_max):
            self.completed_kwh += self.session_max
            self.session_max = value
            self.resets += 1
        else:
            self.session_max = max(self.session_max, value)
        return self.completed_kwh + (self.session_max or 0.0)


class TelemetryNormalizer:
    """Stateful decoder from :class:`TelemetryMessage` to :class:`ZetrabotTelemetry`.

    One instance serves any number of tractors (state is kept per
    ``tractor_id``).  Thread-safe: a replay thread ingests while the API
    thread reads.
    """

    def __init__(self, sinks: Optional[Iterable[TelemetrySink]] = None,
                 tolerate_out_of_order: bool = True):
        self._states: Dict[str, ZetrabotTelemetry] = {}
        self._sessions: Dict[str, _SessionAccumulator] = {}
        self._sinks: List[TelemetrySink] = list(sinks or [])
        self._lock = threading.RLock()
        self.tolerate_out_of_order = tolerate_out_of_order
        self.messages_ingested = 0

    def add_sink(self, sink: TelemetrySink) -> None:
        with self._lock:
            self._sinks.append(sink)

    # -- read side ------------------------------------------------------------
    def state(self, tractor_id: str) -> Optional[ZetrabotTelemetry]:
        with self._lock:
            return self._states.get(str(tractor_id))

    def states(self) -> List[ZetrabotTelemetry]:
        with self._lock:
            return list(self._states.values())

    # -- write side -----------------------------------------------------------
    def ingest(self, msg: TelemetryMessage) -> ZetrabotTelemetry:
        with self._lock:
            state = self._states.get(msg.tractor_id)
            if state is None:
                state = ZetrabotTelemetry(tractor_id=msg.tractor_id, source=msg.source,
                                          first_timestamp=msg.timestamp)
                self._states[msg.tractor_id] = state
                self._sessions[msg.tractor_id] = _SessionAccumulator()
            self._apply(state, msg, self._sessions[msg.tractor_id])
            self.messages_ingested += 1
            sinks = list(self._sinks)
        for sink in sinks:
            sink.update(state, msg)
        return state

    def ingest_all(self, messages: Iterable[TelemetryMessage]) -> None:
        for msg in messages:
            self.ingest(msg)

    # -- decoding -------------------------------------------------------------
    def _apply(self, state: ZetrabotTelemetry, msg: TelemetryMessage,
               session: _SessionAccumulator) -> None:
        ts = msg.timestamp
        late = state.timestamp is not None and ts < state.timestamp
        if late:
            state.out_of_order += 1
            if not self.tolerate_out_of_order:
                return
        else:
            state.timestamp = ts
        if state.first_timestamp is None or ts < state.first_timestamp:
            state.first_timestamp = ts
        if msg.mission_id is not None:
            state.mission_id = msg.mission_id
        if msg.schema_version is not None:
            state.schema_version = msg.schema_version
        state.source = msg.source
        state.messages += 1
        state.message_counts[msg.source_message] = state.message_counts.get(msg.source_message, 0) + 1
        if not late or msg.source_message not in state.message_seen_at:
            state.message_seen_at[msg.source_message] = ts

        known_message = msg.source_message in zetrack.KNOWN_MESSAGES
        if not known_message:
            state.unknown_messages[msg.source_message] = (
                state.unknown_messages.get(msg.source_message, 0) + 1)

        for signal, value in msg.signals.items():
            state.signals_raw[signal] = value
            spec = zetrack.lookup(msg.source_message, signal)
            raw_only = msg.source_message in zetrack.RAW_ONLY_MESSAGES
            if msg.source_message == zetrack.FUNCTION_LED_MESSAGE and signal.endswith("Led"):
                num = _num(value)
                if num is not None:
                    state.function_leds[signal] = int(num)
                    raw_only = True          # known indicator, kept in function_leds
            if spec is None:
                if not raw_only:
                    # Genuinely new to the dictionary: preserved AND flagged.
                    state.unknown_signals[f"{msg.source_message}.{signal}"] = value
                continue
            num = _num(value)
            if num is None:
                state.invalid_values += 1
                continue
            self._set(state, spec, num, ts, late)
            if spec.field == "discharged_energy_session_kwh":
                state.discharged_energy_kwh = session.push(num)
                state.session_resets = session.resets
                state.observed_at["discharged_energy_kwh"] = ts

        self._derive_power(state)

    @staticmethod
    def _set(state: ZetrabotTelemetry, spec: zetrack.SignalSpec, num: float,
             ts: _dt.datetime, late: bool) -> None:
        previous_ts = state.observed_at.get(spec.field if spec.key is None else f"{spec.field}[{spec.key}]")
        if late and previous_ts is not None and previous_ts > ts:
            return   # a fresher value is already in place
        if spec.kind == "bool":
            value: Any = bool(round(num))
        elif spec.kind == "int":
            value = int(round(num))
        else:
            value = num
        if spec.key is None:
            setattr(state, spec.field, value)
            state.observed_at[spec.field] = ts
        else:
            getattr(state, spec.field)[spec.key] = value
            state.observed_at[f"{spec.field}[{spec.key}]"] = ts
            state.observed_at[spec.field] = ts

    @staticmethod
    def _derive_power(state: ZetrabotTelemetry) -> None:
        v, i = state.battery_voltage_v, state.battery_current_a
        tv, ti = state.observed_at.get("battery_voltage_v"), state.observed_at.get("battery_current_a")
        if v is None or i is None or tv is None or ti is None:
            return
        if abs((tv - ti).total_seconds()) > POWER_PAIRING_MAX_S:
            state.battery_power_kw = None
            return
        state.battery_power_kw = v * i / 1000.0
        state.observed_at["battery_power_kw"] = max(tv, ti)


# --------------------------------------------------------------------------- #
#  HARVEST-side registry (a sink)
# --------------------------------------------------------------------------- #
class TelemetryRegistry(TelemetrySink):
    """Latest canonical state per tractor, plus wall-clock receipt times.

    This is what :class:`~harvest_integrations.runtime.FleetRuntime` reads to
    merge real tractors into fleet snapshots, and what the Diagnostics view and
    the FIWARE mirror render.  It stores; it does not interpret.
    """

    def __init__(self) -> None:
        self._latest: Dict[str, ZetrabotTelemetry] = {}
        self._received_at: Dict[str, float] = {}
        self._lock = threading.Lock()
        self.updates = 0

    def update(self, state: ZetrabotTelemetry, message: TelemetryMessage) -> None:
        import time
        with self._lock:
            self._latest[state.tractor_id] = state
            self._received_at[state.tractor_id] = time.time()
            self.updates += 1

    def latest(self, tractor_id: str) -> Optional[ZetrabotTelemetry]:
        with self._lock:
            return self._latest.get(str(tractor_id))

    def all(self) -> List[ZetrabotTelemetry]:
        with self._lock:
            return list(self._latest.values())

    def received_age_s(self, tractor_id: str) -> Optional[float]:
        import time
        with self._lock:
            ts = self._received_at.get(str(tractor_id))
        return None if ts is None else max(0.0, time.time() - ts)


__all__ = ["POWER_PAIRING_MAX_S", "SESSION_RESET_FRACTION", "SESSION_RESET_MIN_KWH",
           "TelemetryNormalizer", "TelemetryRegistry"]
