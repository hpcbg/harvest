"""
TelemetryService -- the wiring between a TelemetrySource and HARVEST.

    TelemetrySource.subscribe() --> TelemetryNormalizer --> TelemetryRegistry
                                                                 |
                              FleetRuntime.snapshot() merges  <--+--> Diagnostics / FIWARE

Built from the ``integrations.telemetry`` section of config.yaml (env
overrides ``HARVEST_TELEMETRY_*``) by :func:`build_telemetry_service`.  It
owns the subscription thread, exposes the canonical states, and maps them
onto ``harvest_control.TractorState`` for the fleet snapshot -- the only
place where telemetry touches HARVEST's domain model.

It never sends anything to the robot.  Control stays in
``harvest_integrations.devices`` (Modbus / OPC-UA).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from harvest_control.interface import TractorState

from .model import (
    SOURCE_CSV_REPLAY,
    Subscription,
    TelemetrySource,
    ZetrabotTelemetry,
    iso_utc,
    parse_timestamp,
)
from .normalizer import TelemetryNormalizer, TelemetryRegistry
from .replay import ReplayOptions

SOURCE_NONE = "none"
SOURCE_CSV = "csv"
SOURCE_AWS = "aws"

# Wall-clock silence after which a *live* source's tractor is reported
# unavailable.  A replay uses replay progress instead (it stops when the
# file ends -- the tractor is not "gone", the recording is).
LIVE_STALE_S = 60.0


def telemetry_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return ((cfg.get("integrations") or {}).get("telemetry") or {})


def _env_or(name: str, default: Any) -> Any:
    value = os.environ.get(name)
    return default if value in (None, "") else value


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- #
#  The service
# --------------------------------------------------------------------------- #
class TelemetryService:
    def __init__(self, source: TelemetrySource, *,
                 tractor_ids: Optional[Dict[str, str]] = None,
                 nominal_capacity_kwh: Optional[float] = None,
                 mode: str = "replay"):
        self.source = source
        self.mode = mode                                    # "replay" | "live"
        self.tractor_ids = {str(k): str(v) for k, v in (tractor_ids or {}).items()}
        self.nominal_capacity_kwh = nominal_capacity_kwh
        self.registry = TelemetryRegistry()
        self.normalizer = TelemetryNormalizer(sinks=[self.registry])
        self._subscription: Optional[Subscription] = None
        self._error = ""
        self._started_at: Optional[float] = None

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> "TelemetryService":
        if self._subscription is not None:
            return self
        try:
            self._subscription = self.source.subscribe(self.normalizer.ingest)
            self._started_at = time.time()
        except Exception as exc:                                   # noqa: BLE001
            self._error = f"{type(exc).__name__}: {exc}"
        return self

    def close(self) -> None:
        if self._subscription is not None:
            self._subscription.stop()
        self.source.close()

    # -- identity -------------------------------------------------------------
    def harvest_id(self, tractor_id: str) -> str:
        """HARVEST fleet id for a robot id (``"1"`` -> ``"zetrabot_1"`` by default)."""
        return self.tractor_ids.get(str(tractor_id), f"zetrabot_{tractor_id}")

    def source_kind(self) -> str:
        return self.source.kind

    # -- state ----------------------------------------------------------------
    def states(self) -> List[ZetrabotTelemetry]:
        return self.registry.all()

    def replay_status(self) -> Optional[Dict[str, Any]]:
        if self._subscription is None:
            return None
        return self._subscription.status()

    def is_live(self) -> bool:
        return self.mode == "live"

    def tractor_available(self, state: ZetrabotTelemetry) -> bool:
        """Whether the tractor counts as *present* for the fleet snapshot.

        Live: fresh data within :data:`LIVE_STALE_S`.  Replay: while the
        replay is delivering, the recorded tractor is present; once the
        recording has ended it is marked unavailable, which is the truth --
        HARVEST has no current information about it.
        """
        if self.is_live():
            age = self.registry.received_age_s(state.tractor_id)
            return age is not None and age <= LIVE_STALE_S
        status = self.replay_status() or {}
        return status.get("state") == "running"

    def tractor_states(self) -> List[TractorState]:
        """Real tractors as ``harvest_control`` dataclasses for the fleet snapshot.

        Only tractors whose SOC has actually been observed are included (a
        TractorState without SOC would be a fabricated number).  ``energy_kwh``
        is SOC x the *configured nominal* capacity -- a model-derived figure,
        labelled as such in the telemetry document.  Position is ``None``:
        this data has no GPS.  Charging state is not in the export; a strongly
        negative current would be the only hint and is deliberately not
        interpreted here.
        """
        rows: List[TractorState] = []
        for state in self.states():
            if state.soc_pct is None:
                continue
            energy = None
            if self.nominal_capacity_kwh:
                energy = state.soc_pct / 100.0 * self.nominal_capacity_kwh
            rows.append(TractorState(
                id=self.harvest_id(state.tractor_id),
                soc_pct=float(state.soc_pct),
                energy_kwh=float(energy) if energy is not None else 0.0,
                available=self.tractor_available(state),
                charging=False,
                current_task=None,
                position=None,
                discharging=False,
                discharge_kw=0.0,
            ))
        return rows

    # -- diagnostics ----------------------------------------------------------
    def device_rows(self) -> List[Dict[str, Any]]:
        """Device-table rows (same shape as the DeviceIO layer's)."""
        rows: List[Dict[str, Any]] = []
        for state in self.states():
            values: Dict[str, Any] = {}
            for key in ("soc_pct", "battery_voltage_v", "battery_current_a",
                        "battery_power_kw", "discharged_energy_kwh", "speed_kmh"):
                value = getattr(state, key)
                if value is not None:
                    values[key] = round(value, 3)
            if state.pto_active is not None:
                values["pto_active"] = float(state.pto_active)
            if state.drive_active is not None:
                values["drive_active"] = float(state.drive_active)
            age = self.registry.received_age_s(state.tractor_id)
            rows.append({
                "id": self.harvest_id(state.tractor_id),
                "kind": "tractor",
                "layer": "telemetry",        # inbound data, not a field-control protocol
                "protocol": self.source_kind(),
                "endpoint": self._endpoint_label(),
                "reachable": self.tractor_available(state),
                "last_read_ts": (time.time() - age) if age is not None else None,
                "values": values,
                "telemetry_timestamp": iso_utc(state.timestamp),
            })
        return rows

    def _endpoint_label(self) -> str:
        info = self.source.describe()
        if "file" in info:
            return f"{Path(info['file']).name}"
        return str(info.get("endpoint") or info.get("kind") or "?")

    def status(self) -> Dict[str, Any]:
        """Short status for ``/api/fleet/status`` and the diagnostics row."""
        replay = self.replay_status()
        return {
            "source": self.source_kind(),
            "mode": self.mode,
            "tractors": [self.harvest_id(s.tractor_id) for s in self.states()],
            "messages_ingested": self.normalizer.messages_ingested,
            "replay_state": (replay or {}).get("state"),
            "error": self._error,
        }

    def document(self) -> Dict[str, Any]:
        """The full telemetry document (``GET /api/telemetry``)."""
        tractors = []
        for state in self.states():
            doc = state.to_dict()
            doc["harvest_id"] = self.harvest_id(state.tractor_id)
            doc["available"] = self.tractor_available(state)
            doc["received_age_s"] = self.registry.received_age_s(state.tractor_id)
            doc["field_ages_s"] = {
                key: state.age_of(key) for key in (
                    "soc_pct", "battery_voltage_v", "battery_current_a",
                    "battery_temp_c", "discharged_energy_session_kwh", "pto_active",
                    "drive_active", "speed_kmh")
                if key in state.observed_at}
            if self.nominal_capacity_kwh and state.soc_pct is not None:
                doc["energy_kwh_nominal"] = round(
                    state.soc_pct / 100.0 * self.nominal_capacity_kwh, 3)
            tractors.append(doc)
        return {
            "state": "failed" if self._error else ("active" if tractors else "waiting"),
            "mode": self.mode,
            "source": self.source.describe(),
            "replay": self.replay_status(),
            "tractors": tractors,
            "nominal_capacity_kwh": self.nominal_capacity_kwh,
            "notes": [
                "position: not present in this telemetry (no GPS in the Zetrack export)",
                "charging state: not present in this telemetry",
                "battery_power_kw is derived (voltage x current); "
                "discharged_energy_kwh accumulates the per-session counter across resets",
                "energy_kwh_nominal is SOC x the configured nominal capacity, not a measurement",
            ],
            "error": self._error,
            "started_at": self._started_at,
        }


# --------------------------------------------------------------------------- #
#  Factory
# --------------------------------------------------------------------------- #
def build_telemetry_source(cfg: Dict[str, Any], repo_root: Optional[Path] = None) -> Optional[TelemetrySource]:
    """A TelemetrySource from config + environment, or ``None`` when off.

    Raises for a misconfigured or not-yet-implemented source (the AWS
    scaffold) -- the caller reports that in Diagnostics instead of crashing.
    """
    settings = telemetry_settings(cfg)
    kind = str(_env_or("HARVEST_TELEMETRY_SOURCE", settings.get("source", SOURCE_NONE))).lower()
    if kind in (SOURCE_NONE, "", "off", "false"):
        return None
    if kind == SOURCE_CSV:
        from .zetrack import CsvTelemetrySource
        csv_cfg = settings.get("csv") or {}
        file = _env_or("HARVEST_TELEMETRY_FILE", csv_cfg.get("file"))
        if not file:
            raise ValueError("integrations.telemetry.csv.file (or HARVEST_TELEMETRY_FILE) is required")
        path = Path(str(file))
        if not path.is_absolute() and repo_root is not None:
            path = repo_root / path
        options = ReplayOptions(
            speed=float(_env_or("HARVEST_TELEMETRY_SPEED", csv_cfg.get("speed", 1.0))),
            max_gap_s=_optional_float(_env_or("HARVEST_TELEMETRY_MAX_GAP_S",
                                              csv_cfg.get("max_gap_s", 30.0))),
            loop=_as_bool(_env_or("HARVEST_TELEMETRY_LOOP", csv_cfg.get("loop", False))),
            start=parse_timestamp(_env_or("HARVEST_TELEMETRY_START", csv_cfg.get("start"))),
            end=parse_timestamp(_env_or("HARVEST_TELEMETRY_END", csv_cfg.get("end"))),
        )
        return CsvTelemetrySource(path, replay=options,
                                  source_label=str(csv_cfg.get("label", SOURCE_CSV_REPLAY)))
    if kind == SOURCE_AWS:
        from harvest_integrations.aws.base import build_aws_source
        return build_aws_source(settings.get("aws") or {})
    raise ValueError(f"Unknown integrations.telemetry.source {kind!r} "
                     f"(expected {SOURCE_NONE!r}, {SOURCE_CSV!r} or {SOURCE_AWS!r})")


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, "", "none", "None", "null"):
        return None
    number = float(value)
    return None if number <= 0 else number


def build_telemetry_service(cfg: Dict[str, Any], repo_root: Optional[Path] = None,
                            start: bool = True) -> Optional[TelemetryService]:
    source = build_telemetry_source(cfg, repo_root)
    if source is None:
        return None
    settings = telemetry_settings(cfg)
    model = (cfg.get("tractors") or {}).get("model") or {}
    capacity = model.get("battery_capacity_kwh")
    service = TelemetryService(
        source,
        tractor_ids=settings.get("tractor_ids") or {},
        nominal_capacity_kwh=float(capacity) if capacity else None,
        mode="replay" if source.kind == SOURCE_CSV_REPLAY or str(source.kind).startswith("csv")
        else "live",
    )
    return service.start() if start else service


__all__ = ["LIVE_STALE_S", "SOURCE_AWS", "SOURCE_CSV", "SOURCE_NONE", "TelemetryService",
           "build_telemetry_service", "build_telemetry_source", "telemetry_settings"]
