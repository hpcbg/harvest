"""
Live fleet runtime.

Builds a :class:`harvest_control.FleetInterface` backend from the
``integrations.fleet`` section of ``config.yaml`` and, for simulation
backends, advances it continuously on a background thread so HTTP/FIWARE/ROS
consumers see a farm that moves in real time.

Backend selection (``integrations.fleet.backend``, env override
``HARVEST_FLEET_BACKEND``):

* ``sim``     -- ``harvest_control.SimulationFleetInterface`` (zero deps);
* ``devices`` -- :class:`DeviceFleetInterface` talking Modbus/OPC-UA to real
  hardware or to the bundled farm simulator.

For the ``devices`` backend the endpoint list is *derived from the farm
config* (fleet, charging stations, energy consumers) using the same wire
conventions the bundled simulator serves, so config.yaml stays the single
source of device identity.  Fully custom endpoints (arbitrary point maps,
extra protocols) can be appended under ``integrations.fleet.custom_devices``.

Real telemetry (``integrations.telemetry``, env ``HARVEST_TELEMETRY_*``) is a
separate, inbound-only layer: a :class:`~harvest_integrations.telemetry.service.TelemetryService`
feeds canonical ZETRABOT states into a registry, and this runtime *merges*
those tractors into every snapshot (as ``zetrabot_<id>`` unless mapped onto
an existing fleet id).  Commands never go through it -- control stays with
the fleet backend (DeviceIO / Modbus / OPC-UA).
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from harvest_control.interface import Command, CommandAck, FleetInterface, FleetSnapshot
from harvest_control.sim_backend import SimulationFleetInterface

from .devices.base import DeviceEndpoint, PointSpec, endpoint_from_dict
from .devices.fleet_backend import DeviceFleetInterface

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Wire conventions shared with harvest_integrations.simulators (see
# modbus_server.py / opcua_server.py for the authoritative maps).
_MODBUS_GRID_POINTS = {
    "clock_min": (0, 1.0), "grid_draw_kw": (1, 0.1), "grid_cap_kw": (2, 0.1),
    "pv_kw": (3, 0.1), "tariff_code": (4, 1.0), "price_eur_per_kwh": (5, 0.001),
}
_CHARGER_BASE, _LOAD_BASE, _STRIDE = 100, 200, 10
_OPCUA_TRACTOR_POINTS = {
    "soc_pct": "SoC_pct", "energy_kwh": "Energy_kWh", "available": "Available",
    "charging": "Charging", "discharging": "Discharging",
    "discharge_kw": "Discharge_kW", "pos_x": "Pos_X", "pos_y": "Pos_Y",
}
_OPCUA_TRACTOR_WRITABLE = {"charge_request": "Charge_Request", "v2l_kw": "V2L_kW"}


def fleet_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return (cfg.get("integrations") or {}).get("fleet") or {}


# --------------------------------------------------------------------------- #
#  Endpoint derivation
# --------------------------------------------------------------------------- #
def build_device_endpoints(cfg: Dict[str, Any]) -> List[DeviceEndpoint]:
    """Derive device endpoints from the farm config + integration settings."""
    settings = fleet_settings(cfg)
    modbus_host = os.environ.get(
        "HARVEST_MODBUS_HOST", str(settings.get("modbus_host", "127.0.0.1")))
    modbus_port = int(os.environ.get(
        "HARVEST_MODBUS_PORT", settings.get("modbus_port", 5020)))
    opcua_endpoint = os.environ.get(
        "HARVEST_OPCUA_ENDPOINT",
        str(settings.get("opcua_endpoint", "opc.tcp://127.0.0.1:4840/harvest/")))
    modbus_opts = {"host": modbus_host, "port": modbus_port}

    endpoints: List[DeviceEndpoint] = [DeviceEndpoint(
        device_id="grid",
        kind="grid",
        protocol="modbus",
        options=dict(modbus_opts),
        points={
            name: PointSpec(name, addr, scale)
            for name, (addr, scale) in _MODBUS_GRID_POINTS.items()
        },
    )]

    fleet = (cfg.get("tractors") or {}).get("fleet") or []
    for entry in fleet:
        if not entry.get("enabled", True):
            continue
        tid = str(entry["id"])
        points = {
            name: PointSpec(name, node) for name, node in _OPCUA_TRACTOR_POINTS.items()
        }
        points.update({
            name: PointSpec(name, node, writable=True)
            for name, node in _OPCUA_TRACTOR_WRITABLE.items()
        })
        endpoints.append(DeviceEndpoint(
            device_id=tid, kind="tractor", protocol="opcua",
            options={"endpoint": opcua_endpoint, "object_path": tid},
            points=points,
        ))

    stations = (cfg.get("charging") or {}).get("stations") or []
    for i, entry in enumerate(stations):
        base = _CHARGER_BASE + i * _STRIDE
        endpoints.append(DeviceEndpoint(
            device_id=str(entry["id"]), kind="charger", protocol="modbus",
            options=dict(modbus_opts),
            points={
                "power_kw": PointSpec("power_kw", base + 0, 0.1),
                "level": PointSpec("level", base + 1, writable=True),
                "occupied": PointSpec("occupied", base + 2),
            },
        ))

    for i, entry in enumerate(cfg.get("energy_consumers") or []):
        base = _LOAD_BASE + i * _STRIDE
        endpoints.append(DeviceEndpoint(
            device_id=str(entry["id"]), kind="load", protocol="modbus",
            options=dict(modbus_opts),
            points={
                "power_kw": PointSpec("power_kw", base + 0, 0.1),
                "shed": PointSpec("shed", base + 1, writable=True),
            },
        ))

    for raw in settings.get("custom_devices") or []:
        endpoints.append(endpoint_from_dict(raw))
    return endpoints


# --------------------------------------------------------------------------- #
#  Runtime
# --------------------------------------------------------------------------- #
class FleetRuntime:
    """Thread-safe facade over a live fleet backend.

    For non-real-time backends a daemon thread advances the simulation by
    ``sim_minutes_per_s`` every second (default 1.0: one real second is one
    simulated minute, matching the bundled farm simulator's 60x speedup).
    """

    def __init__(self, cfg: Dict[str, Any]):
        settings = fleet_settings(cfg)
        self.backend_name = os.environ.get(
            "HARVEST_FLEET_BACKEND", str(settings.get("backend", "sim"))).lower()
        self.sim_minutes_per_s = float(settings.get("sim_minutes_per_s", 1.0))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if self.backend_name == "devices":
            self.fleet: FleetInterface = DeviceFleetInterface(build_device_endpoints(cfg))
        elif self.backend_name == "sim":
            grid_cap = float((cfg.get("grid") or {}).get("max_power_kw", 10.5))
            self.fleet = SimulationFleetInterface(grid_cap_kw=grid_cap)
        else:
            raise ValueError(
                f"Unknown integrations.fleet.backend {self.backend_name!r} "
                "(expected 'sim' or 'devices')")

        if not self.fleet.is_real_time():
            self._thread = threading.Thread(
                target=self._advance_loop, name="fleet-advance", daemon=True)
            self._thread.start()

        # Real telemetry (optional, inbound only).  A misconfigured or
        # not-yet-implemented source (the AWS scaffold) is REPORTED through
        # telemetry_error / Diagnostics, never allowed to take the fleet down.
        self.telemetry = None
        self.telemetry_error = ""
        try:
            from .telemetry.service import build_telemetry_service
            self.telemetry = build_telemetry_service(cfg, repo_root=_REPO_ROOT)
        except Exception as exc:                                   # noqa: BLE001
            self.telemetry_error = f"{type(exc).__name__}: {exc}"

    # -- boundary methods (thread-safe) ---------------------------------------
    def snapshot(self) -> FleetSnapshot:
        with self._lock:
            snap = self.fleet.snapshot()
        return self._merge_telemetry(snap)

    def submit(self, commands: List[Command]) -> List[CommandAck]:
        with self._lock:
            return list(self.fleet.submit(commands))

    def status(self) -> Dict[str, Any]:
        doc = {
            "backend": self.backend_name,
            "real_time": self.fleet.is_real_time(),
            "sim_minutes_per_s": None if self.fleet.is_real_time() else self.sim_minutes_per_s,
        }
        if self.telemetry is not None:
            doc["telemetry"] = self.telemetry.status()
        elif self.telemetry_error:
            doc["telemetry"] = {"source": None, "error": self.telemetry_error}
        return doc

    # -- real telemetry --------------------------------------------------------
    def _merge_telemetry(self, snap: FleetSnapshot) -> FleetSnapshot:
        """Add (or, when mapped onto an existing id, replace) real tractors."""
        if self.telemetry is None:
            return snap
        try:
            real = self.telemetry.tractor_states()
        except Exception:                                          # noqa: BLE001
            return snap
        if not real:
            return snap
        by_id = {t.id: t for t in real}
        merged = [by_id.pop(t.id, t) for t in snap.tractors]
        merged.extend(by_id.values())
        return FleetSnapshot(grid=snap.grid, tractors=merged,
                             chargers=snap.chargers, loads=snap.loads)

    def telemetry_document(self) -> Dict[str, Any]:
        """``GET /api/telemetry`` -- honest even when nothing is configured."""
        if self.telemetry is not None:
            return self.telemetry.document()
        if self.telemetry_error:
            return {"state": "failed", "mode": None, "source": None, "tractors": [],
                    "error": self.telemetry_error}
        return {"state": "inactive", "mode": None, "source": None, "tractors": [],
                "error": "",
                "detail": "no telemetry source configured (integrations.telemetry.source "
                          "/ HARVEST_TELEMETRY_SOURCE=csv, or ./run_harvest_dashboard.sh replay)"}

    def device_diagnostics(self) -> List[Dict[str, Any]]:
        """Per-device rows for the Diagnostics view.

        ``devices`` backend: real protocol/reachability data from
        :meth:`DeviceFleetInterface.diagnostics`.  ``sim`` backend: the same
        row shape synthesised from the reference simulation, with protocol
        ``sim`` — deliberate simulation must read as such, never as a failure
        or as a live protocol (WISEPACK's honesty rule).
        """
        diag = getattr(self.fleet, "diagnostics", None)
        if callable(diag):
            with self._lock:
                rows = list(diag())
            return rows + self._telemetry_rows()
        now = time.time()
        with self._lock:
            snap = self.fleet.snapshot()          # backend only: real rows come from telemetry
        rows: List[Dict[str, Any]] = []
        for t in snap.tractors:
            rows.append(_sim_row(t.id, "tractor", now, {
                "soc_pct": round(t.soc_pct, 2), "charging": float(t.charging),
                "discharging": float(t.discharging)}))
        for c in snap.chargers:
            rows.append(_sim_row(c.id, "charger", now, {
                "power_kw": round(c.power_kw, 3),
                "occupied": 1.0 if c.occupied_by else 0.0}))
        for l in snap.loads:
            rows.append(_sim_row(l.id, "load", now, {
                "power_kw": round(l.power_kw, 3), "shed": float(l.shed)}))
        rows.append(_sim_row("grid", "grid", now, {
            "grid_draw_kw": round(snap.grid.grid_draw_kw, 3),
            "pv_kw": round(snap.grid.pv_kw, 3),
            "grid_cap_kw": round(snap.grid.grid_cap_kw, 3)}))
        return rows + self._telemetry_rows()

    def _telemetry_rows(self) -> List[Dict[str, Any]]:
        """Device rows for real tractors, tagged with the source kind (``csv-replay``)."""
        if self.telemetry is None:
            return []
        try:
            return self.telemetry.device_rows()
        except Exception:                                          # noqa: BLE001
            return []

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.telemetry is not None:
            self.telemetry.close()
        close = getattr(self.fleet, "close", None)
        if callable(close):
            close()

    # -- background advance ----------------------------------------------------
    def _advance_loop(self) -> None:
        while not self._stop.wait(1.0):
            with self._lock:
                self.fleet.advance(self.sim_minutes_per_s)


def _sim_row(dev_id: str, kind: str, now: float, values: Dict[str, float]) -> Dict[str, Any]:
    return {
        "id": dev_id, "kind": kind, "protocol": "sim",
        "endpoint": "in-process reference simulation",
        "reachable": True, "last_read_ts": now, "values": values,
    }


