"""
DeviceFleetInterface -- a :class:`harvest_control.FleetInterface` backend that
reads and actuates *real* field devices through the ``DeviceIO`` seam.

This is the third implementation of the fleet contract, next to
``SimulationFleetInterface`` (pure Python) and the ROS 2 bridge:

    decision layer -> FleetInterface -> { simulation | ROS 2 | field devices }

The HARVEST semantic model stays authoritative: everything returned from
:meth:`snapshot` is a ``harvest_control`` dataclass, and protocol details
(registers, node ids, scaling) never cross this module's boundary.

Wire conventions (matching the bundled farm simulators, overridable per point
in configuration):

======== ==========================================================
kind     expected points
======== ==========================================================
grid     clock_min, grid_draw_kw, grid_cap_kw, pv_kw, tariff_code,
         price_eur_per_kwh
tractor  soc_pct, energy_kwh, available, charging, discharging,
         discharge_kw, pos_x, pos_y  [+ writable charge_request, v2l_kw]
charger  power_kw, level, occupied   [level writable 0=off 1=half 2=full;
         occupied = 1-based index into the configured tractor list, 0=free]
load     power_kw, shed              [shed writable 0/1]
======== ==========================================================
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Sequence

from harvest_control.interface import (
    ChargerLevel,
    ChargerState,
    Command,
    CommandAck,
    CommandType,
    FleetInterface,
    FleetSnapshot,
    GridState,
    LoadState,
    TractorState,
)

from .base import DeviceEndpoint, DeviceIO, create_device_io

_TARIFF_BY_CODE = {0: "valle", 1: "llano", 2: "punta"}
_LEVEL_BY_CODE = {0: ChargerLevel.OFF, 1: ChargerLevel.HALF, 2: ChargerLevel.FULL}
_CODE_BY_LEVEL = {v: k for k, v in _LEVEL_BY_CODE.items()}


class DeviceFleetInterface(FleetInterface):
    """Aggregates one ``DeviceIO`` per endpoint into fleet snapshots/commands."""

    def __init__(self, endpoints: Sequence[DeviceEndpoint]):
        self._endpoints: Dict[str, DeviceEndpoint] = {}
        self._ios: Dict[str, DeviceIO] = {}
        self._last_good: Dict[str, Dict[str, float]] = {}
        # Per-device read health, kept for the diagnostics view:
        # device_id -> {"reachable": bool, "ts": epoch of last attempt,
        #               "ok_ts": epoch of last successful read}
        self._read_health: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        for ep in endpoints:
            self._endpoints[ep.device_id] = ep
            self._ios[ep.device_id] = create_device_io(ep)
        self._tractor_ids: List[str] = [
            ep.device_id for ep in endpoints if ep.kind == "tractor"
        ]

    # -- FleetInterface -------------------------------------------------------
    def snapshot(self) -> FleetSnapshot:
        with self._lock:
            readings = {dev_id: io.read() for dev_id, io in self._ios.items()}
        now = time.time()
        for dev_id, vals in readings.items():
            health = self._read_health.setdefault(dev_id, {"ok_ts": None})
            health["reachable"] = vals is not None
            health["ts"] = now
            if vals is not None:
                health["ok_ts"] = now
        grid = GridState(0, 0.0, 0.0, 0.0, "valle", 0.0)
        tractors: List[TractorState] = []
        chargers: List[ChargerState] = []
        loads: List[LoadState] = []

        for dev_id, ep in self._endpoints.items():
            vals = readings.get(dev_id)
            reachable = vals is not None
            if reachable:
                self._last_good[dev_id] = vals
            else:
                vals = self._last_good.get(dev_id, {})

            if ep.kind == "grid":
                grid = GridState(
                    clock_min=int(vals.get("clock_min", 0)),
                    grid_draw_kw=vals.get("grid_draw_kw", 0.0),
                    grid_cap_kw=vals.get("grid_cap_kw", 0.0),
                    pv_kw=vals.get("pv_kw", 0.0),
                    tariff=_TARIFF_BY_CODE.get(int(vals.get("tariff_code", 0)), "valle"),
                    price_eur_per_kwh=vals.get("price_eur_per_kwh", 0.0),
                )
            elif ep.kind == "tractor":
                pos = None
                if "pos_x" in vals and "pos_y" in vals:
                    pos = (vals["pos_x"], vals["pos_y"])
                tractors.append(TractorState(
                    id=dev_id,
                    soc_pct=vals.get("soc_pct", 0.0),
                    energy_kwh=vals.get("energy_kwh", 0.0),
                    available=reachable and bool(round(vals.get("available", 1.0))),
                    charging=bool(round(vals.get("charging", 0.0))),
                    current_task=None,       # task state lives in HARVEST, not on the wire
                    position=pos,
                    discharging=bool(round(vals.get("discharging", 0.0))),
                    discharge_kw=vals.get("discharge_kw", 0.0),
                ))
            elif ep.kind == "charger":
                occ = int(round(vals.get("occupied", 0.0)))
                occupied_by = (
                    self._tractor_ids[occ - 1]
                    if 0 < occ <= len(self._tractor_ids) else None
                )
                chargers.append(ChargerState(
                    id=dev_id,
                    level=_LEVEL_BY_CODE.get(int(round(vals.get("level", 0.0))), ChargerLevel.OFF),
                    power_kw=vals.get("power_kw", 0.0),
                    occupied_by=occupied_by,
                ))
            elif ep.kind == "load":
                loads.append(LoadState(
                    id=dev_id,
                    name=str(ep.options.get("name", dev_id)),
                    shed=bool(round(vals.get("shed", 0.0))),
                    power_kw=vals.get("power_kw", 0.0),
                ))
        return FleetSnapshot(grid=grid, tractors=tractors, chargers=chargers, loads=loads)

    def submit(self, commands: Sequence[Command]) -> list[CommandAck]:
        return [self._submit_one(cmd) for cmd in commands]

    def is_real_time(self) -> bool:
        return True

    def diagnostics(self) -> List[Dict[str, Any]]:
        """Per-device health rows for the Diagnostics view.

        Reflects the *last snapshot's* read results; call :meth:`snapshot`
        first for fresh data.  Values are last-good engineering units, so a
        temporarily unreachable device still shows what it last reported.
        """
        rows: List[Dict[str, Any]] = []
        for dev_id, ep in self._endpoints.items():
            health = self._read_health.get(dev_id, {})
            rows.append({
                "id": dev_id,
                "kind": ep.kind,
                "protocol": ep.protocol,
                "endpoint": _endpoint_label(ep),
                "reachable": health.get("reachable"),   # None = not read yet
                "last_read_ts": health.get("ok_ts"),
                "values": dict(self._last_good.get(dev_id, {})),
            })
        return rows

    def close(self) -> None:
        with self._lock:
            for io in self._ios.values():
                io.close()

    # -- command mapping ------------------------------------------------------
    def _submit_one(self, cmd: Command) -> CommandAck:
        try:
            if cmd.type == CommandType.SET_CHARGER_LEVEL:
                level = cmd.value if isinstance(cmd.value, ChargerLevel) else ChargerLevel(cmd.value)
                self._write(cmd.target_id, "level", _CODE_BY_LEVEL[level])
            elif cmd.type == CommandType.SHED_LOAD:
                self._write(cmd.target_id, "shed", 1)
            elif cmd.type == CommandType.RESTORE_LOAD:
                self._write(cmd.target_id, "shed", 0)
            elif cmd.type == CommandType.REQUEST_CHARGE:
                self._write(cmd.target_id, "charge_request", 1)
            elif cmd.type == CommandType.RELEASE_CHARGE:
                self._write(cmd.target_id, "charge_request", 0)
            elif cmd.type == CommandType.V2L_START:
                self._write(cmd.target_id, "v2l_kw", float(cmd.value or 0.0))
            elif cmd.type == CommandType.V2L_STOP:
                self._write(cmd.target_id, "v2l_kw", 0.0)
            else:
                # Task assignment stays inside HARVEST's decision layer; field
                # devices have no notion of it.
                return CommandAck(cmd, False,
                                  f"{cmd.type.value} not supported by device backend")
            return CommandAck(cmd, True)
        except Exception as exc:
            return CommandAck(cmd, False, str(exc))

    def _write(self, device_id: str, point: str, value: float) -> None:
        io = self._ios.get(device_id)
        if io is None:
            raise KeyError(f"Unknown device id {device_id!r}")
        with self._lock:
            io.write(point, value)


def _endpoint_label(ep: DeviceEndpoint) -> str:
    """Human-readable connection target (no secrets — host/port/URL only)."""
    if ep.protocol == "modbus":
        return f"{ep.options.get('host', '?')}:{ep.options.get('port', '?')}"
    if ep.protocol == "opcua":
        return str(ep.options.get("endpoint", "?"))
    return ep.protocol
