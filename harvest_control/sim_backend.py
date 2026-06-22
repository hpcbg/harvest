"""
harvest_control.sim_backend
===========================

A :class:`~harvest_control.interface.FleetInterface` backed by a simulation.

Two ways to use it
-------------------
1.  **With the pilot6 engine (production path).**  Construct
    :class:`SimulationFleetInterface` with your existing environment object and
    a small set of adapter callables that map your internal objects onto the
    interface dataclasses.  Nothing in the decision layer changes::

        iface = SimulationFleetInterface(env=my_env, adapters=PilotSixAdapters(...))

2.  **Standalone (out-of-the-box).**  If no ``env`` is supplied, a small,
    self-contained reference simulation (:class:`_ReferenceSim`) is used so the
    interface and the autonomous-control demo run with zero dependencies.  The
    reference sim is intentionally compact -- it exists to exercise the
    *interface contract*, not to replicate pilot6 physics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

try:
    from .interface import (
        ChargerLevel, ChargerState, Command, CommandAck, CommandType,
        FleetInterface, FleetSnapshot, GridState, LoadState, TractorState,
    )
except ImportError:  # running as standalone script from package directory
    from interface import (
        ChargerLevel, ChargerState, Command, CommandAck, CommandType,
        FleetInterface, FleetSnapshot, GridState, LoadState, TractorState,
    )

_LEVEL_FRACTION = {ChargerLevel.OFF: 0.0, ChargerLevel.HALF: 0.5, ChargerLevel.FULL: 1.0}


# --------------------------------------------------------------------------- #
#  Reference simulation (standalone fallback)
# --------------------------------------------------------------------------- #
@dataclass
class _Tractor:
    id: str
    capacity_kwh: float = 45.0
    soc_kwh: float = 13.5            # ~30 %
    available: bool = True
    charging: bool = False
    charger_id: Optional[str] = None
    task: Optional[str] = None
    work_draw_kw: float = 5.0        # consumption while executing a task
    discharging: bool = False        # V2L mode
    discharge_kw: float = 0.0


@dataclass
class _Charger:
    id: str
    rated_kw: float = 6.6
    level: ChargerLevel = ChargerLevel.OFF
    occupied_by: Optional[str] = None

    @property
    def power_kw(self) -> float:
        return self.rated_kw * _LEVEL_FRACTION[self.level] if self.occupied_by else 0.0


@dataclass
class _Load:
    id: str
    name: str
    nominal_kw: float
    shed: bool = False
    critical: bool = False

    @property
    def power_kw(self) -> float:
        return 0.0 if self.shed else self.nominal_kw


class _ReferenceSim:
    """Minimal but coherent farm-energy simulation (15-min granularity)."""

    def __init__(self, grid_cap_kw: float = 10.5):
        self.clock_min = 0
        self.grid_cap_kw = grid_cap_kw
        self.tractors = [_Tractor("T1", soc_kwh=36.0), _Tractor("T2", soc_kwh=24.8),
                         _Tractor("T3", soc_kwh=13.0)]
        self.chargers = [_Charger("CH1"), _Charger("CH2")]
        self.loads = [
            _Load("LD1", "workshop", 1.8),
            _Load("LD2", "office_hvac", 2.4),
            _Load("LD3", "barn_doors", 0.6, critical=True),
            _Load("LD4", "outdoor_lighting", 0.9),
        ]

    # --- environment signals ------------------------------------------------
    def pv_kw(self) -> float:
        # simple daylight bell curve peaking ~13:00 at 5 kW
        h = self.clock_min / 60.0
        if h < 6 or h > 20:
            return 0.0
        return round(5.0 * math.exp(-((h - 13.0) ** 2) / (2 * 2.6 ** 2)), 3)

    def tariff(self) -> tuple[str, float]:
        h = self.clock_min / 60.0
        if h < 8:
            return "valle", 0.10
        if 10 <= h < 14 or 18 <= h < 22:
            return "punta", 0.28
        return "llano", 0.17

    def grid_draw_kw(self) -> float:
        load = sum(c.power_kw for c in self.chargers) + sum(l.power_kw for l in self.loads)
        return round(max(0.0, load - self.pv_kw()), 3)

    # --- dynamics -----------------------------------------------------------
    def step(self, minutes: float) -> None:
        dt_h = minutes / 60.0
        for t in self.tractors:
            if not t.available:
                continue
            if t.discharging and t.discharge_kw > 0:
                t.soc_kwh = max(0.0, t.soc_kwh - t.discharge_kw * dt_h)
                if 100 * t.soc_kwh / t.capacity_kwh <= 35.0:
                    t.discharging, t.discharge_kw = False, 0.0
            elif t.charging and t.charger_id is not None:
                ch = next(c for c in self.chargers if c.id == t.charger_id)
                t.soc_kwh = min(t.capacity_kwh, t.soc_kwh + ch.power_kw * dt_h)
            elif t.task is not None:
                t.soc_kwh = max(0.0, t.soc_kwh - t.work_draw_kw * dt_h)
        self.clock_min += int(minutes)


# --------------------------------------------------------------------------- #
#  Adapters (for binding to the real pilot6 environment)
# --------------------------------------------------------------------------- #
@dataclass
class Adapters:
    """Callables mapping a host environment onto the interface dataclasses.

    Supply these when wrapping the pilot6 engine.  Each callable receives the
    host ``env`` and returns the corresponding interface objects / values.
    Left ``None`` (the default) the :class:`_ReferenceSim` mapping is used.
    """
    read_tractors: Optional[callable] = None
    read_chargers: Optional[callable] = None
    read_loads: Optional[callable] = None
    read_grid: Optional[callable] = None
    apply_command: Optional[callable] = None
    step: Optional[callable] = None


# --------------------------------------------------------------------------- #
#  The simulation-backed interface
# --------------------------------------------------------------------------- #
class SimulationFleetInterface(FleetInterface):
    def __init__(self, env=None, adapters: Optional[Adapters] = None, grid_cap_kw: float = 10.5):
        self.env = env if env is not None else _ReferenceSim(grid_cap_kw=grid_cap_kw)
        self.adapters = adapters or Adapters()
        self._using_reference = env is None

    # -- read ----------------------------------------------------------------
    def snapshot(self) -> FleetSnapshot:
        a = self.adapters
        if not self._using_reference and a.read_grid:           # production path
            return FleetSnapshot(a.read_grid(self.env), a.read_tractors(self.env),
                                 a.read_chargers(self.env), a.read_loads(self.env))
        sim: _ReferenceSim = self.env                            # reference path
        tariff, price = sim.tariff()
        grid = GridState(sim.clock_min, sim.grid_draw_kw(), sim.grid_cap_kw,
                         sim.pv_kw(), tariff, price)
        tractors = [TractorState(t.id, round(100 * t.soc_kwh / t.capacity_kwh, 1),
                                 round(t.soc_kwh, 2), t.available, t.charging, t.task,
                                 discharging=t.discharging, discharge_kw=t.discharge_kw)
                    for t in sim.tractors]
        chargers = [ChargerState(c.id, c.level, c.power_kw, c.occupied_by) for c in sim.chargers]
        loads = [LoadState(l.id, l.name, l.shed, l.power_kw) for l in sim.loads]
        return FleetSnapshot(grid, tractors, chargers, loads)

    # -- actuate -------------------------------------------------------------
    def submit(self, commands: Sequence[Command]) -> list[CommandAck]:
        if not self._using_reference and self.adapters.apply_command:
            return [self.adapters.apply_command(self.env, c) for c in commands]
        return [self._apply_reference(c) for c in commands]

    def _apply_reference(self, c: Command) -> CommandAck:
        sim: _ReferenceSim = self.env
        try:
            if c.type == CommandType.REQUEST_CHARGE:
                t = next(x for x in sim.tractors if x.id == c.target_id)
                if not t.available:
                    return CommandAck(c, False, "tractor offline")
                if t.charging:
                    return CommandAck(c, True, "already charging")
                free = next((ch for ch in sim.chargers if ch.occupied_by is None), None)
                if free is None:
                    return CommandAck(c, False, "no free charger")
                free.occupied_by = t.id
                free.level = ChargerLevel.FULL
                t.charging = True
                t.charger_id = free.id
                t.task = None
                return CommandAck(c, True, f"docked at {free.id}")

            if c.type == CommandType.RELEASE_CHARGE:
                t = next(x for x in sim.tractors if x.id == c.target_id)
                if t.charger_id is not None:
                    ch = next(x for x in sim.chargers if x.id == t.charger_id)
                    ch.occupied_by, ch.level = None, ChargerLevel.OFF
                t.charging, t.charger_id = False, None
                return CommandAck(c, True, "released")

            if c.type == CommandType.SET_CHARGER_LEVEL:
                ch = next(x for x in sim.chargers if x.id == c.target_id)
                ch.level = c.value if isinstance(c.value, ChargerLevel) else ChargerLevel(c.value)
                return CommandAck(c, True, f"level={ch.level.value}")

            if c.type == CommandType.ASSIGN_TASK:
                t = next(x for x in sim.tractors if x.id == c.target_id)
                if t.charging:
                    return CommandAck(c, False, "busy charging")
                t.task = str(c.value)
                return CommandAck(c, True, f"task={c.value}")

            if c.type == CommandType.PREEMPT_TASK:
                t = next(x for x in sim.tractors if x.id == c.target_id)
                t.task = None
                return CommandAck(c, True, "preempted")

            if c.type in (CommandType.SHED_LOAD, CommandType.RESTORE_LOAD):
                l = next(x for x in sim.loads if x.id == c.target_id)
                if l.critical and c.type == CommandType.SHED_LOAD:
                    return CommandAck(c, False, "critical load")
                l.shed = (c.type == CommandType.SHED_LOAD)
                return CommandAck(c, True, "shed" if l.shed else "restored")

            if c.type == CommandType.V2L_START:
                t = next(x for x in sim.tractors if x.id == c.target_id)
                if not t.available:
                    return CommandAck(c, False, "tractor offline")
                if t.task is not None:
                    return CommandAck(c, False, "tractor has active task")
                kw = float(c.value) if c.value else 3.3
                t.discharging, t.discharge_kw = True, kw
                t.charging, t.charger_id = False, None
                return CommandAck(c, True, f"v2l {kw:.1f} kW")

            if c.type == CommandType.V2L_STOP:
                t = next(x for x in sim.tractors if x.id == c.target_id)
                t.discharging, t.discharge_kw = False, 0.0
                return CommandAck(c, True, "v2l stopped")

            return CommandAck(c, False, "unknown command")
        except StopIteration:
            return CommandAck(c, False, "unknown target id")

    # -- time ----------------------------------------------------------------
    def advance(self, minutes: float) -> None:
        if not self._using_reference and self.adapters.step:
            self.adapters.step(self.env, minutes)
        else:
            self.env.step(minutes)

    # -- helper for the breakdown demo --------------------------------------
    def force_offline(self, tractor_id: str) -> None:
        """Inject a breakdown (used by the dynamic-events demo)."""
        if self._using_reference:
            t = next(x for x in self.env.tractors if x.id == tractor_id)
            if t.charger_id is not None:
                ch = next(x for x in self.env.chargers if x.id == t.charger_id)
                ch.occupied_by, ch.level = None, ChargerLevel.OFF
            t.available, t.charging, t.charger_id, t.task = False, False, None, None
