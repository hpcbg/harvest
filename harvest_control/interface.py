"""
harvest_control.interface
==========================

Transport-agnostic control interface for the HARVEST fleet.

This module defines the single boundary between the *decision layer* (the
scheduling / MARL agents) and the *plant* (the simulation today, the real
ZETRABOT tractors in Stage 3).

The decision layer only ever talks to a :class:`FleetInterface`.  It does not
know, and does not care, whether the other side is the pilot6 simulation or a
ROS 2 bridge to real hardware.  Swapping the backend therefore requires **no
change** to the control logic -- which is exactly the property that lets the
autonomy validated in Stage 2 (TPI2) be carried over to the September field
deployment unchanged.

    decision layer  ->  FleetInterface  ->  { SimulationFleetInterface | Ros2FleetInterface }
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple


# --------------------------------------------------------------------------- #
#  Enumerations
# --------------------------------------------------------------------------- #
class ChargerLevel(Enum):
    OFF = "off"
    HALF = "half"      # ~50 % rated power
    FULL = "full"      # rated power


class CommandType(Enum):
    REQUEST_CHARGE = "request_charge"      # dock a tractor and start drawing power
    RELEASE_CHARGE = "release_charge"      # stop charging / undock
    SET_CHARGER_LEVEL = "set_charger_level"
    ASSIGN_TASK = "assign_task"
    PREEMPT_TASK = "preempt_task"
    SHED_LOAD = "shed_load"
    RESTORE_LOAD = "restore_load"


# --------------------------------------------------------------------------- #
#  State reports  (plant -> decision layer)
# --------------------------------------------------------------------------- #
@dataclass
class TractorState:
    id: str
    soc_pct: float                       # 0..100
    energy_kwh: float
    available: bool                      # False when offline / broken down
    charging: bool
    current_task: Optional[str]
    position: Optional[Tuple[float, float]] = None


@dataclass
class ChargerState:
    id: str
    level: ChargerLevel
    power_kw: float
    occupied_by: Optional[str]           # tractor id or None


@dataclass
class LoadState:
    id: str
    name: str
    shed: bool
    power_kw: float


@dataclass
class GridState:
    clock_min: int                       # minutes since midnight
    grid_draw_kw: float
    grid_cap_kw: float
    pv_kw: float
    tariff: str                          # "valle" | "llano" | "punta"
    price_eur_per_kwh: float


@dataclass
class FleetSnapshot:
    grid: GridState
    tractors: list[TractorState]
    chargers: list[ChargerState]
    loads: list[LoadState]

    # convenience look-ups
    def tractor(self, tid: str) -> TractorState:
        return next(t for t in self.tractors if t.id == tid)

    def free_chargers(self) -> list[ChargerState]:
        return [c for c in self.chargers if c.occupied_by is None]

    @property
    def grid_headroom_kw(self) -> float:
        return max(0.0, self.grid.grid_cap_kw - self.grid.grid_draw_kw)


# --------------------------------------------------------------------------- #
#  Commands  (decision layer -> plant)
# --------------------------------------------------------------------------- #
@dataclass
class Command:
    type: CommandType
    target_id: str                       # tractor / charger / load id
    value: object = None                 # ChargerLevel, task id, ... (type-specific)

    # ergonomic constructors --------------------------------------------------
    @staticmethod
    def request_charge(tractor_id: str) -> "Command":
        return Command(CommandType.REQUEST_CHARGE, tractor_id)

    @staticmethod
    def release_charge(tractor_id: str) -> "Command":
        return Command(CommandType.RELEASE_CHARGE, tractor_id)

    @staticmethod
    def set_charger_level(charger_id: str, level: ChargerLevel) -> "Command":
        return Command(CommandType.SET_CHARGER_LEVEL, charger_id, level)

    @staticmethod
    def assign_task(tractor_id: str, task_id: str) -> "Command":
        return Command(CommandType.ASSIGN_TASK, tractor_id, task_id)

    @staticmethod
    def preempt_task(tractor_id: str) -> "Command":
        return Command(CommandType.PREEMPT_TASK, tractor_id)

    @staticmethod
    def shed_load(load_id: str) -> "Command":
        return Command(CommandType.SHED_LOAD, load_id)

    @staticmethod
    def restore_load(load_id: str) -> "Command":
        return Command(CommandType.RESTORE_LOAD, load_id)


@dataclass
class CommandAck:
    command: Command
    accepted: bool
    reason: str = ""


# --------------------------------------------------------------------------- #
#  The interface
# --------------------------------------------------------------------------- #
class FleetInterface(ABC):
    """Contract that every backend (simulation or hardware) must implement.

    Only three methods matter to the decision layer:

    * :meth:`snapshot` -- read the current fleet + grid/PV state;
    * :meth:`submit`   -- issue a batch of actuation commands;
    * :meth:`advance`  -- (simulation only) step time forward.

    A real-hardware backend runs in real time, so :meth:`advance` is optional
    and :meth:`is_real_time` returns ``True``.
    """

    @abstractmethod
    def snapshot(self) -> FleetSnapshot:
        """Return the current state of the whole fleet and the grid/PV signals."""

    @abstractmethod
    def submit(self, commands: Sequence[Command]) -> list[CommandAck]:
        """Apply a batch of actuation commands; return one ack per command."""

    def advance(self, minutes: float) -> None:
        """Advance a *simulation* backend by ``minutes`` (no-op for hardware)."""
        return None

    def is_real_time(self) -> bool:
        return False
