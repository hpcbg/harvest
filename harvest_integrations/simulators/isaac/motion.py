"""
Field kinematics for the GPU-free stand-in simulator (``isaac-demo``).

Pure stdlib, no ROS, no Isaac, no GPU: this is what lets the whole
HARVEST -> ROS 2 -> simulator -> HARVEST loop be exercised on any machine.

IT IS NOT WHAT ISAAC RUNS, and the difference is deliberate.  Here a tractor
INTERPOLATES toward its goal: it slides in a straight line at a constant speed
and counts as *docked* when a charger is assigned and it is within the dock
radius.  In Isaac the same tractor is a PhysX articulation that has to turn its
wheels, accelerate, steer and brake (see ``robots.py``), so it takes a curved
path, overshoots slightly and settles.  Both report the same wire contract and
the same physical-state vocabulary, and Diagnostics always says which one is
connected -- ``simulated`` for this, ``healthy`` for Isaac -- because a
stand-in must never read as the real thing.

What this module is good for: the contract, the goal handling, the docking
predicate and the whole bridge round-trip, none of which need a GPU.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .contract import (DEFAULT_DOCK_RADIUS_M, DEFAULT_SPEED_MPS, activity,
                       physical_state)


@dataclass
class SimEntity:
    id: str
    kind: str                                    # "tractor" | "charger"
    pose: list                                   # [x, y]
    heading_deg: float = 0.0
    target: Optional[list] = None
    docked_charger: Optional[str] = None
    charging: bool = False
    active: bool = False                         # chargers only
    power_kw: float = 0.0
    moving: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)
    # ---- HARVEST task execution (goals in, progress out) --------------------
    # None of this is a decision: the task id, the work radius and the number of
    # seconds all arrive from HARVEST in the command message.  What lives here is
    # a stopwatch and a distance test.
    activity_kind: str = "idle"                  # "task" | "charging" | "idle"
    task_id: Optional[str] = None
    work_seconds: float = 0.0
    work_radius_m: float = 6.0
    at_task: bool = False
    work_started: Optional[float] = None         # time.monotonic()
    task_progress_pct: float = 0.0
    transit_progress_pct: float = 0.0
    task_complete: bool = False
    start_distance_m: Optional[float] = None


class FieldKinematics:
    """Applies contract messages and advances entity motion."""

    def __init__(self, speed_mps: float = DEFAULT_SPEED_MPS,
                 dock_radius_m: float = DEFAULT_DOCK_RADIUS_M):
        self.speed_mps = speed_mps
        self.dock_radius_m = dock_radius_m
        self.entities: Dict[str, SimEntity] = {}
        self.scene_fingerprint = ""
        self.field: Dict[str, float] = {"width": 100.0, "height": 100.0}
        self.sim_time_s = 0.0
        self._started = time.time()

    # -- command intake -------------------------------------------------------
    def apply_command(self, cmd: Dict[str, Any]) -> bool:
        """Apply one command message; returns True if a scene was (re)built."""
        rebuilt = False
        scene = cmd.get("scene")
        fingerprint = str(cmd.get("scene_fingerprint") or "")
        if scene and fingerprint != self.scene_fingerprint:
            self._build_scene(scene, fingerprint)
            rebuilt = True

        for tid, goal in (cmd.get("goals") or {}).items():
            ent = self.entities.get(str(tid))
            if ent is None or ent.kind != "tractor":
                continue
            ent.docked_charger = goal.get("docked_charger")
            ent.charging = bool(goal.get("charging"))
            ent.extras["soc_pct"] = goal.get("soc_pct")
            ent.activity_kind = str(goal.get("activity") or "idle")
            task_id = goal.get("task_id")
            if task_id != ent.task_id:
                # A new assignment (or none): the stopwatch restarts.  Driven
                # entirely by HARVEST -- nothing here ever decides that a task
                # is over and moves on to another.
                ent.task_id = task_id
                ent.at_task = False
                ent.work_started = None
                ent.task_progress_pct = 0.0
                ent.transit_progress_pct = 0.0
                ent.task_complete = False
                ent.start_distance_m = None
            ent.work_seconds = float(goal.get("work_seconds", 0.0) or 0.0)
            ent.work_radius_m = float(
                goal.get("work_radius_m", self.dock_radius_m) or self.dock_radius_m)
            target = goal.get("target")
            # Idle means HOLD, matching the Isaac backend: after finishing a task
            # the tractor is physically at it, and driving back to HARVEST's
            # stale semantic position would be motion nobody asked for.
            if ent.activity_kind == "idle":
                ent.target = None
            else:
                ent.target = ([float(target[0]), float(target[1])]
                              if target else None)

        for cid, state in (cmd.get("chargers") or {}).items():
            ent = self.entities.get(str(cid))
            if ent is None or ent.kind != "charger":
                continue
            ent.active = bool(state.get("active"))
            ent.power_kw = float(state.get("power_kw", 0.0))
        return rebuilt

    def _build_scene(self, scene: Dict[str, Any], fingerprint: str) -> None:
        previous = self.entities
        self.entities = {}
        self.field = dict(scene.get("field") or self.field)
        for spec in scene.get("entities", []):
            eid, kind = str(spec["id"]), str(spec.get("kind", "tractor"))
            pose = spec.get("home") or spec.get("pose") or [50.0, 50.0]
            ent = SimEntity(id=eid, kind=kind,
                            pose=[float(pose[0]), float(pose[1])])
            # A same-id entity keeps its physical pose across additive scene
            # updates -- a scene refresh must not teleport a moving tractor.
            if eid in previous and previous[eid].kind == kind:
                ent.pose = list(previous[eid].pose)
                ent.heading_deg = previous[eid].heading_deg
            self.entities[eid] = ent
        self.scene_fingerprint = fingerprint

    # -- physics --------------------------------------------------------------
    def step(self, dt_s: float) -> None:
        self.sim_time_s += max(0.0, dt_s)
        for ent in self.entities.values():
            if ent.kind != "tractor":
                continue
            self._advance_motion(ent, dt_s)
            self._advance_work(ent)

    def _advance_motion(self, ent: SimEntity, dt_s: float) -> None:
        if not ent.target:
            ent.moving = False
            return
        # Stop at the work radius when driving to a TASK, not at its exact
        # centre: the radius is HARVEST's definition of "there", and the Isaac
        # backend arrives on the same number.  A charger is still approached
        # exactly -- the tractor parks on the pad, and `_docked` already has the
        # dock radius for deciding whether it counts as connected.
        arrival = ent.work_radius_m if ent.activity_kind == "task" else 0.0
        dx = ent.target[0] - ent.pose[0]
        dy = ent.target[1] - ent.pose[1]
        dist = math.hypot(dx, dy)
        if dist <= max(arrival, 1e-6):
            ent.moving = False
            return
        step = self.speed_mps * dt_s
        if dist <= max(step, 1e-6):
            ent.pose = [float(ent.target[0]), float(ent.target[1])]
            ent.moving = False
        else:
            ent.pose[0] += dx / dist * step
            ent.pose[1] += dy / dist * step
            ent.heading_deg = math.degrees(math.atan2(dy, dx))
            ent.moving = True

    def _advance_work(self, ent: SimEntity) -> None:
        """Execute the work HARVEST assigned: a distance test and a stopwatch."""
        if ent.activity_kind != "task" or not ent.task_id or not ent.target:
            ent.at_task = False
            return
        dist = math.hypot(ent.target[0] - ent.pose[0], ent.target[1] - ent.pose[1])
        if ent.start_distance_m is None:
            ent.start_distance_m = max(dist, ent.work_radius_m)
        span = max(1e-6, ent.start_distance_m - ent.work_radius_m)
        ent.transit_progress_pct = max(0.0, min(
            100.0, (1.0 - (dist - ent.work_radius_m) / span) * 100.0))
        if dist <= ent.work_radius_m and not ent.moving:
            ent.at_task = True
            if ent.work_started is None:
                ent.work_started = time.monotonic()
        if ent.work_started is not None:
            seconds = max(1.0, ent.work_seconds or 30.0)
            elapsed = time.monotonic() - ent.work_started
            ent.task_progress_pct = min(100.0, elapsed / seconds * 100.0)
            ent.task_complete = ent.task_progress_pct >= 100.0

    # -- reporting ------------------------------------------------------------
    def _docked(self, ent: SimEntity) -> bool:
        if ent.kind != "tractor" or not ent.docked_charger:
            return False
        charger = self.entities.get(ent.docked_charger)
        if charger is None:
            return False
        return (math.hypot(charger.pose[0] - ent.pose[0],
                           charger.pose[1] - ent.pose[1]) <= self.dock_radius_m)

    def distance_to_target(self, ent: SimEntity) -> float:
        if not ent.target:
            return 0.0
        return math.hypot(ent.target[0] - ent.pose[0], ent.target[1] - ent.pose[1])

    def telemetry_entities(self) -> Dict[str, Any]:
        """Per-entity physical state, from the *simulated* poses (measured,
        never echoed from the goals -- the WISEPACK acknowledgement rule)."""
        out: Dict[str, Any] = {}
        for ent in self.entities.values():
            row: Dict[str, Any] = {
                "kind": ent.kind,
                "pose": [round(ent.pose[0], 2), round(ent.pose[1], 2)],
            }
            if ent.kind == "tractor":
                docked = self._docked(ent)
                row.update({
                    "heading_deg": round(ent.heading_deg, 1),
                    "moving": ent.moving,
                    "docked": docked,
                    "docked_charger": ent.docked_charger,
                    "charging": ent.charging,
                    # The same vocabulary the Isaac backend reports, from the
                    # one definition in contract.py, so Diagnostics reads a
                    # stand-in run and a real one the same way.
                    "physical_state": physical_state(
                        moving=ent.moving, docked=docked, charging=ent.charging),
                    "distance_to_target_m": round(self.distance_to_target(ent), 2),
                    # ---- the task half, reported the same way Isaac does -----
                    "activity": activity(
                        task_id=ent.task_id, at_task=ent.at_task,
                        charging=ent.charging, moving=ent.moving),
                    "task_id": ent.task_id,
                    "at_task": ent.at_task,
                    "transit_progress_pct": round(ent.transit_progress_pct, 1),
                    "task_progress_pct": round(ent.task_progress_pct, 1),
                    "task_complete": ent.task_complete,
                })
            else:
                row.update({"active": ent.active,
                            "power_kw": round(ent.power_kw, 3)})
            out[ent.id] = row
        return out

    def stats(self) -> Dict[str, Any]:
        return {
            "entities_synced": len(self.entities),
            "sim_time_s": round(self.sim_time_s, 1),
            "uptime_s": round(time.time() - self._started, 1),
        }
