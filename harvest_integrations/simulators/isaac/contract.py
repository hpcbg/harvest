"""
HARVEST <-> simulator wire contract (schema ``harvest-sim/1.1``).

One pure-stdlib module imported by *both* ends of the simulation channel --
the ROS 2 Isaac bridge (container), the GPU-free stub simulator (container)
and the standalone app inside Isaac Sim's bundled Python (host) -- so the two
sides cannot drift (the WISEPACK ``isaac_contract`` rule).  It must therefore
import nothing beyond the standard library: Isaac's interpreter cannot see
HARVEST's dependencies, and the bridge deliberately carries none.

Transport: two ``std_msgs/String`` topics carrying versioned JSON (the same
decision as the fleet bridge -- no custom message package, because Isaac's
interpreter cannot import a colcon-built package).  Both are RELIABLE +
TRANSIENT_LOCAL + KEEP_LAST(1): Isaac takes tens of seconds to boot and must
not miss the latched scene command; conversely the latest full state is the
only interesting message in either direction, so idempotent full-state
messages replace WISEPACK's per-item event stream and its dedup gate.

Message shapes
--------------

Command (bridge -> simulator), sent on every fleet snapshot::

    {"schema": "harvest-sim/1.1", "kind": "command", "ts": ...,
     "scene": {...} | null,          # included until the fingerprint is acked
     "goals": {tractor_id: {"target": [x,y], "charging": bool,
                            "docked_charger": id|null, "soc_pct": float,
                            "activity": "task"|"charging"|"idle",
                            "task_id": id|null}},
     "chargers": {charger_id: {"active": bool, "power_kw": float}},
     "tasks": {task_id: {"name": str, "location": [x,y],
                         "work_radius_m": float, "state": TASK_STATES,
                         "assigned_tractor": id|null, "priority": str,
                         "progress_pct": float, "work_seconds": float,
                         "emphasis": bool}}}

``tasks`` was added in 1.1 (additive, so a 1.0 reader still works).  Every field
in it is a CONSEQUENCE of a HARVEST decision: where the work is, how long it
takes, which tractor was given it and what to draw.  There are no priorities to
weigh and no queue to sort, because the simulator does not schedule -- see
``harvest_integrations/tasks.py``.

Telemetry (simulator -> bridge), published periodically::

    {"schema": "harvest-sim/1.1", "kind": "telemetry", "ts": ...,
     "simulator": {"kind": "isaac"|"stub", "state": "starting"|"running"|"error",
                   "version": str, "started_ts": float, "detail": str},
     "scene_fingerprint": str,       # acknowledges the applied scene
     "entities": {id: {"kind": ..., "pose": [x,y], "heading_deg": float,
                       "moving": bool, "docked": bool,
                       "physical_state": one of PHYSICAL_STATES,
                       "activity": one of ACTIVITIES,
                       "task_id": id|null, "at_task": bool,
                       "task_progress_pct": float, "task_complete": bool,
                       "distance_to_target_m": float}},
     "stats": {"entities_synced": int, "sim_time_s": float}}

Both simulators fill the fields above.  The Isaac backend adds measured
physical detail the stand-in has no notion of -- ``speed_mps``,
``distance_to_charger_m``, ``destination``, ``driveable`` -- and its
``simulator`` block carries ``visualization`` (how to watch it),
``robot_model`` (which body is standing in for a tractor) and ``physics``.
Every one of those is OPTIONAL by design: a reader must work with the
stand-in's smaller message, and Diagnostics is written to do exactly that.

The scene fingerprint is acknowledged from the *applied* scene, never echoed
from the request (WISEPACK: an echoed acknowledgement is a tautology).

A message whose schema MAJOR does not match is refused, never best-effort
parsed -- a silently mis-parsed pose is an entity moving somewhere nobody
asked it to.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "harvest-sim/1.1"

# Topic names.  harvest_ros.topics defines the same strings for the bridge
# side; tests/test_isaac_contract.py asserts they stay identical.  Neither
# ends in '/status' (Orion-LD's DDS bridge drops such topics -- WISEPACK
# finding, guarded in tests/test_ros_contract.py).
SIM_COMMAND_TOPIC = "/harvest/sim/command"
SIM_TELEMETRY_TOPIC = "/harvest/sim/telemetry"

# The label the Isaac bridge announces itself with (X-Harvest-Client header
# and /api/integrations/status payloads).
BRIDGE_CLIENT = "isaac-bridge"

# Physical defaults for the field plane (config coordinates are metres).
DEFAULT_SPEED_MPS = 4.0        # believable for a farm tractor, demo-friendly
DEFAULT_DOCK_RADIUS_M = 1.5    # "docked" when within this radius of a charger

#: The physical-state vocabulary, defined HERE because both simulators put it on
#: the wire and Diagnostics renders it: one word for what the physics is doing.
#: It is deliberately NOT HARVEST's operational state -- HARVEST derives that
#: from its own model, and a disagreement between the two is a fact an operator
#: needs to see rather than something either side should paper over.
PHYSICAL_STATES = ("parked", "moving", "docked", "charging")


def physical_state(*, moving: bool, docked: bool, charging: bool) -> str:
    """One word for what a tractor is physically doing.  See PHYSICAL_STATES."""
    if charging and docked:
        return "charging"
    if docked:
        return "docked"
    if moving:
        return "moving"
    return "parked"


#: What a tractor is BUSY WITH, which is a different question from what its body
#: is doing (`physical_state`): a parked tractor may be working a task with its
#: PTO, and a moving one may be on its way to a charger.  Diagnostics shows both.
ACTIVITIES = ("idle", "travelling", "working", "charging")


def activity(*, task_id: Optional[str], at_task: bool, charging: bool,
             moving: bool) -> str:
    """One word for what a tractor is busy with.  See ACTIVITIES."""
    if charging:
        return "charging"
    if task_id:
        return "working" if at_task else "travelling"
    return "travelling" if moving else "idle"


# --------------------------------------------------------------------------- #
#  Task vocabulary
# --------------------------------------------------------------------------- #
#: The five task states the demonstrator distinguishes.  HARVEST's own model
#: (``main.Task.phase``) has more phases than this and is authoritative; these
#: are the ones worth DRAWING, and the mapping between them lives here so the
#: scheduler side and the simulator side cannot describe a task differently.
TASK_STATES = ("pending", "assigned", "active", "completed", "deferred", "missed")

#: main.Task.phase -> drawn state.  INTERRUPTED and DELAYED are both "deferred":
#: from the field's point of view they are the same situation -- work that was
#: due and is now waiting again.
_PHASE_TO_STATE = {
    "PENDING": "pending",
    "TRANSIT": "assigned",
    "EXECUTING": "active",
    "DONE": "completed",
    "DELAYED": "deferred",
    "INTERRUPTED": "deferred",
}


def task_state_from_phase(phase: str, missed: bool = False) -> str:
    """Map a HARVEST task phase to the state the demonstrator draws.

    ``missed`` is passed in rather than derived here: whether a window has
    closed for good is a scheduling judgement, and this module deliberately
    holds no clock.
    """
    state = _PHASE_TO_STATE.get(str(phase).upper(), "pending")
    if missed and state in ("pending", "deferred"):
        return "missed"
    return state


# --------------------------------------------------------------------------- #
#  Schema handling
# --------------------------------------------------------------------------- #
def schema_compatible(schema: Any) -> bool:
    """Same MAJOR as ours; MINOR differences are additive and tolerated."""
    if not isinstance(schema, str) or "/" not in schema:
        return False
    name, _, version = schema.partition("/")
    ours = SCHEMA_VERSION.partition("/")
    return name == ours[0] and version.split(".")[0] == ours[2].split(".")[0]


def parse_message(text: str) -> Optional[Dict[str, Any]]:
    """Parse a wire message; return None (refusal) on bad JSON or schema."""
    try:
        raw = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(raw, dict) or not schema_compatible(raw.get("schema")):
        return None
    return raw


# --------------------------------------------------------------------------- #
#  Scene derivation (from HARVEST's config -- the single source of farm
#  structure, same rule as runtime.build_device_endpoints)
# --------------------------------------------------------------------------- #
def scene_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build the simulator scene description from a HARVEST config dict."""
    entities: List[Dict[str, Any]] = []
    for entry in (cfg.get("tractors") or {}).get("fleet") or []:
        if not entry.get("enabled", True):
            continue
        loc = entry.get("initial_location") or {}
        entities.append({
            "id": str(entry["id"]),
            "kind": "tractor",
            "home": [float(loc.get("x", 50.0)), float(loc.get("y", 50.0))],
        })
    for entry in (cfg.get("charging") or {}).get("stations") or []:
        loc = entry.get("location") or {}
        entities.append({
            "id": str(entry["id"]),
            "kind": "charger",
            "pose": [float(loc.get("x", 40.0)), float(loc.get("y", 40.0))],
        })
    xs = [p for e in entities for p in [e.get("home", e.get("pose"))[0]]]
    ys = [p for e in entities for p in [e.get("home", e.get("pose"))[1]]]
    field = {
        "width": (max(xs) + 20.0) if xs else 100.0,
        "height": (max(ys) + 20.0) if ys else 100.0,
    }
    return {"field": field, "entities": entities}


def scene_fingerprint(scene: Dict[str, Any]) -> str:
    """Deterministic digest of a scene, computed identically on both ends."""
    canonical = json.dumps(scene, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
#  Goal derivation (from a harvest-fleet/1.0 snapshot dict)
# --------------------------------------------------------------------------- #
def goals_from_snapshot(scene: Dict[str, Any],
                        snapshot: Dict[str, Any],
                        task_goals: Optional[Dict[str, Any]] = None
                        ) -> Dict[str, Any]:
    """Map the semantic fleet state (+ HARVEST's task assignments) onto
    physical motion goals.

    HARVEST's device model docks instantly (semantic state); the simulator
    executes the physical part -- driving to the assigned charger or to the
    assigned task -- so the goal for a tractor is, in order:

    1. its assigned CHARGER's pose, while it is actually charging (or has no
       work);
    2. its assigned TASK's location, when HARVEST's scheduler has given it one;
    3. its reported field position (or home) -- i.e. stay put.

    ACTIVE CHARGING WINS over a task, and that ordering is HARVEST's rather than
    a preference of this module: HARVEST's scheduler already refuses to give work
    to a charging tractor and the fleet backend refuses to send a tractor with
    live work to a charger, so the two overlap only for the moment of a
    transition -- and during it the truthful thing to draw is the charger.
    Occupying a bay after a FINISHED charge is not charging, and must not
    override an assignment HARVEST has just made; the alternative was measured
    and it left a full tractor parked on a pad while HARVEST believed it was on
    its way to a task.

    ``task_goals`` is the document from ``GET /api/tasks/goals``.  Absent (no
    task service), behaviour is exactly as before: the task layer is additive.
    """
    charger_pose = {e["id"]: e["pose"] for e in scene.get("entities", [])
                    if e.get("kind") == "charger"}
    tractor_home = {e["id"]: e["home"] for e in scene.get("entities", [])
                    if e.get("kind") == "tractor"}

    assigned = {}   # tractor id -> charger id
    chargers: Dict[str, Any] = {}
    for c in snapshot.get("chargers", []):
        cid = str(c.get("id"))
        if c.get("occupied_by"):
            assigned[str(c["occupied_by"])] = cid
        power = float(c.get("power_kw", 0.0))
        chargers[cid] = {"active": abs(power) > 0.05 or bool(c.get("occupied_by")),
                         "power_kw": round(power, 3)}

    tasks = (task_goals or {}).get("tasks") or {}
    # tractor id -> task id, exactly as HARVEST's scheduler assigned it.
    tractor_task = {str(k): str(v) for k, v in
                    ((task_goals or {}).get("assignments") or {}).items()}
    default_radius = float((task_goals or {}).get("work_radius_m", 6.0))

    goals: Dict[str, Any] = {}
    for t in snapshot.get("tractors", []):
        tid = str(t.get("id"))
        charger_id = assigned.get(tid)
        task_id = tractor_task.get(tid)
        task = tasks.get(task_id) if task_id else None
        # A tractor that is ACTUALLY CHARGING goes to (stays at) its charger;
        # one that merely still occupies a bay after a finished charge does not
        # get to ignore the work HARVEST just gave it.  Measured: with the
        # charger winning on occupancy alone, a full tractor sat on the pad while
        # HARVEST believed it was driving to task_002.
        charging_now = bool(t.get("charging"))
        if charger_id and charger_id in charger_pose and (
                charging_now or not task):
            target = charger_pose[charger_id]
            activity_kind, task_id, task = "charging", None, None
        elif task and task.get("location"):
            target = [float(task["location"][0]), float(task["location"][1])]
            activity_kind = "task"
        else:
            task_id, task, activity_kind = None, None, "idle"
            if t.get("position"):
                target = [float(t["position"][0]), float(t["position"][1])]
            else:
                target = tractor_home.get(tid, [50.0, 50.0])
        goals[tid] = {
            "target": [float(target[0]), float(target[1])],
            "charging": bool(t.get("charging")),
            "docked_charger": charger_id,
            "soc_pct": float(t.get("soc_pct", 0.0)),
            "activity": activity_kind,
            "task_id": task_id,
            # How long the work takes, and how close counts as arrived, are
            # HARVEST's numbers -- passed through so the simulator invents
            # neither one.
            "work_seconds": float(task.get("work_seconds", 0.0)) if task else 0.0,
            "work_radius_m": float((task or {}).get("work_radius_m",
                                                    default_radius)),
        }
    return {"goals": goals, "chargers": chargers, "tasks": tasks}


# --------------------------------------------------------------------------- #
#  Message builders
# --------------------------------------------------------------------------- #
def command_message(scene: Dict[str, Any], derived: Dict[str, Any],
                    include_scene: bool) -> Dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "kind": "command",
        "ts": time.time(),
        "scene": scene if include_scene else None,
        "scene_fingerprint": scene_fingerprint(scene),
        "goals": derived.get("goals", {}),
        "chargers": derived.get("chargers", {}),
        "tasks": derived.get("tasks", {}),
    }


def telemetry_message(simulator: Dict[str, Any], entities: Dict[str, Any],
                      fingerprint: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "kind": "telemetry",
        "ts": time.time(),
        "simulator": simulator,
        "scene_fingerprint": fingerprint,
        "entities": entities,
        "stats": stats,
    }
