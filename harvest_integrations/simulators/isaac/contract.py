"""
HARVEST <-> simulator wire contract (schema ``harvest-sim/1.0``).

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

    {"schema": "harvest-sim/1.0", "kind": "command", "ts": ...,
     "scene": {...} | null,          # included until the fingerprint is acked
     "goals": {tractor_id: {"target": [x,y], "charging": bool,
                            "docked_charger": id|null, "soc_pct": float}},
     "chargers": {charger_id: {"active": bool, "power_kw": float}}}

Telemetry (simulator -> bridge), published periodically::

    {"schema": "harvest-sim/1.0", "kind": "telemetry", "ts": ...,
     "simulator": {"kind": "isaac"|"stub", "state": "starting"|"running"|"error",
                   "version": str, "started_ts": float, "detail": str},
     "scene_fingerprint": str,       # acknowledges the applied scene
     "entities": {id: {"kind": ..., "pose": [x,y], "heading_deg": float,
                       "moving": bool, "docked": bool,
                       "distance_to_target_m": float}},
     "stats": {"entities_synced": int, "sim_time_s": float}}

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

SCHEMA_VERSION = "harvest-sim/1.0"

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
                        snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Map the semantic fleet state onto physical motion goals.

    HARVEST's device model docks instantly (semantic state); the simulator
    executes the physical part -- driving to the assigned charger -- so the
    goal for a tractor with a charger assignment is that charger's pose, and
    otherwise its reported field position (or home).
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

    goals: Dict[str, Any] = {}
    for t in snapshot.get("tractors", []):
        tid = str(t.get("id"))
        charger_id = assigned.get(tid)
        if charger_id and charger_id in charger_pose:
            target = charger_pose[charger_id]
        elif t.get("position"):
            target = [float(t["position"][0]), float(t["position"][1])]
        else:
            target = tractor_home.get(tid, [50.0, 50.0])
        goals[tid] = {
            "target": [float(target[0]), float(target[1])],
            "charging": bool(t.get("charging")),
            "docked_charger": charger_id,
            "soc_pct": float(t.get("soc_pct", 0.0)),
        }
    return {"goals": goals, "chargers": chargers}


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
