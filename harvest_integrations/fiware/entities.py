"""
Fleet snapshot <-> NGSI-LD entity mapping.

The HARVEST device-agent model is authoritative: entities are *derived* from
``harvest_control`` snapshots each sync cycle and never drive internal state
(the single inbound path is the ``FarmCommand`` entity, which carries the same
JSON command schema as ``POST /api/fleet/command``).

Entity types (SAREF-aligned: every device is a saref:Device; power attributes
correspond to saref:PowerMeasurement with unitCode KWT, state flags to
saref:State -- the alignment is documented here rather than imposed through a
custom @context so entities stay usable with the NGSI-LD core context alone):

======================  =====================================================
urn:ngsi-ld:...         attributes
======================  =====================================================
ElectricTractor:<id>    socPct, energyKwh, available, charging, discharging,
                        dischargeKw, currentTask, position {x,y}
ChargingStation:<id>    level (off|half|full), powerKw, occupiedBy
EnergyConsumer:<id>     name, shed, powerKw
FarmEnergySystem:main   clockMin, gridDrawKw, gridCapKw, pvKw, tariff,
                        priceEurPerKwh
FarmCommand:main        command (inbound, JSON string), lastResult, lastNonce
======================  =====================================================

Telemetry attributes carry ``observedAt`` so freshness travels in the data
(the broker mirrors current state, it is not a history log -- WISEPACK's
``runCorrelation`` lesson).
"""
from __future__ import annotations

import datetime as _dt
import json
from typing import Any, Dict, List, Optional

from harvest_control.interface import FleetSnapshot

ENTITY_PREFIX = "urn:ngsi-ld:"
COMMAND_ENTITY_ID = f"{ENTITY_PREFIX}FarmCommand:main"
GRID_ENTITY_ID = f"{ENTITY_PREFIX}FarmEnergySystem:main"
SIMULATION_ENTITY_ID = f"{ENTITY_PREFIX}FarmSimulation:isaac"

_UNIT_KW = "KWT"        # UN/CEFACT: kilowatt
_UNIT_KWH = "KWH"       # kilowatt hour
_UNIT_PCT = "P1"        # percent


def entity_id(entity_type: str, local_id: str) -> str:
    return f"{ENTITY_PREFIX}{entity_type}:{local_id}"


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _prop(value: Any, observed_at: Optional[str] = None,
          unit_code: Optional[str] = None) -> Dict[str, Any]:
    prop: Dict[str, Any] = {"type": "Property", "value": value}
    if observed_at:
        prop["observedAt"] = observed_at
    if unit_code:
        prop["unitCode"] = unit_code
    return prop


def snapshot_to_entities(snap: FleetSnapshot,
                         observed_at: Optional[str] = None) -> List[Dict[str, Any]]:
    """Map one fleet snapshot to a batch-upsert payload."""
    ts = observed_at or _now_iso()
    entities: List[Dict[str, Any]] = []

    g = snap.grid
    entities.append({
        "id": GRID_ENTITY_ID,
        "type": "FarmEnergySystem",
        "clockMin": _prop(g.clock_min, ts),
        "gridDrawKw": _prop(round(g.grid_draw_kw, 3), ts, _UNIT_KW),
        "gridCapKw": _prop(round(g.grid_cap_kw, 3), ts, _UNIT_KW),
        "pvKw": _prop(round(g.pv_kw, 3), ts, _UNIT_KW),
        "tariff": _prop(g.tariff, ts),
        "priceEurPerKwh": _prop(g.price_eur_per_kwh, ts),
    })

    for t in snap.tractors:
        entity: Dict[str, Any] = {
            "id": entity_id("ElectricTractor", t.id),
            "type": "ElectricTractor",
            "socPct": _prop(round(t.soc_pct, 2), ts, _UNIT_PCT),
            "energyKwh": _prop(round(t.energy_kwh, 3), ts, _UNIT_KWH),
            "available": _prop(t.available, ts),
            "charging": _prop(t.charging, ts),
            "discharging": _prop(t.discharging, ts),
            "dischargeKw": _prop(round(t.discharge_kw, 3), ts, _UNIT_KW),
            "currentTask": _prop(t.current_task or "none", ts),
        }
        if t.position is not None:
            # Farm-local metres, not lon/lat -- a plain Property, deliberately
            # not a GeoProperty (which would claim WGS84 coordinates).
            entity["position"] = _prop({"x": t.position[0], "y": t.position[1]}, ts)
        entities.append(entity)

    for c in snap.chargers:
        entities.append({
            "id": entity_id("ChargingStation", c.id),
            "type": "ChargingStation",
            "level": _prop(c.level.value, ts),
            "powerKw": _prop(round(c.power_kw, 3), ts, _UNIT_KW),
            "occupiedBy": _prop(c.occupied_by or "none", ts),
        })

    for l in snap.loads:
        entities.append({
            "id": entity_id("EnergyConsumer", l.id),
            "type": "EnergyConsumer",
            "name": _prop(l.name),
            "shed": _prop(l.shed, ts),
            "powerKw": _prop(round(l.power_kw, 3), ts, _UNIT_KW),
        })

    return entities


def simulation_entity(status: Dict[str, Any],
                      observed_at: Optional[str] = None) -> Dict[str, Any]:
    """Mirror of the optional Isaac Sim layer's state.

    ``status`` is the isaac-bridge document from
    ``GET /api/integrations/status`` (its ``status`` field).

    WHAT IS AND IS NOT MIRRORED.  Summary state and the physical OUTCOME per
    tractor — which state each one is physically in, and which are docked — but
    not the live pose stream.  A pose at 2 Hz is telemetry, and a context broker
    is not a telemetry bus: mirroring it would rewrite this entity on every
    frame for no consumer's benefit, and the live values are already on the
    HARVEST/ROS side (``/api/integrations/status`` and the Diagnostics view).
    What an NGSI-LD consumer needs is the semantic answer -- *did the tractor
    physically arrive* -- which is exactly ``dockedTractors``.
    """
    ts = observed_at or _now_iso()
    sim = (status or {}).get("simulator") or {}
    entities = sim.get("entities") or {}
    tractors = {eid: row for eid, row in entities.items()
                if isinstance(row, dict) and row.get("kind") == "tractor"}
    physical_states = {eid: str(row.get("physical_state") or "unknown")
                       for eid, row in sorted(tractors.items())}
    docked = sorted(eid for eid, row in tractors.items() if row.get("docked"))
    entity = {
        "id": SIMULATION_ENTITY_ID,
        "type": "FarmSimulation",
        "connected": _prop(bool(sim.get("connected")), ts),
        "simulatorKind": _prop(str(sim.get("kind") or "none"), ts),
        "simulatorState": _prop(str(sim.get("state") or "unknown"), ts),
        "entitiesSynced": _prop(int(sim.get("entities_synced") or 0), ts),
        "sceneFingerprint": _prop(str(sim.get("scene_fingerprint") or ""), ts),
        "sceneAcknowledged": _prop(bool(sim.get("scene_acknowledged")), ts),
        # The physical outcome, which is the part HARVEST did not compute.
        "tractorsSynced": _prop(len(tractors), ts),
        "physicalStates": _prop(physical_states, ts),
        "dockedTractors": _prop(docked, ts),
    }
    model = sim.get("robot_model") or {}
    if model.get("id"):
        # Which body is standing in for a tractor, so a consumer can tell a
        # proxy demonstration from a run against the real vehicle model.
        entity["robotModel"] = _prop(str(model["id"]), ts)
    view = sim.get("visualization") or {}
    if view:
        entity["liveViewAvailable"] = _prop(
            view.get("state") == "serving", ts)
    return entity


def command_entity() -> Dict[str, Any]:
    """Initial state of the single inbound command entity.

    External systems PATCH ``command`` with a JSON string::

        {"nonce": "<unique>", "commands": [{"type": "shed_load",
                                            "target_id": "cold_storage"}]}

    The sync daemon executes it against ``POST /api/fleet/command`` and writes
    the acks back into ``lastResult`` (and the nonce into ``lastNonce``, which
    is how replays are suppressed).
    """
    return {
        "id": COMMAND_ENTITY_ID,
        "type": "FarmCommand",
        "command": _prop(""),
        "lastNonce": _prop(""),
        "lastResult": _prop(""),
    }


def parse_command_value(raw_value: Any) -> Optional[Dict[str, Any]]:
    """Decode a ``command`` attribute value into ``{nonce, commands}`` or None."""
    if raw_value in (None, ""):
        return None
    if isinstance(raw_value, str):
        try:
            raw_value = json.loads(raw_value)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw_value, dict) or "commands" not in raw_value:
        return None
    return raw_value
