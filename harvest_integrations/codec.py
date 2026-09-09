"""
JSON codec for the fleet boundary.

One canonical wire representation of ``harvest_control`` snapshots, commands
and acks, shared by:

* ``server.py``            -- ``GET /api/fleet/snapshot`` / ``POST /api/fleet/command``;
* the FIWARE sync daemon   -- entity attribute payloads;
* the ROS 2 bridge         -- ``std_msgs/String`` JSON topics.

Rich objects travel as versioned JSON rather than bespoke message/entity
schemas (the WISEPACK pattern: the typed model lives in Python, the wire is
JSON), so all three transports stay in lock-step automatically.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from harvest_control.interface import (
    ChargerLevel,
    ChargerState,
    Command,
    CommandAck,
    CommandType,
    FleetSnapshot,
    GridState,
    LoadState,
    TractorState,
)

SCHEMA_VERSION = "harvest-fleet/1.0"


# --------------------------------------------------------------------------- #
#  Snapshot -> dict -> snapshot
# --------------------------------------------------------------------------- #
def snapshot_to_dict(snap: FleetSnapshot) -> Dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "grid": {
            "clock_min": snap.grid.clock_min,
            "grid_draw_kw": round(snap.grid.grid_draw_kw, 3),
            "grid_cap_kw": round(snap.grid.grid_cap_kw, 3),
            "pv_kw": round(snap.grid.pv_kw, 3),
            "tariff": snap.grid.tariff,
            "price_eur_per_kwh": snap.grid.price_eur_per_kwh,
        },
        "tractors": [
            {
                "id": t.id,
                "soc_pct": round(t.soc_pct, 2),
                "energy_kwh": round(t.energy_kwh, 3),
                "available": t.available,
                "charging": t.charging,
                "current_task": t.current_task,
                "position": list(t.position) if t.position else None,
                "discharging": t.discharging,
                "discharge_kw": round(t.discharge_kw, 3),
            }
            for t in snap.tractors
        ],
        "chargers": [
            {
                "id": c.id,
                "level": c.level.value,
                "power_kw": round(c.power_kw, 3),
                "occupied_by": c.occupied_by,
            }
            for c in snap.chargers
        ],
        "loads": [
            {"id": l.id, "name": l.name, "shed": l.shed, "power_kw": round(l.power_kw, 3)}
            for l in snap.loads
        ],
    }


def snapshot_from_dict(raw: Dict[str, Any]) -> FleetSnapshot:
    g = raw.get("grid", {})
    grid = GridState(
        clock_min=int(g.get("clock_min", 0)),
        grid_draw_kw=float(g.get("grid_draw_kw", 0.0)),
        grid_cap_kw=float(g.get("grid_cap_kw", 0.0)),
        pv_kw=float(g.get("pv_kw", 0.0)),
        tariff=str(g.get("tariff", "valle")),
        price_eur_per_kwh=float(g.get("price_eur_per_kwh", 0.0)),
    )
    tractors = [
        TractorState(
            id=str(t["id"]),
            soc_pct=float(t.get("soc_pct", 0.0)),
            energy_kwh=float(t.get("energy_kwh", 0.0)),
            available=bool(t.get("available", True)),
            charging=bool(t.get("charging", False)),
            current_task=t.get("current_task"),
            position=tuple(t["position"]) if t.get("position") else None,
            discharging=bool(t.get("discharging", False)),
            discharge_kw=float(t.get("discharge_kw", 0.0)),
        )
        for t in raw.get("tractors", [])
    ]
    chargers = [
        ChargerState(
            id=str(c["id"]),
            level=ChargerLevel(c.get("level", "off")),
            power_kw=float(c.get("power_kw", 0.0)),
            occupied_by=c.get("occupied_by"),
        )
        for c in raw.get("chargers", [])
    ]
    loads = [
        LoadState(
            id=str(l["id"]),
            name=str(l.get("name", l["id"])),
            shed=bool(l.get("shed", False)),
            power_kw=float(l.get("power_kw", 0.0)),
        )
        for l in raw.get("loads", [])
    ]
    return FleetSnapshot(grid=grid, tractors=tractors, chargers=chargers, loads=loads)


# --------------------------------------------------------------------------- #
#  Commands / acks
# --------------------------------------------------------------------------- #
def command_from_dict(raw: Dict[str, Any]) -> Command:
    ctype = CommandType(raw["type"])
    value: Any = raw.get("value")
    if ctype == CommandType.SET_CHARGER_LEVEL and value is not None:
        value = ChargerLevel(value)
    return Command(type=ctype, target_id=str(raw["target_id"]), value=value)


def command_to_dict(cmd: Command) -> Dict[str, Any]:
    value = cmd.value
    if isinstance(value, ChargerLevel):
        value = value.value
    return {"type": cmd.type.value, "target_id": cmd.target_id, "value": value}


def commands_from_payload(payload: Dict[str, Any]) -> List[Command]:
    """Parse ``{"commands": [...]}`` (or a bare list) into Command objects."""
    raw_list = payload.get("commands", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_list, list):
        raise ValueError("expected {'commands': [...]} or a JSON list")
    return [command_from_dict(c) for c in raw_list]


def ack_to_dict(ack: CommandAck) -> Dict[str, Any]:
    return {
        "command": command_to_dict(ack.command),
        "accepted": ack.accepted,
        "reason": ack.reason,
    }


def acks_to_payload(acks: List[CommandAck]) -> Dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "acks": [ack_to_dict(a) for a in acks],
        "accepted": all(a.accepted for a in acks) if acks else True,
    }
