#!/usr/bin/env python3
"""
End-to-end Isaac Sim demonstrator (works identically with the isaac-demo
stub and with real Isaac Sim).

    ./run_harvest_dashboard.sh isaac-demo      # or: isaac + scripts/run_isaac_sim.sh
    python3 examples/isaac_sim_demo.py

Walks the loop from the task description:

1. HARVEST models a tractor fleet and chargers (semantic state);
2. this script asks HARVEST to charge the lowest-SoC tractor — through the
   normal fleet API, i.e. the FleetInterface seam; nothing here is
   simulator-specific;
3. the command flows HARVEST -> ROS 2 fleet bridge -> Isaac bridge -> sim;
4. the simulator *drives* the tractor to its assigned charger (HARVEST's
   docking is semantic/instant; the physical trip is the simulator's job);
5. measured poses stream back: watch distance_to_target_m fall until docked;
6. the same state is visible in Diagnostics and (with FIWARE up) as the
   urn:ngsi-ld:FarmSimulation:isaac entity.

Standard library only.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

HARVEST = os.environ.get("HARVEST_URL", "http://127.0.0.1:8765")
ORION = os.environ.get("ORION_URL", "http://127.0.0.1:1026")


def http_json(method: str, url: str, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def simulator_view():
    body = http_json("GET", f"{HARVEST}/api/integrations/status")
    info = body.get("clients", {}).get("isaac-bridge") or {}
    return (info.get("status") or {}).get("simulator") or {}


def main() -> int:
    sim = simulator_view()
    if not sim.get("connected"):
        print("No simulator connected.  Start one first:")
        print("  ./run_harvest_dashboard.sh isaac-demo        (GPU-free stub)")
        print("  ./run_harvest_dashboard.sh isaac  +  ./scripts/run_isaac_sim.sh")
        return 1
    print(f"Simulator connected: {sim.get('kind')} "
          f"({sim.get('entities_synced', 0)} entities synced)\n")

    snap = http_json("GET", f"{HARVEST}/api/fleet/snapshot")
    candidates = [t for t in snap["tractors"]
                  if t["available"] and not t["charging"]]
    if not candidates:
        print("Every tractor is unavailable or already charging — try again "
              "in a minute.")
        return 1
    tractor = min(candidates, key=lambda t: t["soc_pct"])
    tid = tractor["id"]
    print(f"1. Commanding REQUEST_CHARGE({tid}) at {tractor['soc_pct']:.1f}% "
          "SoC through the normal fleet API ...")
    acks = http_json("POST", f"{HARVEST}/api/fleet/command",
                     {"commands": [{"type": "request_charge", "target_id": tid}]})
    if not acks.get("accepted"):
        print(f"   command rejected: {acks}")
        return 1
    charger = next((c["id"] for c in
                    http_json("GET", f"{HARVEST}/api/fleet/snapshot")["chargers"]
                    if c.get("occupied_by") == tid), "?")
    print(f"   accepted — HARVEST assigned {charger} (semantic state; the "
          "physical trip is the simulator's)\n")

    print("2. Watching the simulator drive the tractor to the charger ...")
    docked = False
    deadline = time.time() + 90
    while time.time() < deadline:
        ent = (simulator_view().get("entities") or {}).get(tid) or {}
        pose = ent.get("pose", ["?", "?"])
        print(f"   {tid}: pose=({pose[0]}, {pose[1]})  "
              f"distance_to_target={ent.get('distance_to_target_m', '?')}m  "
              f"moving={ent.get('moving')}  docked={ent.get('docked')}")
        if ent.get("docked"):
            docked = True
            break
        time.sleep(2)
    print("   DOCKED — physical arrival confirmed by simulator telemetry\n"
          if docked else "   tractor never docked (is the simulator healthy?)\n")

    try:
        entity = http_json(
            "GET", f"{ORION}/ngsi-ld/v1/entities/"
                   "urn:ngsi-ld:FarmSimulation:isaac?options=keyValues")
        print("3. FIWARE mirror (urn:ngsi-ld:FarmSimulation:isaac):")
        print("   " + json.dumps({k: entity.get(k) for k in
                                  ("simulatorKind", "simulatorState",
                                   "entitiesSynced", "connected")}))
    except Exception:
        print("3. FIWARE mirror skipped (Orion-LD not reachable — fiware "
              "profile not up).")

    http_json("POST", f"{HARVEST}/api/fleet/command",
              {"commands": [{"type": "release_charge", "target_id": tid}]})
    print(f"\n4. RELEASE_CHARGE({tid}) issued — demo complete. "
          "See the Diagnostics tab for the Isaac Sim component.")
    return 0 if docked else 1


if __name__ == "__main__":
    sys.exit(main())
