#!/usr/bin/env python3
"""
End-to-end validation of the running HARVEST Docker stack.

Usage: python3 scripts/validate_stack.py  (or ./validate_harvest_stack.sh)

Checks are skipped, not failed, when their service is not running (e.g. the
FIWARE checks in `core` mode).  Exit code = number of FAILED checks
(TEMPO/WISEPACK validation-script discipline).

Standard library only.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

HARVEST = os.environ.get("HARVEST_URL", f"http://127.0.0.1:{os.environ.get('HARVEST_PORT', '8765')}")
ORION = os.environ.get("ORION_URL", f"http://127.0.0.1:{os.environ.get('ORION_PORT', '1026')}")

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def http_json(method: str, url: str, payload=None, headers=None, timeout=10):
    data = json.dumps(payload).encode() if payload is not None else None
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode()
        return resp.status, json.loads(body) if body.strip() else None


def reachable(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=3)
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def get_load_state(load_id: str) -> bool:
    _, snap = http_json("GET", f"{HARVEST}/api/fleet/snapshot")
    return next(l for l in snap["loads"] if l["id"] == load_id)["shed"]


def pick_sheddable_load() -> str:
    _, snap = http_json("GET", f"{HARVEST}/api/fleet/snapshot")
    # Avoid loads whose id/name hints at critical duty (the sim backend
    # refuses to shed critical loads).
    for l in snap["loads"]:
        text = (l["id"] + l["name"]).lower()
        if not any(k in text for k in ("fence", "security", "barn")):
            return l["id"]
    return snap["loads"][0]["id"]


# ─────────────────────────────────────────────────────────────────────────────
def check_harvest_api() -> None:
    if not reachable(f"{HARVEST}/health"):
        record("HARVEST API reachable", FAIL, f"{HARVEST}/health not responding")
        return
    record("HARVEST API reachable", PASS)

    status, body = http_json("GET", f"{HARVEST}/api/fleet/status")
    if status == 200 and body.get("backend") in ("sim", "devices"):
        record("Fleet status endpoint", PASS, f"backend={body['backend']}")
    else:
        record("Fleet status endpoint", FAIL, str(body))

    _, snap = http_json("GET", f"{HARVEST}/api/fleet/snapshot")
    ok = (snap.get("schema") == "harvest-fleet/1.0"
          and snap.get("tractors") and snap.get("chargers") and snap.get("loads"))
    record("Fleet snapshot shape", PASS if ok else FAIL,
           f"{len(snap.get('tractors', []))} tractors, "
           f"{len(snap.get('chargers', []))} chargers, "
           f"{len(snap.get('loads', []))} loads")

    clock_a = snap["grid"]["clock_min"]
    time.sleep(3)
    _, snap_b = http_json("GET", f"{HARVEST}/api/fleet/snapshot")
    clock_b = snap_b["grid"]["clock_min"]
    advancing = clock_b != clock_a
    record("Farm clock advances", PASS if advancing else FAIL,
           f"{clock_a} -> {clock_b}")


def check_command_roundtrip() -> None:
    try:
        load_id = pick_sheddable_load()
        status, body = http_json("POST", f"{HARVEST}/api/fleet/command", {
            "commands": [{"type": "shed_load", "target_id": load_id}]})
        if status != 200 or not body.get("accepted"):
            record("Command round-trip (API)", FAIL, str(body))
            return
        deadline = time.time() + 15
        while time.time() < deadline and not get_load_state(load_id):
            time.sleep(1)
        shed = get_load_state(load_id)
        record("Command round-trip (API)", PASS if shed else FAIL,
               f"{load_id} shed={shed}")
        http_json("POST", f"{HARVEST}/api/fleet/command", {
            "commands": [{"type": "restore_load", "target_id": load_id}]})
    except Exception as exc:
        record("Command round-trip (API)", FAIL, str(exc))


def check_diagnostics() -> None:
    try:
        status, body = http_json("GET", f"{HARVEST}/api/diagnostics")
        services = {s["name"]: s["state"] for s in body.get("services", [])}
        ok = (status == 200 and services.get("HARVEST API") == "healthy"
              and "Fleet backend" in services and body.get("devices"))
        record("Diagnostics endpoint", PASS if ok else FAIL,
               ", ".join(f"{n}={s}" for n, s in sorted(services.items())))
    except Exception as exc:
        record("Diagnostics endpoint", FAIL, str(exc))
        return

    # The scripted cross-protocol demonstration must pass end-to-end.
    try:
        status, body = http_json("POST", f"{HARVEST}/api/diagnostics/demo", {})
        if status not in (200,):
            record("Cross-protocol demo", FAIL, str(body))
            return
        deadline = time.time() + 90
        while time.time() < deadline:
            time.sleep(2)
            _, demo = http_json("GET", f"{HARVEST}/api/diagnostics/demo")
            if demo.get("state") != "running":
                break
        steps = {s["title"]: s["status"] for s in demo.get("steps", [])}
        record("Cross-protocol demo",
               PASS if demo.get("state") == "passed" else FAIL,
               f"{demo.get('state')}: " +
               ", ".join(f"{t}={s}" for t, s in steps.items()))
    except Exception as exc:
        record("Cross-protocol demo", FAIL, str(exc))


def check_fiware() -> None:
    if not reachable(f"{ORION}/version"):
        record("FIWARE checks", SKIP, "Orion-LD not running (non-fiware mode?)")
        return
    record("Orion-LD reachable", PASS)

    # Entities mirrored (allow a few sync cycles).
    found = 0
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            _, entities = http_json(
                "GET", f"{ORION}/ngsi-ld/v1/entities?type=ElectricTractor&options=keyValues")
            found = len(entities or [])
            if found:
                break
        except Exception:
            pass
        time.sleep(2)
    record("Tractor entities mirrored", PASS if found else FAIL,
           f"{found} ElectricTractor entities")

    try:
        _, grid = http_json(
            "GET", f"{ORION}/ngsi-ld/v1/entities/urn:ngsi-ld:FarmEnergySystem:main?options=keyValues")
        ok = grid and "gridCapKw" in grid
        record("Grid entity mirrored", PASS if ok else FAIL,
               f"gridCapKw={grid.get('gridCapKw') if grid else None}")
    except Exception as exc:
        record("Grid entity mirrored", FAIL, str(exc))

    # Inbound path: PATCH the FarmCommand entity -> load sheds in HARVEST.
    try:
        load_id = pick_sheddable_load()
        nonce = uuid.uuid4().hex[:8]
        command_value = json.dumps({
            "nonce": nonce,
            "commands": [{"type": "shed_load", "target_id": load_id}]})
        status, _ = http_json(
            "PATCH",
            f"{ORION}/ngsi-ld/v1/entities/urn:ngsi-ld:FarmCommand:main/attrs",
            {"command": {"type": "Property", "value": command_value}})
        deadline = time.time() + 30
        while time.time() < deadline and not get_load_state(load_id):
            time.sleep(1)
        shed = get_load_state(load_id)
        record("Inbound NGSI-LD command", PASS if shed else FAIL,
               f"PATCH FarmCommand -> {load_id} shed={shed}")

        # Result written back to the broker.
        _, cmd_entity = http_json(
            "GET", f"{ORION}/ngsi-ld/v1/entities/urn:ngsi-ld:FarmCommand:main?options=keyValues")
        ok = cmd_entity and cmd_entity.get("lastNonce") == nonce
        record("Command result in broker", PASS if ok else FAIL,
               f"lastNonce={cmd_entity.get('lastNonce') if cmd_entity else None}")

        restore = json.dumps({
            "nonce": uuid.uuid4().hex[:8],
            "commands": [{"type": "restore_load", "target_id": load_id}]})
        http_json("PATCH",
                  f"{ORION}/ngsi-ld/v1/entities/urn:ngsi-ld:FarmCommand:main/attrs",
                  {"command": {"type": "Property", "value": restore}})
    except Exception as exc:
        record("Inbound NGSI-LD command", FAIL, str(exc))


def check_ros2() -> None:
    probe = subprocess.run(
        ["docker", "ps", "--filter", "name=ros2-bridge", "--format", "{{.Names}}"],
        capture_output=True, text=True)
    container = probe.stdout.strip().splitlines()
    if probe.returncode != 0 or not container:
        record("ROS 2 checks", SKIP, "ros2-bridge container not running (non-full mode?)")
        return
    name = container[0]
    cmd = ("source /opt/ros/jazzy/setup.bash && "
           "source /tmp/ros_install/setup.bash 2>/dev/null; "
           "timeout 20 ros2 topic echo --once /harvest/fleet/snapshot std_msgs/msg/String")
    proc = subprocess.run(["docker", "exec", name, "bash", "-lc", cmd],
                          capture_output=True, text=True, timeout=60)
    ok = proc.returncode == 0 and "harvest-fleet/1.0" in proc.stdout
    record("ROS 2 snapshot topic", PASS if ok else FAIL,
           "received /harvest/fleet/snapshot" if ok
           else (proc.stderr.strip()[:200] or "no message received"))


def check_isaac() -> None:
    """Optional Isaac Sim layer: bridge status, scene sync, and — when a
    simulator (Isaac or the isaac-demo stub) is connected — the full physical
    loop: HARVEST charge command -> ROS 2 -> simulator drives the tractor to
    its charger -> docked telemetry returns to HARVEST."""
    try:
        _, body = http_json("GET", f"{HARVEST}/api/integrations/status")
        info = (body or {}).get("clients", {}).get("isaac-bridge")
    except Exception:
        info = None
    if not info or info.get("age_s", 1e9) > 30:
        record("Isaac Sim checks", SKIP,
               "isaac bridge not running (non-isaac mode?)")
        return
    record("Isaac bridge reporting", PASS, f"status age {info['age_s']:.0f}s")

    sim = (info.get("status") or {}).get("simulator") or {}
    if not sim.get("connected"):
        record("Isaac simulator", SKIP,
               "bridge up but no simulator connected — start Isaac Sim "
               "(scripts/run_isaac_sim.sh) or use isaac-demo")
        return
    kind = sim.get("kind", "?")

    # Give the latched scene a few cycles to be applied and acknowledged.
    deadline = time.time() + 30
    while time.time() < deadline and not sim.get("scene_acknowledged"):
        time.sleep(2)
        _, body = http_json("GET", f"{HARVEST}/api/integrations/status")
        info = (body or {}).get("clients", {}).get("isaac-bridge") or {}
        sim = (info.get("status") or {}).get("simulator") or {}
    ok = sim.get("scene_acknowledged") and sim.get("entities_synced", 0) > 0
    record("Simulator scene sync", PASS if ok else FAIL,
           f"{kind}: {sim.get('entities_synced', 0)} entities, "
           f"scene acked={bool(sim.get('scene_acknowledged'))}")

    _, diag = http_json("GET", f"{HARVEST}/api/diagnostics")
    row = next((s for s in diag.get("services", [])
                if s["name"] == "Isaac Sim"), None)
    expect = "simulated" if kind == "stub" else "healthy"
    record("Isaac diagnostics state",
           PASS if row and row["state"] == expect else FAIL,
           f"state={row['state'] if row else None} (expected {expect})")

    def sim_entities():
        _, b = http_json("GET", f"{HARVEST}/api/integrations/status")
        i = (b or {}).get("clients", {}).get("isaac-bridge") or {}
        return ((i.get("status") or {}).get("simulator") or {}).get("entities") or {}

    try:
        _, snap = http_json("GET", f"{HARVEST}/api/fleet/snapshot")
        candidates = [t for t in snap["tractors"]
                      if t["available"] and not t["charging"]]
        if not candidates:
            record("Physical charge loop", SKIP,
                   "no available non-charging tractor")
            return
        tractor = min(candidates, key=lambda t: t["soc_pct"])["id"]
        http_json("POST", f"{HARVEST}/api/fleet/command", {
            "commands": [{"type": "request_charge", "target_id": tractor}]})
        deadline = time.time() + 60
        docked = False
        while time.time() < deadline and not docked:
            time.sleep(2)
            docked = bool((sim_entities().get(tractor) or {}).get("docked"))
        record("Physical charge loop", PASS if docked else FAIL,
               f"{tractor} " + ("drove to its charger and docked in the simulator"
                                if docked else "never docked in the simulator"))
        http_json("POST", f"{HARVEST}/api/fleet/command", {
            "commands": [{"type": "release_charge", "target_id": tractor}]})
    except Exception as exc:
        record("Physical charge loop", FAIL, str(exc))


def main() -> int:
    print(f"Validating HARVEST stack (API {HARVEST}, broker {ORION})\n")
    check_harvest_api()
    check_command_roundtrip()
    check_diagnostics()
    check_fiware()
    check_ros2()
    check_isaac()

    failures = sum(1 for _, status, _ in results if status == FAIL)
    passes = sum(1 for _, status, _ in results if status == PASS)
    skips = sum(1 for _, status, _ in results if status == SKIP)
    print(f"\n{passes} passed, {failures} failed, {skips} skipped")
    return failures


if __name__ == "__main__":
    sys.exit(main())
