#!/usr/bin/env python3
"""
Real ZETRABOT telemetry replay -- watch the mission flow through HARVEST.

Start HARVEST with the supplied mission replaying (host, no Docker):

    ./run_harvest_dashboard.sh replay --file telemetry/telemetria_mision_63.csv --speed 20

or with the full stack, so the same state is also mirrored to Orion-LD:

    ./run_harvest_dashboard.sh replay --file telemetry/telemetria_mision_63.csv \\
                                      --speed 20 --stack fiware

Then run this script.  It polls ``GET /api/telemetry`` (the canonical
ZETRABOT state + replay progress), shows the replayed tractor inside the
fleet snapshot, and -- when a broker answers -- the ``TractorTelemetry``
NGSI-LD entity.  Standard library only.

    python examples/telemetry_replay_demo.py [--harvest URL] [--broker URL] [--seconds 30]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


def get(url: str):
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode())


def fmt(value, unit="", nd=2):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{nd}f}{unit}"
    return f"{value}{unit}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--harvest", default="http://127.0.0.1:8765")
    parser.add_argument("--broker", default="http://127.0.0.1:1026")
    parser.add_argument("--seconds", type=float, default=30.0, help="how long to watch")
    parser.add_argument("--period", type=float, default=3.0)
    args = parser.parse_args()

    doc = get(f"{args.harvest}/api/telemetry")
    if doc.get("state") == "inactive":
        print("No telemetry source configured.  Start HARVEST with:\n"
              "  ./run_harvest_dashboard.sh replay --file telemetry/telemetria_mision_63.csv --speed 20")
        return 1
    if doc.get("state") == "failed":
        print(f"Telemetry source failed: {doc.get('error')}")
        return 1
    src = doc.get("source") or {}
    print(f"Source: {src.get('kind')}  {src.get('file', '')}  mode={doc.get('mode')}")
    print(f"Notes:  " + "\n        ".join(doc.get("notes", [])))
    print()

    deadline = time.time() + args.seconds
    while True:
        doc = get(f"{args.harvest}/api/telemetry")
        replay = doc.get("replay") or {}
        line = (f"replay {replay.get('state')} {replay.get('progress_pct', 0):5.1f}% "
                f"pos {replay.get('position')}  ({replay.get('speed')}x, "
                f"{replay.get('gaps_compressed', 0)} gaps compressed)")
        print(line)
        for t in doc.get("tractors") or []:
            print(f"  {t['harvest_id']} (ZETRABOT {t['tractor_id']}, mission {t.get('mission_id')}) "
                  f"@ {t.get('timestamp')}")
            print(f"    SOC {fmt(t.get('soc_pct'), ' %', 1)}  "
                  f"{fmt(t.get('battery_voltage_v'), ' V', 1)} x {fmt(t.get('battery_current_a'), ' A', 1)} "
                  f"= {fmt(t.get('battery_power_kw'), ' kW')} (derived)   "
                  f"discharged {fmt(t.get('discharged_energy_kwh'), ' kWh')} "
                  f"(session {fmt(t.get('discharged_energy_session_kwh'), ' kWh')})")
            print(f"    PTO {fmt(t.get('pto_active'))} {fmt(t.get('pto_speed_rpm'), ' rpm', 0)}  "
                  f"drive {fmt(t.get('drive_active'))}  speed {fmt(t.get('speed_kmh'), ' km/h', 1)}  "
                  f"battery {fmt(t.get('battery_temp_c'), ' C', 0)}  motors "
                  + " ".join(f"{k}={fmt(v, '', 0)}" for k, v in sorted((t.get('motor_temp_c') or {}).items())))
            print(f"    position: {t.get('position')}  (no GPS in this telemetry)  "
                  f"unknown signals kept: {list((t.get('unknown_signals') or {}).keys()) or 'none'}")
        snap = get(f"{args.harvest}/api/fleet/snapshot")
        real = [t for t in snap["tractors"] if t["id"].startswith("zetrabot") or
                any(t["id"] == x.get("harvest_id") for x in doc.get("tractors") or [])]
        print(f"  fleet snapshot: {[(t['id'], t['soc_pct'], t['available']) for t in real]}")
        try:
            for t in doc.get("tractors") or []:
                entity = get(f"{args.broker}/ngsi-ld/v1/entities/urn:ngsi-ld:TractorTelemetry:"
                             f"{t['harvest_id']}?options=keyValues")
                print(f"  NGSI-LD TractorTelemetry:{t['harvest_id']}: source={entity.get('source')} "
                      f"socPct={entity.get('socPct')} batteryPowerKw={entity.get('batteryPowerKw')} "
                      f"replayState={entity.get('replayState')}")
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            print("  NGSI-LD: broker not reachable (start with --stack fiware to see the mirror)")
        if time.time() >= deadline or replay.get("state") in ("finished", "failed", "stopped"):
            break
        time.sleep(args.period)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
