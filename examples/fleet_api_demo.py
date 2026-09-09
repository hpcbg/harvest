#!/usr/bin/env python3
"""
Live fleet over the HTTP API: read a snapshot, shed the cheapest sheddable
load, watch grid draw change, restore it.

    python3 examples/fleet_api_demo.py [--harvest http://127.0.0.1:8765]
"""
import argparse
import json
import time
import urllib.request


def http_json(method, url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def show(snap):
    g = snap["grid"]
    print(f"  clock={g['clock_min']//60:02.0f}:{g['clock_min']%60:02.0f}  "
          f"draw={g['grid_draw_kw']:.2f} kW  pv={g['pv_kw']:.2f} kW  "
          f"tariff={g['tariff']} ({g['price_eur_per_kwh']:.2f} EUR/kWh)")
    for t in snap["tractors"]:
        state = "charging" if t["charging"] else ("V2L" if t["discharging"] else "idle")
        print(f"  {t['id']}: SoC {t['soc_pct']:.1f}%  {state}")
    for l in snap["loads"]:
        print(f"  {l['id']}: {l['power_kw']:.2f} kW" + ("  [SHED]" if l["shed"] else ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--harvest", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    base = args.harvest.rstrip("/")

    print("Fleet backend:", http_json("GET", f"{base}/api/fleet/status"))
    snap = http_json("GET", f"{base}/api/fleet/snapshot")
    print("\nInitial state:")
    show(snap)

    active = [l for l in snap["loads"] if l["power_kw"] > 0 and not l["shed"]]
    # Prefer clearly deferrable loads for the demo; criticality policy lives in
    # the HARVEST decision layer, not in the raw actuation path used here.
    deferrable = [l for l in active
                  if not any(k in (l["id"] + l["name"]).lower()
                             for k in ("fence", "security", "barn"))]
    if not (deferrable or active):
        print("\nNo active load to shed right now — try later in the (sim) day.")
        return
    target = (deferrable or active)[0]["id"]

    print(f"\nShedding {target} ...")
    acks = http_json("POST", f"{base}/api/fleet/command",
                     {"commands": [{"type": "shed_load", "target_id": target}]})
    print("  ack:", acks["acks"][0])

    time.sleep(3)
    print("\nAfter shedding:")
    show(http_json("GET", f"{base}/api/fleet/snapshot"))

    print(f"\nRestoring {target} ...")
    http_json("POST", f"{base}/api/fleet/command",
              {"commands": [{"type": "restore_load", "target_id": target}]})


if __name__ == "__main__":
    main()
