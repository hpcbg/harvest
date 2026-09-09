#!/usr/bin/env python3
"""
FIWARE/NGSI-LD view of the farm: list the mirror entities, then actuate the
farm the way any external FIWARE application would — by PATCHing the
FarmCommand entity (never by touching HARVEST directly).

    python3 examples/fiware_demo.py [--orion http://127.0.0.1:1026] [--shed LOAD_ID]
"""
import argparse
import json
import time
import urllib.request
import uuid

COMMAND_ENTITY = "urn:ngsi-ld:FarmCommand:main"


def http_json(method, url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode()
        return json.loads(body) if body.strip() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orion", default="http://127.0.0.1:1026")
    parser.add_argument("--shed", default=None,
                        help="EnergyConsumer id to shed+restore via FarmCommand")
    args = parser.parse_args()
    base = args.orion.rstrip("/")

    for etype in ("FarmEnergySystem", "ElectricTractor", "ChargingStation",
                  "EnergyConsumer"):
        entities = http_json(
            "GET", f"{base}/ngsi-ld/v1/entities?type={etype}&options=keyValues") or []
        print(f"\n{etype} ({len(entities)}):")
        for e in entities:
            attrs = {k: v for k, v in e.items() if k not in ("id", "type", "@context")}
            print(f"  {e['id']}\n    {json.dumps(attrs, default=str)[:160]}")

    if not args.shed:
        print("\n(pass --shed <load_id> to demonstrate the inbound command path)")
        return

    nonce = uuid.uuid4().hex[:8]
    print(f"\nPATCHing {COMMAND_ENTITY}: shed {args.shed} (nonce {nonce}) ...")
    http_json("PATCH", f"{base}/ngsi-ld/v1/entities/{COMMAND_ENTITY}/attrs", {
        "command": {"type": "Property", "value": json.dumps({
            "nonce": nonce,
            "commands": [{"type": "shed_load", "target_id": args.shed}],
        })}})

    for _ in range(15):
        time.sleep(2)
        entity = http_json(
            "GET", f"{base}/ngsi-ld/v1/entities/{COMMAND_ENTITY}?options=keyValues")
        if entity and entity.get("lastNonce") == nonce:
            print("Executed. lastResult:", entity.get("lastResult"))
            break
    else:
        print("Command not picked up — is the fiware-sync service running?")
        return

    print(f"Restoring {args.shed} ...")
    http_json("PATCH", f"{base}/ngsi-ld/v1/entities/{COMMAND_ENTITY}/attrs", {
        "command": {"type": "Property", "value": json.dumps({
            "nonce": uuid.uuid4().hex[:8],
            "commands": [{"type": "restore_load", "target_id": args.shed}],
        })}})


if __name__ == "__main__":
    main()
