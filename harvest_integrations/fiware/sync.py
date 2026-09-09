"""
FIWARE context synchronisation daemon.

    python -m harvest_integrations.fiware.sync \
        [--harvest http://127.0.0.1:8765] [--broker http://127.0.0.1:1026] \
        [--period 2.0] [--listen-port 8766] [--notify-url http://host:8766/notify] \
        [--once]

Outbound: polls the HARVEST fleet API (``/api/fleet/snapshot``) and batch-
upserts the NGSI-LD mirror entities into the context broker every ``period``
seconds.

Inbound: external systems PATCH the ``command`` attribute of
``urn:ngsi-ld:FarmCommand:main``.  Commands reach the daemon two ways:

* an NGSI-LD *subscription* notification (registered when ``--notify-url`` is
  given and served by the built-in listener) -- low latency;
* a *poll* of the command entity every cycle -- the fallback that also works
  when the broker cannot reach the daemon (NAT, container networking).

Both paths deduplicate on the command ``nonce``, so running them together is
safe.  Accepted commands are forwarded verbatim to
``POST /api/fleet/command`` -- the broker never touches HARVEST state
directly, keeping the semantic device-agent model authoritative.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional

from .client import ContextBrokerClient, ContextBrokerError
from .entities import (
    COMMAND_ENTITY_ID,
    command_entity,
    parse_command_value,
    simulation_entity,
    snapshot_to_entities,
)

SUBSCRIPTION_ID = "urn:ngsi-ld:Subscription:harvest-fleet-command"


# --------------------------------------------------------------------------- #
#  HARVEST fleet API client
# --------------------------------------------------------------------------- #
class HarvestApiClient:
    def __init__(self, base_url: str, timeout_s: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _json(self, method: str, path: str, payload: Any = None) -> Any:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=data, method=method,
            headers={"Content-Type": "application/json",
                     # Liveness signal for the server's diagnostics view.
                     "X-Harvest-Client": "fiware-sync"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode())

    def snapshot(self) -> Dict[str, Any]:
        return self._json("GET", "/api/fleet/snapshot")

    def submit_commands(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._json("POST", "/api/fleet/command", payload)

    def integrations_status(self) -> Dict[str, Any]:
        return self._json("GET", "/api/integrations/status")


# --------------------------------------------------------------------------- #
#  Subscription notification listener
# --------------------------------------------------------------------------- #
class _NotificationHandler(BaseHTTPRequestHandler):
    inbox: "queue.Queue[Dict[str, Any]]" = queue.Queue()

    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        self.send_response(200)
        self.end_headers()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return
        for entity in payload.get("data", []):
            attr = entity.get("command")
            value = attr.get("value") if isinstance(attr, dict) else attr
            parsed = parse_command_value(value)
            if parsed:
                self.inbox.put(parsed)


def start_listener(host: str, port: int) -> HTTPServer:
    server = HTTPServer((host, port), _NotificationHandler)
    threading.Thread(target=server.serve_forever, daemon=True,
                     name="fiware-notify").start()
    return server


# --------------------------------------------------------------------------- #
#  The sync engine
# --------------------------------------------------------------------------- #
class ContextSync:
    def __init__(self, harvest: HarvestApiClient, broker: ContextBrokerClient,
                 period_s: float = 2.0):
        self.harvest = harvest
        self.broker = broker
        self.period_s = period_s
        self.last_nonce: Optional[str] = None
        self.cycles = 0
        self.errors = 0

    # -- setup ----------------------------------------------------------------
    def ensure_command_entity(self) -> None:
        existing = self.broker.get_entity(COMMAND_ENTITY_ID)
        if existing is None:
            self.broker.upsert_entities([command_entity()])
        else:
            # Resume nonce tracking across restarts so old commands don't replay.
            self.last_nonce = str(existing.get("lastNonce") or "") or None

    def register_subscription(self, notify_url: str) -> None:
        self.broker.ensure_subscription({
            "id": SUBSCRIPTION_ID,
            "type": "Subscription",
            "entities": [{"id": COMMAND_ENTITY_ID, "type": "FarmCommand"}],
            "watchedAttributes": ["command"],
            "notification": {
                "endpoint": {"uri": notify_url, "accept": "application/json"},
                "attributes": ["command"],
            },
        })

    # -- one cycle ------------------------------------------------------------
    def sync_once(self) -> int:
        """Push one snapshot; returns the number of entities upserted."""
        snap = self.harvest.snapshot()
        from harvest_integrations.codec import snapshot_from_dict
        entities = snapshot_to_entities(snapshot_from_dict(snap))
        entities.extend(self._simulation_entities())
        self.broker.upsert_entities(entities)
        self.cycles += 1
        return len(entities)

    def _simulation_entities(self) -> list:
        """Mirror the optional Isaac Sim layer's state, when its bridge is
        alive.  Absence of the bridge is normal (isaac modes only) and mirrors
        nothing — an optional layer that is off must not produce a stale
        broker entity claiming otherwise."""
        try:
            clients = (self.harvest.integrations_status() or {}).get("clients", {})
        except Exception:
            return []
        info = clients.get("isaac-bridge")
        if not info or info.get("age_s", 1e9) > 30.0:
            return []
        return [simulation_entity(info.get("status") or {})]

    def poll_inbound(self) -> None:
        entity = self.broker.get_entity(COMMAND_ENTITY_ID)
        if entity is None:
            return
        parsed = parse_command_value(entity.get("command"))
        if parsed:
            self.process_command(parsed)

    def process_command(self, parsed: Dict[str, Any]) -> bool:
        nonce = str(parsed.get("nonce") or "")
        if nonce and nonce == self.last_nonce:
            return False
        result = self.harvest.submit_commands({"commands": parsed.get("commands", [])})
        self.last_nonce = nonce or self.last_nonce
        self.broker.patch_attrs(COMMAND_ENTITY_ID, {
            "lastNonce": {"type": "Property", "value": nonce},
            "lastResult": {"type": "Property", "value": json.dumps(result)},
        })
        return True

    def drain_notifications(self) -> None:
        while True:
            try:
                parsed = _NotificationHandler.inbox.get_nowait()
            except queue.Empty:
                return
            self.process_command(parsed)


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="HARVEST FIWARE context sync")
    parser.add_argument("--harvest",
                        default=os.environ.get("HARVEST_URL", "http://127.0.0.1:8765"))
    parser.add_argument("--broker",
                        default=os.environ.get("ORION_URL", "http://127.0.0.1:1026"))
    parser.add_argument("--period", type=float,
                        default=float(os.environ.get("HARVEST_FIWARE_PERIOD", "2.0")))
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int,
                        default=int(os.environ.get("HARVEST_FIWARE_LISTEN_PORT", "8766")))
    parser.add_argument("--notify-url",
                        default=os.environ.get("HARVEST_FIWARE_NOTIFY_URL", ""),
                        help="URL the broker uses to reach this daemon; empty "
                             "disables the subscription (poll-only inbound)")
    parser.add_argument("--once", action="store_true",
                        help="push a single snapshot and exit (used by validation)")
    args = parser.parse_args()

    harvest = HarvestApiClient(args.harvest)
    broker = ContextBrokerClient(args.broker)
    sync = ContextSync(harvest, broker, period_s=args.period)

    # Bounded wait for the broker (WISEPACK-style checked health gate).
    deadline = time.time() + float(os.environ.get("HARVEST_FIWARE_READY_TIMEOUT", "90"))
    while not broker.is_alive():
        if time.time() > deadline:
            raise SystemExit(f"context broker at {args.broker} not reachable")
        print(f"[fiware-sync] waiting for broker at {args.broker} ...", flush=True)
        time.sleep(2.0)

    sync.ensure_command_entity()
    if args.once:
        n = sync.sync_once()
        print(f"[fiware-sync] pushed {n} entities to {args.broker}", flush=True)
        return

    if args.notify_url:
        start_listener(args.listen_host, args.listen_port)
        try:
            sync.register_subscription(args.notify_url)
            print(f"[fiware-sync] subscription -> {args.notify_url}", flush=True)
        except ContextBrokerError as exc:
            print(f"[fiware-sync] subscription failed ({exc}); poll-only", flush=True)

    print(f"[fiware-sync] {args.harvest} -> {args.broker} every {args.period}s", flush=True)
    while True:
        started = time.time()
        try:
            sync.drain_notifications()
            sync.sync_once()
            sync.poll_inbound()
        except Exception as exc:
            sync.errors += 1
            print(f"[fiware-sync] cycle failed: {exc}", flush=True)
        time.sleep(max(0.2, sync.period_s - (time.time() - started)))


if __name__ == "__main__":
    main()
