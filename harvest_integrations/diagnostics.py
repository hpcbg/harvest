"""
Operational diagnostics for the HARVEST integration stack.

Feeds the dashboard's Diagnostics view (``GET /api/diagnostics``) and hosts
the cross-protocol demonstration (``POST /api/diagnostics/demo``).

Design rules, adapted from WISEPACK's diagnostics page:

* **read-only and allowlisted** — the collector reports a fixed set of
  services and per-device rows; no environment dumps, no Docker socket, no
  arbitrary shell commands;
* **honest states** — every component is one of ``healthy`` / ``simulated`` /
  ``inactive`` (optional, not started) / ``failed`` (expected but not
  answering) / ``unknown``.  Deliberate simulation must never read as a
  failure, and an optional service that was never started must never read as
  an error;
* **no protocol logic upstream** — the demo drives the fleet exclusively
  through the generic ``FleetInterface`` commands; which wire protocol serves
  which device is *reported* from the device diagnostics, never assumed.

Service reachability probes use short timeouts and a small TTL cache so the
dashboard's periodic polling stays cheap.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from harvest_control.interface import Command

# Component states (the vocabulary the UI colour-codes).
HEALTHY = "healthy"
SIMULATED = "simulated"
INACTIVE = "inactive"     # optional component, not started — not an error
FAILED = "failed"         # expected/configured but not answering
UNKNOWN = "unknown"

_STARTED_AT = time.time()
_PROBE_TTL_S = 2.5


# --------------------------------------------------------------------------- #
#  Client heartbeats (daemons announce themselves via X-Harvest-Client)
# --------------------------------------------------------------------------- #
class ClientRegistry:
    """Last-seen timestamps of known API clients (fiware-sync, ros2-bridge,
    isaac-bridge), plus an optional status document per client.

    The daemons already poll the fleet API; tagging their requests with an
    ``X-Harvest-Client`` header turns that existing traffic into a liveness
    signal — no extra heartbeat endpoint needed.  Clients with richer state
    (the Isaac bridge) additionally push a status blob to
    ``POST /api/integrations/status``; the registry only stores and ages it —
    interpretation stays in :func:`collect`, so no integration-specific logic
    leaks into server.py.
    """

    KNOWN = ("fiware-sync", "ros2-bridge", "isaac-bridge")

    def __init__(self) -> None:
        self._seen: Dict[str, float] = {}
        self._status: Dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def mark(self, label: Optional[str]) -> None:
        if label:
            with self._lock:
                self._seen[str(label)] = time.time()

    def age_s(self, label: str) -> Optional[float]:
        with self._lock:
            ts = self._seen.get(label)
        return None if ts is None else max(0.0, time.time() - ts)

    def report(self, label: str, status: Any) -> None:
        """Store a client's pushed status document (and mark it seen)."""
        now = time.time()
        with self._lock:
            self._seen[str(label)] = now
            self._status[str(label)] = (now, status)

    def latest(self, label: str) -> Optional[Dict[str, Any]]:
        """The freshest status document for ``label`` with its age, or None."""
        with self._lock:
            hit = self._status.get(label)
        if hit is None:
            return None
        return {"age_s": max(0.0, time.time() - hit[0]), "status": hit[1]}

    def statuses(self) -> Dict[str, Any]:
        """All pushed statuses (for ``GET /api/integrations/status``)."""
        with self._lock:
            labels = list(self._status)
        return {label: self.latest(label) for label in labels}


# --------------------------------------------------------------------------- #
#  Probes (cached)
# --------------------------------------------------------------------------- #
class _TtlCache:
    def __init__(self, ttl_s: float = _PROBE_TTL_S):
        self.ttl_s = ttl_s
        self._data: Dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str, producer: Callable[[], Any]) -> Any:
        now = time.time()
        with self._lock:
            hit = self._data.get(key)
            if hit and now - hit[0] < self.ttl_s:
                return hit[1]
        value = producer()
        with self._lock:
            self._data[key] = (now, value)
        return value


_cache = _TtlCache()


def _http_get_json(url: str, timeout: float = 1.5) -> Optional[Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            return json.loads(body) if body.strip() else {}
    except Exception:
        return None


def _tcp_alive(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def orion_url() -> str:
    return os.environ.get("ORION_URL", "http://127.0.0.1:1026").rstrip("/")


def _mongo_target() -> tuple[str, int]:
    raw = os.environ.get("HARVEST_MONGO_HOST", "127.0.0.1:27017")
    host, _, port = raw.partition(":")
    return host or "127.0.0.1", int(port or 27017)


def _probe_orion() -> Optional[Dict[str, Any]]:
    return _cache.get("orion", lambda: _http_get_json(f"{orion_url()}/version"))


def _probe_mongo() -> bool:
    host, port = _mongo_target()
    return _cache.get("mongo", lambda: _tcp_alive(host, port))


def _probe_entities() -> Optional[Dict[str, Any]]:
    """Entity counts + freshness/sync info from the broker (one cached pass)."""
    def produce():
        base = orion_url()
        counts: Dict[str, int] = {}
        for etype in ("ElectricTractor", "ChargingStation", "EnergyConsumer",
                      "FarmEnergySystem"):
            rows = _http_get_json(
                f"{base}/ngsi-ld/v1/entities?type={etype}&limit=100&options=keyValues")
            if rows is None:
                return None
            counts[etype] = len(rows)
        grid = _http_get_json(
            f"{base}/ngsi-ld/v1/entities/urn:ngsi-ld:FarmEnergySystem:main")
        observed = None
        if isinstance(grid, dict):
            observed = (grid.get("gridDrawKw") or {}).get("observedAt")
        command = _http_get_json(
            f"{base}/ngsi-ld/v1/entities/urn:ngsi-ld:FarmCommand:main?options=keyValues")
        return {
            "counts": counts,
            "grid_observed_at": observed,
            "last_nonce": (command or {}).get("lastNonce") or "",
        }
    return _cache.get("entities", produce)


def _observed_age_s(observed_at: Optional[str]) -> Optional[float]:
    if not observed_at:
        return None
    try:
        import datetime as dt
        ts = dt.datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
        return max(0.0, dt.datetime.now(dt.timezone.utc).timestamp() - ts.timestamp())
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
#  The collector
# --------------------------------------------------------------------------- #
def collect(runtime, clients: ClientRegistry, tasks: Any = None) -> Dict[str, Any]:
    """Build the full diagnostics document.

    ``runtime`` is the live :class:`FleetRuntime` (may be ``None`` when
    harvest_integrations could not be created — reported, not raised).
    ``tasks`` is the optional :class:`~harvest_integrations.tasks.LiveTaskService`;
    absent, the farm-tasks row says so rather than disappearing.
    """
    now = time.time()
    services: List[Dict[str, Any]] = []

    def svc(name: str, state: str, detail: str, **extra: Any) -> None:
        services.append({"name": name, "state": state, "detail": detail, **extra})

    # HARVEST API — if this code runs, the server is answering.
    svc("HARVEST API", HEALTHY,
        f"server.py up {int(now - _STARTED_AT)}s, endpoints /simulate /api/roi /api/fleet/*")

    # Fleet backend + devices.
    devices: List[Dict[str, Any]] = []
    if runtime is None:
        svc("Fleet backend", FAILED, "FleetRuntime could not be created")
    else:
        status = runtime.status()
        try:
            runtime.snapshot()                     # refresh device health
            devices = runtime.device_diagnostics()
        except Exception as exc:
            svc("Fleet backend", FAILED, f"snapshot failed: {exc}")
            devices = []
        else:
            if status["backend"] == "devices":
                unreachable = [d["id"] for d in devices if d["reachable"] is False]
                if unreachable:
                    svc("Fleet backend", FAILED,
                        f"devices backend — unreachable: {', '.join(unreachable[:6])}",
                        backend="devices")
                else:
                    svc("Fleet backend", HEALTHY,
                        f"devices backend — {len(devices)} endpoints via Modbus/OPC-UA",
                        backend="devices")
            else:
                svc("Fleet backend", SIMULATED,
                    f"reference simulation, {status['sim_minutes_per_s']:g} sim-min/s "
                    "(switch with HARVEST_FLEET_BACKEND=devices)",
                    backend=status["backend"])

    # Farm simulator / field devices: summarised from the device rows.
    if runtime is not None and runtime.status()["backend"] == "devices":
        by_proto: Dict[str, List[Dict[str, Any]]] = {}
        for d in devices:
            by_proto.setdefault(d["protocol"], []).append(d)
        for proto, rows in sorted(by_proto.items()):
            ok = sum(1 for r in rows if r["reachable"])
            label = {"modbus": "Modbus adapter", "opcua": "OPC-UA adapter"}.get(
                proto, f"{proto} adapter")
            state = HEALTHY if ok == len(rows) else (FAILED if ok == 0 else FAILED)
            svc(label, state,
                f"{ok}/{len(rows)} devices answering ({rows[0]['endpoint']})")
    else:
        svc("Modbus adapter", INACTIVE,
            "not in use — sim backend (start with ./run_harvest_dashboard.sh devices)")
        svc("OPC-UA adapter", INACTIVE,
            "not in use — sim backend (start with ./run_harvest_dashboard.sh devices)")

    # FIWARE: broker, datastore, sync daemon.
    version = _probe_orion()
    entities = _probe_entities() if version is not None else None
    if version is None:
        svc("Orion-LD broker", INACTIVE,
            f"not reachable at {orion_url()} — optional "
            "(start with ./run_harvest_dashboard.sh fiware)")
    else:
        counts = (entities or {}).get("counts", {})
        svc("Orion-LD broker", HEALTHY,
            f"NGSI-LD {version.get('orionld version', '?')} — "
            f"{sum(counts.values())} mirrored entities", counts=counts)
    svc("MongoDB", HEALTHY if _probe_mongo() else INACTIVE,
        "broker datastore answering" if _probe_mongo()
        else f"not reachable at {':'.join(map(str, _mongo_target()))} — optional")

    sync_age = clients.age_s("fiware-sync")
    mirror_age = _observed_age_s((entities or {}).get("grid_observed_at"))
    if sync_age is None or sync_age > 120.0:   # long-stopped optional != error
        svc("FIWARE sync", INACTIVE if version is None else FAILED,
            "daemon not seen — optional" if version is None
            else "broker is up but the sync daemon is not polling this API")
    else:
        fresh = sync_age < 15.0
        detail = f"last poll {sync_age:.0f}s ago"
        if mirror_age is not None:
            detail += f", broker mirror age {mirror_age:.0f}s"
        if (entities or {}).get("last_nonce"):
            detail += f", last command nonce {entities['last_nonce']}"
        svc("FIWARE sync", HEALTHY if fresh else FAILED,
            detail if fresh else f"stale — {detail}")

    # ROS 2 bridge: optional, detected from its own polling traffic.  A short
    # silence reads as failed (it was alive moments ago); a long one reads as
    # inactive again — a deliberately stopped optional service is not an error.
    ros_age = clients.age_s("ros2-bridge")
    if ros_age is None or ros_age > 120.0:
        svc("ROS 2 bridge", INACTIVE,
            "not running — optional (start with ./run_harvest_dashboard.sh full)")
    elif ros_age < 15.0:
        svc("ROS 2 bridge", HEALTHY,
            f"polling this API (last {ros_age:.0f}s ago) and publishing /harvest/* topics")
    else:
        svc("ROS 2 bridge", FAILED, f"was running but silent for {ros_age:.0f}s")

    # Isaac Sim: optional physical/3D simulation layer.  The isaac-bridge
    # pushes its status (bridge + simulator view) to
    # POST /api/integrations/status; every state below must stay honest:
    # a deliberately-not-started simulator is INACTIVE, a stand-in is
    # SIMULATED, only a lost connection is FAILED.
    name, state, detail, extra = _isaac_row(clients)
    svc(name, state, detail, **extra)

    # Farm tasks: HARVEST's own work schedule, and which layer is executing it.
    name, state, detail, extra = _tasks_row(tasks)
    svc(name, state, detail, **extra)

    return {
        "ts": now,
        "services": services,
        "devices": devices,
        "fleet": runtime.status() if runtime is not None else None,
        "demo": DEMO.status(),
    }


def _isaac_row(clients: ClientRegistry) -> tuple[str, str, str, Dict[str, Any]]:
    """The Isaac Sim service row: (name, state, detail, extras).

    Distinguishes: never enabled (INACTIVE/optional), bridge up but waiting
    for a simulator (INACTIVE with instructions), stand-in connected
    (SIMULATED), Isaac connected (HEALTHY), and connection lost or simulator
    error (FAILED).
    """
    name = "Isaac Sim"
    info = clients.latest("isaac-bridge")
    if info is None or info["age_s"] > 120.0:
        return (name, INACTIVE,
                "not running — optional (start with "
                "./run_harvest_dashboard.sh isaac, or isaac-demo for the "
                "GPU-free stand-in)", {})
    if info["age_s"] > 20.0:
        return (name, FAILED,
                f"isaac bridge was running but silent for {info['age_s']:.0f}s",
                {})

    status = info["status"] if isinstance(info["status"], dict) else {}
    sim = status.get("simulator") or {}
    extra: Dict[str, Any] = {"simulator": sim}
    if not sim.get("connected"):
        if sim.get("ever_connected"):
            age = sim.get("telemetry_age_s")
            return (name, FAILED,
                    "simulator connection lost — last telemetry "
                    f"{age:.0f}s ago" if isinstance(age, (int, float))
                    else "simulator connection lost", extra)
        return (name, INACTIVE,
                "bridge running, waiting for a simulator — start Isaac Sim "
                "with ./scripts/run_isaac_sim.sh, or use isaac-demo for the "
                "GPU-free stand-in", extra)

    kind = str(sim.get("kind") or "?")
    detail = (f"{kind} connected — sim state {sim.get('state', '?')}, "
              f"{sim.get('entities_synced', 0)} entities synced, "
              f"telemetry {sim.get('telemetry_age_s', 0):.0f}s ago")
    if not sim.get("scene_acknowledged"):
        detail += ", scene not yet acknowledged"
    extra["facts"] = _isaac_facts(sim, status.get("ros") or {})
    if sim.get("state") == "error":
        return (name, FAILED, f"{kind} reported an error — {detail}", extra)
    # A stand-in must read as deliberate simulation, never as a live Isaac.
    return (name, SIMULATED if kind == "stub" else HEALTHY, detail, extra)


def _tasks_row(service: Any) -> tuple[str, str, str, Dict[str, Any]]:
    """The farm-tasks row: what work exists, who has it, who is executing it.

    HEALTHY when HARVEST's scheduler is running the task layer.  The row states
    plainly whether execution is driven by the PHYSICAL simulator (Isaac has the
    tractor at the work zone and is reporting progress) or by HARVEST's own clock
    (no simulator connected) — the same honesty rule as everywhere else here: a
    clock-driven task must not read as a physically executed one.
    """
    name = "Farm tasks"
    if service is None:
        return (name, INACTIVE,
                "task service not running — optional (needs the simulation "
                "module; see harvest_integrations/tasks.py)", {})
    try:
        document = service.document()
        lines = service.summary_lines()
    except Exception as exc:                                   # noqa: BLE001
        return (name, FAILED, f"task service error: {exc}", {})

    counts = document.get("counts") or {}
    execution = str(document.get("execution") or "?")
    detail = (f"{document.get('clock', '?')} — "
              + ", ".join(f"{counts.get(k, 0)} {k}"
                          for k in ("active", "assigned", "pending",
                                    "completed", "deferred", "missed"))
              + f"; executed by {execution}")
    facts = [f"scheduler: {document.get('scheduler', '?')} — HARVEST decides "
             "assignment, priorities and completion"]
    if execution == "physical":
        facts.append(f"execution: physical, reported by "
                     f"{document.get('physical_source') or 'the simulator'}")
    else:
        facts.append("execution: HARVEST's clock (no simulator reporting "
                     "physical progress)")
    facts.extend(lines)
    if document.get("last_error"):
        facts.append(f"problem: {document['last_error']}")
    extra: Dict[str, Any] = {"facts": facts, "tasks": {
        "counts": counts, "execution": execution,
        "clock": document.get("clock"),
        "assignments": document.get("assignments") or {}}}
    # A task layer that has fallen behind its own clock is a real fault, but a
    # farm with nothing due is not: only an error makes this row unhealthy.
    return (name, HEALTHY, detail, extra)


#: How many tractors get their own line before the list is summarised.  This is
#: an OPERATIONAL diagnostics view, not a simulation monitor: enough to see that
#: the fleet is synchronised and what the one interesting tractor is doing, and
#: no more.
_ISAAC_FACT_TRACTORS = 6


def _isaac_facts(sim: Dict[str, Any], ros: Dict[str, Any]) -> List[str]:
    """A handful of short lines about the live simulation.

    Everything here is REPORTED by the simulator and the bridge; nothing is
    inferred or recomputed, so a line that disagrees with HARVEST's own view is
    a real disagreement worth seeing rather than a rendering artefact.
    """
    facts: List[str] = []

    model = sim.get("robot_model") or {}
    if model:
        size = ""
        if model.get("length_m") and model.get("width_m"):
            size = f", {model['length_m']:.2g}×{model['width_m']:.2g} m"
        facts.append(f"model: {model.get('id', '?')} "
                     f"({model.get('provider', '?')}{size})")
    if sim.get("physics"):
        facts.append(f"physics: {sim['physics']}")

    view = sim.get("visualization") or {}
    if view.get("state") == "serving" and view.get("viewer_url"):
        facts.append(
            f"WebRTC: enabled, {view['viewer_url']} "
            f"({view.get('signal_port', '?')}/TCP + "
            f"{view.get('stream_port', '?')}/UDP) — open it with the NVIDIA "
            "Isaac Sim WebRTC Streaming Client")
    elif view.get("state") == "failed":
        facts.append(f"WebRTC: FAILED — {view.get('detail', 'no detail')}")
    elif view:
        facts.append("WebRTC: not enabled "
                     "(HARVEST_ISAAC_VIEW_MODE=webrtc to stream)")

    if ros:
        facts.append(
            f"ROS 2/DDS: connected, domain {ros.get('domain_id', '?')}, "
            f"{ros.get('rmw', '?')}, transport {ros.get('transport', '?')}")

    synced = sim.get("tractors_synced")
    driveable = sim.get("tractors_driveable")
    if synced is not None:
        run = ("running" if sim.get("state") == "running"
               else str(sim.get("state") or "?"))
        facts.append(
            f"simulation: {run}, {synced} tractor(s) synchronised"
            + (f", {driveable} driveable" if driveable is not None else "")
            + (f", sim time {sim['sim_time_s']:.0f}s"
               if isinstance(sim.get("sim_time_s"), (int, float)) else ""))

    entities = sim.get("entities") or {}
    tractors = [(eid, row) for eid, row in sorted(entities.items())
                if isinstance(row, dict) and row.get("kind") == "tractor"]
    for eid, row in tractors[:_ISAAC_FACT_TRACTORS]:
        pose = row.get("pose") or [0.0, 0.0]
        line = (f"{eid}: {row.get('physical_state', '?')} at "
                f"({pose[0]:.1f}, {pose[1]:.1f})")
        if isinstance(row.get("heading_deg"), (int, float)):
            line += f" hdg {row['heading_deg']:.0f}°"
        if isinstance(row.get("speed_mps"), (int, float)):
            line += f" {row['speed_mps']:.1f} m/s"
        charger = row.get("docked_charger")
        if charger:
            distance = row.get("distance_to_charger_m")
            line += (f" → {charger}"
                     + (f" {distance:.1f} m" if isinstance(distance, (int, float))
                        else ""))
            line += ", docked" if row.get("docked") else ", approaching"
        elif row.get("destination"):
            destination = row["destination"]
            line += f" → ({destination[0]:.1f}, {destination[1]:.1f})"
        if row.get("error"):
            line += f" — {row['error']}"
        facts.append(line)
    if len(tractors) > _ISAAC_FACT_TRACTORS:
        facts.append(f"… and {len(tractors) - _ISAAC_FACT_TRACTORS} more tractor(s)")

    for problem in (sim.get("problems") or [])[:3]:
        facts.append(f"problem: {problem}")
    return facts


# --------------------------------------------------------------------------- #
#  Cross-protocol demonstration
# --------------------------------------------------------------------------- #
class CrossProtocolDemo:
    """Scripted, observable demonstration of the DeviceIO abstraction.

    Sequence (all through the generic ``FleetInterface`` — no protocol code):

    1. read a fleet snapshot and note grid draw + charger power (served over
       **Modbus** in the devices backend);
    2. issue ``REQUEST_CHARGE`` for the lowest-SoC tractor (written over
       **OPC-UA** in the devices backend);
    3. wait until the tractor reports charging and a charger reports power —
       i.e. the OPC-UA command became visible through the Modbus meter;
    4. if Orion-LD is reachable, show the same state change mirrored as
       NGSI-LD (``charging`` on the ElectricTractor entity);
    5. release the charge request and confirm the fleet returns to rest.

    Runs on its own thread; the UI polls :meth:`status` for the step trace.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._state: Dict[str, Any] = {"state": "idle", "steps": []}

    # -- public ---------------------------------------------------------------
    def start(self, runtime) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._state = {"state": "running", "started_ts": time.time(),
                           "steps": []}
            self._thread = threading.Thread(
                target=self._run, args=(runtime,), name="crossproto-demo",
                daemon=True)
            self._thread.start()
            return True

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state))   # deep copy

    # -- internals ------------------------------------------------------------
    def _step(self, title: str, status: str, detail: str = "",
              data: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            self._state["steps"].append({
                "ts": time.time(), "title": title, "status": status,
                "detail": detail, "data": data or {},
            })

    def _finish(self, ok: bool, summary: str) -> None:
        with self._lock:
            self._state["state"] = "passed" if ok else "failed"
            self._state["summary"] = summary
            self._state["finished_ts"] = time.time()

    def _grid_view(self, snap) -> Dict[str, Any]:
        return {
            "grid_draw_kw": round(snap.grid.grid_draw_kw, 2),
            "pv_kw": round(snap.grid.pv_kw, 2),
            "chargers": {c.id: round(c.power_kw, 2) for c in snap.chargers},
        }

    def _run(self, runtime) -> None:
        try:
            protocols = {d["id"]: d["protocol"] for d in runtime.device_diagnostics()}
            snap = runtime.snapshot()
            candidates = [t for t in snap.tractors if t.available and not t.charging]
            if not candidates:
                self._finish(False, "no available non-charging tractor to demonstrate with")
                return
            tractor = min(candidates, key=lambda t: t.soc_pct)
            t_proto = protocols.get(tractor.id, "?")
            meter_proto = protocols.get("grid", "?")

            self._step(
                "Baseline read", "ok",
                f"grid meter via {meter_proto}: draw {snap.grid.grid_draw_kw:.2f} kW; "
                f"{tractor.id} at {tractor.soc_pct:.1f}% SoC (via {t_proto})",
                self._grid_view(snap))
            baseline_draw = snap.grid.grid_draw_kw

            acks = runtime.submit([Command.request_charge(tractor.id)])
            if not acks[0].accepted:
                self._step("Charge command", "fail",
                           f"REQUEST_CHARGE({tractor.id}) rejected: {acks[0].reason}")
                self._finish(False, "charge command rejected")
                return
            self._step(
                "Charge command", "ok",
                f"REQUEST_CHARGE({tractor.id}) accepted — written via {t_proto} "
                "through the common DeviceIO seam")

            deadline = time.time() + 25.0
            charging_snap = None
            while time.time() < deadline:
                time.sleep(1.0)
                snap = runtime.snapshot()
                t = snap.tractor(tractor.id)
                active = [c for c in snap.chargers if c.power_kw > 0.05]
                if t.charging and active:
                    charging_snap = snap
                    break
            if charging_snap is None:
                self._step("Cross-protocol effect", "fail",
                           "tractor never showed as charging with charger power")
                runtime.submit([Command.release_charge(tractor.id)])
                self._finish(False, "charging not observed")
                return
            active = [c for c in charging_snap.chargers if c.power_kw > 0.05]
            self._step(
                "Cross-protocol effect", "ok",
                f"{tractor.id} charging at {active[0].id} "
                f"({active[0].power_kw:.2f} kW); grid draw "
                f"{baseline_draw:.2f} → {charging_snap.grid.grid_draw_kw:.2f} kW "
                f"— observed via {meter_proto}",
                self._grid_view(charging_snap))

            # FIWARE reflection (optional — reported honestly either way).
            if _http_get_json(f"{orion_url()}/version", timeout=2.0) is None:
                self._step("FIWARE reflection", "skip",
                           "Orion-LD not running — start the fiware profile to "
                           "see the same state as NGSI-LD")
            else:
                entity_url = (f"{orion_url()}/ngsi-ld/v1/entities/"
                              f"urn:ngsi-ld:ElectricTractor:{tractor.id}"
                              "?options=keyValues")
                # The sync daemon pushes every ~2 s; give it a few cycles.
                entity = None
                for _ in range(5):
                    entity = _http_get_json(entity_url, timeout=3.0)
                    if entity and entity.get("charging"):
                        break
                    time.sleep(2.0)
                if entity and entity.get("charging"):
                    self._step("FIWARE reflection", "ok",
                               f"urn:ngsi-ld:ElectricTractor:{tractor.id} shows "
                               "charging=true in the context broker",
                               {"entity": {k: entity.get(k) for k in
                                           ("socPct", "charging", "energyKwh")}})
                elif entity is None:
                    self._step("FIWARE reflection", "fail",
                               f"broker is up but has no entity for {tractor.id} "
                               "— is the fiware-sync service running?")
                else:
                    self._step("FIWARE reflection", "fail",
                               "entity did not reflect charging=true in time "
                               "(is the fiware-sync service healthy?)")

            runtime.submit([Command.release_charge(tractor.id)])
            time.sleep(2.0)
            snap = runtime.snapshot()
            self._step(
                "Release", "ok",
                f"RELEASE_CHARGE({tractor.id}) issued; grid draw back to "
                f"{snap.grid.grid_draw_kw:.2f} kW", self._grid_view(snap))

            failed = [s for s in self.status()["steps"] if s["status"] == "fail"]
            self._finish(not failed,
                         "command in over one protocol, effect observed over the "
                         "other, mirrored to NGSI-LD — all through one DeviceIO seam"
                         if not failed else "one or more steps failed")
        except Exception as exc:
            self._step("Demo error", "fail", str(exc))
            self._finish(False, f"demo crashed: {exc}")


DEMO = CrossProtocolDemo()
