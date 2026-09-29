# External integrations and Docker deployment

Fleet control interface, DeviceIO (Modbus / OPC-UA), FIWARE / NGSI-LD, ROS 2, Isaac Sim, the launcher and the Diagnostics view. Real ZETRABOT telemetry has its own page: [telemetry.md](telemetry.md).

## Fleet Control Interface

The `harvest_control/` package provides the **transport-agnostic boundary** between the decision layer (scheduler / MARL agents) and the plant — the pilot6 simulation today, the real ZETRABOT tractors in Stage 3.

```
decision layer  -->  FleetInterface  -->  { SimulationFleetInterface  (now)
                                         { Ros2FleetInterface          (Stage 3)
```

The decision layer only ever calls three methods, so the control logic validated here carries over to the September field deployment **without modification**:

| Method | Direction | Description |
|---|---|---|
| `snapshot()` | plant → decision | Returns `FleetSnapshot`: grid state, all tractor/charger/load states |
| `submit(cmds)` | decision → plant | Issues a batch of `Command` objects; returns one `CommandAck` per command |
| `advance(minutes)` | — | Steps a simulation backend forward (no-op on real hardware) |

### Command types

| Command | Effect |
|---|---|
| `Command.request_charge(tractor_id)` | Dock a tractor and start drawing power |
| `Command.release_charge(tractor_id)` | Stop charging / undock |
| `Command.set_charger_level(charger_id, level)` | `OFF` / `HALF` (~50 % rated) / `FULL` |
| `Command.assign_task(tractor_id, task_id)` | Begin a new farm task |
| `Command.preempt_task(tractor_id)` | Interrupt the current task |
| `Command.shed_load(load_id)` | Suppress a deferrable consumer |
| `Command.restore_load(load_id)` | Re-enable a previously shed consumer |

### TPI2 demo — 100 % autonomous control

```bash
# from the repo root
python -m harvest_control.demo_autonomous_control            # nominal day
python -m harvest_control.demo_autonomous_control --events   # with 4 mid-day plan changes
```

The demo drives a full simulated day with a tariff/PV/headroom-aware policy — no human in the loop. Output:

```
decisions executed              : 31
without manual intervention     : 100.0 %   (TPI2 target >= 70 %)
peak grid draw                  : 10.53 kW (cap 10.5 kW)
final state of charge           : {'T1': 85.5, 'T2': 88.1, 'T3': 85.7}
execution log written           : execution_log.csv
```

With `--events`, all four mid-day disruptions (urgent task injection, tractor breakdown, grid cap reduction, grid cap restore) are handled autonomously:

```
dynamic events handled          :
   - 10:00 emergency task injected
   - 13:30 Tractor 2 breakdown
   - 16:00 grid cap -> 7 kW
   - 18:00 grid cap restored
```

`execution_log.csv` (written to the current directory) is the TPI2 evidence file: one row per actuation command, with `manual_intervention: false` on every row.

### Wiring to the pilot6 engine

`SimulationFleetInterface` uses a compact built-in reference sim when constructed with no arguments. To wrap the real pilot6 engine instead, pass it and a set of `Adapters` callables:

```python
from harvest_control import SimulationFleetInterface, Adapters, TractorState, GridState

iface = SimulationFleetInterface(env=my_pilot6_env, adapters=Adapters(
    read_tractors = lambda env: [TractorState(t.id, t.soc_pct, t.kwh, t.online,
                                               t.charging, t.task) for t in env.tractors],
    read_chargers = lambda env: [...],
    read_loads    = lambda env: [...],
    read_grid     = lambda env: GridState(env.clock, env.grid_kw, env.cap,
                                          env.pv, env.tariff, env.price),
    apply_command = lambda env, c: env.handle(c),
    step          = lambda env, mins: env.step(mins),
))
```

### ROS 2 bridge (Stage 3)

A working, containerised ROS 2 bridge now lives in `ros2_ws/src/harvest_ros`
(see [External Integrations & Docker Deployment](#external-integrations--docker-deployment) — start it with
`./run_harvest_dashboard.sh full`).  It mirrors the live fleet onto canonical
`/harvest/*` topics and forwards command messages to the fleet API, so ROS
never needs to be installed on the host.  Rich objects travel as versioned
JSON in `std_msgs/String` (custom `harvest_msgs` remain deferred until the
ZETRABOT signal set is confirmed — the same decision WISEPACK documents for
its Orion-LD DDS-bridge compatibility).

`harvest_control/ros2_bridge.py` remains as the in-process skeleton variant
for a future `Ros2FleetInterface` running *inside* a ROS-native deployment;
the containerised bridge is the recommended path today.

## External Integrations & Docker Deployment

The `harvest_integrations` package connects HARVEST to industrial/agricultural
field devices (Modbus TCP, OPC-UA), to FIWARE (NGSI-LD context broker) and to
ROS 2 — while keeping HARVEST's semantic device-agent model
(`harvest_control`) the single authoritative domain model.  Everything in this
section is **optional**: `python main.py`, `python server.py`, MARL, the
predictor and ROI run unchanged without any of it.

Architecture (protocol abstraction adapted from TEMPO's adapter seam; Docker
orchestration, single-command startup and the state-mirror broker pattern
adapted from WISEPACK):

```
   external FIWARE apps          ROS 2 nodes / rqt
        │ NGSI-LD (HTTP)              │ /harvest/* topics (DDS, in-container)
        v                             v
  +-----------+   HTTP    +---------------------+
  | Orion-LD  | <-------- |  fiware-sync daemon |          docker compose
  | + MongoDB | PATCH cmd |  (mirror + inbound) |          profiles:
  +-----------+ --------> +----------+----------+            fiware
                                     │ /api/fleet/*           ros2
                          +----------+----------+             devices
                          |  server.py          |
                          |  FleetRuntime       |
                          +----------+----------+
                                     │ FleetInterface (harvest_control)
                 +-------------------+--------------------+
                 v                                        v
     SimulationFleetInterface                DeviceFleetInterface
     (reference sim, zero deps)              (DeviceIO seam: Modbus / OPC-UA /
                                              fake / register_protocol(...))
                                                          │
                                             farm-sim container *or* real
                                             chargers, loads, tractor BMS
```

### One-command startup

```bash
./run_harvest_dashboard.sh [mode]
```

| Mode | What runs | Fleet backend |
|---|---|---|
| `lite` | `server.py` on the host, no Docker (original behaviour) | lazy `sim` |
| `core` | Dashboard/API container only | `sim` |
| `devices` | + farm-device simulator (real Modbus TCP + OPC-UA on the wire) | `devices` |
| `fiware` *(default)* | + Orion-LD, MongoDB, NGSI-LD sync daemon | `devices` |
| `full` | + ROS 2 fleet bridge (`ros:jazzy`, DDS stays in-container) | `devices` |
| `isaac` | fiware + fleet & Isaac bridges on the **host network**, plus real Isaac Sim started on the host (`HARVEST_ISAAC_AUTOSTART=0` to start it yourself with `./scripts/run_isaac_sim.sh`) | `devices` |
| `isaac-demo` | isaac + a GPU-free simulator stand-in — the full HARVEST→ROS 2→simulator loop with no Isaac install | `devices` |
| `replay --file <csv> [--speed N] [--stack <mode>]` | **Real ZETRABOT telemetry** replayed on its original timestamps (see [telemetry.md](telemetry.md)); `lite` unless `--stack` names a mode above | as per stack |

`./run_harvest_dashboard.sh stop` tears everything down; `status` and
`logs [service]` are also available.  After every successful start the
launcher prints (and, if the stack is already up, re-prints) the
`HARVEST is up` summary box with the dashboard URL and the stop/logs/validate
commands.  Then validate the running stack end-to-end (exit code = number of
failed checks):

```bash
./validate_harvest_stack.sh
```

### Live fleet API

`server.py` gains three endpoints (created lazily — plain dashboard usage
starts no background fleet):

| Endpoint | Description |
|---|---|
| `GET /api/fleet/snapshot` | Current fleet state as `harvest-fleet/1.0` JSON |
| `POST /api/fleet/command` | `{"commands": [{"type": "shed_load", "target_id": "...", "value": ...}]}` → one ack per command |
| `GET /api/fleet/status` | Active backend (`sim` \| `devices`) |

The same JSON schema (`harvest_integrations/codec.py`) is used on the FIWARE
command entity and the ROS topics, so the three transports cannot drift apart.

### Device / protocol abstraction (Modbus, OPC-UA)

`harvest_integrations/devices` adapts TEMPO's `CellIO` pattern: the single
seam is `DeviceIO` (`read() -> {point: value}`, `write(point, value)`), and
*where* a point lives on the wire is configuration (`PointSpec`: register
address / node name, scale, writability), not code.  Protocol backends are
looked up in a registry:

```python
from harvest_integrations.devices import register_protocol
register_protocol("mqtt", MyMqttDeviceIO.from_endpoint)   # no core changes
```

`DeviceFleetInterface` composes one `DeviceIO` per endpoint into the standard
`FleetInterface` contract — the decision layer cannot tell it apart from the
simulation.  Device identity/count derive from `config.yaml`'s existing
`tractors.fleet` / `charging.stations` / `energy_consumers` sections; hosts
and ports live under `integrations.fleet` (env overrides
`HARVEST_FLEET_BACKEND`, `HARVEST_MODBUS_HOST`, `HARVEST_MODBUS_PORT`,
`HARVEST_OPCUA_ENDPOINT`).

The bundled simulator (`python -m harvest_integrations.simulators.farm_sim`)
serves **both protocols from one farm state** — chargers/loads/grid meter as
Modbus holding registers, tractor BMS as OPC-UA nodes — so a charge request
written over OPC-UA becomes charger power on the Modbus side within a tick.
Register/node maps are documented in
`harvest_integrations/simulators/modbus_server.py` and `opcua_server.py`.

### FIWARE / NGSI-LD

The `fiware-sync` daemon mirrors each fleet snapshot into Orion-LD as NGSI-LD
entities (SAREF-aligned, documented in `harvest_integrations/fiware/entities.py`):

| Entity | Content |
|---|---|
| `urn:ngsi-ld:ElectricTractor:<id>` | SoC, energy, availability, charging/V2L state, position |
| `urn:ngsi-ld:ChargingStation:<id>` | level, power, occupancy |
| `urn:ngsi-ld:EnergyConsumer:<id>` | name, shed state, power |
| `urn:ngsi-ld:FarmEnergySystem:main` | grid draw/cap, PV, tariff, price |
| `urn:ngsi-ld:FarmCommand:main` | **inbound**: `command` attr; `lastNonce`/`lastResult` write-back |
| `urn:ngsi-ld:FarmSimulation:isaac` | optional Isaac Sim layer: simulator kind/state, synced entities, per-tractor physical state, which tractors physically docked, robot model, whether a live view is being served (mirrored only while its bridge is alive) |

Telemetry attributes carry `observedAt` and `unitCode`; the broker holds
*current state*, not history.  External systems actuate the farm by PATCHing
the command entity:

```bash
curl -X PATCH http://localhost:1026/ngsi-ld/v1/entities/urn:ngsi-ld:FarmCommand:main/attrs \
  -H 'Content-Type: application/json' \
  -d '{"command": {"type": "Property", "value":
        "{\"nonce\": \"n42\", \"commands\": [{\"type\": \"shed_load\", \"target_id\": \"workshop_tools\"}]}"}}'
```

The daemon receives it via an NGSI-LD subscription (plus a polling fallback
that survives broker-cannot-reach-daemon topologies), forwards it to
`POST /api/fleet/command`, and writes the acks back into `lastResult`.
Replays are suppressed by nonce.  The broker never touches HARVEST state
directly.

### ROS 2 topics (`full` mode)

| Topic | Type | QoS |
|---|---|---|
| `/harvest/fleet/snapshot` | `String` (fleet JSON) | reliable, latched |
| `/harvest/fleet/command` | `String` (command JSON, inbound) | reliable |
| `/harvest/fleet/ack` | `String` (acks) | reliable, latched |
| `/harvest/grid/draw_kw`, `/harvest/grid/pv_kw`, `/harvest/grid/price_eur_per_kwh` | `Float32` | best-effort |
| `/harvest/grid/tariff` | `String` | reliable, latched |
| `/harvest/tractors/<id>/soc_pct` | `Float32` | best-effort |

The bridge (`ros2_ws/src/harvest_ros`) needs only `rclpy` + `std_msgs` and
talks to HARVEST over HTTP, so the coupling is one-way and fully
containerised — plain `ros:jazzy-ros-base`, no Vulcanexus requirement.

### Isaac Sim (optional physical/3D layer)

NVIDIA Isaac Sim can act as an optional physical simulation layer for
entities HARVEST already models — tractors driving to chargers, docking,
charger activity — following WISEPACK's proven Isaac architecture adapted to
farm mobility.  HARVEST remains authoritative for energy management,
scheduling and semantic state; the simulator only *executes/visualises*
physical behaviour and reports the measured result back:

```text
HARVEST agents / scheduler → FleetInterface/DeviceIO → ROS 2 → Isaac bridge
      → Isaac Sim (host) or GPU-free stub — telemetry flows back the same way
```

```bash
./run_harvest_dashboard.sh isaac-demo   # whole loop, no GPU/Isaac needed

# Real Isaac Sim 6.0.1, watched over WebRTC — stack, bridges and simulator:
HARVEST_ISAAC_VIEW_MODE=webrtc HARVEST_ISAAC_STREAMING=1 \
HARVEST_ISAAC_HEADLESS=1 ./run_harvest_dashboard.sh isaac
# then open the NVIDIA Isaac Sim WebRTC Streaming Client on http://127.0.0.1:49100
python3 examples/isaac_sim_demo.py      # command a charge, watch the tractor drive & dock
./run_harvest_dashboard.sh isaac-restart   # restart only Isaac; HARVEST keeps running
```

In Isaac each tractor is a **PhysX articulation** (chassis, four wheels, four
revolute joints with velocity drives), so it reaches a charger — or a task —
because its wheels turn against the ground.  Nothing is teleported, and the
poses HARVEST receives are measured from the simulated bodies.

**HARVEST's agricultural tasks are drawn in the field** as work zones with poles,
signs and assignment flags, colour-coded pending / assigned / active / completed
/ deferred / missed.  The assigned tractor drives there, works for as long as
HARVEST said the work takes, and reports arrival and progress back; HARVEST
declares completion and decides what the tractor does next.  Task creation,
assignment, priorities, deadlines and completion stay with HARVEST — the live
task service (`harvest_integrations/tasks.py`) *calls* `main.Scheduler`, so there
is no second scheduler and none at all inside Isaac.  `GET /api/tasks` is the
whole schedule, `GET /api/tasks/goals` is the subset the simulator is told, and
Diagnostics summarises it in three lines
(`tractor_2 -> task_018, travelling, 248 m remaining`).  Which body represents a
tractor is isolated in
[`robot_models.yaml`](../harvest_integrations/simulators/isaac/robot_models.yaml):
the default is a procedurally built compact utility vehicle (ZETRABOT class, no
asset download), and a real ZETRABOT USD model replaces it by editing that file
alone — no HARVEST change.

Isaac Sim itself runs **on the host, never in Docker** (its bundled Python +
GPU stack); the `isaac` profile puts the two bridge nodes on the host network
because Fast DDS discovery does not cross a bridged Docker network, and both
ends pin `FASTDDS_BUILTIN_TRANSPORTS=UDPv4` because Fast DDS's shared-memory
transport silently delivers nothing between a host process and a root
container (discovery succeeds, data never arrives — measured; see the layer
README).  The wire contract (`harvest-sim/1.0`, two latched JSON topics
`/harvest/sim/command` / `/harvest/sim/telemetry`), the robot-model registry,
the WebRTC setup and the full demonstrator walkthrough are documented in
[`harvest_integrations/simulators/isaac/README.md`](../harvest_integrations/simulators/isaac/README.md).
The Isaac bridge pushes its state to `POST /api/integrations/status`
(readable at `GET /api/integrations/status`), which feeds the Diagnostics
`Isaac Sim` component and the `FarmSimulation` NGSI-LD mirror.  Computer
vision and manipulation are deliberately out of scope.

### Diagnostics view

The dashboard's **Diagnostics** tab (modelled on WISEPACK's diagnostics page)
is the operational/debug/demo surface for everything in this section — served
by `GET /api/diagnostics`, read-only and allowlisted (no environment dumps, no
Docker socket).

* **Service health** — HARVEST API, fleet backend, Modbus/OPC-UA adapters,
  Orion-LD, MongoDB, FIWARE sync, the ROS 2 bridge and the Isaac Sim layer,
  each in one of five
  honest states: `healthy`, `simulated` (deliberate simulation never reads as
  a failure), `inactive` (optional component not started — with the launcher
  command that starts it), `failed` (expected but not answering), `unknown`.
  The sync daemon and ROS bridge are detected from their own API polling (an
  `X-Harvest-Client` header), so "running container" is never confused with
  "actually working".  The `Isaac Sim` card distinguishes: not enabled
  (`inactive`, optional), bridge up but waiting for a simulator (`inactive`,
  with the start command), the GPU-free stand-in (`simulated`), real Isaac
  connected (`healthy` — with sim state, synced entity count and telemetry
  age), and a lost connection or simulator error (`failed`).
* **Devices** — one row per endpoint from the `DeviceIO` layer: protocol,
  connection target, reachability, latest values and their age.  In `sim`
  mode the same rows appear tagged `sim`; a replayed real tractor appears
  tagged `csv-replay` (it is never counted as a field device or an adapter).
* **Real ZETRABOT mission — KPIs & model validation** — headline KPI cards
  with provenance badges, the real-vs-HARVEST-model table, rule-based
  validation findings, calibration proposals (never applied) and the
  explicit limitations, with JSON/CSV export — see
  [telemetry.md](telemetry.md#mission-kpis-and-model-validation).
* **Real telemetry** — the `Real telemetry` service card (`inactive` with the
  replay command when no source is configured, `failed` with the reason when
  the source cannot deliver — a missing file, the AWS scaffold — and
  `healthy` while a replay runs or a live feed is fresh) plus the **Real
  telemetry (ZETRABOT)** panel: mission and tractor ids, replay/live state and
  progress, source type, latest telemetry timestamp, SOC, voltage/current,
  derived power, cumulative discharged energy, temperatures, PTO/drive state
  and every other canonical field, with `n/a` for anything the data never
  carried (position included).
* **Cross-protocol demonstration** — a scripted, observable proof of the
  abstraction (button in the sidebar, or `POST /api/diagnostics/demo`):
  a charge command is issued through the generic fleet interface (written via
  **OPC-UA** to the tractor BMS in devices mode), the resulting charger power
  and grid-draw change are read back via the **Modbus** grid meter, and the
  same state change is shown mirrored as **NGSI-LD** in Orion-LD when the
  fiware profile is up (the step reports `skip` when it isn't).  The UI never
  touches a protocol — it only renders the server-side trace.

`./validate_harvest_stack.sh` runs the same demonstration headlessly as part
of the end-to-end validation, and — when a simulator is connected (`isaac` /
`isaac-demo` modes) — the physical charge loop as well: a charge command
issued over the fleet API must end with the simulator reporting the tractor
physically docked at its charger.

### Examples & tests

Client examples live in `examples/` (HTTP, NGSI-LD, direct `DeviceIO`).  All
integration-layer tests are stdlib `unittest` and run with the rest of the
suite:

```bash
python -m unittest discover tests            # protocol tests auto-skip
pip install -r requirements-integrations.txt # enables live Modbus/OPC-UA loopback tests
python -m unittest tests.test_telemetry -v   # real-telemetry layer: parsing, normalisation,
                                             # ordering, replay, FIWARE mapping, analysis
python -m unittest tests.test_telemetry_kpi -v  # mission KPIs: provenance, missing data,
                                             # async timestamps, real vs model, export, API/UI
```

`tests/test_telemetry.py` runs on `tests/fixtures/zetrabot_mission_fixture.csv`
(432 rows copied verbatim from the mission export, deliberately scrambled,
plus ten synthetic edge rows: unknown signal/message, empty/invalid/null
signals, a non-numeric value, a row without a timestamp, a second tractor);
the cross-checks against the Zetrack report run only when the full export is
present.

