# HARVEST × NVIDIA Isaac Sim (optional physical/3D layer)

Isaac Sim provides an **optional** physical simulation layer for entities
HARVEST already models: tractors driving across the farm, docking at charging
stations, charger activity. HARVEST remains authoritative for energy
management, scheduling, agent decisions and semantic state; Isaac only
*executes and visualises* the physical behaviour and reports the measured
result back. The architecture follows WISEPACK's proven Isaac integration,
adapted from manipulation to farm mobility.

```text
HARVEST agents / scheduler
        |
   FleetInterface / DeviceIO           (unchanged; single seam for all backends)
        |
   /api/fleet/* HTTP API  ──►  ros2-bridge (fleet)  ──►  /harvest/fleet/* topics
                                                              |
                                                      isaac-bridge (ROS 2 node)
                                                              |  /harvest/sim/command
                                                              |  /harvest/sim/telemetry
                                                              ▼
                                     Isaac Sim (host)  or  GPU-free stub (isaac-demo)
```

The bridge derives motion **goals** from the semantic fleet state (a tractor
with a charger assignment gets that charger's field coordinates as its drive
target); the simulator *drives* the tractor there, detects docking physically
and publishes measured poses back.

In Isaac each tractor is a **PhysX articulation** — a chassis rigid body, four
wheel rigid bodies, four revolute joints with angular velocity drives — so it
reaches a charger because its wheels turn against the ground and the body
accelerates, steers and rolls to a stop. Nothing is teleported, and the poses
HARVEST receives are *measured* from the simulated bodies, which is what makes
arrival and docking observations rather than assertions. (The GPU-free stub
instead interpolates in a straight line; Diagnostics always says which of the
two is connected.) Telemetry surfaces in the Diagnostics tab
(`Isaac Sim` component), in `GET /api/integrations/status`, and — when the
FIWARE profile is up — as the `urn:ngsi-ld:FarmSimulation:isaac` NGSI-LD
entity.

## Running

### Without Isaac installed (recommended first)

```bash
./run_harvest_dashboard.sh isaac-demo
```

Starts the whole stack plus a **GPU-free stand-in** (`stub.py`) driven by the
same `contract.py` + `motion.py` code the real Isaac app runs — the full
HARVEST → ROS 2 → simulator → HARVEST loop, end to end, on any machine.
Diagnostics shows the component as `SIMULATED` (honesty rule: a stand-in never
pretends to be Isaac).

### With Isaac Sim, watched over WebRTC

One command starts everything — the stack, the bridges and Isaac Sim itself:

```bash
HARVEST_ISAAC_VIEW_MODE=webrtc \
HARVEST_ISAAC_STREAMING=1 \
HARVEST_ISAAC_HEADLESS=1 \
./run_harvest_dashboard.sh isaac
```

Then **open the NVIDIA Isaac Sim WebRTC Streaming Client** and connect it to the
URL the launcher prints (`http://127.0.0.1:49100` by default). A browser cannot
show this stream: the installed Isaac Sim 6.0.1 livestream package ships no
in-browser client at all, and an HTTP GET to the signal port returns 501. The
client needs **both** ports — `49100/TCP` for signalling and `47998/UDP` for
media — so a TCP-only SSH tunnel negotiates a connection and then shows no
picture. From another machine:

```bash
ssh -L 49100:127.0.0.1:49100 <user>@<this-host>   # plus UDP 47998, see below
```

Isaac takes a minute or two on a cold start (shader compilation). The launcher
does **not** block on it: `tail -f /tmp/harvest-isaac-$(id -u)/isaac.log` and
wait for `[harvest-isaac] READY`, or watch the Diagnostics tab go from
`inactive` to `healthy`. `./run_harvest_dashboard.sh stop` stops the stack *and*
Isaac; `status` reports whether Isaac is running.

To run Isaac yourself instead (a second terminal, a debugger, a different
model):

```bash
HARVEST_ISAAC_AUTOSTART=0 ./run_harvest_dashboard.sh isaac
./scripts/run_isaac_sim.sh                       # GUI if a display exists
```

#### How you watch it: one variable

`HARVEST_ISAAC_VIEW_MODE` settles the whole viewing configuration, because the
individual switches interact — WebRTC needs headless, a desktop window needs a
display, and telemetry-only needs neither. Setting them separately is how you
end up asking for a GUI stream on a machine with no display and getting neither.

| mode | what happens | how you watch |
|---|---|---|
| `desktop` | a window on the host display | your own NoMachine / Sunshine+Moonlight / VNC session |
| `webrtc` | headless + Isaac's WebRTC livestream | NVIDIA Isaac Sim WebRTC Streaming Client |
| `none` | headless, no stream | telemetry only (Diagnostics, `/api/integrations/status`) |

Streaming knobs, all optional: `HARVEST_ISAAC_STREAM_HOST` (the **advertised**
address, not the bind address — Kit binds `0.0.0.0` regardless, so this cannot
restrict access), `HARVEST_ISAAC_SIGNAL_PORT` (49100/TCP),
`HARVEST_ISAAC_STREAM_PORT` (47998/UDP), `HARVEST_ISAAC_STREAM_URL` (for a
reverse proxy), `HARVEST_ISAAC_STREAM_FPS`.

**The stream has no authentication and no encryption.** Advertising loopback
does not restrict who can reach the port; that is a firewall decision. Prefer
loopback plus an SSH forward, a firewall rule scoped to one client address, or
an authenticated reverse proxy.

#### Why the launcher scrubs the environment

Isaac Sim runs **on the host in its own bundled Python, never in Docker**
(WISEPACK rule: it needs the GPU, its own Python ABI and its own ROS 2 build).
`scripts/run_isaac_sim.sh` is a port of WISEPACK's proven
`scripts/run_wisepack_isaac.sh`. It locates the installation (`ISAAC_SIM_ROOT`
overrides auto-detection), **scrubs the ROS environment** — sourcing a host
`/opt/ros` puts an ABI-incompatible `rclpy` ahead of Isaac's internal ROS 2 and
crashes its C extension — then sources Isaac's *own* `setup_ros_env.sh` to put
Isaac's ROS libraries back (without it `librmw_implementation.so` fails to load
and `import rclpy` raises), keeps only `ROS_DOMAIN_ID` (default 42), and pins
`RMW_IMPLEMENTATION=rmw_fastrtps_cpp` on both ends. The `isaac`-mode containers
run with `network_mode: host` because DDS discovery does not cross a bridged
Docker network.

#### DDS transport: UDPv4, and why it is pinned

Both ends set `FASTDDS_BUILTIN_TRANSPORTS=UDPv4`
(`scripts/run_isaac_sim.sh` and the `ros2-bridge-isaac` / `isaac-sim-stub`
services). This is a fix for a measured failure, not a preference.

Fast DDS prefers **shared memory** between participants on one host. Isaac runs
on the host as the invoking user; the ROS 2 containers run as root with
`ipc: host`, so their SHM segments are root-owned mode `0644` and the host user
cannot map them read-write. Fast DDS does not then fall back to UDP for a pair
it has already matched on a SHM locator, so the failure mode is the worst
possible one: **discovery succeeds and no data ever arrives.** `ros2 topic list`
shows `/harvest/sim/telemetry`, `ros2 topic echo` times out forever, and nothing
anywhere logs an error.

Measured on this machine (host Isaac `rclpy` ↔ `harvest:ros2`, domain 42, both
Fast DDS 2.14.6):

| configuration | discovery | data |
|---|---|---|
| default transports (SHM + UDP) | topic visible | **never delivered** |
| container as `--user 1000:1000` | topic visible | delivered |
| `FASTDDS_BUILTIN_TRANSPORTS=UDPv4` | topic visible | delivered |

UDPv4 is what HARVEST uses because it does not couple the container's user to
whoever happens to run Isaac. The traffic is two latched JSON strings at 2 Hz,
so loopback UDP costs nothing measurable. `HARVEST_DDS_TRANSPORT=DEFAULT`
restores Fast DDS's own preference on both ends if you want to re-test this.

Stale segments from a killed participant hang every later `ros2` command with no
error at all, so `scripts/clean_dds_shm.sh` (ported from WISEPACK) is run before
Isaac starts and after it stops.

#### Useful flags

`--headless`, `--self-test` (validate the Isaac + ROS 2 boot without HARVEST),
`--self-test-drive` (build a local scene and prove a tractor *physically*
drives, reporting measured displacement — the test that catches a vehicle whose
wheels spin without traction), `--list-models`, `--robot-model ID`,
`--max-runtime S`, `--speed M_PER_S`, `--dock-radius M`.

## Which robot represents a tractor

Isaac needs a *body* for something HARVEST models only as an id, a position and
a state of charge. That choice is isolated in
[`robot_models.yaml`](robot_models.yaml) + [`robots.py`](robots.py), and
**nothing above the simulator knows what a tractor looks like** — the
`harvest-sim/1.0` contract, the bridge, FleetInterface, Diagnostics and FIWARE
address tractors by HARVEST id and field coordinates only.

```bash
./scripts/run_isaac_sim.sh --list-models
HARVEST_ISAAC_ROBOT_MODEL=proxy_utility_tractor ./scripts/run_isaac_sim.sh
```

| model | provider | what it is |
|---|---|---|
| `proxy_utility_tractor` (default) | procedural | a 2.6 × 1.35 m four-wheel skid-steer vehicle with a cab and a load bed, built in the stage. Approximately a compact electric agricultural utility vehicle (ZETRABOT class) |
| `zetrabot_usd` | usd | **the seam for the real model** — disabled until a `zetrabot.usd` exists on this machine |
| `nova_carter` | usd | a stock NVIDIA AMR, for checking that the integration is not proxy-specific |

The proxy is the default deliberately: it needs no asset download, no Omniverse
Nucleus connection and no network, so on any machine with Isaac Sim installed it
always works. It is **not** a ZETRABOT CAD model and does not pretend to be one.

**Replacing the proxy with a real ZETRABOT model** is an edit to
`robot_models.yaml` and nothing else: point `asset_path_candidates` at the USD
file, name its articulation root, base link and per-side wheel joints, set
`enabled: true`, and select it with `HARVEST_ISAAC_ROBOT_MODEL=zetrabot_usd`. No
HARVEST code, no bridge change, no contract change.

## HARVEST's tasks in the field

The demonstrator shows the farm's actual work, not just its vehicles.  Every
task HARVEST has scheduled for the day is drawn where it is, and the tractor
assigned to it drives there, works for as long as HARVEST said the work takes,
and reports back.

**The division of labour is the point, so it is worth stating plainly.**

| decision | who |
|---|---|
| what work exists, where, with what priority and deadline | HARVEST (`main.build_simulation_config` → `task_generator`) |
| which tractor gets which task, in what order, and when | HARVEST (`main.Scheduler.assign_tasks` — *called*, not reimplemented) |
| whether a task is finished | HARVEST (`harvest_integrations/tasks.py`) |
| where the tractor actually is, whether it reached the work zone, how far through the work it is | Isaac Sim (reported as evidence) |

Isaac Sim has no scheduler, no queue and no priorities.  It is told a location,
a work radius and a number of seconds; a report about a task a tractor was not
given is ignored, and no report can create, reassign or cancel anything.

```text
HARVEST task scheduler ─► FleetInterface/DeviceIO ─► ROS 2 ─► Isaac Sim
pose / arrival / progress / completion ─► ROS 2 ─► HARVEST ─► FIWARE + Diagnostics
```

### What you see on the stream

Each task is a **work zone** (a disc of exactly the work radius HARVEST sent), a
**pole and sign**, an **assignment flag** and, for the task being travelled to
or worked, a **tall beam** visible across the field.  Colour carries the state:

| state | colour | meaning |
|---|---|---|
| pending | grey | known, not due or not yet assigned |
| assigned | amber | a tractor is on its way |
| active | green | being worked, right now |
| completed | dark green | done — deliberately dim, so the eye stops counting it |
| deferred | orange | was due, preempted or its window slipped; waiting again |
| missed | red | the window closed |

Only `assigned` and `active` tasks get the beam, so an 800 m field with twenty
markers still answers "where is the action?" at a glance.

**Which tractor is going to which task** needs no caption: every tractor carries
an identity colour as a roof stripe, and its task's flag is painted the same
colour.  **What a tractor is busy with** is its beacon: amber travelling, green
working, blue charging, grey idle.

`integrations.tasks` in `config.yaml` tunes this: `work_radius_m` (6),
`seconds_per_work_minute` (1.0 — matching the 60x farm clock, so a 30-minute
task takes 30 s to watch), `max_visualised_tasks` (24 — the rest are still
scheduled, just not drawn).

### Charging still wins

Charging is not special-cased anywhere in the task layer.  HARVEST's scheduler
already refuses to hand work to a charging tractor below 40 % (and may take a
well-charged one off a charger as a last resort, releasing it properly), and the
fleet backend already refuses to send a tractor with live work to a charger.
Where a charger assignment and a task assignment overlap for the moment of a
transition, the simulator drives to the charger — see `goals_from_snapshot`.

### The task day

Task windows are datetimes on the configured day; the live farm clock is a
time-of-day that runs at 60x and wraps at midnight.  The service anchors one to
the other, so windows open and close while you watch, and **rolls the day over**
when the last deadline passes — regenerating the day's work through the same
authoritative builder.  A stack left running therefore keeps having work to do
instead of becoming a museum piece.  A freshly started stack begins at 08:00,
which is when the first windows open.

### Watching it

```bash
curl -s localhost:8765/api/tasks | python3 -m json.tool     # the whole schedule
curl -s localhost:8765/api/tasks/goals                      # what the simulator is told
```

and the Diagnostics **Farm tasks** row, which is deliberately three or four
lines:

```text
tractor_2 -> task_018, travelling, 248 m remaining
tractor_1 -> task_003, working, 62%
tractor_3 -> charger_1, charging
```

It also states which layer is executing: `physical` when Isaac is reporting
progress, `clock` when HARVEST is advancing the work itself because no simulator
is connected.  A clock-driven task must never read as a physically executed one.

## The initial world

Deliberately simple, and built from HARVEST's own `config.yaml` (via
`contract.scene_from_config`) so there is no second list of farm structure: a
flat field with crop rows sized to contain everything, **three tractors**
(`tractor_1`, `tractor_2`, `tractor_3` at their configured field coordinates,
clearly separated), **two charging stations** (`charger_1`, `charger_2` — a
bright pad, a post, and a head that turns green while the station is delivering
power), **HARVEST's scheduled tasks** as work zones with poles and signs (see
above), sky and sun lighting, and a fixed spectator camera framing the tractors,
the chargers and the work. Deliberately absent: buildings, terrain, obstacles, crops
as geometry, vehicle cameras, and perception of any kind.

## The demonstrator

1. Start `./run_harvest_dashboard.sh isaac` (with the WebRTC variables above),
   or `isaac-demo` for the GPU-free stand-in.
2. Open the dashboard's **Diagnostics** tab → the `Isaac Sim` card shows the
   connection, simulator state and telemetry age, and beneath it a few live
   lines: the robot model, the WebRTC endpoint, the ROS 2/DDS domain and
   transport, how many tractors are synchronised and driveable, and one line per
   tractor with its pose, heading, speed, physical state, assigned charger,
   distance to it and whether it has docked. Deliberately a handful of lines:
   this stays an operational diagnostics view, not a simulation monitor.
3. Command a charge through the normal abstraction (any of these):
   - press **RUN CROSS-PROTOCOL DEMO** in Diagnostics;
   - `python3 examples/isaac_sim_demo.py`;
   - `curl -X POST localhost:8765/api/fleet/command -d '{"commands":[{"type":"request_charge","target_id":"tractor_1"}]}'`
4. HARVEST assigns a charger (semantic, instant, from its own energy model).
   The bridge turns that assignment into a physical goal; Isaac turns the goal
   into wheel velocities, and the tractor **visibly drives** across the field,
   turns toward the pad and rolls to a stop on it. Watch
   `distance_to_charger_m` fall and `docked` flip in
   `GET /api/integrations/status`, and watch the charger head turn green when
   HARVEST reports the station delivering power.
5. Isaac reports the arrival; HARVEST moves the tractor's operational state on
   from *its* side. Charging power and grid behaviour continue to come from the
   existing Modbus/OPC-UA devices and the HARVEST energy model — Isaac computes
   no SOC, no energy and no schedule.
6. With FIWARE up, the same state appears at
   `http://localhost:1026/ngsi-ld/v1/entities/urn:ngsi-ld:FarmSimulation:isaac`.

`./validate_harvest_stack.sh` runs this loop automatically when a simulator is
connected (skipped otherwise).

## Wire contract (`harvest-sim/1.0`)

Defined once in [`contract.py`](contract.py) — pure stdlib, imported by the
bridge, the stub and Isaac's interpreter alike, so the ends cannot drift.
Two `std_msgs/String` topics carrying versioned JSON (no custom message
package: Isaac's interpreter cannot import colcon-built packages, same as the
fleet bridge's rationale):

| topic | writer | QoS | payload |
|---|---|---|---|
| `/harvest/sim/command` | isaac-bridge | RELIABLE, TRANSIENT_LOCAL, depth 1 | scene spec (until acknowledged) + per-tractor goals + charger state |
| `/harvest/sim/telemetry` | simulator | RELIABLE, TRANSIENT_LOCAL, depth 1 | simulator identity/state + measured entity poses + stats |

Latching matters: Isaac takes tens of seconds to boot and must not miss the
scene command; a message with a mismatched schema MAJOR is refused, never
best-effort parsed. The scene carries a fingerprint computed identically on
both ends; telemetry acknowledges the fingerprint of the scene *as applied*,
never echoes the request, and the bridge keeps re-sending the scene until the
acknowledgement matches.

## Files

| file | runs where | role |
|---|---|---|
| `contract.py` | everywhere | topics, schema, scene/goal derivation, fingerprints |
| `motion.py` | stub | straight-line kinematics + work stopwatch (the stand-in's physics) |
| `../../tasks.py` | HARVEST API | the live task service — HARVEST's scheduler, driven |
| `stub.py` | Docker (`isaac-demo`) | GPU-free stand-in simulator |
| `robot_models.yaml` | host | **the only robot list** — the ZETRABOT seam |
| `robots.py` | host, Isaac's `python.sh` | registry, the procedural vehicle, skid-steer control |
| `scene.py` | host, Isaac's `python.sh` | the demonstration world (field, chargers, camera) |
| `streaming.py` | host, Isaac's `python.sh` | WebRTC livestream configuration |
| `harvest_isaac.py` | host, Isaac's `python.sh` | the Isaac Sim app and its loop |
| `../../../scripts/run_isaac_sim.sh` | host | env-scrubbing, view-mode-settling launcher |
| `../../../scripts/clean_dds_shm.sh` | host | removes orphaned Fast DDS segments |
| `../../../ros2_ws/src/harvest_ros/harvest_ros/isaac_bridge.py` | Docker | HARVEST ↔ simulator bridge node |

Tests: `tests/test_isaac_contract.py` (wire contract + kinematics),
`tests/test_isaac_robots.py` (registry, steering arithmetic, streaming
configuration, Diagnostics facts), `tests/test_live_tasks.py` (task assignment
through the fleet interface, physical execution, and the rule that a simulator
report is evidence and never instruction). Both run without Isaac, a GPU or ROS.
Anything that needs PhysX is covered by `--self-test-drive`, which measures
actual displacement rather than asserting that a command was sent.

## Notes for the next person

Three failures cost real time here, all with the same signature — the wheels
turn at exactly their commanded velocity, with no load and no error, and the
vehicle does not move:

* **A scaled rigid body carrying joints.** A joint's `localPos` is expressed in
  its body's own space, and that space carries the prim's scale, so an anchor of
  0.775 m on a chassis scaled 2.6 becomes 2.0 m. Bodies that carry joints are
  unscaled Xforms with a scaled child holding the collider.
* **Wheel gprim colliders.** `UsdGeom.Cylinder` + `CollisionAPI` produced no
  usable contact on this install: the vehicle settled onto its hull (chassis z
  0.72 → 0.50 = `height/2`) with the wheels turning in the air. Same-radius
  sphere colliders, with the cylinders kept as visuals, carry it.
* **Instrumentation that breaks what it measures.** Adding a rigid body to a
  *playing* stage, and wrapping articulation links in a separate `RigidPrim`
  view, each invalidated every PhysX tensor view in the process: reads and drive
  commands then failed with `Failed to get ... from backend` and a whole test run
  was meaningless. If a prim is part of an articulation, ask the articulation.
  To change the stage: stop, author, play, re-bind.

The single most useful number when a vehicle will not move is the **root
height**: authored `ground_clearance + height/2` means the wheels are carrying
it; `height/2` means they are not.

Then three more, from making the tractors actually *arrive* somewhere:

* **Cap the turn by the speed, never the speed by the turn.** A skid-steer
  vehicle cannot turn tightly at speed, so `drive_towards` pivots on the spot
  beyond 20° of misalignment and otherwise clamps the yaw to
  `degrees(v / MIN_TURN_RADIUS)`.  The inverse (`v ≤ ω·R`) looks equally
  plausible and is a lower bound, not an upper one: it clamped a 5° heading
  error to 0.55 m/s and crawled across an 800 m farm.
* **The minimum turn radius is a measurement.** 6 m is fine on paper (0.27 g at
  4 m/s) and skidded in PhysX — forward speed collapsed from 3.0 to 0.6 m/s
  mid-corner with the wheels still turning at 10 rad/s.  15 m keeps every turn
  inside the tyres' grip.
* **The instantaneous linear velocity is noisy.** A tractor making steady 3 m/s
  progress reported 0.8, 0.8, 3.5, 1.9, 0.5 m/s on consecutive samples, so
  "it has stopped" needs several consecutive frames (`_ARRIVAL_FRAMES`), not one
  low reading — otherwise a tractor starts "working" while driving through the
  zone.  Judge progress from the *position*, not from the speed.

The regression test for all of this is `--self-test-drive`: 334 m with the
target BEHIND the vehicle, which must pivot, cross the field and stop.  The
short forward hop it used to do passed happily while the controller could not
turn at all.

Richer agricultural simulation (field paths, implements, terrain, more device
types) extends `robots.py`/`scene.py` behind the same contract — the bridge,
Diagnostics and the HARVEST core need no changes. Computer vision, pose
estimation, cameras and manipulation are deliberately out of scope.
