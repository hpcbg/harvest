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
target); the simulator moves the tractor, detects docking physically and
publishes measured poses back. Telemetry surfaces in the Diagnostics tab
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

### With Isaac Sim

```bash
./run_harvest_dashboard.sh isaac        # HARVEST + host-network ROS 2 bridges
./scripts/run_isaac_sim.sh              # Isaac Sim itself, on the host
```

Isaac Sim runs **on the host in its own bundled Python, never in Docker**
(WISEPACK rule). The launcher locates the installation (`ISAAC_SIM_ROOT`
overrides auto-detection), **scrubs the ROS environment** — sourcing a host
`/opt/ros` puts an ABI-incompatible `rclpy` ahead of Isaac's internal ROS 2
and crashes its C extension — keeps only `ROS_DOMAIN_ID` (default 42), and
pins `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` on both ends so the containers and
Isaac actually discover each other. The `isaac`-mode containers run with
`network_mode: host` for the same reason: DDS discovery does not cross a
bridged Docker network.

Useful flags: `--headless`, `--self-test` (validate Isaac + ROS 2 boot without
HARVEST), `--max-runtime S`, `--speed M_PER_S`.

## The demonstrator

1. Start `./run_harvest_dashboard.sh isaac-demo` (or `isaac` + Isaac Sim).
2. Open the dashboard's **Diagnostics** tab → the `Isaac Sim` card shows the
   connection, simulator state, synced entity count and telemetry age.
3. Command a charge through the normal abstraction (any of these):
   - press **RUN CROSS-PROTOCOL DEMO** in Diagnostics;
   - `python3 examples/isaac_sim_demo.py`;
   - `curl -X POST localhost:8765/api/fleet/command -d '{"commands":[{"type":"request_charge","target_id":"tractor_1"}]}'`
4. HARVEST assigns a charger (semantic, instant). The simulator *drives* the
   tractor to the charger's coordinates and reports poses until it physically
   docks — watch `distance_to_target_m` fall and `docked` flip in
   `GET /api/integrations/status`.
5. With FIWARE up, the same state appears at
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
| `motion.py` | stub + Isaac | shared field kinematics (drive, dock) |
| `stub.py` | Docker (`isaac-demo`) | GPU-free stand-in simulator |
| `harvest_isaac.py` | host, Isaac's `python.sh` | the Isaac Sim app |
| `../../../scripts/run_isaac_sim.sh` | host | env-scrubbing Isaac launcher |
| `../../../ros2_ws/src/harvest_ros/harvest_ros/isaac_bridge.py` | Docker | HARVEST ↔ simulator bridge node |

Richer agricultural simulation (field paths, implements, terrain, more device
types) extends `motion.py`/`harvest_isaac.py` behind the same contract — the
bridge, diagnostics and HARVEST core need no changes. Computer vision and
manipulation are deliberately out of scope.
