"""
HARVEST Isaac Sim standalone app (Isaac Sim 6.x) -- the physical execution and
visualisation layer for the tractor fleet HARVEST already models.

    ./scripts/run_isaac_sim.sh                       # GUI if a display exists
    HARVEST_ISAAC_VIEW_MODE=webrtc ./scripts/run_isaac_sim.sh
    ./scripts/run_isaac_sim.sh --self-test           # boot, report READY, exit
    ./scripts/run_isaac_sim.sh --self-test-drive     # ...and prove a tractor drives
    ./scripts/run_isaac_sim.sh --list-models

Runs inside NVIDIA Isaac Sim's bundled Python (``python.sh``), never inside
HARVEST's environment -- start it with ``scripts/run_isaac_sim.sh``, which finds
the installation, scrubs and rebuilds the ROS environment and settles the
viewing mode (see that script and README.md here for why each step is
load-bearing).

WHAT THIS PROCESS IS RESPONSIBLE FOR, and nothing else: receiving latched scene
and goal commands on ``/harvest/sim/command`` (schema harvest-sim/1.0), building
the demonstration field, DRIVING the tractors there with wheel torques, and
publishing MEASURED poses, motion and docking on ``/harvest/sim/telemetry``.

HARVEST remains authoritative for everything that is not physics: battery SOC,
the energy model, charging schedules and decisions, task scheduling, prices, PV,
grid constraints, the MARL agents and the NGSI-LD semantic state.  None of that
is computed, cached or second-guessed here.  A tractor drives to a charger
because HARVEST assigned it one; whether it is charging, and at what power, is
HARVEST's answer, not this simulator's.

Import order is load-bearing (WISEPACK finding): argparse first so bad flags
fail in milliseconds, ``SimulationApp`` before any ``isaacsim``/``omni`` import,
the ``isaacsim.ros2.bridge`` extension before ``rclpy`` -- Isaac ships its own
ABI-compatible ROS 2 build and a host rclpy crashes inside its C extension.

Graceful absence of HARVEST: the app builds nothing until a scene command
arrives; it sits publishing "starting" telemetry, and because the command topic
is latched it also catches a scene published before it booted.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# The contract, the kinematics helpers, the robot registry and the streaming
# configuration are imported from the repository by path: Isaac's interpreter has
# no HARVEST packages installed, and these modules are pure stdlib (+PyYAML,
# which Isaac bundles) by design.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3]))          # repo root
from harvest_integrations.simulators.isaac import contract           # noqa: E402
from harvest_integrations.simulators.isaac import robots as robot_registry  # noqa: E402
from harvest_integrations.simulators.isaac import streaming          # noqa: E402

_TELEMETRY_PERIOD_S = 0.5
_LOG = "[harvest-isaac]"
_VERSION = "harvest-isaac/2.0"

#: A tractor counts as stopped below this speed.  Physics never settles at
#: exactly zero: the wheels keep micro-rolling against the ground.
_STOPPED_MPS = 0.15


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HARVEST Isaac Sim layer")
    parser.add_argument("--headless", action="store_true",
                        help="run without the Isaac viewport")
    parser.add_argument("--robot-model", default=None,
                        help=f"model id from robot_models.yaml (or ${robot_registry.MODEL_ENV})")
    parser.add_argument("--list-models", action="store_true",
                        help="list the configured robot models and exit")
    parser.add_argument("--speed", type=float, default=contract.DEFAULT_SPEED_MPS,
                        help="tractor cruising speed, m/s (capped by the model)")
    parser.add_argument("--dock-radius", type=float,
                        default=contract.DEFAULT_DOCK_RADIUS_M,
                        help="a tractor is docked within this radius of its charger, m")
    parser.add_argument("--max-runtime", type=float, default=0.0,
                        help="exit after this many seconds (0 = run forever)")
    parser.add_argument("--self-test", action="store_true",
                        help="boot Isaac + the ROS 2 bridge, publish one "
                             "'starting' telemetry message and exit 0 -- "
                             "validates the environment without HARVEST")
    parser.add_argument("--self-test-drive", action="store_true",
                        help="build a local scene and drive one tractor under "
                             "physics, reporting measured displacement -- "
                             "validates the vehicle without HARVEST")
    return parser.parse_args()


def _list_models() -> int:
    registry = robot_registry.load_registry()
    print(f"{_LOG} robot models in {robot_registry.REGISTRY_FILE.name} "
          f"(default: {registry['default_model']}):")
    for model in registry["models"].values():
        mark = " " if model.enabled else "-"
        print(f"  [{mark}] {model.id:24s} {model.provider:11s} "
              f"{model.display_name}")
        print(f"      {model.resembles}")
    print(f"{_LOG} '-' = disabled; select one with "
          f"{robot_registry.MODEL_ENV}=<id>")
    return 0


def main() -> int:                                     # noqa: PLR0915, PLR0912
    args = _parse_args()
    if args.list_models:
        return _list_models()

    # Resolve the robot model BEFORE booting Kit: a bad model id should fail in
    # milliseconds, not after ninety seconds of shader compilation.
    try:
        model = robot_registry.resolve_model(args.robot_model)
    except robot_registry.RobotModelError as exc:
        print(f"{_LOG} ERROR: {exc}", file=sys.stderr)
        return 4
    stream_config = streaming.StreamingConfig.from_env()
    print(f"{_LOG} robot model : {model.id} ({model.display_name})", flush=True)
    print(f"{_LOG} streaming   : "
          f"{'webrtc ' + stream_config.resolved_viewer_url() if stream_config.enabled else 'disabled'}",
          flush=True)

    # ---- Isaac boot (only now do the heavyweight imports become legal) ------
    from isaacsim import SimulationApp                       # noqa: PLC0415
    simulation_app = SimulationApp(
        streaming.launch_config(stream_config, args.headless))

    import isaacsim.core.experimental.utils.app as app_utils  # noqa: PLC0415

    # ---- WebRTC, before anything slow, so a watching operator sees a picture
    # while the first stage is still being built.
    visualization = {"enabled": False, "state": "disabled", "detail": ""}
    try:
        visualization = streaming.enable(simulation_app, stream_config)
        if stream_config.enabled:
            print(f"{_LOG} WebRTC      : {visualization['detail']}", flush=True)
            for line in stream_config.client_hint().splitlines():
                print(f"{_LOG}   {line}", flush=True)
    except Exception as exc:                                 # noqa: BLE001
        # Reported, not fatal: the physical loop is the product, the stream is
        # how you watch it.  Telemetry carries the failure to Diagnostics.
        visualization = {"enabled": stream_config.enabled, "state": "failed",
                         "detail": f"{type(exc).__name__}: {exc}",
                         **stream_config.to_dict()}
        print(f"{_LOG} WARNING: WebRTC streaming unavailable: {exc}",
              file=sys.stderr, flush=True)

    from harvest_integrations.simulators.isaac.scene import HarvestWorld  # noqa: E402,PLC0415

    world = HarvestWorld(model)
    world_problems: list = []
    attached = 0

    def build_world(scene: dict, fingerprint: str) -> None:
        """Stop, rebuild, play, re-bind.  The only way the stage changes shape."""
        nonlocal attached, world_problems
        app_utils.stop()
        simulation_app.update()
        world.build(scene, fingerprint)
        app_utils.play()
        # A few updates before binding: PhysX publishes its articulation views
        # on the first simulation steps after play, and Articulation() on an
        # unstepped stage finds no DOFs.
        for _ in range(6):
            simulation_app.update()
        attached, world_problems = world.attach()
        for problem in world_problems:
            print(f"{_LOG} WARNING: {problem}", file=sys.stderr, flush=True)
        print(f"{_LOG} scene applied: {len(world.robots)} tractors "
              f"({attached} driveable), {len(world.chargers)} chargers, "
              f"fingerprint {fingerprint}", flush=True)

    # ---- the self-test that needs no ROS at all -----------------------------
    if args.self_test_drive:
        return _self_test_drive(simulation_app, app_utils, world, build_world,
                                args, model)

    # ---- ROS 2: the bridge extension first, THEN rclpy (Isaac's own build) --
    app_utils.enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()

    import rclpy                                             # noqa: PLC0415
    from rclpy.qos import (DurabilityPolicy, HistoryPolicy,   # noqa: PLC0415
                           QoSProfile, ReliabilityPolicy)
    from std_msgs.msg import String                           # noqa: PLC0415

    latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
    rclpy.init()
    node = rclpy.create_node("harvest_isaac_sim")
    pending: list = []
    node.create_subscription(String, contract.SIM_COMMAND_TOPIC,
                             lambda m: pending.append(m.data), latched)
    telemetry_pub = node.create_publisher(
        String, contract.SIM_TELEMETRY_TOPIC, latched)

    started_ts = time.time()
    goals: dict = {}
    charger_states: dict = {}

    def telemetry_entities() -> dict:
        """Per-entity physical state, MEASURED from the simulated bodies.

        Never echoed back from the goals: a pose that merely repeats what was
        asked for cannot tell HARVEST that a tractor arrived, and would make
        docking an assertion instead of an observation.
        """
        out: dict = {}
        for rid, robot in world.robots.items():
            goal = goals.get(rid) or {}
            charger_id = goal.get("docked_charger")
            charger_pose = world.charger_pose(charger_id)
            distance_to_charger = (
                math.hypot(charger_pose[0] - robot.pose[0],
                           charger_pose[1] - robot.pose[1])
                if charger_pose else None)
            moving = robot.speed_mps > _STOPPED_MPS
            docked = bool(charger_pose is not None
                          and distance_to_charger is not None
                          and distance_to_charger <= args.dock_radius
                          and not moving)
            target = goal.get("target")
            out[rid] = {
                "kind": "tractor",
                "pose": [round(robot.pose[0], 2), round(robot.pose[1], 2)],
                "heading_deg": round(robot.heading_deg, 1),
                "speed_mps": round(robot.speed_mps, 2),
                "moving": moving,
                "docked": docked,
                "docked_charger": charger_id,
                "charging": bool(goal.get("charging")),
                # One definition of the vocabulary, in the contract, shared
                # with the GPU-free stand-in.
                "physical_state": contract.physical_state(
                    moving=moving, docked=docked,
                    charging=bool(goal.get("charging"))),
                "destination": ([round(float(target[0]), 2),
                                 round(float(target[1]), 2)] if target else None),
                "distance_to_target_m": round(
                    math.hypot(target[0] - robot.pose[0],
                               target[1] - robot.pose[1]), 2) if target else 0.0,
                "distance_to_charger_m": (round(distance_to_charger, 2)
                                          if distance_to_charger is not None else None),
                "drive": robot.last_command.reason,
                "driveable": robot.ready,
            }
            if robot.attach_error:
                out[rid]["error"] = robot.attach_error
        for cid, info in world.chargers.items():
            state = charger_states.get(cid) or {}
            out[cid] = {
                "kind": "charger",
                "pose": [round(info["pose"][0], 2), round(info["pose"][1], 2)],
                "active": bool(state.get("active")),
                "power_kw": round(float(state.get("power_kw", 0.0)), 3),
            }
        return out

    def publish_telemetry(state: str, detail: str = "") -> None:
        entities = telemetry_entities()
        stats = {
            "entities_synced": len(entities),
            "tractors_synced": len(world.robots),
            "tractors_driveable": attached,
            "sim_time_s": round(time.time() - started_ts, 1),
            "uptime_s": round(time.time() - started_ts, 1),
        }
        telemetry_pub.publish(String(data=json.dumps(contract.telemetry_message(
            simulator={
                "kind": "isaac",
                "state": state,
                "version": _VERSION,
                "started_ts": started_ts,
                "detail": detail or "NVIDIA Isaac Sim physical layer",
                # What the operator needs to watch it, and what is being
                # simulated -- both reported, never inferred downstream.
                "visualization": visualization,
                "robot_model": model.describe(),
                "physics": "PhysX articulation, skid-steer wheel drives",
                "problems": world_problems,
            },
            entities=entities, fingerprint=world.fingerprint, stats=stats))))

    app_utils.play()
    simulation_app.update()
    publish_telemetry("starting", "waiting for a HARVEST scene command")
    print(f"{_LOG} READY -- waiting for {contract.SIM_COMMAND_TOPIC} "
          f"(ROS_DOMAIN_ID {os.environ.get('ROS_DOMAIN_ID', '0')}, "
          f"RMW {os.environ.get('RMW_IMPLEMENTATION', '?')})", flush=True)

    if args.self_test:
        for _ in range(60):                       # let the publication match
            simulation_app.update()
            rclpy.spin_once(node, timeout_sec=0.0)
        print(f"{_LOG} self-test ok", flush=True)
        rclpy.shutdown()
        simulation_app.close()
        return 0

    started = time.time()
    last_telemetry = 0.0
    try:
        while simulation_app.is_running():
            # ONE update per iteration: it renders a frame, steps physics and
            # services the livestream.  Never block the loop on ROS.
            simulation_app.update()
            rclpy.spin_once(node, timeout_sec=0.0)

            while pending:
                command = contract.parse_message(pending.pop(0))
                if command is None:
                    print(f"{_LOG} refused message with bad JSON/schema",
                          flush=True)
                    continue
                scene = command.get("scene")
                fingerprint = str(command.get("scene_fingerprint") or "")
                if scene and fingerprint != world.fingerprint:
                    build_world(scene, fingerprint)
                goals = dict(command.get("goals") or {})
                charger_states = dict(command.get("chargers") or {})
                for cid, state in charger_states.items():
                    world.set_charger_active(cid, bool(state.get("active")))
                publish_telemetry("running")

            # ---- physics-in-the-loop: measure, then steer ------------------
            for rid, robot in world.robots.items():
                robot.measure()
                goal = goals.get(rid) or {}
                target = goal.get("target")
                robot.drive_towards(
                    (float(target[0]), float(target[1])) if target else None,
                    arrival_radius_m=args.dock_radius,
                    speed_mps=args.speed)

            if time.time() - last_telemetry >= _TELEMETRY_PERIOD_S:
                publish_telemetry("running" if world.fingerprint else "starting")
                last_telemetry = time.time()

            if args.max_runtime and time.time() - started > args.max_runtime:
                print(f"{_LOG} max runtime reached, exiting", flush=True)
                break
    except Exception as exc:
        # Tell the bridge the backend died rather than letting it time out
        # (publish, then spin so the RELIABLE publication reaches the wire).
        publish_telemetry("error", f"{type(exc).__name__}: {exc}")
        print(f"{_LOG} fatal: {exc}", flush=True)
        for _ in range(30):
            simulation_app.update()
        raise
    finally:
        rclpy.shutdown()
        simulation_app.close()
    return 0


# --------------------------------------------------------------------------- #
#  Physics self-test: no ROS, no HARVEST, no network
# --------------------------------------------------------------------------- #
def _self_test_drive(simulation_app, app_utils, world, build_world, args,
                     model) -> int:
    """Build a one-tractor scene locally and prove the vehicle DRIVES.

    This is the test that would have caught every way a wheeled articulation can
    look fine and not move: wheels spinning without traction, a drive that is
    position- not velocity-controlled, a units error between rad/s and deg/s, a
    joint whose axis is wrong.  It reports MEASURED displacement and speed, and
    fails if the tractor did not get appreciably closer to its target.
    """
    scene = {
        "field": {"width": 90.0, "height": 90.0},
        "entities": [
            {"id": "tractor_selftest", "kind": "tractor", "home": [70.0, 50.0]},
            {"id": "charger_selftest", "kind": "charger", "pose": [40.0, 40.0]},
        ],
    }
    build_world(scene, "selftest")
    robot = world.robots["tractor_selftest"]
    if not robot.ready:
        print(f"{_LOG} self-test-drive FAILED: articulation not bound: "
              f"{robot.attach_error}", file=sys.stderr, flush=True)
        simulation_app.close()
        return 5

    target = tuple(world.charger_pose("charger_selftest") or (40.0, 40.0))
    robot.measure()
    start_pose = robot.pose
    start_distance = math.hypot(target[0] - start_pose[0],
                                target[1] - start_pose[1])
    print(f"{_LOG} self-test-drive: {model.id} at {start_pose} -> {target} "
          f"({start_distance:.1f} m), {robot.articulation.num_dofs} DOFs "
          f"{robot.articulation.dof_names}", flush=True)

    # ---- where IS the vehicle, actually? -----------------------------------
    # The chassis HEIGHT answers the question that the drive numbers alone
    # cannot: this vehicle is authored with its hull centre at
    # ground_clearance + height/2, so if the wheels are carrying it the root
    # stays at that height, and if the hull has settled onto the field it drops
    # to height/2 with the wheels turning in the air.  Measured before the
    # chassis was restructured: exactly the latter.
    #
    # READ FROM THE ARTICULATION ITSELF, and nothing else is wrapped.  Two
    # earlier versions of this check broke the very thing they measured: adding a
    # rigid body to a PLAYING stage, and wrapping articulation links in a
    # separate RigidPrim view, each invalidated the PhysX tensor views so every
    # read and every drive command failed with "Failed to get ... from backend".
    # If it is part of an articulation, ask the articulation.
    authored_hull_z = (float(model.body.get("ground_clearance_m", 0.22))
                       + float(model.body.get("height_m", 1.0)) / 2.0)
    print(f"{_LOG} chassis height authored at {authored_hull_z:.2f} m "
          f"(hull alone would rest at "
          f"{float(model.body.get('height_m', 1.0)) / 2.0:.2f} m)", flush=True)
    # A WINDOW INTO THE DRIVE, because "the tractor did not move" has half a
    # dozen possible causes that look identical from outside: a fixed-base
    # articulation, a joint axis in the wrong direction, a position drive where a
    # velocity drive was meant, a rad/s vs deg/s units error, wheels not touching
    # the ground, or torque saturation.  Each of those shows a different
    # signature in these four numbers.
    def _drive_report(tag: str) -> None:
        try:
            targets = robot.articulation.get_dof_velocity_targets().numpy()[0]
            actual = robot.articulation.get_dof_velocities().numpy()[0]
            linear, angular = robot.articulation.get_velocities()
            root_z = float(robot.articulation.get_world_poses()[0].numpy()[0][2])
            print(f"{_LOG}   {tag}: z={root_z:.3f} cmd={robot.last_command.reason} "
                  f"v={robot.last_command.linear_mps:.2f} m/s "
                  f"yaw={robot.last_command.yaw_rate_dps:.1f} deg/s | "
                  f"dof targets={[round(float(t), 2) for t in targets]} "
                  f"dof actual={[round(float(a), 2) for a in actual]} | "
                  f"body linear={[round(float(x), 2) for x in linear.numpy()[0]]} "
                  f"angular={[round(float(x), 2) for x in angular.numpy()[0]]}",
                  flush=True)
        except Exception as exc:                             # noqa: BLE001
            print(f"{_LOG}   {tag}: could not read DOF state: {exc}", flush=True)
        if robot.attach_error:
            print(f"{_LOG}   {tag}: error: {robot.attach_error}", flush=True)

    print(f"{_LOG} links={robot.articulation.num_links} "
          f"{robot.articulation.link_names} joints={robot.articulation.num_joints}",
          flush=True)

    deadline = time.time() + 60.0
    top_speed = 0.0
    report_at = time.time() + 2.0
    while time.time() < deadline:
        simulation_app.update()
        robot.measure()
        top_speed = max(top_speed, robot.speed_mps)
        robot.drive_towards(target, arrival_radius_m=args.dock_radius,
                            speed_mps=args.speed)
        remaining = math.hypot(target[0] - robot.pose[0],
                               target[1] - robot.pose[1])
        if time.time() >= report_at:
            _drive_report(f"t+{60.0 - (deadline - time.time()):.0f}s")
            report_at = time.time() + 10.0
        if remaining <= args.dock_radius:
            break
    robot.measure()
    remaining = math.hypot(target[0] - robot.pose[0], target[1] - robot.pose[1])
    travelled = math.hypot(robot.pose[0] - start_pose[0],
                           robot.pose[1] - start_pose[1])
    print(f"{_LOG} self-test-drive: pose {tuple(round(p, 2) for p in robot.pose)} "
          f"heading {robot.heading_deg:.0f} deg, travelled {travelled:.1f} m, "
          f"{remaining:.1f} m to go, top speed {top_speed:.2f} m/s", flush=True)
    docked = remaining <= args.dock_radius
    if docked:
        print(f"{_LOG} self-test-drive ok -- physically docked", flush=True)
    elif travelled > 2.0:
        print(f"{_LOG} self-test-drive ok -- moved under physics but did not "
              f"reach the charger within 60 s", flush=True)
    else:
        print(f"{_LOG} self-test-drive FAILED: the tractor did not move "
              f"({travelled:.2f} m in 60 s)", file=sys.stderr, flush=True)
        simulation_app.close()
        return 5
    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
