"""
HARVEST Isaac Sim standalone app.

    ./scripts/run_isaac_sim.sh [--headless] [--max-runtime S] [--self-test]

Runs inside NVIDIA Isaac Sim's bundled Python (``python.sh``), never inside
HARVEST's environment -- start it with ``scripts/run_isaac_sim.sh``, which
locates the Isaac installation and scrubs the ROS environment first (see
that script and README.md here for why the scrub is load-bearing).

Role (adapted from WISEPACK): a physical/3D layer for entities HARVEST
already models.  It receives latched scene + goal commands on
``/harvest/sim/command`` (schema harvest-sim/1.0), builds a procedural farm
scene (ground plane, cuboid tractors, cylinder chargers at their configured
field coordinates), drives tractor prims with the shared
:mod:`motion.FieldKinematics` core and publishes measured prim poses on
``/harvest/sim/telemetry``.  HARVEST stays authoritative for energy
management; this app only executes/visualises motion and docking.

Import order is load-bearing (WISEPACK finding): argparse first so bad flags
fail in milliseconds, ``SimulationApp`` before any ``omni``/``isaacsim``
import, the ``isaacsim.ros2.bridge`` extension before ``rclpy`` (Isaac ships
its own ABI-compatible ROS 2 -- the host's rclpy would crash the C
extension).

Graceful absence of HARVEST: the app builds nothing until a scene command
arrives; it sits publishing "starting" telemetry, and the latched command
topic means it also catches a scene published before it booted.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# The contract + kinematics are imported from the repository by path: Isaac's
# interpreter has no HARVEST packages installed, and both modules are pure
# stdlib by design.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3]))          # repo root
from harvest_integrations.simulators.isaac import contract          # noqa: E402
from harvest_integrations.simulators.isaac.motion import FieldKinematics  # noqa: E402

_TELEMETRY_PERIOD_S = 0.5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HARVEST Isaac Sim layer")
    parser.add_argument("--headless", action="store_true",
                        help="run without the Isaac viewport")
    parser.add_argument("--speed", type=float, default=contract.DEFAULT_SPEED_MPS,
                        help="tractor drive speed, m/s")
    parser.add_argument("--max-runtime", type=float, default=0.0,
                        help="exit after this many seconds (0 = run forever)")
    parser.add_argument("--self-test", action="store_true",
                        help="boot, publish one 'starting' telemetry message "
                             "and exit 0 -- validates the Isaac + ROS 2 "
                             "environment without HARVEST running")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # ---- Isaac boot (only now do the heavyweight imports become legal) ------
    from isaacsim import SimulationApp                       # noqa: PLC0415
    simulation_app = SimulationApp({"headless": args.headless})

    from isaacsim.core.api import World                      # noqa: PLC0415
    from isaacsim.core.api.objects import (FixedCylinder,    # noqa: PLC0415
                                           VisualCuboid)
    from isaacsim.core.utils import extensions               # noqa: PLC0415

    extensions.enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()

    import rclpy                                             # noqa: PLC0415
    from rclpy.qos import (DurabilityPolicy, HistoryPolicy,  # noqa: PLC0415
                           QoSProfile, ReliabilityPolicy)
    from std_msgs.msg import String                          # noqa: PLC0415

    latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)

    rclpy.init()
    node = rclpy.create_node("harvest_isaac_sim")
    kin = FieldKinematics(speed_mps=args.speed)
    pending: list = []
    node.create_subscription(String, contract.SIM_COMMAND_TOPIC,
                             lambda m: pending.append(m.data), latched)
    telemetry_pub = node.create_publisher(
        String, contract.SIM_TELEMETRY_TOPIC, latched)

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    prims: dict = {}

    def build_prims() -> None:
        """(Re)create one prim per scene entity at its current pose."""
        for eid in list(prims):
            world.scene.remove_object(eid)
        prims.clear()
        for ent in kin.entities.values():
            if ent.kind == "charger":
                prims[ent.id] = world.scene.add(FixedCylinder(
                    prim_path=f"/World/{ent.id}", name=ent.id,
                    position=[ent.pose[0], ent.pose[1], 0.6],
                    radius=0.5, height=1.2, color=[0.15, 0.45, 0.9]))
            else:
                prims[ent.id] = world.scene.add(VisualCuboid(
                    prim_path=f"/World/{ent.id}", name=ent.id,
                    position=[ent.pose[0], ent.pose[1], 0.8],
                    scale=[3.2, 1.8, 1.6], color=[0.2, 0.65, 0.2]))

    def publish_telemetry(state: str) -> None:
        # Measured poses: read the prim transforms back rather than echoing
        # the kinematics targets (the WISEPACK acknowledgement rule).
        entities = kin.telemetry_entities()
        for eid, prim in prims.items():
            try:
                position, _ = prim.get_world_pose()
                entities[eid]["pose"] = [round(float(position[0]), 2),
                                         round(float(position[1]), 2)]
            except Exception:
                pass
        telemetry_pub.publish(String(data=json.dumps(contract.telemetry_message(
            simulator={"kind": "isaac", "state": state,
                       "version": "harvest-isaac/1.0",
                       "started_ts": kin._started,
                       "detail": "NVIDIA Isaac Sim physical layer"},
            entities=entities, fingerprint=kin.scene_fingerprint,
            stats=kin.stats()))))

    world.reset()
    publish_telemetry("starting")
    print("[harvest-isaac] READY -- waiting for "
          f"{contract.SIM_COMMAND_TOPIC}", flush=True)

    if args.self_test:
        for _ in range(60):                       # let the publication match
            simulation_app.update()
            rclpy.spin_once(node, timeout_sec=0.0)
        print("[harvest-isaac] self-test ok", flush=True)
        rclpy.shutdown()
        simulation_app.close()
        return 0

    started = time.time()
    last_step = time.monotonic()
    last_telemetry = 0.0
    try:
        while simulation_app.is_running():
            world.step(render=not args.headless)
            # Never block the render loop on ROS (WISEPACK loop skeleton).
            rclpy.spin_once(node, timeout_sec=0.0)

            while pending:
                cmd = contract.parse_message(pending.pop(0))
                if cmd is None:
                    print("[harvest-isaac] refused message with bad "
                          "JSON/schema", flush=True)
                elif kin.apply_command(cmd):
                    build_prims()
                    print(f"[harvest-isaac] scene applied: "
                          f"{len(kin.entities)} entities, fingerprint "
                          f"{kin.scene_fingerprint}", flush=True)
                    publish_telemetry("running")

            now = time.monotonic()
            kin.step(now - last_step)
            last_step = now
            for ent in kin.entities.values():
                if ent.kind == "tractor" and ent.id in prims:
                    position, orientation = prims[ent.id].get_world_pose()
                    prims[ent.id].set_world_pose(
                        [ent.pose[0], ent.pose[1], float(position[2])],
                        orientation)

            if time.time() - last_telemetry >= _TELEMETRY_PERIOD_S:
                publish_telemetry("running" if kin.scene_fingerprint
                                  else "starting")
                last_telemetry = time.time()

            if args.max_runtime and time.time() - started > args.max_runtime:
                print("[harvest-isaac] max runtime reached, exiting", flush=True)
                break
    except Exception as exc:
        # Tell the bridge the backend died rather than letting it time out
        # (publish, then spin so the RELIABLE publication reaches the wire).
        publish_telemetry("error")
        print(f"[harvest-isaac] fatal: {exc}", flush=True)
        for _ in range(30):
            simulation_app.update()
        raise
    finally:
        rclpy.shutdown()
        simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
