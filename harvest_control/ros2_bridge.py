"""
harvest_control.ros2_bridge
===========================

Skeleton ROS 2 bridge that exposes a :class:`~harvest_control.interface.FleetInterface`
over ROS 2 topics and services.  It shows that adding ROS integration is a thin
adapter on top of the same interface -- the decision layer is untouched.

Importing this module does **not** require ``rclpy``: the ROS 2 parts are
guarded so the rest of ``harvest_control`` (and the simulation demo) work with
no ROS installation.  On a machine with ROS 2 the node becomes live.

Proposed topic / service map
----------------------------
    publish    /harvest/fleet/snapshot        harvest_msgs/FleetSnapshot   (state out)
    service    /harvest/command               harvest_msgs/SubmitCommand   (actuation in)
    publish    /harvest/tractor/{id}/state    harvest_msgs/TractorState
    subscribe  /harvest/tractor/{id}/cmd      harvest_msgs/Command

Stage 3
-------
For the September deployment the *same* node is pointed at a
:class:`Ros2FleetInterface` whose backend subscribes to the real ZETRABOT
topics instead of the simulation.  Confirming the exact ZETRABOT signal set
(see the mentor correspondence) is the only remaining input needed to finalise
the message definitions below.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

try:
    from .interface import (
        ChargerLevel, Command, CommandAck, CommandType, FleetInterface, FleetSnapshot,
    )
except ImportError:  # running as standalone script from package directory
    from interface import (
        ChargerLevel, Command, CommandAck, CommandType, FleetInterface, FleetSnapshot,
    )

try:
    import rclpy
    from rclpy.node import Node
    _HAVE_ROS = True
except Exception:                       # rclpy not installed -> module still importable
    _HAVE_ROS = False

    class Node:                          # minimal stand-in so the class body parses
        def __init__(self, *_a, **_k): ...
        def create_timer(self, *_a, **_k): ...
        def get_logger(self):
            import logging
            return logging.getLogger("harvest_ros2")


# --------------------------------------------------------------------------- #
#  (de)serialisation helpers  --  JSON-friendly dicts standing in for msgs
# --------------------------------------------------------------------------- #
def snapshot_to_msg(snap: FleetSnapshot) -> dict:
    return {
        "grid": asdict(snap.grid),
        "tractors": [asdict(t) for t in snap.tractors],
        "chargers": [{**asdict(c), "level": c.level.value} for c in snap.chargers],
        "loads": [asdict(l) for l in snap.loads],
    }


def msg_to_command(msg: dict) -> Command:
    ctype = CommandType(msg["type"])
    value = msg.get("value")
    if ctype == CommandType.SET_CHARGER_LEVEL and value is not None:
        value = ChargerLevel(value)
    return Command(ctype, msg["target_id"], value)


def ack_to_msg(ack: CommandAck) -> dict:
    return {"type": ack.command.type.value, "target_id": ack.command.target_id,
            "accepted": ack.accepted, "reason": ack.reason}


# --------------------------------------------------------------------------- #
#  Bridge node
# --------------------------------------------------------------------------- #
class HarvestRos2Bridge(Node):
    """Publishes fleet snapshots and turns incoming ROS 2 commands into
    :class:`FleetInterface` calls.  Backend-agnostic: pass any FleetInterface."""

    def __init__(self, iface: FleetInterface, publish_period_s: float = 1.0):
        super().__init__("harvest_fleet_bridge")
        self.iface = iface
        if _HAVE_ROS:
            # Real wiring (pseudo-API; concrete msg types defined in harvest_msgs):
            #   self._snap_pub = self.create_publisher(FleetSnapshotMsg, "/harvest/fleet/snapshot", 10)
            #   self._cmd_srv  = self.create_service(SubmitCommand, "/harvest/command", self._on_command)
            #   self.create_timer(publish_period_s, self._publish_snapshot)
            self.get_logger().info("HARVEST ROS 2 bridge up (backend=%s)" % type(iface).__name__)
        else:
            self.get_logger().info("rclpy not present -- bridge in offline/parse-only mode")

    def _publish_snapshot(self) -> None:
        msg = snapshot_to_msg(self.iface.snapshot())
        # self._snap_pub.publish(to_ros_msg(msg))
        return msg

    def _on_command(self, request, response=None):
        cmds: Sequence[Command] = [msg_to_command(m) for m in getattr(request, "commands", [])]
        acks = self.iface.submit(cmds)
        out = [ack_to_msg(a) for a in acks]
        if response is not None:
            response.acks = out
            return response
        return out


def main():  # entry point: `ros2 run harvest_control bridge`
    if not _HAVE_ROS:
        print("rclpy not installed; install ROS 2 to run the live bridge.")
        return
    try:
        from .sim_backend import SimulationFleetInterface
    except ImportError:
        from sim_backend import SimulationFleetInterface
    rclpy.init()
    node = HarvestRos2Bridge(SimulationFleetInterface())
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
