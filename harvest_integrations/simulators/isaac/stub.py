"""
GPU-free stand-in for Isaac Sim (``isaac-demo`` mode).

    python3 -m harvest_integrations.simulators.isaac.stub

Joins the same DDS domain the Isaac bridge uses, subscribes to the latched
``/harvest/sim/command`` topic and publishes ``/harvest/sim/telemetry`` --
exactly what the real Isaac app does, driven by the *same*
:mod:`~harvest_integrations.simulators.isaac.motion` kinematics and
:mod:`~harvest_integrations.simulators.isaac.contract` wire code.  The full
HARVEST -> ROS 2 -> simulator -> HARVEST loop therefore runs end-to-end on
any machine, and swapping in real Isaac changes only where the entities are
rendered.

Reports itself honestly as ``"kind": "stub"`` so Diagnostics shows SIMULATED,
never a fake HEALTHY Isaac (the WISEPACK honesty rule).

Needs only rclpy + std_msgs (runs in the harvest:ros2 container with the
repository on PYTHONPATH).
"""
from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from std_msgs.msg import String

import json

from . import contract
from .motion import FieldKinematics

_STEP_PERIOD_S = 0.1
_TELEMETRY_PERIOD_S = 0.5


def _latched_qos() -> QoSProfile:
    # Must match the bridge side (harvest_ros.qos.state_qos): RELIABLE +
    # TRANSIENT_LOCAL + KEEP_LAST(1) -- mismatched durability silently never
    # delivers the latched scene command.
    return QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST, depth=1)


class IsaacSimStub(Node):
    def __init__(self) -> None:
        super().__init__("harvest_isaac_stub")
        self.declare_parameter("speed_mps", contract.DEFAULT_SPEED_MPS)
        self.kin = FieldKinematics(
            speed_mps=float(self.get_parameter("speed_mps").value))
        self._last_step = time.monotonic()
        self._pub = self.create_publisher(
            String, contract.SIM_TELEMETRY_TOPIC, _latched_qos())
        self.create_subscription(
            String, contract.SIM_COMMAND_TOPIC, self._on_command, _latched_qos())
        self.create_timer(_STEP_PERIOD_S, self._step)
        self.create_timer(_TELEMETRY_PERIOD_S, self._publish_telemetry)
        self._publish_telemetry(state="starting")
        self.get_logger().info(
            f"stub simulator up -- waiting for {contract.SIM_COMMAND_TOPIC}")

    def _on_command(self, msg: String) -> None:
        cmd = contract.parse_message(msg.data)
        if cmd is None:
            self.get_logger().warning("refused command with bad JSON/schema")
            return
        if self.kin.apply_command(cmd):
            self.get_logger().info(
                f"scene applied: {len(self.kin.entities)} entities, "
                f"fingerprint {self.kin.scene_fingerprint}")
            self._publish_telemetry()   # acknowledge the scene promptly

    def _step(self) -> None:
        now = time.monotonic()
        self.kin.step(now - self._last_step)
        self._last_step = now

    def _publish_telemetry(self, state: str = "running") -> None:
        message = contract.telemetry_message(
            simulator={
                "kind": "stub",
                "state": state if self.kin.scene_fingerprint else "starting",
                "version": "harvest-stub/1.0",
                "started_ts": self.kin._started,
                "detail": "GPU-free stand-in (isaac-demo mode) -- same sync "
                          "core Isaac Sim runs",
            },
            entities=self.kin.telemetry_entities(),
            fingerprint=self.kin.scene_fingerprint,
            stats=self.kin.stats(),
        )
        self._pub.publish(String(data=json.dumps(message)))


def main() -> None:
    rclpy.init()
    node = IsaacSimStub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
