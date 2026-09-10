"""
harvest_ros.isaac_bridge -- the HARVEST <-> simulator bridge node.

    ros2 run harvest_ros isaac_bridge --ros-args -p harvest_url:=http://host:8765

Sits between the fleet side and the (optional) physical simulator:

* subscribes to ``/harvest/fleet/snapshot`` (published by the fleet bridge --
  the single HARVEST->ROS gateway) and to HARVEST's ``/api/config`` (fetched
  once over HTTP) to derive the simulator scene and per-tractor motion goals;
* publishes latched ``harvest-sim/1.0`` commands on ``/harvest/sim/command``;
* subscribes to ``/harvest/sim/telemetry`` from Isaac Sim or the GPU-free
  stub;
* reports its own and the simulator's state to HARVEST via
  ``POST /api/integrations/status`` (tagged ``X-Harvest-Client:
  isaac-bridge``), which feeds the Diagnostics view and the FIWARE
  ``FarmSimulation`` mirror.

No simulator-specific business logic lives here or upstream: the node only
translates between the semantic fleet state and the neutral sim contract.
Scene/goal derivation comes from the shared contract module
(``harvest_integrations/simulators/isaac/contract.py``, pure stdlib) imported
from the bind-mounted repository, so bridge and simulator cannot drift.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from . import topics
from .qos import state_qos

# The shared contract is pure stdlib; find the repo root (bind-mounted at
# /harvest in Docker; resolved relative to this file for local runs, through
# colcon's --symlink-install).
_REPO = Path(os.environ.get("HARVEST_REPO_ROOT")
             or Path(__file__).resolve().parents[4])
sys.path.insert(0, str(_REPO))
from harvest_integrations.simulators.isaac import contract  # noqa: E402

_SIM_STALE_S = 12.0        # telemetry older than this = simulator not connected


class HarvestIsaacBridge(Node):
    def __init__(self) -> None:
        super().__init__("harvest_isaac_bridge")
        self.declare_parameter("harvest_url", "http://127.0.0.1:8765")
        self.declare_parameter("status_period_s", 3.0)
        self._url = str(self.get_parameter("harvest_url").value).rstrip("/")

        self._scene = None                 # built from /api/config
        self._fingerprint = ""
        self._acked_fingerprint = ""
        self._last_telemetry = None        # parsed telemetry dict
        self._last_telemetry_ts = 0.0
        self._ever_connected = False
        self._started = time.time()

        self._pub_command = self.create_publisher(
            String, topics.SIM_COMMAND, state_qos())
        self.create_subscription(
            String, topics.FLEET_SNAPSHOT, self._on_snapshot, state_qos())
        self.create_subscription(
            String, topics.SIM_TELEMETRY, self._on_telemetry, state_qos())
        self.create_timer(
            float(self.get_parameter("status_period_s").value),
            self._report_status)
        self.create_timer(5.0, self._ensure_scene)
        self._ensure_scene()
        self.get_logger().info(
            f"bridging {self._url} <-> {topics.SIM_COMMAND}")

    # -- scene (from HARVEST config, the single source of farm structure) -----
    def _ensure_scene(self) -> None:
        if self._scene is not None:
            return
        cfg = self._http_json("GET", "/api/config")
        if not isinstance(cfg, dict):
            self.get_logger().warning(
                f"HARVEST config not readable at {self._url}/api/config yet")
            return
        self._scene = contract.scene_from_config(cfg)
        self._fingerprint = contract.scene_fingerprint(self._scene)
        self.get_logger().info(
            f"scene derived from config: "
            f"{len(self._scene['entities'])} entities, "
            f"fingerprint {self._fingerprint}")

    # -- HARVEST -> simulator -------------------------------------------------
    def _on_snapshot(self, msg: String) -> None:
        if self._scene is None:
            return
        try:
            snapshot = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        derived = contract.goals_from_snapshot(self._scene, snapshot)
        # Keep sending the scene until the simulator acknowledges its
        # fingerprint from applied state (never assume the latched message
        # arrived -- the simulator may boot minutes later, or restart).
        include_scene = self._acked_fingerprint != self._fingerprint
        message = contract.command_message(self._scene, derived, include_scene)
        self._pub_command.publish(String(data=json.dumps(message)))

    # -- simulator -> HARVEST -------------------------------------------------
    def _on_telemetry(self, msg: String) -> None:
        telemetry = contract.parse_message(msg.data)
        if telemetry is None:
            self.get_logger().warning("refused telemetry with bad JSON/schema")
            return
        first = self._last_telemetry is None
        self._last_telemetry = telemetry
        self._last_telemetry_ts = time.time()
        self._ever_connected = True
        self._acked_fingerprint = str(telemetry.get("scene_fingerprint") or "")
        if first:
            sim = telemetry.get("simulator") or {}
            self.get_logger().info(
                f"simulator connected: {sim.get('kind')} ({sim.get('state')})")

    # -- status reporting (feeds Diagnostics + FIWARE mirror) -----------------
    def _simulator_view(self) -> dict:
        age = time.time() - self._last_telemetry_ts
        connected = self._last_telemetry is not None and age < _SIM_STALE_S
        view: dict = {
            "connected": connected,
            "ever_connected": self._ever_connected,
            "scene_fingerprint": self._fingerprint,
            "scene_acknowledged": self._acked_fingerprint == self._fingerprint
                                  and bool(self._fingerprint),
        }
        if self._last_telemetry is not None:
            sim = self._last_telemetry.get("simulator") or {}
            stats = self._last_telemetry.get("stats") or {}
            view.update({
                "kind": sim.get("kind"),
                "state": sim.get("state"),
                "version": sim.get("version"),
                "detail": sim.get("detail"),
                "telemetry_age_s": round(age, 1),
                "entities_synced": stats.get("entities_synced", 0),
                "tractors_synced": stats.get("tractors_synced"),
                "tractors_driveable": stats.get("tractors_driveable"),
                "sim_time_s": stats.get("sim_time_s"),
                "entities": self._last_telemetry.get("entities") or {},
                # Passed through verbatim, never interpreted here: how the
                # operator can watch the simulation, which robot model is
                # standing in for a tractor, and anything the simulator is
                # unhappy about.  Diagnostics renders them; the bridge has no
                # opinion about any of it.
                "visualization": sim.get("visualization") or {},
                "robot_model": sim.get("robot_model") or {},
                "physics": sim.get("physics"),
                "problems": sim.get("problems") or [],
            })
        return view

    def _ros_view(self) -> dict:
        """The DDS facts an operator needs when the two sides cannot see each
        other -- which is the failure this integration actually has.

        Reported from THIS process's environment, because that is the half of
        the pair HARVEST controls; the simulator reports its own in telemetry.
        A domain or transport mismatch between the two is invisible from either
        side alone and is the first thing to check.
        """
        return {
            "domain_id": os.environ.get("ROS_DOMAIN_ID", "0"),
            "rmw": os.environ.get("RMW_IMPLEMENTATION", "<default>"),
            "transport": os.environ.get("FASTDDS_BUILTIN_TRANSPORTS",
                                        "<Fast DDS default>"),
        }

    def _report_status(self) -> None:
        payload = {
            "client": contract.BRIDGE_CLIENT,
            "status": {
                "bridge_uptime_s": round(time.time() - self._started, 1),
                "scene_ready": self._scene is not None,
                "ros": self._ros_view(),
                "simulator": self._simulator_view(),
            },
        }
        self._http_json("POST", "/api/integrations/status", payload)

    # -- HTTP helper ----------------------------------------------------------
    def _http_json(self, method: str, path: str, payload: dict | None = None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            f"{self._url}{path}", data=data, method=method,
            headers={"Content-Type": "application/json",
                     "X-Harvest-Client": contract.BRIDGE_CLIENT})
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return None


def main() -> None:
    rclpy.init()
    node = HarvestIsaacBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
