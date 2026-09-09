"""
harvest_ros.bridge -- the HARVEST <-> ROS 2 fleet bridge node.

    ros2 run harvest_ros fleet_bridge --ros-args -p harvest_url:=http://host:8765

Architecture (WISEPACK observer pattern + TEMPO adapter pattern):

* the node polls ``GET /api/fleet/snapshot`` at ``poll_period_s`` and mirrors
  it onto the canonical topics (JSON snapshot + scalar KPIs);
* messages on ``/harvest/fleet/command`` are forwarded verbatim to
  ``POST /api/fleet/command`` and each response is published on
  ``/harvest/fleet/ack``.

Talking HTTP instead of importing HARVEST keeps the coupling one-way and
containerisable: this node needs only ``rclpy`` + ``std_msgs`` + the standard
library, and HARVEST itself never needs ROS installed.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String

from . import topics
from .qos import command_qos, state_qos, telemetry_qos


class HarvestFleetBridge(Node):
    def __init__(self) -> None:
        super().__init__("harvest_fleet_bridge")
        self.declare_parameter("harvest_url", "http://127.0.0.1:8765")
        self.declare_parameter("poll_period_s", 1.0)
        self._url = str(self.get_parameter("harvest_url").value).rstrip("/")
        period = float(self.get_parameter("poll_period_s").value)

        self._pub_snapshot = self.create_publisher(
            String, topics.FLEET_SNAPSHOT, state_qos())
        self._pub_ack = self.create_publisher(String, topics.FLEET_ACK, state_qos())
        self._pub_tariff = self.create_publisher(String, topics.GRID_TARIFF, state_qos())
        self._pub_draw = self.create_publisher(
            Float32, topics.GRID_DRAW_KW, telemetry_qos())
        self._pub_pv = self.create_publisher(Float32, topics.GRID_PV_KW, telemetry_qos())
        self._pub_price = self.create_publisher(
            Float32, topics.GRID_PRICE, telemetry_qos())
        self._pub_soc: dict[str, object] = {}      # created on first sight

        self.create_subscription(
            String, topics.FLEET_COMMAND, self._on_command, command_qos())
        self.create_timer(period, self._poll)

        self._offline_logged = False
        self.get_logger().info(f"bridging {self._url} every {period}s")

    # -- HARVEST -> ROS -------------------------------------------------------
    def _poll(self) -> None:
        snap = self._http_json("GET", "/api/fleet/snapshot")
        if snap is None:
            if not self._offline_logged:
                self.get_logger().warning(f"HARVEST API at {self._url} unreachable")
                self._offline_logged = True
            return
        if self._offline_logged:
            self.get_logger().info("HARVEST API reachable again")
            self._offline_logged = False

        self._pub_snapshot.publish(String(data=json.dumps(snap)))
        grid = snap.get("grid", {})
        self._pub_draw.publish(Float32(data=float(grid.get("grid_draw_kw", 0.0))))
        self._pub_pv.publish(Float32(data=float(grid.get("pv_kw", 0.0))))
        self._pub_price.publish(Float32(data=float(grid.get("price_eur_per_kwh", 0.0))))
        self._pub_tariff.publish(String(data=str(grid.get("tariff", ""))))
        for tractor in snap.get("tractors", []):
            tid = tractor.get("id")
            if tid not in self._pub_soc:
                self._pub_soc[tid] = self.create_publisher(
                    Float32, topics.tractor_soc(tid), telemetry_qos())
            self._pub_soc[tid].publish(
                Float32(data=float(tractor.get("soc_pct", 0.0))))

    # -- ROS -> HARVEST -------------------------------------------------------
    def _on_command(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self._publish_ack({"error": f"invalid command JSON: {exc}"})
            return
        result = self._http_json("POST", "/api/fleet/command", payload)
        self._publish_ack(result if result is not None
                          else {"error": "HARVEST API unreachable"})

    def _publish_ack(self, payload: dict) -> None:
        self._pub_ack.publish(String(data=json.dumps(payload)))

    # -- HTTP helper ----------------------------------------------------------
    def _http_json(self, method: str, path: str, payload: dict | None = None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            f"{self._url}{path}", data=data, method=method,
            headers={"Content-Type": "application/json",
                     # Liveness signal for the server's diagnostics view.
                     "X-Harvest-Client": "ros2-bridge"})
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            try:
                return json.loads(exc.read().decode())
            except Exception:
                return {"error": f"HTTP {exc.code}"}
        except Exception:
            return None


def main() -> None:
    rclpy.init()
    node = HarvestFleetBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
