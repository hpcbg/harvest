"""
QoS profiles for the HARVEST bridge, built in code and passed explicitly to
every publisher/subscription (TEMPO ``tempo_bringup.qos`` pattern -- no XML
profile file that nothing loads).

* ``state_qos``     -- RELIABLE + TRANSIENT_LOCAL, depth 1: late joiners
  (dashboards, loggers) immediately receive the current snapshot/ack/tariff.
* ``telemetry_qos`` -- BEST_EFFORT + VOLATILE, depth 5: scalar KPI streams
  where the latest value is the only interesting one.
* ``command_qos``   -- RELIABLE, depth 10: commands must not be dropped.
  Unlike TEMPO's sub-second control loop, HARVEST decisions run at
  minutes-scale, so no Deadline/Liveliness watchdog is imposed here; loss of
  the bridge is detected by the snapshot topic going stale.
"""
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)


def state_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def telemetry_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
    )


def command_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )
