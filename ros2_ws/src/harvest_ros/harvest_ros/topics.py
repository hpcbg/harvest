"""
Canonical HARVEST ROS 2 topic contract (TEMPO ``tempo_bringup.topics`` pattern:
one module owns every name, everyone imports it, nothing is stringly typed
twice).

Rich objects travel as versioned JSON in ``std_msgs/String`` -- the same
``harvest-fleet/1.0`` schema served by ``GET /api/fleet/snapshot`` -- so the
ROS view, the HTTP API and the FIWARE entities can never drift apart.  Scalar
KPI topics are provided alongside for plotting/rqt convenience.

Single-writer discipline: the bridge is the only publisher of every topic
below except FLEET_COMMAND, which belongs to decision/operator nodes.
"""

# Fleet-wide JSON snapshot (schema harvest-fleet/1.0), latched.
FLEET_SNAPSHOT = "/harvest/fleet/snapshot"

# Inbound: JSON command batches {"commands": [{type, target_id, value}]}.
FLEET_COMMAND = "/harvest/fleet/command"

# Outbound: JSON acks for each command batch, latched.
FLEET_ACK = "/harvest/fleet/ack"

# Scalar KPI telemetry (std_msgs/Float32).
GRID_DRAW_KW = "/harvest/grid/draw_kw"
GRID_PV_KW = "/harvest/grid/pv_kw"
GRID_PRICE = "/harvest/grid/price_eur_per_kwh"

# Latched String: current tariff period name (valle | llano | punta).
GRID_TARIFF = "/harvest/grid/tariff"

# Simulation channel (optional Isaac Sim / stub layer).  JSON, schema
# harvest-sim/1.0.  These strings are also defined in
# harvest_integrations/simulators/isaac/contract.py -- the simulator side
# cannot import this colcon package -- and tests/test_isaac_contract.py
# asserts the two stay identical.  SIM_COMMAND is written by the Isaac
# bridge; SIM_TELEMETRY by the simulator (Isaac or the isaac-demo stub).
SIM_COMMAND = "/harvest/sim/command"
SIM_TELEMETRY = "/harvest/sim/telemetry"


def tractor_soc(tractor_id: str) -> str:
    """Per-tractor SoC telemetry topic (std_msgs/Float32)."""
    return f"/harvest/tractors/{tractor_id}/soc_pct"
