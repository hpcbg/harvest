#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Launch the HARVEST Isaac Sim layer on the HOST (never inside Docker).
#
#   ./scripts/run_isaac_sim.sh [--headless] [--self-test] [--max-runtime S]
#
# Start the HARVEST side first:  ./run_harvest_dashboard.sh isaac
# The two meet only on the DDS wire (shared ROS_DOMAIN_ID, default 42).
#
# Environment:
#   ISAAC_SIM_ROOT   Isaac Sim installation dir (auto-detected otherwise)
#   ROS_DOMAIN_ID    must match the Docker stack (default 42)
#
# WISEPACK pattern, and the scrub below is load-bearing: sourcing a host
# /opt/ros environment puts an ABI-incompatible rclpy ahead of Isaac's own
# internally-built ROS 2 and crashes in rclpy's C extension.  Only
# ROS_DOMAIN_ID survives; Isaac's bundled ROS libraries do the talking.
# ----------------------------------------------------------------------------
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Locate Isaac Sim ─────────────────────────────────────────────────────────
find_isaac() {
  [ -n "${ISAAC_SIM_ROOT:-}" ] && { echo "$ISAAC_SIM_ROOT"; return; }
  # Common install locations, newest first.
  ls -d "$HOME"/isaacsim* "$HOME"/isaac-sim* \
        "$HOME"/.local/share/ov/pkg/isaac-sim-* \
        /opt/isaacsim* /isaac-sim 2>/dev/null | sort -rV | head -1
}

ISAAC_ROOT="$(find_isaac || true)"
if [ -z "$ISAAC_ROOT" ] || [ ! -x "$ISAAC_ROOT/python.sh" ]; then
  echo "Isaac Sim not found (looked for python.sh under \$ISAAC_SIM_ROOT and" >&2
  echo "common install paths).  Set ISAAC_SIM_ROOT=/path/to/isaac-sim, or use" >&2
  echo "the GPU-free stand-in instead:  ./run_harvest_dashboard.sh isaac-demo" >&2
  exit 3
fi
echo "Using Isaac Sim at: $ISAAC_ROOT"

# ── Scrub the ROS environment (keep only the domain id) ──────────────────────
DOMAIN="${ROS_DOMAIN_ID:-42}"
unset PYTHONPATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH \
      LD_LIBRARY_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION \
      ROS_LOCALHOST_ONLY RMW_IMPLEMENTATION 2>/dev/null || true
export ROS_DOMAIN_ID="$DOMAIN"

# Isaac >= 4.5 ships a helper that wires its internal ROS 2 libraries.
if [ -f "$ISAAC_ROOT/setup_ros_env.sh" ]; then
  # shellcheck disable=SC1091
  source "$ISAAC_ROOT/setup_ros_env.sh"
fi
# Same RMW on both ends or the nodes simply never discover each other.
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export PYTHONUNBUFFERED=1

echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID  RMW=rmw_fastrtps_cpp"
exec "$ISAAC_ROOT/python.sh" \
  "$REPO/harvest_integrations/simulators/isaac/harvest_isaac.py" "$@"
