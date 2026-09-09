#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# HARVEST — one-command startup (WISEPACK-style launcher)
#
#   ./run_harvest_dashboard.sh [mode]
#
# Modes (cumulative — each adds services to the previous one):
#   lite       Run server.py directly on the host (no Docker; uses .venv if
#              present).  Exactly the pre-integration HARVEST experience.
#   core       Dashboard/API in Docker, reference-sim fleet backend.
#   devices    + farm-device simulator; fleet backend switches to real
#              Modbus TCP / OPC-UA clients talking to it.
#   fiware     + Orion-LD context broker, MongoDB and the NGSI-LD sync daemon.
#              [DEFAULT — the full integration stack without ROS]
#   full       + ROS 2 fleet bridge (ros:jazzy image, DDS stays in-container).
#   isaac      fiware + the ROS 2 fleet & Isaac bridges on the HOST network so
#              NVIDIA Isaac Sim (running on the host, outside Docker) can join
#              over DDS.  Start the simulator separately with
#              ./scripts/run_isaac_sim.sh — see
#              harvest_integrations/simulators/isaac/README.md.
#   isaac-demo isaac + a GPU-free stand-in for the simulator (the same sync
#              core Isaac uses), so the full HARVEST->ROS 2->simulator loop
#              runs end-to-end without Isaac installed.
#
# Management:
#   ./run_harvest_dashboard.sh stop      Stop and remove the whole stack
#   ./run_harvest_dashboard.sh clean     stop + delete the broker database
#                                        (mirrored NGSI-LD entities persist
#                                        across restarts otherwise)
#   ./run_harvest_dashboard.sh status    Show service state and health
#   ./run_harvest_dashboard.sh logs [service]
#
# Environment overrides:
#   HARVEST_PORT (8765) | ORION_PORT (1026) | ROS_DOMAIN_ID (42)
#   HARVEST_READY_TIMEOUT (120) seconds per health gate before failing
#
# Exit codes: 0 ok | 2 usage | 6 port in use by a foreign process
#             7 services failed a health gate
# ----------------------------------------------------------------------------
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

MODE="${1:-fiware}"
HARVEST_PORT="${HARVEST_PORT:-8765}"
READY_TIMEOUT="${HARVEST_READY_TIMEOUT:-120}"

usage() { awk '/^# ----/{n++} n==1' "$0" | sed 's/^# \{0,1\}//'; }

compose() { docker compose "$@"; }

# All profiles, for stop/status/logs regardless of how the stack was started.
ALL="--profile devices --profile fiware --profile ros2 --profile isaac --profile isaac-demo"

# ── The startup summary box ──────────────────────────────────────────────────
# The one piece of output that must survive every successful path: the user's
# next step is always a URL or a command from this box.
hline() { local l="$1" m="$2" r="$3" i; printf '%s' "$l"
          for ((i=0;i<65;i++)); do printf '%s' "$m"; done; printf '%s\n' "$r"; }
row()   { printf '║  %-11s : %-48s ║\n' "$1" "$2"; }
title() { printf '║  %-63s║\n' "$1"; }

print_banner() {   # $1 = headline
  echo
  hline "╔" "═" "╗"
  title "$1"
  hline "╟" "─" "╢"
  row "Dashboard"   "http://localhost:$HARVEST_PORT"
  row "Diagnostics" "http://localhost:$HARVEST_PORT  (Diagnostics tab)"
  row "Fleet API"   "http://localhost:$HARVEST_PORT/api/fleet/snapshot"
  case "$MODE" in fiware|full|isaac|isaac-demo)
    row "NGSI-LD"   "http://localhost:${ORION_PORT:-1026}/ngsi-ld/v1/entities" ;;
  esac
  case "$MODE" in
    full)       row "ROS 2" "/harvest/* topics (inside ros2-bridge)" ;;
    isaac)      row "ROS 2" "/harvest/* topics (host network, domain ${ROS_DOMAIN_ID:-42})"
                row "Isaac Sim" "waiting - start with ./scripts/run_isaac_sim.sh" ;;
    isaac-demo) row "ROS 2" "/harvest/* topics (host network, domain ${ROS_DOMAIN_ID:-42})"
                row "Isaac Sim" "GPU-free stub active (isaac-demo mode)" ;;
  esac
  hline "╟" "─" "╢"
  row "Stop"     "./run_harvest_dashboard.sh stop"
  row "Logs"     "./run_harvest_dashboard.sh logs [service]"
  row "Validate" "./validate_harvest_stack.sh"
  hline "╚" "═" "╝"
}

case "$MODE" in
  -h|--help|help) usage; exit 0 ;;

  stop)
    compose $ALL down --remove-orphans
    echo "HARVEST stack stopped."
    exit 0 ;;

  clean)
    compose $ALL down --remove-orphans --volumes
    echo "HARVEST stack stopped and broker database removed."
    exit 0 ;;

  status)
    compose $ALL ps
    exit 0 ;;

  logs)
    shift || true
    compose $ALL logs -f "$@"
    exit 0 ;;

  lite)
    # No Docker: the original lightweight path.
    PY="python3"
    [ -x "$REPO/.venv/bin/python" ] && PY="$REPO/.venv/bin/python"
    echo "Starting HARVEST server on the host ($PY) ..."
    exec "$PY" server.py ;;

  core|devices|fiware|full|isaac|isaac-demo) ;;

  *)
    echo "Unknown mode: $MODE" >&2
    usage
    exit 2 ;;
esac

# ── Profile / backend selection ───────────────────────────────────────────────
PROFILES=""
BACKEND="sim"
case "$MODE" in
  core)    ;;
  devices) PROFILES="--profile devices";                              BACKEND="devices" ;;
  fiware)  PROFILES="--profile devices --profile fiware";             BACKEND="devices" ;;
  full)    PROFILES="--profile devices --profile fiware --profile ros2"; BACKEND="devices" ;;
  isaac)   PROFILES="--profile devices --profile fiware --profile isaac"; BACKEND="devices" ;;
  isaac-demo)
           PROFILES="--profile devices --profile fiware --profile isaac --profile isaac-demo"
           BACKEND="devices" ;;
esac
export HARVEST_FLEET_BACKEND="$BACKEND"

# ── Port guard ────────────────────────────────────────────────────────────────
# WISEPACK pattern, with one refinement: if the port is held by OUR harvest
# container the stack is simply already up — reprint the summary (this is the
# most common way the URLs used to vanish) instead of failing.
if python3 - "$HARVEST_PORT" <<'PY'
import socket, sys
s = socket.socket()
try:
    s.settimeout(0.5); s.connect(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)   # free
sys.exit(0)       # in use
PY
then
  if docker ps --filter "name=harvest-harvest" --format '{{.Names}}' | grep -q harvest; then
    print_banner "HARVEST is already running"
    echo
    echo "Note: mode/profile changes need './run_harvest_dashboard.sh stop' first."
    exit 0
  fi
  echo "Port $HARVEST_PORT is in use by another process." >&2
  echo "Free it or start with HARVEST_PORT=<other> $0 $MODE" >&2
  exit 6
fi

echo "── HARVEST mode: $MODE  (fleet backend: $BACKEND) ──────────────────────"
compose $PROFILES up --build -d

# ── Bounded, checked health gates (each gate gets its own full timeout) ──────
wait_http() {   # $1 = label, $2 = url
  local deadline=$(( $(date +%s) + READY_TIMEOUT ))
  echo -n "Waiting for $1 "
  until python3 -c "import urllib.request as u; u.urlopen('$2', timeout=2)" 2>/dev/null; do
    if [ "$(date +%s)" -gt "$deadline" ]; then
      echo
      echo "$1 did not become healthy within ${READY_TIMEOUT}s:" >&2
      compose $ALL ps >&2
      echo "Inspect with: ./run_harvest_dashboard.sh logs" >&2
      exit 7
    fi
    echo -n "."; sleep 2
  done
  echo " ok"
}

wait_http "HARVEST API on :$HARVEST_PORT" "http://127.0.0.1:$HARVEST_PORT/health"
case "$MODE" in fiware|full|isaac|isaac-demo)
  wait_http "Orion-LD on :${ORION_PORT:-1026}" "http://127.0.0.1:${ORION_PORT:-1026}/version" ;;
esac

print_banner "HARVEST is up"
case "$MODE" in
  core|devices)
    echo
    echo "Optional services not started in this mode: FIWARE (fiware), ROS 2 (full)," ;
    echo "Isaac Sim (isaac / isaac-demo)." ;;
  fiware)
    echo
    echo "Optional services not started in this mode: ROS 2 (full), Isaac Sim (isaac)." ;;
  full)
    echo
    echo "Optional services not started in this mode: Isaac Sim (isaac / isaac-demo)." ;;
  isaac)
    echo
    echo "Isaac Sim itself runs on the host, outside Docker.  Start it with"
    echo "'./scripts/run_isaac_sim.sh' (see harvest_integrations/simulators/isaac/"
    echo "README.md), or use './run_harvest_dashboard.sh isaac-demo' for the"
    echo "GPU-free stand-in.  The Diagnostics tab shows the connection state"
    echo "either way." ;;
esac
