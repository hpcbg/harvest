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
#              over DDS, AND the real Isaac Sim itself, started for you on the
#              host by scripts/run_isaac_sim.sh — see
#              harvest_integrations/simulators/isaac/README.md.
#              Watch it over WebRTC (the WISEPACK-proven recipe):
#                HARVEST_ISAAC_VIEW_MODE=webrtc \\
#                HARVEST_ISAAC_STREAMING=1 \\
#                HARVEST_ISAAC_HEADLESS=1 \\
#                ./run_harvest_dashboard.sh isaac
#              then open the NVIDIA Isaac Sim WebRTC Streaming Client on the
#              URL this script prints.  HARVEST_ISAAC_AUTOSTART=0 keeps the
#              stack-only behaviour and leaves Isaac for you to start.
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
#   ./run_harvest_dashboard.sh isaac-stop     Stop only Isaac Sim
#   ./run_harvest_dashboard.sh isaac-restart  Restart only Isaac Sim — the stack
#                                        keeps running and the bridge
#                                        re-discovers the simulator on its own
#
# Environment overrides:
#   HARVEST_PORT (8765) | ORION_PORT (1026) | ROS_DOMAIN_ID (42)
#   HARVEST_READY_TIMEOUT (120) seconds per health gate before failing
#   isaac mode:  HARVEST_ISAAC_AUTOSTART (1) | HARVEST_ISAAC_VIEW_MODE
#                (desktop|webrtc|none) | HARVEST_ISAAC_STREAMING |
#                HARVEST_ISAAC_HEADLESS | HARVEST_ISAAC_STREAM_HOST |
#                HARVEST_ISAAC_SIGNAL_PORT (49100) |
#                HARVEST_ISAAC_STREAM_PORT (47998) |
#                HARVEST_ISAAC_ROBOT_MODEL | ISAAC_SIM_ROOT
#                — all read by scripts/run_isaac_sim.sh, which owns them
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

# ── Isaac Sim on the host: started, owned and stopped BY PID ──────────────────
# Isaac cannot run in the HARVEST containers (WISEPACK rule: it needs the GPU,
# its own bundled Python and its own ROS 2 build), so `isaac` mode starts it on
# the host as a separate process group and remembers only its pid.  Ownership by
# pid, never by name: a `pkill -f isaac` on a shared machine would take out
# another project's simulator, and this machine has WISEPACK's on it.
#
# The run directory outlives this invocation because `stop` is a different one.
ISAAC_RUN_DIR="${HARVEST_ISAAC_RUN_DIR:-${TMPDIR:-/tmp}/harvest-isaac-$(id -u)}"
ISAAC_PIDFILE="$ISAAC_RUN_DIR/isaac.pid"
ISAAC_LOGFILE="$ISAAC_RUN_DIR/isaac.log"

isaac_pid() {   # echoes a LIVE pid, or nothing
  [ -f "$ISAAC_PIDFILE" ] || return 0
  local pid; pid="$(cat "$ISAAC_PIDFILE" 2>/dev/null || true)"
  case "$pid" in ''|*[!0-9]*) return 0 ;; esac
  kill -0 "$pid" 2>/dev/null && printf '%s' "$pid"
}

isaac_start() {
  if [ -n "$(isaac_pid)" ]; then
    echo "Isaac Sim is already running (pid $(isaac_pid)); leaving it alone."
    echo "  log: $ISAAC_LOGFILE"
    return 0
  fi
  mkdir -p "$ISAAC_RUN_DIR"
  # setsid so Isaac gets its own process group: Kit spawns a tree of helpers
  # (renderer, telemetry transmitter, streaming server) and stopping the group
  # is the only way to stop all of them.  Its output is a file, not this
  # terminal: the stack is started detached and Kit logs thousands of lines.
  setsid "$REPO/scripts/run_isaac_sim.sh" >"$ISAAC_LOGFILE" 2>&1 &
  local pid=$!
  echo "$pid" >"$ISAAC_PIDFILE"
  # Give it long enough to fail LOUDLY (a bad view mode, no Isaac install, an
  # occupied signal port) rather than reporting a pid that is already gone.
  local waited=0
  while [ "$waited" -lt 6 ]; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1; waited=$((waited + 1))
  done
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Isaac Sim exited immediately — it did not start:" >&2
    sed 's/^/    /' "$ISAAC_LOGFILE" | tail -20 >&2
    rm -f "$ISAAC_PIDFILE"
    return 1
  fi
  echo "Isaac Sim starting on the host (pid $pid, process group $pid)."
  echo "  log:  $ISAAC_LOGFILE"
  echo "  NOTE: the first launch compiles shaders and can take a few minutes."
  echo "        HARVEST does not wait for it; the Diagnostics tab shows its state."
  return 0
}

isaac_stop() {
  local pid; pid="$(isaac_pid)"
  [ -n "$pid" ] || { rm -f "$ISAAC_PIDFILE" 2>/dev/null || true; return 0; }
  echo "Stopping Isaac Sim (process group $pid) ..."
  kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  local waited=0
  while [ "$waited" -lt 20 ]; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1; waited=$((waited + 1))
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "Isaac Sim did not exit on TERM — sending KILL." >&2
    kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$ISAAC_PIDFILE"
  # Isaac's own Fast DDS segments go with it; a killed participant leaves its
  # shared memory behind and the next run then hangs scanning it.
  [ -x "$REPO/scripts/clean_dds_shm.sh" ] && "$REPO/scripts/clean_dds_shm.sh" || true
  return 0
}

isaac_stream_url() {
  local host="${HARVEST_ISAAC_STREAM_HOST:-127.0.0.1}"
  local port="${HARVEST_ISAAC_SIGNAL_PORT:-49100}"
  printf '%s' "${HARVEST_ISAAC_STREAM_URL:-http://${host}:${port}}"
}

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
                if [ -n "$(isaac_pid)" ]; then
                  row "Isaac Sim" "running on the host, pid $(isaac_pid)"
                else
                  row "Isaac Sim" "not running - ./scripts/run_isaac_sim.sh"
                fi
                if [ "${HARVEST_ISAAC_STREAMING:-0}" = "1" ] \
                   || [ "${HARVEST_ISAAC_VIEW_MODE:-}" = "webrtc" ]; then
                  row "Live view" "$(isaac_stream_url)  (WebRTC)"
                fi ;;
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
    isaac_stop
    compose $ALL down --remove-orphans
    echo "HARVEST stack stopped."
    exit 0 ;;

  clean)
    isaac_stop
    compose $ALL down --remove-orphans --volumes
    echo "HARVEST stack stopped and broker database removed."
    exit 0 ;;

  isaac-stop)
    isaac_stop
    echo "Isaac Sim stopped; the HARVEST stack is untouched."
    echo "Diagnostics will report the Isaac Sim component as FAILED"
    echo "(connection lost) — which is the honest state, not INACTIVE."
    exit 0 ;;

  isaac-restart)
    # The point of having this at all: the bridge re-discovers a restarted
    # simulator over DDS and re-sends the latched scene until the new process
    # acknowledges its fingerprint, so HARVEST itself never has to restart.
    isaac_stop
    isaac_start || exit 1
    exit 0 ;;

  status)
    compose $ALL ps
    if [ -n "$(isaac_pid)" ]; then
      echo
      echo "Isaac Sim: running on the host, pid $(isaac_pid) (log $ISAAC_LOGFILE)"
    else
      echo
      echo "Isaac Sim: not running on this host (start it with"
      echo "           ./run_harvest_dashboard.sh isaac, or ./scripts/run_isaac_sim.sh)"
    fi
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

  # (isaac-stop / isaac-restart are handled above, before profile selection.)

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

# ── Isaac Sim, started AFTER the stack is healthy ─────────────────────────────
# In that order on purpose: the Isaac bridge derives its scene from
# GET /api/config, so a simulator that boots first would spend its first minute
# retrying HTTP.  The stack is quick and Isaac is slow, so this costs nothing.
#
# Not started, and reported as such, when HARVEST_ISAAC_AUTOSTART=0 -- the way to
# run Isaac in the foreground of another terminal (where you can watch its log)
# or under a debugger.
if [ "$MODE" = "isaac" ] && [ "${HARVEST_ISAAC_AUTOSTART:-1}" = "1" ]; then
  echo
  isaac_start || {
    echo "The HARVEST stack is up, but Isaac Sim did not start." >&2
    echo "Diagnostics will show 'Isaac Sim: inactive' until it does." >&2
  }
fi

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
    echo "Isaac Sim runs on the HOST, outside Docker (it needs the GPU and its"
    echo "own bundled Python + ROS 2 build).  The Diagnostics tab shows its live"
    echo "state: inactive -> healthy once it has booted and synchronised."
    if [ "${HARVEST_ISAAC_STREAMING:-0}" = "1" ] \
       || [ "${HARVEST_ISAAC_VIEW_MODE:-}" = "webrtc" ]; then
      echo
      echo "TO WATCH THE SIMULATION:"
      echo "  1. wait for '[harvest-isaac] READY' in $ISAAC_LOGFILE"
      echo "       tail -f $ISAAC_LOGFILE"
      echo "  2. open the NVIDIA Isaac Sim WebRTC Streaming Client (a browser"
      echo "     cannot show this stream -- the installed Isaac Sim 6.0.1 ships"
      echo "     no in-browser client)"
      echo "  3. connect it to  $(isaac_stream_url)"
      echo "     The client needs BOTH ports: ${HARVEST_ISAAC_SIGNAL_PORT:-49100}/TCP"
      echo "     (signalling) and ${HARVEST_ISAAC_STREAM_PORT:-47998}/UDP (media);"
      echo "     a TCP-only SSH tunnel negotiates a connection and shows no picture."
      echo "     From another machine:"
      echo "       ssh -L ${HARVEST_ISAAC_SIGNAL_PORT:-49100}:127.0.0.1:${HARVEST_ISAAC_SIGNAL_PORT:-49100} <user>@<this-host>"
    else
      echo
      echo "No live view in this mode.  For the WebRTC stream, restart with:"
      echo "  ./run_harvest_dashboard.sh stop"
      echo "  HARVEST_ISAAC_VIEW_MODE=webrtc HARVEST_ISAAC_STREAMING=1 \\"
      echo "  HARVEST_ISAAC_HEADLESS=1 ./run_harvest_dashboard.sh isaac"
    fi
    echo
    echo "Stop everything (stack + Isaac): ./run_harvest_dashboard.sh stop"
    echo "GPU-free stand-in instead:       ./run_harvest_dashboard.sh isaac-demo" ;;
esac
