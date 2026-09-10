#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Launch the HARVEST Isaac Sim layer on the HOST (never inside Docker).
#
#   ./scripts/run_isaac_sim.sh                       # GUI if a display exists
#   HARVEST_ISAAC_VIEW_MODE=webrtc ./scripts/run_isaac_sim.sh
#   ./scripts/run_isaac_sim.sh --self-test           # boot, report READY, exit
#   ./scripts/run_isaac_sim.sh --self-test-drive     # prove a tractor drives
#   ./scripts/run_isaac_sim.sh --list-models
#
# Start the HARVEST side first:  ./run_harvest_dashboard.sh isaac
# (that launcher starts this script for you unless HARVEST_ISAAC_AUTOSTART=0).
# The two meet only on the DDS wire: a shared ROS_DOMAIN_ID and nothing else.
#
# This script is a port of WISEPACK's scripts/run_wisepack_isaac.sh, which is
# the recipe PROVEN on this machine.  Every deviation is either a name (HARVEST_
# for WISEPACK_) or is commented with what was measured here.
#
# Configuration precedence:
#   1. explicit environment (ISAAC_SIM_ROOT, ROS_DOMAIN_ID, HARVEST_ISAAC_*)
#   2. config.local.env in the repo root (git-ignored KEY=VALUE file, parsed
#      with an allowlist -- never sourced, so a stray backtick is data)
#   3. known install locations, newest major first (see find_isaac_root)
#
# HOW YOU WATCH IT -- one variable, because the switches interact:
#   HARVEST_ISAAC_VIEW_MODE=desktop  a window on the host display; use your own
#                                    remote-desktop tool.  No stream.
#   HARVEST_ISAAC_VIEW_MODE=webrtc   headless + Isaac's WebRTC livestream, for
#                                    the NVIDIA Isaac Sim WebRTC Streaming
#                                    Client.  No window.
#   HARVEST_ISAAC_VIEW_MODE=none     headless, no stream.  Telemetry only.
#
# WHY THIS SCRIPT SCRUBS THE ROS ENVIRONMENT (WISEPACK finding, measured):
# Isaac Sim ships its OWN ROS 2 build compiled against its own Python ABI.  If
# the calling shell has sourced /opt/ros/<distro>, PYTHONPATH carries a second,
# ABI-incompatible rclpy ahead of Isaac's and the failure is an import-time
# crash deep inside rclpy's C extension.  Scrubbing alone is not enough either:
# Isaac's ROS libraries are on no default search path, so after the scrub
# Isaac's own setup_ros_env.sh is sourced to put them back.  ROS_DOMAIN_ID is
# the one ROS variable deliberately kept and passed through.
#
# Exit codes: 0 ok | 3 no Isaac Sim | 4 bad robot model | 5 self-test failed
#             6 streaming unavailable/port busy | 7 cannot create runtime dir
#             8 bad view mode
# ----------------------------------------------------------------------------
set -u   # no -e: vendor scripts are sourced below and must not abort us

REPO="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"
LOG="[isaac-launch]"

# ── Local machine config (allowlisted KEY=VALUE, never sourced) ──────────────
load_local_env() {
  local file="$REPO/config.local.env" line key value
  [ -f "$file" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|'#'*) continue ;; esac
    key="${line%%=*}"; value="${line#*=}"
    case "$key" in
      # Only keys this project owns; an explicit export always wins.
      ISAAC_SIM_ROOT)           [ -n "${ISAAC_SIM_ROOT:-}" ]           || export ISAAC_SIM_ROOT="$value" ;;
      ROS_DOMAIN_ID)            [ -n "${ROS_DOMAIN_ID:-}" ]            || export ROS_DOMAIN_ID="$value" ;;
      HARVEST_ISAAC_HEADLESS)   [ -n "${HARVEST_ISAAC_HEADLESS:-}" ]   || export HARVEST_ISAAC_HEADLESS="$value" ;;
      HARVEST_ISAAC_VIEW_MODE)  [ -n "${HARVEST_ISAAC_VIEW_MODE:-}" ]  || export HARVEST_ISAAC_VIEW_MODE="$value" ;;
      HARVEST_ISAAC_STREAMING)  [ -n "${HARVEST_ISAAC_STREAMING:-}" ]  || export HARVEST_ISAAC_STREAMING="$value" ;;
      HARVEST_ISAAC_STREAM_HOST)[ -n "${HARVEST_ISAAC_STREAM_HOST:-}" ]|| export HARVEST_ISAAC_STREAM_HOST="$value" ;;
      HARVEST_ISAAC_SIGNAL_PORT)[ -n "${HARVEST_ISAAC_SIGNAL_PORT:-}" ]|| export HARVEST_ISAAC_SIGNAL_PORT="$value" ;;
      HARVEST_ISAAC_STREAM_PORT)[ -n "${HARVEST_ISAAC_STREAM_PORT:-}" ]|| export HARVEST_ISAAC_STREAM_PORT="$value" ;;
      HARVEST_ISAAC_ROBOT_MODEL)[ -n "${HARVEST_ISAAC_ROBOT_MODEL:-}" ]|| export HARVEST_ISAAC_ROBOT_MODEL="$value" ;;
      HARVEST_SSH_PORT)         [ -n "${HARVEST_SSH_PORT:-}" ]         || export HARVEST_SSH_PORT="$value" ;;
    esac
  done < "$file"
}
load_local_env

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

# ── Isaac Sim discovery ──────────────────────────────────────────────────────
# Explicit ISAAC_SIM_ROOT wins.  Otherwise known locations NEWEST FIRST -- and
# never silently settle for an older major: the app is written against Isaac
# Sim 6.x (isaacsim.core.experimental API) and a 5.x install fails with import
# errors that look like our bugs.
find_isaac_root() {
  if [ -n "${ISAAC_SIM_ROOT:-}" ]; then
    printf '%s' "$ISAAC_SIM_ROOT"; return
  fi
  local candidates=(
    /data/isaac-sim/isaac-sim-6.0.1
    /data/isaac-sim/isaac-sim-6.0
    "$HOME/isaacsim"
    "$HOME/.local/share/ov/pkg/isaac-sim-6.0.1"
    /opt/isaac-sim
    /isaac-sim
  )
  local path
  for path in "${candidates[@]}"; do
    [ -x "$path/python.sh" ] && { printf '%s' "$path"; return; }
  done
  # Last resort: any 6.x under the standard local install tree, newest first.
  for path in $(ls -d /data/isaac-sim/isaac-sim-6.* 2>/dev/null | sort -Vr); do
    [ -x "$path/python.sh" ] && { printf '%s' "$path"; return; }
  done
  printf ''
}

ISAAC_ROOT="$(find_isaac_root)"
if [ -z "$ISAAC_ROOT" ]; then
  echo "$LOG ERROR: no Isaac Sim installation found." >&2
  echo "$LOG   set ISAAC_SIM_ROOT=/path/to/isaac-sim (env or config.local.env)," >&2
  echo "$LOG   or install Isaac Sim 6.x.  Searched: /data/isaac-sim/isaac-sim-6.*," >&2
  echo "$LOG   ~/isaacsim, ~/.local/share/ov/pkg, /opt/isaac-sim, /isaac-sim." >&2
  echo "$LOG   No GPU/Isaac?  Use the stand-in: ./run_harvest_dashboard.sh isaac-demo" >&2
  exit 3
fi
# Verify the bundled launcher BEFORE starting anything: a root without
# python.sh produces a confusing 'command not found' several steps later.
if [ ! -x "$ISAAC_ROOT/python.sh" ]; then
  echo "$LOG ERROR: $ISAAC_ROOT has no executable python.sh --" >&2
  echo "$LOG   that is not an Isaac Sim installation root." >&2
  exit 3
fi

ISAAC_VERSION="unknown"
[ -f "$ISAAC_ROOT/VERSION" ] && ISAAC_VERSION="$(cat "$ISAAC_ROOT/VERSION")"
case "$ISAAC_VERSION" in
  6.*) ;;
  *) echo "$LOG WARNING: $ISAAC_ROOT reports version '$ISAAC_VERSION'." >&2
     echo "$LOG   harvest_isaac.py targets Isaac Sim 6.x (isaacsim.core.experimental);" >&2
     echo "$LOG   older majors will fail to import." >&2 ;;
esac

# ── Headless / display ───────────────────────────────────────────────────────
# GUI by default only when a display actually exists: defaulting to GUI over
# SSH crashes inside the renderer instead of printing anything useful.
HEADLESS="${HARVEST_ISAAC_HEADLESS:-}"
if [ -z "$HEADLESS" ]; then
  if [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
    HEADLESS=0
  else
    HEADLESS=1
    echo "$LOG no DISPLAY -- falling back to headless mode"
  fi
fi

# ── How the operator will WATCH this run ─────────────────────────────────────
# ONE variable decides the whole viewing configuration, because the individual
# switches interact: WebRTC needs headless, desktop needs a display, and "none"
# needs neither.  Setting them separately is how you end up asking for a GUI
# stream on a machine with no display and getting neither (WISEPACK).
VIEW_MODE="${HARVEST_ISAAC_VIEW_MODE:-}"
if [ -z "$VIEW_MODE" ]; then
  # Infer from the older switches so existing invocations keep working.
  if [ "${HARVEST_ISAAC_STREAMING:-0}" = "1" ]; then VIEW_MODE="webrtc"
  elif [ "$HEADLESS" = "0" ]; then VIEW_MODE="desktop"
  else VIEW_MODE="none"; fi
fi

case "$VIEW_MODE" in
  desktop)
    if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
      echo "$LOG ERROR: view mode 'desktop' needs a display, and neither" >&2
      echo "$LOG   DISPLAY nor WAYLAND_DISPLAY is set.  Either run where a" >&2
      echo "$LOG   desktop session exists, or use HARVEST_ISAAC_VIEW_MODE=webrtc." >&2
      exit 8
    fi
    HEADLESS=0
    HARVEST_ISAAC_STREAMING=0
    ;;
  webrtc)
    # Isaac Sim 6.0.1 captures the application framebuffer for the stream and is
    # run headless for it; a GUI window is not additionally opened.
    HEADLESS=1
    HARVEST_ISAAC_STREAMING=1
    ;;
  none)
    HEADLESS=1
    HARVEST_ISAAC_STREAMING=0
    ;;
  *)
    echo "$LOG ERROR: unknown HARVEST_ISAAC_VIEW_MODE='$VIEW_MODE'." >&2
    echo "$LOG   expected one of: desktop | webrtc | none" >&2
    exit 8 ;;
esac
export HARVEST_ISAAC_HEADLESS="$HEADLESS"

ISAAC_ARGS=()
[ "$HEADLESS" = "1" ] && ISAAC_ARGS+=(--headless)
ISAAC_ARGS+=("$@")

# ── WebRTC livestreaming ─────────────────────────────────────────────────────
# ADVERTISED host, not the bind address.  Kit listens on 0.0.0.0 whatever this
# says; this only controls the endpoint HARVEST tells a client to dial.  The two
# are kept apart because conflating them is a real reporting bug: a client can
# connect through the server's routable address while the UI displays 127.0.0.1.
STREAMING="${HARVEST_ISAAC_STREAMING:-0}"
STREAM_HOST_EXPLICIT=0
[ -n "${HARVEST_ISAAC_STREAM_HOST:-}" ] && STREAM_HOST_EXPLICIT=1
STREAM_HOST="${HARVEST_ISAAC_STREAM_HOST:-127.0.0.1}"
KIT_BIND_ADDRESS="0.0.0.0"
SIGNAL_PORT="${HARVEST_ISAAC_SIGNAL_PORT:-49100}"
STREAM_PORT="${HARVEST_ISAAC_STREAM_PORT:-47998}"
VIEWER_PORT="${HARVEST_ISAAC_VIEWER_PORT:-0}"

if [ "$STREAMING" = "1" ]; then
  # Verify the extensions EXIST before launching.  Kit reports a missing
  # livestream extension as a warning buried in a few thousand startup lines and
  # then runs perfectly happily with no stream at all, so the operator waits for
  # a viewer URL that is never coming.
  missing=""
  for ext in omni.kit.livestream.app omni.kit.livestream.webrtc; do
    ls -d "$ISAAC_ROOT"/ext*/"$ext"* >/dev/null 2>&1 || missing="$missing $ext"
  done
  if [ -n "$missing" ]; then
    echo "$LOG ERROR: Isaac Sim at $ISAAC_ROOT is missing required" >&2
    echo "$LOG   livestream extension(s):$missing" >&2
    echo "$LOG   run with HARVEST_ISAAC_VIEW_MODE=none, or install them." >&2
    exit 6
  fi

  # Refuse to start a SECOND stream server on an occupied signal port.  Kit
  # silently falls back to another free port, so the URL published here would
  # point at a different, older stream -- which shows a picture of the wrong
  # thing, the hardest kind of wrong to notice.
  if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":${SIGNAL_PORT} "; then
    echo "$LOG ERROR: signal port ${SIGNAL_PORT} is already in use." >&2
    echo "$LOG   another Isaac stream is probably running.  Stop it, or set" >&2
    echo "$LOG   HARVEST_ISAAC_SIGNAL_PORT to a free port." >&2
    exit 6
  fi

  VIEWER_URL="${HARVEST_ISAAC_STREAM_URL:-http://${STREAM_HOST}:$([ "$VIEWER_PORT" != "0" ] && echo "$VIEWER_PORT" || echo "$SIGNAL_PORT")}"
  export HARVEST_ISAAC_STREAMING=1
  # Exported ONLY when the operator set it.  Exporting the resolved default
  # would make the app believe loopback was a deliberate choice and suppress the
  # "set HARVEST_ISAAC_STREAM_HOST for a remote client" guidance exactly when it
  # is needed.
  if [ "$STREAM_HOST_EXPLICIT" = "1" ]; then
    export HARVEST_ISAAC_STREAM_HOST="$STREAM_HOST"
  else
    unset HARVEST_ISAAC_STREAM_HOST
  fi
  export HARVEST_ISAAC_SIGNAL_PORT="$SIGNAL_PORT"
  export HARVEST_ISAAC_STREAM_PORT="$STREAM_PORT"
  export HARVEST_ISAAC_VIEWER_PORT="$VIEWER_PORT"
  export HARVEST_ISAAC_STREAM_URL="$VIEWER_URL"
else
  export HARVEST_ISAAC_STREAMING=0
fi

# ── Environment isolation (scrub, then Isaac's own ROS environment) ──────────
unset PYTHONPATH
unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset ROS_DISTRO
unset ROS_VERSION
unset ROS_PYTHON_VERSION
unset LD_LIBRARY_PATH
unset ROS_PACKAGE_PATH

# setup_ros_env.sh ships WITH Isaac and only acts when ROS_DISTRO is unset --
# exactly the state the scrub leaves.  It adds Isaac's internal ROS libraries
# (librmw_implementation.so etc.) to the search path; without it the ros2 bridge
# extension fails to load and `import rclpy` raises.  `set +u` around it: the
# vendor script tests variables it has not defined, and under nounset that
# aborts before any of the environment is applied.
if [ -f "$ISAAC_ROOT/setup_ros_env.sh" ]; then
  set +u
  # shellcheck source=/dev/null
  source "$ISAAC_ROOT/setup_ros_env.sh"
  set -u
fi

# Fast DDS on both ends (docker-compose pins the same): different RMWs simply
# never discover each other.
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

# ── DDS TRANSPORT: UDPv4 ONLY, and this is the fix for a real, measured bug ──
#
# Fast DDS prefers SHARED MEMORY between participants on the same host.  Isaac
# runs here on the host as the invoking user; HARVEST's ros2 containers run with
# `ipc: host` as root.  A Fast DDS SHM segment is created mode 0644 owned by its
# creator, so the container (root) can map the host's segments but the host user
# CANNOT map the container's read-write.  Fast DDS does not then fall back to
# UDP for a pair it has already matched on a SHM locator, so the result is the
# worst possible failure mode: discovery SUCCEEDS -- `ros2 topic list` shows
# /harvest/sim/telemetry -- and not one sample is ever delivered.
#
# Measured on this machine, host Isaac rclpy <-> harvest:ros2 container, domain
# 42, both Fast DDS 2.14.6:
#     default transports  : topic visible, `ros2 topic echo` times out
#     container as uid 1000: works        (segments then owned by the same user)
#     FASTDDS_BUILTIN_TRANSPORTS=UDPv4: works
# UDPv4 is the setting HARVEST uses because it does not couple the container's
# user to whoever happens to run Isaac.  The traffic is two latched JSON strings
# at 2 Hz, so loopback UDP costs nothing measurable.
#
# Set it on BOTH ends or neither: docker-compose.yml pins the same value for
# ros2-bridge-isaac and isaac-sim-stub.  Override with
# HARVEST_DDS_TRANSPORT=DEFAULT to get Fast DDS's own preference back.
DDS_TRANSPORT="${HARVEST_DDS_TRANSPORT:-UDPv4}"
if [ "$DDS_TRANSPORT" = "DEFAULT" ]; then
  unset FASTDDS_BUILTIN_TRANSPORTS
else
  export FASTDDS_BUILTIN_TRANSPORTS="$DDS_TRANSPORT"
fi

# Stale segments from a crashed participant hang every later ros2 command with
# no error message at all, so they are cleared before we add another.
[ -x "$REPO/scripts/clean_dds_shm.sh" ] && "$REPO/scripts/clean_dds_shm.sh"

# Unbuffered so the READY/telemetry progress lines stream to pipes and logs
# instead of sitting in an 8 KiB buffer (WISEPACK, measured: 23 kB of Kit's own
# unbuffered C++ logging with not one of our lines among it).
export PYTHONUNBUFFERED=1

# ── Resolved configuration, printed once ─────────────────────────────────────
# Everything an operator needs to find the simulation, and nothing that should
# not be published: no secrets, and never this host's SSH port.
echo "$LOG ----------------------------------------------------------------"
echo "$LOG  Isaac Sim     : $ISAAC_ROOT (version $ISAAC_VERSION)"
echo "$LOG  robot model   : ${HARVEST_ISAAC_ROBOT_MODEL:-<registry default>}"
echo "$LOG  view mode     : $VIEW_MODE"
echo "$LOG  headless      : $([ "$HEADLESS" = "1" ] && echo yes || echo no)"
echo "$LOG  DISPLAY       : ${DISPLAY:-<none>}"
echo "$LOG  ROS_DOMAIN_ID : $ROS_DOMAIN_ID"
echo "$LOG  ROS runtime   : Isaac-internal ${ROS_DISTRO:-?} (${RMW_IMPLEMENTATION})"
echo "$LOG  DDS transport : ${FASTDDS_BUILTIN_TRANSPORTS:-<Fast DDS default>}"
if [ "$STREAMING" = "1" ]; then
  echo "$LOG  streaming     : enabled (WebRTC)"
  echo "$LOG  bind/listen   : ${KIT_BIND_ADDRESS}:${SIGNAL_PORT} (Kit binds every interface)"
  echo "$LOG  advertised    : ${STREAM_HOST}$([ "$STREAM_HOST_EXPLICIT" = "1" ] && echo " (explicit)" || echo " (default -- local/forwarded)")"
  echo "$LOG  signalling    : ${SIGNAL_PORT}/TCP"
  echo "$LOG  media         : ${STREAM_PORT}/UDP  (a TCP-only tunnel carries no video)"
  echo "$LOG  watch it with : NVIDIA Isaac Sim WebRTC Streaming Client -> $VIEWER_URL"
  echo "$LOG                  (a browser cannot display this stream)"
  echo "$LOG  remote view   : ssh -L ${SIGNAL_PORT}:127.0.0.1:${SIGNAL_PORT} <user>@<this-host>"
  if [ "$STREAM_HOST_EXPLICIT" != "1" ]; then
    echo "$LOG    note        : local/forwarded endpoint.  For a remote client, set"
    echo "$LOG                  HARVEST_ISAAC_STREAM_HOST to an address it can reach."
  fi
  # The stream is unauthenticated.  Say so once, loudly, at the point of use.
  case "$STREAM_HOST" in
    127.0.0.1|localhost|::1) ;;
    *) echo "$LOG WARNING: the WebRTC stream has NO authentication and NO" >&2
       echo "$LOG   encryption, and you have advertised it as '$STREAM_HOST'." >&2
       echo "$LOG   Prefer loopback + SSH port forwarding, or restrict the" >&2
       echo "$LOG   ports by firewall to a single client address." >&2 ;;
  esac
elif [ "$VIEW_MODE" = "desktop" ]; then
  echo "$LOG  streaming     : disabled"
  echo "$LOG  watch it on   : the host desktop (${DISPLAY:-?}) via your own"
  echo "$LOG                  NoMachine / Sunshine+Moonlight / VNC session"
else
  echo "$LOG  streaming     : disabled"
  echo "$LOG  watch it via  : telemetry only -- no video in this mode"
fi
echo "$LOG ----------------------------------------------------------------"

# ── Runtime working directory ────────────────────────────────────────────────
# NVIDIA's streaming stack writes trace files (NvStreamer-*.etli, ~7 MB each,
# one per minute of streaming) into the PROCESS WORKING DIRECTORY.  Launched
# from the repository root, that is the repository root -- WISEPACK measured
# 53 MB of binary traces among its source after one WebRTC session.  So run from
# a directory the launcher owns and removes.  Every path handed to the simulator
# is absolute: changing the working directory breaks relative asset lookups and
# sys.path entries derived from the script location.
ISAAC_APP="$REPO/harvest_integrations/simulators/isaac/harvest_isaac.py"
RUNTIME_DIR="${TMPDIR:-/tmp}/harvest-isaac-runtime/isaac-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$RUNTIME_DIR" || { echo "$LOG ERROR: cannot create $RUNTIME_DIR" >&2; exit 7; }
trap 'rm -rf -- "$RUNTIME_DIR"' EXIT INT TERM
cd "$RUNTIME_DIR" || exit 7

exec "$ISAAC_ROOT/python.sh" "$ISAAC_APP" "${ISAAC_ARGS[@]}"
