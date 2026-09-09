#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Validate the running HARVEST stack end-to-end.
#
#   ./run_harvest_dashboard.sh [mode]     # start the stack first
#   ./validate_harvest_stack.sh
#
# Checks the fleet API, the command round-trip, the FIWARE mirror + inbound
# NGSI-LD command path, and the ROS 2 snapshot topic.  Checks whose service
# is not running are skipped, not failed.
#
# Exit code = number of failed checks.
# ----------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec python3 scripts/validate_stack.py "$@"
