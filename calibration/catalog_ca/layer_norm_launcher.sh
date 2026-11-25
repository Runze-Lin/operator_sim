#!/usr/bin/env bash
# Keep both a possible materialization and the reduction kernel.
set -eu
program="$1"
action="$2"
shift 2
args=("$program" "$action" --launch-count=8 "$@")
if [[ -n "${OPERATOR_SIM_CANN_LAUNCHER:-}" ]]; then
    exec bash "$OPERATOR_SIM_CANN_LAUNCHER" "${args[@]}"
elif command -v "$program" >/dev/null; then
    exec "${args[@]}"
fi
project=$(cd "$(dirname "$0")/../.." && pwd)
if [[ -f "$project/scripts/cann-run.sh" ]]; then
    exec bash "$project/scripts/cann-run.sh" "${args[@]}"
fi
echo 'Activate CANN 9.2.1 or set OPERATOR_SIM_CANN_LAUNCHER to its environment wrapper.' >&2
exit 127
