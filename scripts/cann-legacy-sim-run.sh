#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$project_root/scripts/cann-run.sh" bash --noprofile --norc -c '
  export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/dav_3510/lib:$LD_LIBRARY_PATH"
  exec "$@"
' legacy-sim "$@"
