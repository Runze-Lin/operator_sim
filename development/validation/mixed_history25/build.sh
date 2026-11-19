#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
mkdir -p "$root/build/mixed-stream25"
exec bash "$root/scripts/cann-run.sh" bash --noprofile --norc -c '
set -euo pipefail
g++ -std=c++17 -O2 "$1/development/validation/mixed_history25/host.cpp" -I"$ASCEND_HOME_PATH/include" -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath-link,"$ASCEND_HOME_PATH/lib64" -lopapi -lnnopbase -lascendcl -o "$1/build/mixed-stream25/host"
' mixed-host "$root"
