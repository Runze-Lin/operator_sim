#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
mkdir -p "$root/build/smoke"
exec bash "$root/scripts/cann-run.sh" bash -c '
g++ -std=c++17 -O2 -Wall -Wextra "$1/smoke/large_vector.cpp" -I"$ASCEND_HOME_PATH/include" -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath-link,"$ASCEND_HOME_PATH/lib64" -lopapi -lnnopbase -lascendcl -o "$1/build/smoke/large_vector"
' build "$root"
