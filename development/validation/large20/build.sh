#!/usr/bin/env bash
set -euo pipefail
project=$(cd "$(dirname "$0")/../../.." && pwd)
mkdir -p "$project/build/smoke"
exec bash "$project/scripts/cann-run.sh" bash -c '
 g++ -O2 -std=c++17 "$1/development/validation/large20/cast.cpp" -I"$ASCEND_HOME_PATH/include" -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath-link,"$ASCEND_HOME_PATH/lib64" -lopapi -lnnopbase -lascendcl -o "$1/build/smoke/large_cast20"
' build "$project"
