#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$project_root/build/smoke"
exec bash "$project_root/scripts/cann-run.sh" bash --noprofile --norc -c '
    set -euo pipefail
    g++ -std=c++17 -O2 -g -Wall -Wextra "$1/smoke/extension_kda.cpp" \
        -I"$ASCEND_HOME_PATH/include" -L"$ASCEND_HOME_PATH/lib64" \
        -Wl,-rpath-link,"$ASCEND_HOME_PATH/lib64" \
        -lopapi -lnnopbase -lascendcl -o "$1/build/smoke/extension_kda"
' build-extension "$project_root"
