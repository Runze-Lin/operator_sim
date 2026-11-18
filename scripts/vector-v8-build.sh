#!/bin/bash
set -eu
root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$root/build/vector-v8"
bash "$root/scripts/cann-run.sh" bash -c '
set -eu
for kind in NORM MHC KDA; do
g++ -std=c++17 -O2 -D"V8_$kind" "$1/smoke/vector_v8.cpp" -I"$ASCEND_HOME_PATH/include" -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath-link,"$ASCEND_HOME_PATH/lib64" -lopapi -lnnopbase -lascendcl -o "$1/build/vector-v8/$kind"
done
' bash "$root"
