#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$project_root/build/capture"
g++ -std=c++17 -O2 -Wall -Wextra -fPIC -shared "$project_root/capture/sim_soc_adapter.cpp" -ldl -o "$project_root/build/capture/libsim_soc_adapter.so"
