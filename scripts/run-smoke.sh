#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
case_name="${1:-}"
case "$case_name" in quant|matmul|fa) ;; *) echo "Usage: bash scripts/run-smoke.sh quant|matmul|fa" >&2; exit 2 ;; esac
output="$project_root/runs/manual-smoke/$(date +%Y%m%d-%H%M%S)-$case_name-$$"
mkdir -p "$output"
echo "Output: $output"
cd "$output"
/usr/bin/time -o "$output/host-time.txt" -f 'wall_seconds=%e exit_status=%x' \
    timeout --signal=TERM --kill-after=15s 360s \
    bash "$project_root/scripts/cann-run.sh" msopprof simulator \
    --soc-version=Ascend950DT_9582 --dump=true --launch-count=10 --timeout=2 \
    --output="$output/profile" "$project_root/build/smoke/three_ops" "$case_name" \
    > "$output/run.log" 2>&1
cat "$output/run.log"
grep -q '^SMOKE_PASS$' "$output/run.log"
grep -q 'Profiling data parse finished.' "$output/run.log"
