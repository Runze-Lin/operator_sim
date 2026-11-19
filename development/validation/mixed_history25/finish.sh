#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
py=toolchains/envs/cann-9.2.1/bin/python
dir=development/validation/mixed_history25
while ! grep -q '"state": "finished"' results/v0.25/mixed-stream/independent-status.json; do sleep 15; done
"$py" "$dir/verify_windows.py" independent
"$py" "$dir/score.py"
"$py" "$dir/occupancy.py"
"$py" "$dir/score.py"
"$py" "$dir/recover.py" DT
"$py" "$dir/score.py"
"$py" "$dir/recover.py" PR
"$py" "$dir/score.py"
