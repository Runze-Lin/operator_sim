#!/usr/bin/env bash
# Usage after CANN activation: launch.sh NEW_DIRECTORY PROBE SOC M N K
set -euo pipefail
if [[ $# != 6 ]]; then
    echo "usage: $0 NEW_DIRECTORY PROBE SOC M N K" >&2
    exit 2
fi
root=$(cd "$(dirname "$0")/../.." && pwd)
dest=$(realpath -m "$1")
probe=$(realpath -e "$2")
soc=$3
shift 3
launcher=()
if [[ -n ${CANN_LAUNCHER:-} ]]; then
    launcher=(bash "$(realpath -e "$CANN_LAUNCHER")")
elif ! command -v msopprof >/dev/null; then
    if [[ -f "$root/scripts/cann-run.sh" ]]; then
        launcher=(bash "$root/scripts/cann-run.sh")
    else
        echo "Activate CANN 9.2.1 or set CANN_LAUNCHER to an environment initialization wrapper." >&2
        exit 2
    fi
fi
mkdir "$dest"
cd "$dest"
export GOMP_SPINCOUNT=0 OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=1
exec "${launcher[@]}" msopprof simulator "--soc-version=$soc" --launch-count=1 \
    --timeout=10 --dump=true "--output=$dest/profile" "$probe" a8w8 "$@" >run.log 2>&1
