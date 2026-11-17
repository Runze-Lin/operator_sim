#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
environment_prefix="$project_root/toolchains/envs/cann-9.2.1"
cann_prefix="$project_root/toolchains/cann/9.2.1/cann-9.2.1"
export MAMBA_ROOT_PREFIX="$project_root/toolchains/mamba-root"
export XDG_CACHE_HOME="$project_root/toolchains/cache"
if [[ $# -eq 0 ]]; then
    echo "Usage: bash scripts/cann-run.sh COMMAND [ARGS...]" >&2
    exit 2
fi
if [[ ! -f "$cann_prefix/set_env.sh" ]]; then
    echo "CANN installation is not ready: $cann_prefix" >&2
    exit 1
fi
exec "$project_root/toolchains/micromamba/bin/micromamba" --no-rc run \
    -p "$environment_prefix" bash --noprofile --norc -c '
        set -e
        unset PYTHONHOME PYTHONPATH ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_OPP_PATH ASCEND_AICPU_PATH
        export PYTHONNOUSERSITE=1
        export PATH="$CONDA_PREFIX/bin:/usr/local/bin:/usr/bin:/bin"
        export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
        source "$1/set_env.sh"
        shift
        exec "$@"
    ' cann-run "$cann_prefix" "$@"
