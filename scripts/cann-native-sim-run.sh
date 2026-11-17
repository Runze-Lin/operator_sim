#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$project_root/scripts/cann-run.sh" bash --noprofile --norc -c '
    record_dir="$1"; adapter="$2"; shift 2
    mkdir -p "$record_dir/log_ca" "$record_dir/log/ub_log" "$record_dir/run_log"
    export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/dav_3510/camodel:$LD_LIBRARY_PATH"
    export LD_PRELOAD="$adapter"
    export soc_version=Ascend950DT
    export OPERATOR_SIM_TARGET_SOC=Ascend950DT_9582
    export CAMODEL_LOG_PATH="$record_dir/log_ca"
    export STARS_LOG_PATH="$record_dir/log_ca"
    unset ENABLE_CA_LOG_TRANS ENABLE_CA_RAW_INSTR_DUMP
    if [[ "${OPERATOR_CAPTURE_HELPER:-0}" == 1 ]]; then
        mkdir -p "$record_dir/helper"
        export LD_PRELOAD="$ASCEND_HOME_PATH/tools/simulator/bin/libcannsim_helper.so:$LD_PRELOAD"
        export CANNSIM_ARCH=Ascend950DT
        export CANNSIM_REPORT_ARCH=v100
        export CANNSIM_HELPER_LAUNCH_DB_DIR="$record_dir/helper"
        export CANNSIM_HELPER_LOG="$record_dir/helper/cannsim_helper.log"
    fi
    exec "$@"
' native-sim "$1" "$project_root/build/capture/libsim_soc_adapter.so" "${@:2}"
