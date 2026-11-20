#!/usr/bin/env bash
set -euo pipefail
root="$1"
chip="$2"
shift 2
mkdir -p log_ca log/ub_log run_log
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/dav_3510/camodel:$LD_LIBRARY_PATH"
export OPERATOR_SIM_TARGET_SOC="$chip"
case "$chip" in
    Ascend950DT_9582) export soc_version=Ascend950DT LD_PRELOAD="$root/build/capture/libsim_soc_adapter.so" ;;
    Ascend950PR_957b) export soc_version=Ascend950PR LD_PRELOAD="$root/build/capture/libpr15.so" ;;
    *) exit 2 ;;
esac
export CAMODEL_LOG_PATH="$PWD/log_ca" STARS_LOG_PATH="$PWD/log_ca"
unset ENABLE_CA_LOG_TRANS ENABLE_CA_RAW_INSTR_DUMP CAMODEL_CONFIG_PATH CORE_ENABLE_MASK
exec "$@"
