#!/usr/bin/env bash
set -eu
: "${ASCEND_HOME_PATH:?invoke under scripts/cann-run.sh}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
mkdir -p helper log_ca log/ub_log run_log
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/dav_3510/camodel:$LD_LIBRARY_PATH"
export LD_PRELOAD="$ASCEND_HOME_PATH/tools/simulator/bin/libcannsim_helper.so:$root/build/capture/libpr15.so"
export soc_version=Ascend950PR OPERATOR_SIM_TARGET_SOC=Ascend950PR_957b CANNSIM_ARCH=Ascend950 CANNSIM_REPORT_ARCH=v100
export CANNSIM_HELPER_LAUNCH_DB_DIR="$PWD/helper" CANNSIM_HELPER_LOG="$PWD/helper/cannsim_helper.log" CAMODEL_LOG_PATH="$PWD/log_ca" STARS_LOG_PATH="$PWD/log_ca"
unset CAMODEL_CONFIG_PATH ENABLE_CA_LOG_TRANS ENABLE_CA_RAW_INSTR_DUMP CORE_ENABLE_MASK
exec "$@"
