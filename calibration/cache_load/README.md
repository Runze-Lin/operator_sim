# 校准 CacheLoad 的两个原生候选实现

`inference::cache_load` 是系统图中的物化语义。`probe.cpp` 单次发起连续 E2M1 + 成对 block32 E8M0 → BF16 的 `aclnnAntiMxQuant`，`probe_fp8.cpp` 单次发起 E4M3FN + 逐行 FP32 → BF16 的 `aclnnAscendAntiQuant`。两个驱动逐个比对 BF16 输出。收据校验实际 SKU、内存模型、唯一 launch、安装 OPP 与实际加载 ELF 的一致性，以及全核名册中每核唯一 END 和最终退休。[工作量、设备模型、逐价推导与适用边界](../../docs/examples/cache_load.md)

**查询已归档的完整窗。** 从仓根运行，源码包或可编辑安装均可使用。

```sh
python -m calibration.cache_load.lookup --chip 950DT --shape 1 128 512
python -m calibration.cache_load.lookup --dtype fp8_e4m3fn --chip 950PR --shape 1 128 512
```

FP4 的开发、独立归档分别包含 `freeze` 和 `observations`。每个 observation 含完整 `receipt`、按公共 `ingest` 重算的 `measurement`、原生探针 SHA、采集秒数、970 查询墙钟与最后退休核的 `terminal_phases`。FP8 记录有完整收据与逐核观测，仅提供原生参考。独立 FP4 的已验证几何、当时芯片配置及源码指纹都在冻结文件中，重绘示例可执行以下命令。

```sh
python -m calibration.cache_load.plot calibration/cache_load/fp4-independent.json.gz /tmp/cache-load-antimx.png
```

**用已激活的 CANN 9.2.1 重新采集。** 设置 `OPP_BIN` 为该安装的 `opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_nn` 目录。下面的驱动是 CPU 宿主 C++，器件侧执行已安装的库核。宿主没有物理设备时可在链接参数中追加 CANN 的 `aarch64-linux/devlib/linux/aarch64` rpath-link。

```sh
c++ -std=c++17 -O2 calibration/cache_load/probe.cpp -I"$ASCEND_HOME_PATH/include" \
    -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath-link,"$ASCEND_HOME_PATH/lib64" \
    -lopapi -lnnopbase -lascendcl -o /tmp/cache4-probe
c++ -std=c++17 -O2 calibration/cache_load/probe_fp8.cpp -I"$ASCEND_HOME_PATH/include" \
    -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath-link,"$ASCEND_HOME_PATH/lib64" \
    -lopapi -lnnopbase -lascendcl -o /tmp/cache8-probe
python -m calibration.cache_load.fp8_smoke --chip Ascend950DT_9582 \
    --raw /tmp/cache8-dt-new --probe /tmp/cache8-probe --opp "$OPP_BIN" --minutes 4
```

FP4 在访问新几何之前用 `python -m calibration freeze cases.json freeze.json` 冻结。每行指定 `op=cache_load`、`contract=cache_load_mxfp4_antimx`、`attrs={"implementation":"aclnnAntiMxQuant"}`、`dtype=mxfp4`、`api=aclnnAntiMxQuant` 与 `outputs={"Y":逻辑输出元素数}`。独立样本通过冻结指纹绑定每次采集。开发样本可以用 `--development cases.json` 单独采集，以晚于采集的冻结重放，保留其开发角色。每个命令只启动一个 CA。

```sh
python -m calibration.cache_load.collect --freeze freeze.json --case CASE_ID \
    --raw /tmp/cache-results/CASE_ID --probe /tmp/cache4-probe --opp "$OPP_BIN" --minutes 4
python -m calibration.cache_load.bundle --freeze freeze.json --raw /tmp/cache-results \
    --opp "$OPP_BIN" --output /tmp/fp4-observations.json.gz
```

`--raw` 属于本地运行目录，不应提交到版本库。已经存在完整 profile 时，向单个采集命令追加 `--convert-only` 即可重新检查并出具收据；新的采集需使用新的目录。
