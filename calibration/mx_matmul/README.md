# 950 稠密 MX Matmul 的 CA 验证

在已激活的 CANN 9.2.1 开发环境中，从仓库根目录编译驱动。它向设备传入真正打包的 FP4 权重、A 和 W 各自的 E8M0 scale，调用一次 `aclnnQuantMatmulV5`。CPU 对 K 向每 32 项先应用两侧的 scale，累加后按最近偶数舍入到 BF16，逐位比对全部 M×N 个输出。该驱动要求 K 为 64 的倍数。

```sh
c++ -std=c++17 -O2 calibration/mx_matmul/native.cpp -I"$ASCEND_HOME_PATH/include" \
    -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath-link,"$ASCEND_HOME_PATH/lib64" \
    -lopapi -lnnopbase -lascendcl -o /tmp/mx-matmul-ca
```

在仓根为新几何准备清单并冻结。原生模式用 `dtype="mxfp4", attrs={}`，混合模式用 `dtype="fp8_e4m3fn", attrs={"weight_dtype":"mxfp4"}`。每条 case 都填实际 SKU、`shape=[M,N,K]`、`api="aclnnQuantMatmulV5"` 及 `outputs={"Y":M*N}`。下面的 `OPP_BIN` 是含有 `quant_batch_matmul_v3/`、`quant_batch_matmul_v4/` 对象文件的已安装芯片二进制目录。

```sh
python -m calibration freeze cases.json freeze.json
python -m calibration.mx_matmul.collect --freeze freeze.json --case native-m9-dt \
    --raw /tmp/native-m9-dt --probe /tmp/mx-matmul-ca --opp "$OPP_BIN"
python -m calibration score freeze.json /tmp/native-m9-dt/measurement.json
```

采集器先写冻结指纹，再运行 msopprof simulator。每个已完成的样例校验内存模型、单次派发、实际加载的 ELF 与 OPP 一致、全输出以及所有发射核的 END 和退休时间。CA 运行的墙钟时间单独写入 `cost.json`。需要重新解析时加 `--convert-only` 并使用原始目录和冻结文件。源码安装于主机或直接运行驱动的墙钟均不构成设备计时证据。[校准工具](../README.md)

一次运行已冻结的多个形状，可以用批量入口。它为每个形状创建独立工作目录，写入收据、计量结果及逐核流水摘要。默认同时运行 4 个 CA，每个 CA 的墙钟预算为 20 分钟。 `--jobs`、`--minutes`、`--reserve-disk-gib` 和 `--reserve-memory-gib` 可按宿主资源设置。

```sh
python -m calibration.mx_matmul.campaign --freeze freeze.json --raw-root /tmp/mx-ca \
    --probe /tmp/mx-matmul-ca --opp "$OPP_BIN" --jobs 4 --minutes 20
```

用同一目录和同一冻结文件再次执行，会复用已完成的收据和流水摘要；已有完整采集、缺少解析产物的形状只重新解析。进程中断或超时的形状保留日志以便定位，恢复采集可在新的输出目录使用 `--case CASE_ID` 单独执行。

逐核指令与数值收据可以分别复核。对已完成的原始目录运行 `python -m calibration.mx_matmul.native_phases /tmp/native-m9-dt` 可列出真正执行 MMAD 或向量指令的核数、指令类型、矩形和描述符时间。安装 matplotlib 后，可用 `python -m calibration.mx_matmul.plot docs/examples/assets/mx-matmul.json /tmp/mx.png` 从保留的时间数据重绘算例。在线预测不依赖绘图库。

若保留了本轮开发 raw，可重算 1.2.0 的阶段参数。先从 [参考快照](../reference/mx-matmul.json) 取 `development.freeze` 存为 `development-freeze.json`，再执行下列命令。原始目录需要与采集时的文件哈希匹配。计算只读取八个开发窗。

```sh
python -m calibration.mx_matmul.fit_phases --freeze development-freeze.json \
    --raw-root runs/mx12 --output /tmp/mx-phases-replay.json
```

### 同一套投影比较两个已冻结版本

现有 40 个独立整窗可以直接复核。从仓根运行以下命令即可查看两端 WAPE、活跃 AIC 和生产 AIV 命中、DT/PR 耗时比误差，以及每个几何的配对结果。

```sh
python -m calibration.mx_matmul.compare
python -m calibration.mx_matmul.compare --pairs
python -m calibration.verify --reprice
```

要用新的一套投影复现实验，先用 `python -m calibration.mx_matmul.projection_cases /tmp/mx-cases` 导出场景和配置来源。分别在两个版本各自的 Python 环境中，对同一个 cases.json 执行 `python -m calibration freeze`，再用 `python -m calibration.mx_matmul.projection_freeze FREEZE.json PLAN.json` 固化活跃核数、L1 页深、MMAD 片深和阶段时间。详细分核冻结工具随本版发布；复算旧版时从本仓根运行校准工具，令 Python 加载旧版的 `operator_model`。两个冻结文件均与各自的定价包哈希绑定。CA 收据绑定新版本指纹，两套计划必须早于同一批新采集。每个芯片配对都应完成后再统计加速比。[压缩参考包](../reference/mx-projections.json.gz) · [投影结果与误差](../../docs/error-analysis.md#稠密-mx-matmul--双端同形验证)

完成整组采集后可从工作目录生成新的紧凑证据包。保留两版冻结、逐窗身份、全核计时和指令摘要，完整 trace 继续存放在 raw 目录。

```sh
python -m calibration.mx_matmul.bundle --geometries geometries.json --budget budget.json \
    --old-timing v12-freeze.json --old-plans v12-plans.json \
    --new-timing v13-freeze.json --new-plans v13-plans.json \
    --raw /tmp/mx-ca --output /tmp/mx-projections.json.gz
python -m calibration.mx_matmul.compare /tmp/mx-projections.json.gz
```
