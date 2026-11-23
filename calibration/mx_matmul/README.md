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

逐核指令与数值收据可以分别复核。对已完成的原始目录运行 `python -m calibration.mx_matmul.native_phases /tmp/native-m9-dt` 可列出真正执行 MMAD 或向量指令的核数、指令类型、矩形和描述符时间。安装 matplotlib 后，可用 `python -m calibration.mx_matmul.plot docs/examples/assets/mx-matmul.json /tmp/mx.png` 从保留的时间数据重绘算例。在线预测不依赖绘图库。

若保留了本轮开发 raw，可重算 1.2.0 的阶段参数。先从 [参考快照](../reference/mx-matmul.json) 取 `development.freeze` 存为 `development-freeze.json`，再执行下列命令。原始目录需要与采集时的文件哈希匹配。计算只读取八个开发窗。

```sh
python -m calibration.mx_matmul.fit_phases --freeze development-freeze.json \
    --raw-root runs/mx12 --output /tmp/mx-phases-replay.json
```
