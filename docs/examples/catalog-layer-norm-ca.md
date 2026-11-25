# Indexer LayerNorm 的 950DT / 950PR 校准对照

GLM-5.3 Flash 将投影输出分割为 key 与权重，key 的逻辑 shape 为 [R,128]，物理行步长为 160 个 BF16 元素。gamma 与 beta 各为 128 个 BF16，输出仅为连续 BF16 的 Y。源捕获声明 epsilon=1e-6。实验用 CANN 9.2.1 的 `aclnnLayerNorm`，显式传入同一行步长、同精度的输入、gamma、beta 和空的可选 mean/rstd 输出。探针给行尾填充不同值，并按行变换输入符号模式，逐元素核对 Y。DeepSeek 只有在上述合同也匹配时才可复用这组对照。

设备模型将每行分成两个规约和一遍仿射，计每行约 8D+3 个 Vector 运算，其中 D 是最后一维宽度。默认 VF 方法把读取、两次规约、倒平方根、乘加与输出写回折为指令类发起数。例中 D=128、9 行，scale 和 bias 的逻辑读为 2×128×2=512 B；X 和 Y 各为 9×128×2=2304 B。公式取实际的行数和宽度，当前模型未单列跨步输入物化的启动。预测在采集前冻结。实际生产宽度 128 选取 decode 1 行、推测校验 9 行和 prefill 512 行。额外比较 512 行连续输入与 32 行、宽 256 的设计敏感性，后一项是合成形状。

每个 CA 窗要求所有已启动核出现 END 并与 profiler 的逐核时间一致，设备时间取最后一个核退休减最早一个核首次发起；每个 ELF 须与已安装 OPP 对象的 SHA-256 一致。多 kernel 调用按顺序累加各自活跃窗，这个数不包含 kernel 间启动空隙及 host API 耗时。计算意义与完整服务延迟不同。

[冻结的 10 次逐模式预测](../../results/v2.4/layer-norm-freeze.json) · [ACLNN 探针与张量步长](../../calibration/catalog_ca/layer_norm.cpp) · [采集命令实现](../../calibration/catalog_ca/layer_norm.py) · [逐核校验实现](../../calibration/catalog_ca/layer_norm_receipt.py)

## 冻结预测对比

| SKU | 行×宽 / 行步长 | 自动预测 µs | Slice µs | LayerNormV4 µs | 设备窗合计 µs | 有符号误差 |
|---|---|---:|---:|---:|---:|---:|
| 950DT | 1×128 / 160 | 2.689 | 无此启动 | 2.958 | 2.958 | -9.07% |
| 950PR | 1×128 / 160 | 2.689 | 无此启动 | 2.754 | 2.754 | -2.35% |
| 950DT | 9×128 / 160 | 2.684 | 2.686 | 3.042 | 5.728 | -53.15% |
| 950PR | 9×128 / 160 | 2.684 | 2.441 | 2.975 | 5.416 | -50.45% |
| 950DT | 512×128 / 160 | 2.795 | 3.468 | 3.573 | 7.042 | -60.31% |
| 950PR | 512×128 / 160 | 2.854 | 3.555 | 3.174 | 6.728 | -57.58% |
| 950DT | 512×128 / 128 | 2.795 | 无此启动 | 4.470 | 4.470 | -37.49% |
| 950PR | 512×128 / 128 | 2.854 | 无此启动 | 4.125 | 4.125 | -30.81% |
| 950DT | 32×256 / 288 | 2.699 | 2.680 | 5.440 | 8.120 | -66.76% |
| 950PR | 32×256 / 288 | 2.699 | 2.591 | 5.420 | 8.011 | -66.31% |

误差 = 100 ×（预测/设备窗合计 − 1）。Slice 只在需要物化跨步输入时出现。一次行宽为 128 的单行输入即使带有 160 的行步长，也能直接传入 LayerNormV4。此表按已知 ABI、dtype、形状和物理布局计分；正确性核对的是声明的唯一输出 Y。

- Ascend950DT_9582 的 5 个窗口，默认 WAPE 51.76%，最大绝对误差 66.76%；仅 3 个实际捕获布局时 WAPE 为 48.07%。
- Ascend950PR_957b 的 5 个窗口，默认 WAPE 49.03%，最大绝对误差 66.31%；仅 3 个实际捕获布局时 WAPE 为 44.78%。

预测在多行时漏记了 Slice 窗口。连续 512 行对照仍有误差，说明剩余差距还包含归一化内核本身的阶段定价。相同行宽下，物化后的 LayerNormV4 与直接连续输入的运行窗也会不同，已有窗口无法独立识别缓存初态与核调度各自的影响。没有把窗口用于回调参数。
10 个合格窗口的离线 CA 墙钟和为 1857.8 秒。 部分首次受共享主机竞争触发 5 分钟保护，改为低并发的多 kernel 采集并沿用冻结预测。失败尝试未计入该成功窗口费用。
[实际对象、各核时标摘要、预测与数值核对](../../results/v2.4/layer-norm-qualified.json)

![Indexer LayerNorm 两芯片实测分段与冻结预测](../figures/catalog-layer-norm-ca.png)

条长为实际 Slice 和 LayerNormV4 各自核窗之和；黑线为冻结预测。相同的 512×128 输入仅改变行步长，安装包的核序列就发生了变化。

## 复现

在已安装 CANN 9.2.1、支持 950DT/PR 的 CA 包和已加载 OPP 的主机上，先激活 CANN 环境，再从仓库根目录运行下列命令。示例变量 `CANN_ARCH_DIR` 指向带 `include/`、`lib64/` 的 CANN 架构目录。`--workers 1` 适合共享主机。如采集环境另由 shell 脚本激活，可将 `OPERATOR_SIM_CANN_LAUNCHER` 设为该脚本的绝对路径，脚本应将参数作为命令执行。使用发布的冻结文件，原始输出目录请选新的本地路径。

~~~bash
g++ -O2 -std=c++17 -I"$CANN_ARCH_DIR/include" calibration/catalog_ca/layer_norm.cpp \
    -L"$CANN_ARCH_DIR/lib64" -Wl,--allow-shlib-undefined -lascendcl -lopapi -lnnopbase -o /tmp/layer_norm_probe
PYTHONPATH=src:. python -m calibration.catalog_ca.layer_norm collect \
    --raw /tmp/layer_norm_frozen --probe /tmp/layer_norm_probe --workers 1
PYTHONPATH=src:. python -m calibration.catalog_ca.layer_norm_receipt \
    --raw /tmp/layer_norm_frozen \
    --opp "$ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/kernel/ascend950" \
    --out /tmp/layer_norm_verified.json
~~~
