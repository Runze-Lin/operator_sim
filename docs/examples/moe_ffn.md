# MoE FFN · 专家 [1,7] 的两段前馈

8 个 token、输入 hidden=256、中间 F=128、2 个专家。上投影 1,048,576 ops，下投影 524,288 ops，有用 Cube 工作合计 1,572,864。

## 单卡输入与调用

```python
from operator_model import estimate

shape = [8, 256, 128, 2]
attrs = {'expert_counts': [1, 7]}
result = estimate('moe_ffn', shape, attrs, chip='Ascend950PR_957b')
print(result.latency_us, result.active_aic, result.active_aiv)
print(result.work["logical_read_bytes"], result.work["logical_write_bytes"])
```

## 从工作量到设备时间

GMM1 输出的 FP8 tensor 和 E8M0 scales 随后作为 GMM2 输入。两次真实调用的库间依赖是串行，PR 收尾按 9.964+3.701 µs 估计。T、E 一样而 expert_counts 不同，可产生不同的尾块。

公开工作量统计为 Cube **1,572,864**、Vector **5,120** 个有用标量等效操作。逻辑读 **205,952 B**、写 **5,152 B**。片外实际流量取决于重用与缓存。

此处两段为顺序调用，PR 上 9.964+3.701≈13.666 µs。每个子窗保留本身的首发和最终退休，系统侧的 host 间隔可由调用方独立加入。

下表取 950PR 的公开阶段分账。各段已处理模型可辨识的重叠，数值保留三位小数。

| 阶段字段 | 暴露时间 µs |
|---|---:|
| `grouped_matmul_swiglu_quant_v2` | 9.964 |
| `gmm2` | 3.701 |
| **整窗** | **13.666** |

### 两个 kernel 怎样组成一次设备估计

在公共接口的 `[T,D,F,E]=[8,256,128,2]` 中，两个专家各有 1、7 行。上投影输出 `2F=256` 列，归约 `D=256`；下投影输出 `D=256` 列，归约 `F=128`。对任一段，记归约长 `k`、输出宽 `n`，活跃专家数 `e_a=2`。两侧每个 FP8 行占 `b(k)=k+2⌈k/64⌉` B，含 E8M0。于是读 `(T+e_a n)b(k)+8E`，Cube 操作为 `2Tnk`。这给出 GMM1 读 137,296 B、1,048,576 ops，GMM2 读 68,656 B、524,288 ops。

GMM1 的门控中间有 `TF=1024` 个元素，共 5,120 个有用 Vector 操作。输出为 1,024 B FP8 和 32 B scale；GMM2 消费这 1,056 B 并写出 4,096 B BF16。所以接口返回读 `137296+68656=205952` B、写 `1056+4096=5152` B。跨 kernel 依赖会使 1,056 B 同时出现在一段的写和下一段的读里。若数据保留在片上，实际 HBM 流量可能更小。

两段各有 `(⌈1/16⌉+⌈7/16⌉)⌈256/128⌉=4` 个候选 Cube 任务。只有上投影另有 `⌈128/256⌉+⌈896/256⌉=5` 个 Vector 任务。各段的关键波都为一波，但首段有 4 个 K64 单位，下段为 2 个。分别用这些工作单位与各自的 DT 模板单位相比，并按 PR 每核峰值和带宽预算重定价。应先在每个组件内取最慢核、再对其四个候选取中位数，因为两次调用各有独立的启动与退休。

| 组件 | 候选窗按从短到长排列（µs） | 整窗的贡献 |
|---|---|---:|
| GMM1，FP8 上投影、SwiGLU、量化 | 8.69526，8.92982，10.99889，11.13200 | `(8.92982+10.99889)/2=9.96436` µs |
| GMM2，FP8 下投影、BF16 输出 | 3.55058，3.69999，3.70293，4.36893 | `(3.69999+3.70293)/2=3.70146` µs |

GMM2 依赖 GMM1 的量化结果，故设备工作之和为 **13.66582 µs**。取两个 CA 窗分别约 7.393 和 3.663 µs，对应误差约 +2.571 和 +0.038 µs；合计高估约 2.610 µs。这里无法识别实际系统的 host 间隙、专家通信或不同流的并发。两个子算例给出如何从每一组实测重叠区间、当前工作单位和资源倍率算出表里的候选窗。[GMM1 完整推导](grouped_matmul_swiglu_quant_v2.md) · [GMM2 完整推导](gmm2.md) · [串行组合实现](../../src/operator_model/operators/fused/templates.py)

### 串行组合是否还需要拟合一个开销

组合器的两个权数都是 1，来自数据依赖。早期的 950DT 开发几何 [2,64,64,1]、[2,64,128,1]、[4,128,64,2]、[6,128,128,2] 各含两份独立的 kernel 指令数据库。按顺序取各 stage 的最后退休代表核，对同一 kernel 的重叠只算一次。以第一种形状（1 个专家，2 个 token）为例，GMM1 的三类核结束于 11558、11382、11112 ticks，得到 11558/1650=7.00485 µs；GMM2 的末核为 5359 ticks，即 3.24788 µs。按两窗和的统一口径，实测设备工作为 (11558+5359)/1650=10.25273 µs。两份数据库的指纹分别以 `13cae4827228`、`5d807ce297ef` 开头。其余三对样本同样分别进入两个组件的模板训练。跨 kernel 数据必须先写后读，接口默认的 host gap 为零；这些各自归零的 CA 窗无法校准真实应用中相邻 kernel 之间的 host 间隔。前文 PR 输出用各组件先完成资源重定价与中位聚合的窗，最后相加。[含 shape 和数据库指纹的冻结模板](https://github.com/Runze-Lin/operator_sim/blob/e99f2eb6e645de0e6e187379ba7f9d89d05d3297/results/v0.8/fusion/model.json) · [按指令边界提取不重叠区间](https://github.com/Runze-Lin/operator_sim/blob/e99f2eb6e645de0e6e187379ba7f9d89d05d3297/scripts/fusion-v8-freeze.py)

## 与完整 CA 核对

| 目标 | 公开预测 µs | 完整 CA µs | 有符号误差 | 预测有效 AIC / AIV |
|---|---:|---:|---:|---:|
| Ascend950DT_9582 | 13.636 | 11.119 | +22.63% | — / — |
| Ascend950PR_957b | 13.666 | 11.056 | +23.61% | 4 / 5 |
| Ascend970DT（外推） | 12.490 | N/A | N/A | 4 / 5 |

核数字段的 “—” 表示该参考路径未返回这一项。

比较时分别检查第一段的量化输出与 scale、第二段的 BF16 输出。在这组形状，PR 的主要误差来自 GMM1，第一段高估不会被第二段较小的偏差消除。

示例收据的调用为 `aclnnGroupedMatmulSwigluQuantV2 + aclnnGroupedMatmulV5`，provider 为 CANN 9.2.1 ops-transformer，实际记录的 kernel 为 `GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic`、`GroupedMatmul_037cec40685a17f15b75759e999767c3_0`。

复核时，在逐窗记录中筛选 `contract=moe_ffn`、此页 shape 和属性。有多个前序的样本逐窗保留。示例用于解释该次计算。校准统计、冻结后独立样本数及完整误差范围以总报告为准。[逐窗数据](../../calibration/reference/windows.jsonl) · [模型实现](../../src/operator_model/operators/fused/templates.py) · [总建模方法](../modeling-strategy.md) · [误差分析](../error-analysis.md)
