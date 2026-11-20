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
