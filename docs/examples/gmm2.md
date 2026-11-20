# GMM2 · 两个专家的下投影

上一级输入宽 F=128，目标 hidden=256，8 行按 [1,7] 分给专家。下投影共 2×8×128×256=524,288 ops，输出 2048 个 BF16 值。

## 单卡输入与调用

```python
from operator_model import estimate

shape = [8, 256, 128, 2]
attrs = {'expert_counts': [1, 7]}
result = estimate('gmm2', shape, attrs, chip='Ascend950PR_957b')
print(result.latency_us, result.active_aic, result.active_aiv)
print(result.work["logical_read_bytes"], result.work["logical_write_bytes"])
```

## 从工作量到设备时间

对专家分别分行，沿 256 个输出列分列，归约深度此时是 128。PR 两个中位模板预测约 3.700、3.703 µs。按同一重叠预算方法得到整窗。

公开工作量统计为 Cube **524,288**、Vector **0** 个有用标量等效操作。逻辑读 **68,656 B**、写 **4,096 B**。片外实际流量取决于重用与缓存。

模板中 CUBE+MTE2 指这段 µs 内两种资源重叠，伸缩时取两项工作比的较大者。GMM2 的输入激活是 MX 量化值，累计与 BF16 输出另计，同一专家列表中 T=1 的块也单独占用调度单元。 [参数及单位](../../src/operator_model/data/parameters/fusion_overlap.json)

下表取 950PR 的公开阶段分账。各段已处理模型可辨识的重叠，数值保留三位小数。

| 阶段字段 | 暴露时间 µs |
|---|---:|
| `gmm2` | 3.701 |
| **整窗** | **3.701** |

## 与完整 CA 核对

| 目标 | 公开预测 µs | 完整 CA µs | 有符号误差 | 预测有效 AIC / AIV |
|---|---:|---:|---:|---:|
| Ascend950DT_9582 | 3.695 | 3.630 | +1.79% | — / — |
| Ascend950PR_957b | 3.701 | 3.663 | +1.05% | 4 / 0 |
| Ascend970DT（外推） | 3.385 | N/A | N/A | 4 / 0 |

核数字段的 “—” 表示该参考路径未返回这一项。

PR 的 17 条 MMAD_MX 分布在 17 个核上。两个模板总时长相近却有不同的 Cube 首发。整体误差小不能单独验证内部分核。

示例收据的调用为 `aclnnGroupedMatmulV5`，provider 为 CANN 9.2.1 ops-transformer，实际记录的 kernel 为 `GroupedMatmul_037cec40685a17f15b75759e999767c3_0`。

复核时，在逐窗记录中筛选 `contract=gmm2`、此页 shape 和属性。有多个前序的样本逐窗保留。示例用于解释该次计算。校准统计、冻结后独立样本数及完整误差范围以总报告为准。[逐窗数据](../../calibration/reference/windows.jsonl) · [模型实现](../../src/operator_model/operators/fused/templates.py) · [总建模方法](../modeling-strategy.md) · [误差分析](../error-analysis.md)
