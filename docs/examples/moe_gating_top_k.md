# MoE Gating Top-K · 两行选 16 个专家

FP32 logits[2,896] 和 bias[896] 总计 10,752 B。每行返回 16 个 FP32 权重与 16 个 int32 下标，合计写 256 B。

## 单卡输入与调用

```python
from operator_model import estimate

shape = [2, 896, 16]
attrs = {}
result = estimate('moe_gating_top_k', shape, attrs, chip='Ascend950PR_957b')
print(result.latency_us, result.active_aic, result.active_aiv)
print(result.work["logical_read_bytes"], result.work["logical_write_bytes"])
```

## 从工作量到设备时间

2 个 AIV 各一行。每行 896/64=14 个评分向量组。单组排名、偏置和归一化的固定及增量开销按 DT 已测的 128/896 专家宽度定价，随后换成目标每核 FP32 等效服务能力。

公开工作量统计为 Cube **0**、Vector **9,024** 个有用标量等效操作。逻辑读 **10,752 B**、写 **256 B**。片外实际流量取决于重用与缓存。

把两种宽度的观测换算成参考 tick 后，单行排序约为 139.33+208.33×g，g=⌈E/64⌉。本例 g=14，需要约 3056 tick 的排序服务。根据目标每核 FP32 速率伸缩，额外的两次写和收尾独立计价。

下表取 950PR 的公开阶段分账。各段已处理模型可辨识的重叠，数值保留三位小数。

| 阶段字段 | 暴露时间 µs |
|---|---:|
| `control_and_tiling` | 2.546 |
| `read_ready` | 0.753 |
| `rank_and_normalize` | 1.902 |
| `drain_weights_ids` | 0.189 |
| **整窗** | **5.391** |

## 与完整 CA 核对

| 目标 | 公开预测 µs | 完整 CA µs | 有符号误差 | 预测有效 AIC / AIV |
|---|---:|---:|---:|---:|
| Ascend950DT_9582 | 5.340 | 5.472 | -2.40% | 0 / 2 |
| Ascend950PR_957b | 5.391 | 5.446 | -1.02% | 0 / 2 |
| Ascend970DT（外推） | 4.893 | N/A | N/A | 0 / 2 |

top16 的写量远小于扫描 896 个候选所需的读量。只有两行时，更多 AIV 不会缩短本核排序，控制与首读占据明显比例。

示例收据的调用为 `aclnnMoeGatingTopK`，provider 为 CANN 9.2.1 ops-transformer，实际记录的 kernel 为 `MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005`。

复核时，在逐窗记录中筛选 `contract=moe_gating_top_k`、此页 shape 和属性。有多个前序的样本逐窗保留。示例用于解释该次计算。校准统计、冻结后独立样本数及完整误差范围以总报告为准。[逐窗数据](../../calibration/reference/windows.jsonl) · [模型实现](../../src/operator_model/operators/vector/router.py) · [总建模方法](../modeling-strategy.md) · [误差分析](../error-analysis.md)
