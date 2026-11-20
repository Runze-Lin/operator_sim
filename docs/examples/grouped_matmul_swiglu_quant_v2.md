# GMM1 · 两个专家的上投影和量化

8 个 token 分配为 [1,7]。hidden K=256，中间宽 F=128。两路上投影合计 4×8×256×128=1,048,576 算术操作，随后做 SwiGLU 并写 1024 个 FP8 值、32 个 scale。

## 单卡输入与调用

```python
from operator_model import estimate

shape = [8, 256, 128, 2]
attrs = {'expert_counts': [1, 7]}
result = estimate('grouped_matmul_swiglu_quant_v2', shape, attrs, chip='Ascend950PR_957b')
print(result.latency_us, result.active_aic, result.active_aiv)
print(result.work["logical_read_bytes"], result.work["logical_write_bytes"])
```

## 从工作量到设备时间

每个专家独立生成行块。参考阶段按两组 16 行粒度尾块定价，再与向量激活和量化重叠。PR 的两个中位候选排程分别为 8.930 和 10.999 µs，默认时间取两者均值。

公开工作量统计为 Cube **1,048,576**、Vector **5,120** 个有用标量等效操作。逻辑读 **137,296 B**、写 **1,056 B**。片外实际流量取决于重用与缓存。

参数表为每一参考 shape、每一种核及同时活跃的队列集合记录持续时间（µs）。按该块实际的 Cube、搬运和 SwiGLU/量化工作比伸缩各个区间，再取最慢核。中间两份模板的不同首发位置在图中保留。 [参数及单位](../../src/operator_model/data/parameters/fusion_overlap.json)

下表取 950PR 的公开阶段分账。各段已处理模型可辨识的重叠，数值保留三位小数。

| 阶段字段 | 暴露时间 µs |
|---|---:|
| `grouped_matmul_swiglu_quant_v2` | 9.964 |
| **整窗** | **9.964** |

## 与完整 CA 核对

| 目标 | 公开预测 µs | 完整 CA µs | 有符号误差 | 预测有效 AIC / AIV |
|---|---:|---:|---:|---:|
| Ascend950DT_9582 | 9.941 | 7.489 | +32.74% | — / — |
| Ascend950PR_957b | 9.964 | 7.393 | +34.79% | 4 / 5 |
| Ascend970DT（外推） | 9.104 | N/A | N/A | 4 / 5 |

核数字段的 “—” 表示该参考路径未返回这一项。

PR 全核实测只有 2 个 AIC 各发 4 条 MMAD_MX，两个参考模板的 Cube 首发都较晚。这解释了本例明显高估的一个来源，也说明核数预测只是使用范围的近似。

示例收据的调用为 `aclnnGroupedMatmulSwigluQuantV2`，provider 为 CANN 9.2.1 ops-transformer，实际记录的 kernel 为 `GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic`。

![两个参考模板与两类核心的实际占用](assets/gmm1.png)

两条上栏模板各自保留资源重叠。虚线是二者得到的公开点估计，尚无依据将它视为唯一物理排程。 [图上使用的数值](assets/gmm1.json)

复核时，在逐窗记录中筛选 `contract=grouped_matmul_swiglu_quant_v2`、此页 shape 和属性。有多个前序的样本逐窗保留。示例用于解释该次计算。校准统计、冻结后独立样本数及完整误差范围以总报告为准。[逐窗数据](../../calibration/reference/windows.jsonl) · [模型实现](../../src/operator_model/operators/fused/templates.py) · [总建模方法](../modeling-strategy.md) · [误差分析](../error-analysis.md)
