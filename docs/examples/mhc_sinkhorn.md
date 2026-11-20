# mHC Sinkhorn · 4097 张四阶矩阵

4097 张 4×4 的 FP32 矩阵，4 次行列交替迭代，逻辑读写各 4097×16×4=262,208 B，标量除法等效工作 2×4×4097×16=524,416 次。

## 单卡输入与调用

```python
from operator_model import estimate

shape = [4097, 4]
attrs = {'num_iters': 4}
result = estimate('mhc_sinkhorn', shape, attrs, chip='Ascend950PR_957b')
print(result.latency_us, result.active_aic, result.active_aiv)
print(result.work["logical_read_bytes"], result.work["logical_write_bytes"])
```

## 从工作量到设备时间

PR 每核最多 ⌈4097/56⌉=74 张，至多分成 10 个寄存器组，每个迭代函数内部遍历这些组。首个 softmax、每轮两次归一化与最后输出依次处理。对同一张矩阵不可把四次迭代并行。

公开工作量统计为 Cube **0**、Vector **1,278,264** 个有用标量等效操作。逻辑读 **262,208 B**、写 **262,208 B**。片外实际流量取决于重用与缓存。

按首次、扩核和重用三种前序分别记录从 tiling 发起到输入、从 softmax 到每次行列归一以及写回的参考 tick。FP32 的单核基准为 255.68 ops/tick，用来缩放能归属向量服务的部分。发起控制另计。 [参数及单位](../../src/operator_model/data/parameters/sinkhorn_history.json)

下表取 950PR 的公开阶段分账。各段已处理模型可辨识的重叠，数值保留三位小数。

| 阶段字段 | 暴露时间 µs |
|---|---:|
| `tiling_and_setup` | 1.833 |
| `input_and_softmax_issue` | 0.412 |
| `first_column` | 2.502 |
| `normalization_loop` | 3.418 |
| `scatter_to_output_issue` | 0.400 |
| `output_and_retirement` | 0.250 |
| **整窗** | **8.815** |

## 与完整 CA 核对

| 目标 | 公开预测 µs | 完整 CA µs | 有符号误差 | 预测有效 AIC / AIV |
|---|---:|---:|---:|---:|
| Ascend950DT_9582 | 8.105 | 8.964 | -9.58% | 0 / 64 |
| Ascend950PR_957b | 8.815 | 9.112 | -3.26% | 0 / 56 |
| Ascend970DT（外推） | 5.546 | N/A | N/A | 0 / 118 |

归档的设备调用此前有 DT57、PR52 个 AIV 的同族高水位。上表模型按调用方省略 resident_aiv 计算。生产常用的 20 次迭代是另一条输入合同。

示例收据的调用为 `aclnnMhcSinkhorn`，provider 为 CANN 9.2.1 ops-transformer，实际记录的 kernel 为 `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0`。

复核时，在逐窗记录中筛选 `contract=mhc_sinkhorn`、此页 shape 和属性。有多个前序的样本逐窗保留。示例用于解释该次计算。校准统计、冻结后独立样本数及完整误差范围以总报告为准。[逐窗数据](../../calibration/reference/windows.jsonl) · [模型实现](../../src/operator_model/operators/mhc/sink_pipeline.py) · [总建模方法](../modeling-strategy.md) · [误差分析](../error-analysis.md)
