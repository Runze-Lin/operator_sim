# Sigmoid · Kimi 八个 128Ki 请求的 beta

Kimi 的 96 个 KDA 头在 attention TP8 下每卡有 12 个。8×131072=1,048,576 行。FP32 输入、输出各 12,582,912×4=48 MiB。

## 单卡输入与调用

```python
from operator_model import estimate

shape = [1048576, 12]
attrs = {}
result = estimate('sigmoid_fp32', shape, attrs, chip='Ascend950PR_957b')
print(result.latency_us, result.active_aic, result.active_aiv)
print(result.work["logical_read_bytes"], result.work["logical_write_bytes"])
```

## 从工作量到设备时间

PR 按 56 个 AIV 分发，关键核有 224,768 个元素。UB 单块 15,872 个元素，需要 15 块。每 64 个元素是一组向量工作，分别考虑首读响应、向量等待、写回，以及后续块与第一块的重叠。

公开工作量统计为 Cube **0**、Vector **50,331,648** 个有用标量等效操作。逻辑读 **50,331,648 B**、写 **50,331,648 B**。片外实际流量取决于重用与缓存。

参考 Vector 服务为 79+4(g−1) tick，g=⌈n/64⌉。首组 79 tick、后继每组 4 tick 来自同一条向量函数的发起与连续执行。首读还包括 1159 tick 固定响应，写回有 159 tick。目标共享带宽和缓冲复用可比本地计算更慢。

下表取 950PR 的公开阶段分账。各段已处理模型可辨识的重叠，数值保留三位小数。

| 阶段字段 | 暴露时间 µs |
|---|---:|
| `front` | 2.505 |
| `first_read` | 2.540 |
| `first_vector` | 0.664 |
| `first_write_and_flags` | 0.715 |
| `exposed_following_tiles` | 33.207 |
| **整窗** | **39.630** |

## 与完整 CA 核对

| 目标 | 公开预测 µs | 完整 CA µs | 有符号误差 | 预测有效 AIC / AIV |
|---|---:|---:|---:|---:|
| Ascend950DT_9582 | 19.720 | 18.553 | +6.29% | 0 / 64 |
| Ascend950PR_957b | 39.630 | 34.285 | +15.59% | 0 / 56 |
| Ascend970DT（外推） | 11.321 | N/A | N/A | 0 / 120 |

PR 末核首笔有效读在约 1.906 µs 发起，模型给出 2.505 µs。随后满块读间隔的实测中位值约 2.21 µs，模型 2.540 µs。整窗误差随循环累积。

示例收据的调用为 `aclnnSigmoid`，provider 为 CANN 9.2.1 installed OPP，实际记录的 kernel 为 `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100`。

复核时，在逐窗记录中筛选 `contract=sigmoid_fp32`、此页 shape 和属性。有多个前序的样本逐窗保留。示例用于解释该次计算。校准统计、冻结后独立样本数及完整误差范围以总报告为准。[逐窗数据](../../calibration/reference/windows.jsonl) · [模型实现](../../src/operator_model/operators/vector/sigmoid.py) · [总建模方法](../modeling-strategy.md) · [误差分析](../error-analysis.md)
