# DynamicMXQuant · 一行为什么会分到五个 AIV

9 行，每行 5120 个 BF16 值，每组 32 个值，得到 1440 个量化组。读 9×5120×2=92,160 B。量化输出 46,080 B，scale 输出 9×2×⌈5120/64⌉=1440 B。

## 单卡输入与调用

```python
from operator_model import estimate

shape = [9, 5120]
attrs = {}
result = estimate('quant', shape, attrs, chip='Ascend950PR_957b')
print(result.latency_us, result.active_aic, result.active_aiv)
print(result.work["logical_read_bytes"], result.work["logical_write_bytes"])
```

## 从工作量到设备时间

950DT/PR 各使用 45 个 AIV。每条输入行沿列切成 5 片，每个核承担 1024 元素和 32 个量化组，UB 能一次容纳。取绝对值 46,080 次，求组最大值需 46,080−1440 次比较。输入就绪后才可算 scale，量化输出和 scale 走各自写路径。

公开工作量统计为 Cube **0**、Vector **90,720** 个有用标量等效操作。逻辑读 **92,160 B**、写 **47,520 B**。片外实际流量取决于重用与缓存。

PR 这条资源模型的先验来自 DT 局部 DMA 观测。参考首读固定响应约 972.53 tick，响应的数据率为 48.36 B/tick，连续发起的数据率为 68.91 B/tick。前者决定数据何时可算，后者限制第二个缓冲槽最早何时发起。目标若给出 MTE2 端口可替换参考持续速率。 [参数及单位](../../src/operator_model/data/parameters/quant_resources.json)

下表取 950PR 的公开阶段分账。各段已处理模型可辨识的重叠，数值保留三位小数。

| 阶段字段 | 暴露时间 µs |
|---|---:|
| `dispatch` | 3.336 |
| `ub_pipeline` | 1.031 |
| `drain` | 0.010 |
| **整窗** | **4.377** |

## 与完整 CA 核对

| 目标 | 公开预测 µs | 完整 CA µs | 有符号误差 | 预测有效 AIC / AIV |
|---|---:|---:|---:|---:|
| Ascend950DT_9582 | 4.980 | 4.867 | +2.31% | 0 / 45 |
| Ascend950PR_957b | 4.377 | 5.025 | -12.90% | 0 / 45 |
| Ascend970DT（外推） | 7.084 | N/A | N/A | 0 / 100 |

两端都算对活动核数，PR 仍较慢。首批读的返回、tile 内控制及多个核写邻近 scale 的竞争，对短形的影响与纯字节带宽同量级。换更新芯片时，活动核数反而可能增加短形的派发成本。

示例收据的调用为 `aclnnDynamicMxQuant`，provider 为 CANN 9.2.1 ops-nn，实际记录的 kernel 为 `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv`。

## 一个 AIV 连续处理五个 UB tile 时

对 [2048,5120]，950PR 启用 56 个 AIV，最忙核心分得 74 行、每行 2560 列。一个完整 tile 是 15×2560，前四块均为满块，第五块处理 14 行。第一块读 76,800 B，再产生 38,400 B 量化输出和 1200 B scale。下图的竖刻度表示相应指令发起，较长的底色包含等待。图中末核首次有效读在 3.532 µs 发起、7.760 µs 退休。其间可向第二个 UB 槽预取。整窗预测 17.834 µs，CA 为 18.504 µs，低估 3.62%。

![五个 UB tile 的模型与真实 CA 流水](assets/quant.png)

两栏使用同一设备零点与每块的行列数，便于检查偏差来自首次读、稳态还是最后写回。 [图上使用的数值](assets/quant.json)

复核时，在逐窗记录中筛选 `contract=quant`、此页 shape 和属性。有多个前序的样本逐窗保留。示例用于解释该次计算。校准统计、冻结后独立样本数及完整误差范围以总报告为准。[逐窗数据](../../calibration/reference/windows.jsonl) · [模型实现](../../src/operator_model/operators/quant/resources.py) · [总建模方法](../modeling-strategy.md) · [误差分析](../error-analysis.md)
