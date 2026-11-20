# AttnResFwd · 带 bank 写回的两行

本例 [T,D,S,N]=[2,4096,8,1]。对两行 4096 维向量评分并混合，S=8 是 bank 容量，实际仅 N=1 份缓存残差有效。materialize=True、write_idx=1 产生独立输出并写入指定 bank。

## 单卡输入与调用

```python
from operator_model import estimate

shape = [2, 4096, 8, 1]
attrs = {'addend': True, 'materialize': True, 'write_idx': 1}
result = estimate('attn_res_fwd', shape, attrs, chip='Ascend950PR_957b')
print(result.latency_us, result.active_aic, result.active_aiv)
print(result.work["logical_read_bytes"], result.work["logical_write_bytes"])
```

## 从工作量到设备时间

同一轮活跃 2 个 AIV，各做一行。每行 4096/64=64 个向量组。调度含公共权重读取、前缀加与存储、bank 评分及加权归一化，实际返回 Y、prefix 和 materialized 合同决定读写次数。

公开工作量统计为 Cube **0**、Vector **163,866** 个有用标量等效操作。逻辑读 **73,728 B**、写 **65,536 B**。片外实际流量取决于重用与缓存。

校准把读权重和计算得分时段展开成固定项加 64 通道向量组项。例如 bank_scores 的参考价格为 918.05 tick 加每组 5.84 tick，weighted_norm_write 为 1486.71 tick 加每组 49.21 tick。有效残差数和输出合同决定参与各段的组数。 [参数及单位](../../src/operator_model/data/parameters/attention_residual.json)

下表取 950PR 的公开阶段分账。各段已处理模型可辨识的重叠，数值保留三位小数。

| 阶段字段 | 暴露时间 µs |
|---|---:|
| `dispatch` | 2.744 |
| `score_weights` | 1.436 |
| `prefix_add_write` | 1.070 |
| `bank_scores` | 0.892 |
| `weighted_norm_write` | 2.502 |
| `exposed_transfer_floor` | 0.000 |
| **整窗** | **8.643** |

## 与完整 CA 核对

| 目标 | 公开预测 µs | 完整 CA µs | 有符号误差 | 预测有效 AIC / AIV |
|---|---:|---:|---:|---:|
| Ascend950DT_9582 | 8.603 | 8.113 | +6.04% | 0 / 2 |
| Ascend950PR_957b | 8.643 | 7.984 | +8.24% | 0 / 2 |
| Ascend970DT（外推） | 7.885 | N/A | N/A | 0 / 2 |

本例的实际 tiling key 是 40020。Kimi 驻留 prefill 合同 materialize=False、write_idx=−1 改变输出别名、写量和分核，应另查该合同的直接 CA 资格。

示例收据的调用为 `aclnnAttnResFwd`，provider 为 vLLM Ascend custom OPP，实际记录的 kernel 为 `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020`。

复核时，在逐窗记录中筛选 `contract=attn_res_fwd`、此页 shape 和属性。有多个前序的样本逐窗保留。示例用于解释该次计算。校准统计、冻结后独立样本数及完整误差范围以总报告为准。[逐窗数据](../../calibration/reference/windows.jsonl) · [模型实现](../../src/operator_model/operators/attention/residual_timing.py) · [总建模方法](../modeling-strategy.md) · [误差分析](../error-analysis.md)
