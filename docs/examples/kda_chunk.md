# Chunk KDA · 64+64+17 的三个块

[1,145,2,2,128,128] 令每头依次处理 64、64、17 个 token。两个头共 6 个块头任务，每头三块的合法下三角对数为 2080+2080+153=4313。

## 单卡输入与调用

```python
from operator_model import estimate

shape = [1, 145, 2, 2, 128, 128]
attrs = {'implementation': 'fla_26_9_1_v2'}
result = estimate('kda_chunk', shape, attrs, chip='Ascend950PR_957b')
print(result.latency_us, result.active_aic, result.active_aiv)
print(result.work["logical_read_bytes"], result.work["logical_write_bytes"])
```

## 从工作量到设备时间

显式 FLA V2 的 Prepare、FwdH、Finalize 串行发出，分别对块间并行工作、同一头的带状态递推、输出恢复计价。FwdH 中各头可以并行，同头的 17-token 尾块依赖前两个整块产生的状态。

公开工作量统计为 Cube **45,877,248**、Vector **233,728** 个有用标量等效操作。逻辑读 **430,224 B**、写 **279,552 B**。片外实际流量取决于重用与缓存。

同一数据搬运同时有固定响应与持续成本。例如参考 GM→L1 首响应为 391.14 tick，数据率 47.15 B/tick。VF 价格使用短块与整块的向量指令数估计长度 17 的尾块。Prepare 的增长还包含每个 16 行带的有效因果前缀。 [参数及单位](../../src/operator_model/data/parameters/kda_fla.json)

下表取 950PR 的公开阶段分账。各段已处理模型可辨识的重叠，数值保留三位小数。

| 阶段字段 | 暴露时间 µs |
|---|---:|
| `chunk_kda_fwd_prepare` | 19.854 |
| `chunk_fwd_h` | 11.701 |
| `chunk_kda_fwd_finalize` | 3.759 |
| **整窗** | **35.313** |

## 与完整 CA 核对

| 目标 | 公开预测 µs | 完整 CA µs | 有符号误差 | 预测有效 AIC / AIV |
|---|---:|---:|---:|---:|
| Ascend950DT_9582 | 35.185 | 34.586 | +1.73% | 6 / 12 |
| Ascend950PR_957b | 35.313 | 35.012 | +0.86% | 6 / 12 |
| Ascend970DT（外推） | 32.246 | N/A | N/A | 6 / 12 |

CA 合格性要求三个完整主核、全量 attention 输出、Aqk/Akk 及最终状态通过。145 长度是显式三段实现验证。部署中某个短形若走单 kernel 分支，不能沿用三段 CA。

示例收据的调用为 `aclnnChunkKdaFwdV2`，provider 为 FLA 26.9.1 V2，实际记录的 kernel 为 `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic`、`ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic`、`ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0`。

复核时，在逐窗记录中筛选 `contract=fla_kda_v2`、此页 shape 和属性。有多个前序的样本逐窗保留。示例用于解释该次计算。校准统计、冻结后独立样本数及完整误差范围以总报告为准。[逐窗数据](../../calibration/reference/windows.jsonl) · [模型实现](../../src/operator_model/operators/kda/fla.py) · [总建模方法](../modeling-strategy.md) · [误差分析](../error-analysis.md)
