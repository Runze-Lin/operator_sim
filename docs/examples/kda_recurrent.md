# Recurrent KDA · 单链递推三步

[B,S,Hq,Hv,K,V]=[1,3,1,1,128,128]。一个 FP32 状态矩阵占 128×128×4=65,536 B，入口与最终状态各一份。3 个 token 都要更新这张矩阵。

## 单卡输入与调用

```python
from operator_model import estimate

shape = [1, 3, 1, 1, 128, 128]
attrs = {'chunk_size': None,
 'dtype': 'BF16',
 'export_gk': False,
 'gate_dtype': 'FP32',
 'layout': 'BSND',
 'output_final_state': True,
 'state_dtype': 'FP32',
 'state_v_first': False,
 'use_gate_in_kernel': False}
result = estimate('kda_recurrent', shape, attrs, chip='Ascend950PR_957b')
print(result.latency_us, result.active_aic, result.active_aiv)
print(result.work["logical_read_bytes"], result.work["logical_write_bytes"])
```

## 从工作量到设备时间

一个 AIV 保持同一条状态链，每步依次处理键/门、状态半片、秩一更新与输出，第三步必须等待第二步的状态。预测用关键核 SIMD、MTE2、MTE3 与标量控制的互斥成本分账。

公开工作量统计为 Cube **0**、Vector **344,832** 个有用标量等效操作。逻辑读 **69,388 B**、写 **66,304 B**。片外实际流量取决于重用与缓存。

单链用工作单位 u=S×K×V/16384，本例为 3。参考时钟上的每段价格由启动 µs 加单位工作量的斜率给出，DMA 段还乘 log₂(1+活动核数)。阶段是关键核时间轴上互斥的区间。 [参数及单位](../../src/operator_model/data/parameters/vector_instructions.json)

下表取 950PR 的公开阶段分账。各段已处理模型可辨识的重叠，数值保留三位小数。

| 阶段字段 | 暴露时间 µs |
|---|---:|
| `arrival` | 0.000 |
| `SIMD` | 13.471 |
| `MTE3` | 1.590 |
| `MTE2` | 1.497 |
| `SCALAR` | 4.303 |
| `other` | 1.526 |
| **整窗** | **22.388** |

## 与完整 CA 核对

| 目标 | 公开预测 µs | 完整 CA µs | 有符号误差 | 预测有效 AIC / AIV |
|---|---:|---:|---:|---:|
| Ascend950DT_9582 | 22.022 | 21.932 | +0.41% | — / 1 |
| Ascend950PR_957b | 22.388 | 21.221 | +5.50% | 0 / 1 |
| Ascend970DT（外推） | 20.169 | N/A | N/A | 0 / 1 |

核数字段的 “—” 表示该参考路径未返回这一项。

按 token 提升并行度会违反状态依赖。此形只验证一个状态链和显式 FP32 gate/状态。Kimi 的多本地头与分页 gather 要另算。

示例收据的调用为 `aclnnRecurrentKda`，provider 为 CANN 9.2.1 ops-transformer，实际记录的 kernel 为 `RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0`。

复核时，在逐窗记录中筛选 `contract=kda_recurrent`、此页 shape 和属性。有多个前序的样本逐窗保留。示例用于解释该次计算。校准统计、冻结后独立样本数及完整误差范围以总报告为准。[逐窗数据](../../calibration/reference/windows.jsonl) · [模型实现](../../src/operator_model/operators/vector/instructions.py) · [总建模方法](../modeling-strategy.md) · [误差分析](../error-analysis.md)
