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

**按一个位置展开。** 状态衰减 $KV$ 次乘，读取状态得到预测 $2KV$，沿当前键作秩一更新 $2KV$，查询更新后的状态 $2KV$；修正和缩放 value 另计 $2V$。按 FMA=2 的运算量约定，三步合计 $3(7×128×128+2×128)=344832$。该数覆盖主体递推。输入包括 BF16 的 Q、K、V（$2×3×(2×128+128)=2304$ B）、FP32 预激活 gate 和 beta（$3(4×128+4)=1548$ B）及初始状态（65536 B），共 69388 B。返回 BF16 输出 $2×3×128=768$ B 和最终状态 65536 B，共 66304 B。[状态与递推口径](../../src/operator_model/extensions/kda_moe.py)

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

## 本例各阶段怎样定价

输入 [1,3,1,1,128,128] 是一条 FP32 状态链的三个串行更新。16,384 个状态元素作为单位，归一化工作 x=S×K×V/16384=3。950DT 的每段预算是 a+b×x；arrival 使用 log₂(活跃核)，搬运使用 x×log₂(1+活跃核)。本例只有一个核心，后者仍为 3。a 为 950DT 的阶段固定 µs，b 为每单位增量 µs，以有完整窗口的单链样本分段拟合。下表按目标每核 SIMD、传输或时钟倍率换算，展示参数可辨识的粒度。

| 阶段 | (a+b×单位数)×PR 资源倍率，单位 µs |
|---|---|
| arrival | 0 |
| SIMD | (4.302424+2.937403×3)×1.027174=13.47101 |
| MTE3 | (0.241212+0.448225×3)×1.002861=1.59042 |
| MTE2 | (1.414848+0.0260173×3)×1.002861=1.49717 |
| SCALAR | (4.111818+0.0637662×3)×1=4.30312 |
| other | 1.525859×1=1.52586 |

合计 **22.38758 µs**。各段是关键核独占成本的近似，未能从它们反解每个 recurrent 除法或状态描述符的价格。多状态链缺少相同计时资格。

这组系数可从三份 DT 开发窗独立核对。选择最晚结束的 AIV，先取到达偏移，再依次把重叠区间归到 SIMD、MTE3、MTE2、SCALAR，剩余时间归 other。取 $S=1,2,4$ 的阶段时长如下，单位 µs。对每行以 $x=S$ 作非负直线拟合 $a+bx$；例如 SIMD 的 $b=\sum(x-\bar x)(y-\bar y)/\sum(x-\bar x)^2=2.937403$，$a=\bar y-b\bar x=4.302424$。other 的原始趋势向下，非负斜率取 0，截距取三个时长的平均。

| 阶段 | $S=1$ | $S=2$ | $S=4$ |
|---|---:|---:|---:|
| arrival | 0 | 0 | 0 |
| SIMD | 7.29333 | 10.09697 | 16.07879 |
| MTE3 | 0.74121 | 1.06000 | 2.06000 |
| MTE2 | 1.52788 | 1.33636 | 1.56242 |
| SCALAR | 4.02485 | 4.46545 | 4.29152 |
| other | 1.63212 | 1.50848 | 1.43697 |

[单链形状合同与工作量](../../src/operator_model/operators/vector/instructions.py) · [目标端定价](../../src/operator_model/scheduling/projection.py) · [阶段截距、单位价格](../../src/operator_model/data/parameters/vector_instructions.json)

**PR 倍率如何得到。** 单链两端均一波，标量倍率 $1.65/1.65=1$，每核 FP32 Vector 时间比 $(27/64)/(23/56)=1.027174$。搬运比例是投影模型对本地端口和片外带宽的线性混合。令 $p=(69388+66304)/(4000×1000×22.022395)\approx0.00154$，分母是 DT 以 HBM 峰值持续服务整个参考窗能传的字节数，$p$ 是逻辑字节压力代理。DT/PR 没有独立 MTE 端口价，本地比例回退到时钟和波数（本例为 1）。因此 $f_{MTE}=(1-p)×1+p×(4000/1400)=1.002861$。这个比例不能观测出实际缓存命中率。固定项与工作斜率按最晚核心的互斥阶段校准，模型没有单指令地址或索引高低位对应的价格。

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
