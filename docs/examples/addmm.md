# Addmm · 单波列偏置

X[512,256]、W[256,512]，另有 BF16 列偏置 b[512]。有用乘加量 2×512×256×512=134,217,728 FLOP，偏置加发生在输出通路。公开接口将偏置归入投影输出通路。

## 单卡输入与调用

```python
from operator_model import estimate

shape = [512, 512, 256]
attrs = {'alpha': 1, 'beta': 1, 'self_shape': [512]}
result = estimate('addmm', shape, attrs, chip='Ascend950PR_957b')
print(result.latency_us, result.active_aic, result.active_aiv)
print(result.work["logical_read_bytes"], result.work["logical_write_bytes"])
```

## 从工作量到设备时间

PR 输出按 80×128 为代表矩形，整次 28 个 AIC 在一波完成，56 条 K=128 的 MMAD。读一次偏置增加 1024 B，输出矩阵写 524,288 B。分块选择复用 MatMulV3 的偏置变体。

公开工作量统计为 Cube **134,217,728**、Vector **0** 个有用标量等效操作。逻辑读 **525,312 B**、写 **524,288 B**。片外实际流量取决于重用与缓存。

此 PR 排程沿用 MatMulV3 的逐页资源价格。MTE1 的参考首响应为 33.69 tick，持续 180.65 B/tick。偏置改变读入量和输出变体，图形核上的附加排队只在该实际分支的证据允许时计入。 [参数及单位](../../src/operator_model/data/parameters/matmul_resources.json)

下表取 950PR 的公开阶段分账。各段已处理模型可辨识的重叠，数值保留三位小数。

| 阶段字段 | 暴露时间 µs |
|---|---:|
| `read_fill` | 1.557 |
| `l0_lead` | 0.131 |
| `page_intervals` | 0.000 |
| `last_mmad` | 0.508 |
| `write_tail` | 0.345 |
| `additional_output_waves` | 0.000 |
| `shared_link_floor` | 0.000 |
| `launch` | 2.315 |
| **整窗** | **4.856** |

## 本例各阶段怎样定价

整片峰值先除以 AIC 数与频率，得到每核 8181.82 BF16 ops/周期。一个输出矩形有面积 A，每个 K 碎片的计算服务为 2×补齐面积×补齐 K /8181.82。R（首读）取固定读响应加上每核 MTE2、共享 L2、HBM 服务成本的最大值；L（首个 L0 装载）按 L1→L0 字节/180.65 B/周期，加 33.69 周期响应。C（一页末次计算完成）由碎片间隔及末条的 99.92 周期响应和算术服务构成。P 为（页数−1）×页发起间隔，间隔取读发起、读数据、两槽周转和计算服务的最大值。W（末次输出）取 FIX 端口、L2 或 HBM 写服务最大值，再加 381.81 周期尾部。首发统一为 3381.33+91.25×log₂(28)=3820.01 周期。这些固定周期和局部服务斜率来自 950DT 分段 CA；芯片重定价会改变峰值、分核和共享链路。

本例有一页 K256，分两条 K128，alpha=beta=1 的列 bias 被带入同一设备窗。偏置每核需读相应列的 2 B 值，另付 L1 到 bias table 装载；读偏置时与共享 L2 形成上限。忙波数据为 R=2568.89、L=216.49、P=0、C=838.64、W=569.06 周期。全片只有一波，共享读写下界 1219.66 小于这些阶段的 4193.07 周期。总计 (3820.01+4193.07)/1650 = **4.856 µs**。额外 MN 次偏置加法是逻辑工作；本合同没有独立的 Vector 加法 kernel。

[Addmm 入口](../../src/operator_model/operators/matmul/addmm.py) · [带偏置波次计算](../../src/operator_model/operators/matmul/resources.py) · [阶段参数](../../src/operator_model/data/parameters/matmul_resources.json)

## 与完整 CA 核对

| 目标 | 公开预测 µs | 完整 CA µs | 有符号误差 | 预测有效 AIC / AIV |
|---|---:|---:|---:|---:|
| Ascend950DT_9582 | 4.773 | 4.479 | +6.57% | 32 / 0 |
| Ascend950PR_957b | 4.856 | 4.687 | +3.62% | 28 / 0 |
| Ascend970DT（外推） | 3.761 | N/A | N/A | 88 / 0 |

两端逻辑尺寸相同，PR 实测 28 核、DT 32 核。将已分块的 DT 核数直接乘比例会在边界凭空增加一波。

示例收据的调用为 `aclnnAddmm`，provider 为 CANN 9.2.1 ops-nn，实际记录的 kernel 为 `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2`。

复核时，在逐窗记录中筛选 `contract=addmm`、此页 shape 和属性。有多个前序的样本逐窗保留。示例用于解释该次计算。校准统计、冻结后独立样本数及完整误差范围以总报告为准。[逐窗数据](../../calibration/reference/windows.jsonl) · [模型实现](../../src/operator_model/operators/matmul/resources.py) · [总建模方法](../modeling-strategy.md) · [误差分析](../error-analysis.md)
