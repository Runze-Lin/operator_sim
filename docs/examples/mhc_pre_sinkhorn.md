# 融合 mHC PreSinkhorn · 投影交给向量归约

80 个 token，N=4、D=4096，投影输出 24 列。输入行展平为 16,384 元素，投影有 2×80×16,384×24=62,914,560 ops，随后对每张 4×4 矩阵做 20 次迭代。

## 单卡输入与调用

```python
from operator_model import estimate

shape = [80, 4, 4096]
attrs = {'need_backward': False, 'num_iters': 20}
result = estimate('mhc_pre_sinkhorn', shape, attrs, chip='Ascend950PR_957b')
print(result.latency_us, result.active_aic, result.active_aiv)
print(result.work["logical_read_bytes"], result.work["logical_write_bytes"])
```

## 从工作量到设备时间

PR 划分 22 个 K 分片，每份至多 768 个元素。22 个 AIC 与 44 个 AIV 参与。代表块 [80,768] 继续沿 K 用 384 元素 L1 页和 96 元素 L0 片段，临界 AIC 8 条 MMAD。第二阶段有 40 个 AIV 承担有效行，每个两行。

公开工作量统计为 Cube **62,914,560**、Vector **5,029,280** 个有用标量等效操作。逻辑读 **4,194,412 B**、写 **661,760 B**。片外实际流量取决于重用与缓存。

在合作核的分 K 分支，参考模型为每个 MMAD 发起约 76.52 tick，额外每 1024 个填充后的 MAC 约 0.497 tick。阶段二每个有效行约 3299 tick，由 950DT 的末次 VF 观测约束。HF32 在目标上的相对吞吐用 16 位 Cube 比例近似。

下表取 950PR 的公开阶段分账。各段已处理模型可辨识的重叠，数值保留三位小数。

| 阶段字段 | 暴露时间 µs |
|---|---:|
| `dispatch_and_handoff` | 7.633 |
| `stage1_cast` | 1.926 |
| `stage1_cube` | 0.949 |
| `stage2_rows` | 4.107 |
| `exposed_hbm_floor` | 0.000 |
| **整窗** | **14.616** |

## 本例各阶段怎样定价

80 个 token、N=4、D=4096，K=16384、实际输出 N²+2N=24 列。目标 M/K 分块为 [80,768]，K 上 22 个块，对应 22 AIC、44 AIV。忙 AIV 转换 80/2×768=30720 个值。输出对齐到 32 列，L1 的 K 页为 384，L0 的 K 片为 96，忙 AIC 发起 2×4=8 次 MMAD，发出的乘加为 80×32×768=1,966,080。后半段 40 个 AIV 各处理 2 行。

| 阶段 | 数字代入（时间为 µs） |
|---|---|
| 派发与交接 | 一行模板的整窗 9.98971 减去该模板已有三段 2.35663，得到 7.63307。该合并固定项覆盖无法单独识别的调度与释放。 |
| Cast | max(30720×0.1007028×1.027174/1650, 端口连续读下界)=1.92585。 |
| 投影 | (8×76.5241 + (1966080/1024)×0.4968364×1.0)/1650=0.94916。后半项以千 MAC 为单位。 |
| 行归一化 | 2×3299.0000×1.027174/1650=4.10745。 |

上述价格分别为 950DT 观测的周期/元素、周期/MMAD、周期/1024 MAC 和周期/行。目标每核 FP32 Vector 比为 1.027174；HF32 投影使用 16-bit Cube 每核峰值比例代理。整窗取 max(分段和, HBM 3.46405)=**14.61554 µs**。128 token 起的独立 Sinkhorn 团队，或 AIV:AIC 与 2:1 不同的芯片，改走重叠模板；该分支的独立阶段仍缺同等证据。

[联合流水与单行锚点](../../src/operator_model/operators/mhc/pre_sinkhorn.py) · [分核与重叠模板](../../src/operator_model/operators/mhc/sink_resources.py) · [锚点队列的来源](../../src/operator_model/data/parameters/fusion_overlap.json)

## 与完整 CA 核对

| 目标 | 公开预测 µs | 完整 CA µs | 有符号误差 | 预测有效 AIC / AIV |
|---|---:|---:|---:|---:|
| Ascend950DT_9582 | 13.651 | 13.782 | -0.95% | 32 / 64 |
| Ascend950PR_957b | 14.616 | 13.653 | +7.05% | 22 / 44 |
| Ascend970DT（外推） | 14.378 | N/A | N/A | 120 / 120 |

FIXP 描述符可能在同组 MMAD 退休前已发起。公开估计中启动/交接和第二向量阶段占较大份额，实际最后退休的也是第二阶段 AIV。此例仅对应 4096 宽融合路线。

示例收据的调用为 `aclnnMhcPreSinkhorn`，provider 为 CANN 9.2.1 ops-transformer，实际记录的 kernel 为 `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic`。

复核时，在逐窗记录中筛选 `contract=mhc_pre_sinkhorn`、此页 shape 和属性。有多个前序的样本逐窗保留。示例用于解释该次计算。校准统计、冻结后独立样本数及完整误差范围以总报告为准。[逐窗数据](../../calibration/reference/windows.jsonl) · [模型实现](../../src/operator_model/operators/mhc/pre_sinkhorn.py) · [总建模方法](../modeling-strategy.md) · [误差分析](../error-analysis.md)
