# mHC Pre · 四个 token 的 24 列投影

T=4、N=4、D=1024，展平行长 4096，投影宽 N²+2N=24。有效投影 2×4×4096×24=786,432 ops，同时产生倒数 RMS、门控权重与外部层输入。

## 单卡输入与调用

```python
from operator_model import estimate

shape = [4, 4, 1024]
attrs = {'alpha_count': 3, 'dtype': 'bf16', 'has_gamma': False, 'return_intermediates': True}
result = estimate('mhc_pre', shape, attrs, chip='Ascend950PR_957b')
print(result.latency_us, result.active_aic, result.active_aiv)
print(result.work["logical_read_bytes"], result.work["logical_write_bytes"])
```

## 从工作量到设备时间

K 向远长于 M 向，PR 投影划分到 16 个 AIC，4 个 AIV 处理对应系数/归约。保留辅助输出，读写量包含预、后混合系数以及残差矩阵。按匹配分核的 DT 阶段模板重算，只返回可辨识的 Pre 部分合计。

公开工作量统计为 Cube **786,432**、Vector **61,800** 个有用标量等效操作。逻辑读 **426,092 B**、写 **8,976 B**。片外实际流量取决于重用与缓存。

参考阶段按投影的分 K 或普通布局分别给出截距和单位服务时间（µs）。特征由当前形状的投影块、向量、流量及控制工作决定。阶段标签只表示归属在关键核的暴露成本。 [参数及单位](../../src/operator_model/data/parameters/mhc_partition.json)

下表取 950PR 的公开阶段分账。各段已处理模型可辨识的重叠，数值保留三位小数。

| 阶段字段 | 暴露时间 µs |
|---|---:|
| `pre` | 9.247 |
| **整窗** | **9.247** |

## 与完整 CA 核对

| 目标 | 公开预测 µs | 完整 CA µs | 有符号误差 | 预测有效 AIC / AIV |
|---|---:|---:|---:|---:|
| Ascend950DT_9582 | 9.226 | 8.730 | +5.68% | — / — |
| Ascend950PR_957b | 9.247 | 9.420 | -1.84% | 16 / 4 |
| Ascend970DT（外推） | 8.450 | N/A | N/A | 16 / 4 |

核数字段的 “—” 表示该参考路径未返回这一项。

较宽形状中参与核的局部同步可能主导误差。参考预算不能定位一条指定的 MMAD 首发。独立 Pre 与含 Sinkhorn 融合实现的输出合同不同。

示例收据的调用为 `aclnnMhcPre`，provider 为 CANN 9.2.1 ops-transformer，实际记录的 kernel 为 `MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic`。

复核时，在逐窗记录中筛选 `contract=mhc_pre`、此页 shape 和属性。有多个前序的样本逐窗保留。示例用于解释该次计算。校准统计、冻结后独立样本数及完整误差范围以总报告为准。[逐窗数据](../../calibration/reference/windows.jsonl) · [模型实现](../../src/operator_model/operators/mhc/pre_dispatch.py) · [总建模方法](../modeling-strategy.md) · [误差分析](../error-analysis.md)
