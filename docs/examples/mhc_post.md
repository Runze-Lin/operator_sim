# 内置 mHC Post · 相同形状的两次运行

T=196、N=4、D=1536，输出 1,204,224 个 BF16 元素。有用混合的乘/加共 196×4×1536×9=10,838,016 次，另计输入/输出转换。

## 单卡输入与调用

```python
from operator_model import estimate

shape = [196, 4, 1536]
attrs = {'has_h_res': True}
result = estimate('mhc_post', shape, attrs, chip='Ascend950PR_957b')
print(result.latency_us, result.active_aic, result.active_aiv)
print(result.work["logical_read_bytes"], result.work["logical_write_bytes"])
```

## 从工作量到设备时间

两个 950 芯片都按 49 个 AIV 排，每核四行，通道不再分片。公开 shape-only 入口分别计算首用、热态及扩核等适用情景，将快慢边界的调和中心作为单值。两项加权边界贡献只表示这个数值的构成。

公开工作量统计为 Cube **0**、Vector **12,042,240** 个有用标量等效操作。逻辑读 **3,026,240 B**、写 **2,408,448 B**。片外实际流量取决于重用与缓存。

行内切通道与 token 分核分开校准。连续通道段的参考相邻发起约 63 tick，数据交接约 26 tick。同核连续 token 上紧邻的下一行改用该分支的读/写返回与首次间隔。前一次使用过的核数是独立输入。 [参数及单位](../../src/operator_model/data/parameters/post_history.json)

下表取 950PR 的公开阶段分账。各段已处理模型可辨识的重叠，数值保留三位小数。

| 阶段字段 | 暴露时间 µs |
|---|---:|
| `lower_scenario_component` | 2.380 |
| `upper_scenario_component` | 2.380 |
| **整窗** | **4.760** |

## 与完整 CA 核对

| 目标 | 公开预测 µs | 完整 CA µs | 有符号误差 | 预测有效 AIC / AIV |
|---|---:|---:|---:|---:|
| Ascend950DT_9582 | 4.704 | 3.955、3.800 | +18.93%、+23.78% | 0 / 49 |
| Ascend950PR_957b | 4.760 | 5.273、3.865 | -9.72%、+23.15% | 0 / 49 |
| Ascend970DT（外推） | 3.110 | N/A | N/A | 0 / 98 |

同一 PR 几何的两次实际时间明显不同。同一个预测对一个窗口偏慢、对另一个偏快。模型范围并非置信区间。部署能观测同一流前序时，应传状态后再与相同前序的 CA 比较。

示例收据的调用为 `aclnnMhcPost`，provider 为 CANN 9.2.1 ops-transformer，实际记录的 kernel 为 `MhcPost_59130ccee6717ff2c0f15b485e731162_6`。

复核时，在逐窗记录中筛选 `contract=mhc_post · shape`、此页 shape 和属性。有多个前序的样本逐窗保留。示例用于解释该次计算。校准统计、冻结后独立样本数及完整误差范围以总报告为准。[逐窗数据](../../calibration/reference/windows.jsonl) · [模型实现](../../src/operator_model/operators/mhc/history.py) · [总建模方法](../modeling-strategy.md) · [误差分析](../error-analysis.md)
