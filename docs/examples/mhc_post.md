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

## 本例各阶段怎样定价

196 行、4 路和 1536 个通道按每核 4 行分给 49 个 AIV。对每个前序状态先做 MTE2 → VF → MTE3 的行流水，在当前行等输入、跨行允许读写与计算重叠，最后与共享链路下界取大。本例冷读 3,026,240 B / (1400×1000 B/µs) = 2.162 µs。以“上一轮按通道切分，本轮扩核”为例，排程后的各段为到达 0.40121、Scalar 1.22182、MTE2 2.50623、SIMD 2.21092、MTE3 0.53820、退休 0.01005 µs，合计 6.88843 µs。

调用方只给 shape，前序内核与状态未知。按首次、已驻留且前序按通道或按行、扩核且前序按通道或按行、连续重复计算，六种设备时间分别为 5.70178、5.03981、4.44922、6.88843、5.48767、3.63665 µs。令最短 L=3.63665、最长 H=6.88843；点估计 2LH/(L+H)=**4.76021 µs**，使两个端点的相对误差上界相等。公共表中的两个 2.38010 µs 是这个点的代数分账。真实核上六种路径互斥，只有选定状态后才能解释相应流水。状态转移采用已校准的 950 行或通道分支；局部 DMA/VF 价由参考核窗口约束，分核数和字节数由目标分块给出。

[状态区间与点估计](../../src/operator_model/operators/mhc/history.py) · [行流水](../../src/operator_model/operators/mhc/token_pipeline.py) · [历史状态参数](../../src/operator_model/data/parameters/post_history.json) · [读写与 VF 参数](../../src/operator_model/data/parameters/post_token.json)

## 与完整 CA 核对

| 目标 | 公开预测 µs | 完整 CA µs | 有符号误差 | 预测有效 AIC / AIV |
|---|---:|---:|---:|---:|
| Ascend950DT_9582 | 4.704 | 3.955、3.800 | +18.93%、+23.78% | 0 / 49 |
| Ascend950PR_957b | 4.760 | 5.273、3.865 | -9.72%、+23.15% | 0 / 49 |
| Ascend970DT（外推） | 3.110 | N/A | N/A | 0 / 98 |

同一 PR 几何的两次实际时间明显不同。同一个预测对一个窗口偏慢、对另一个偏快。模型范围并非置信区间。部署能观测同一流前序时，应传状态后再与相同前序的 CA 比较。

示例收据的调用为 `aclnnMhcPost`，provider 为 CANN 9.2.1 ops-transformer，实际记录的 kernel 为 `MhcPost_59130ccee6717ff2c0f15b485e731162_6`。

复核时，在逐窗记录中筛选 `contract=mhc_post · shape`、此页 shape 和属性。有多个前序的样本逐窗保留。示例用于解释该次计算。校准统计、冻结后独立样本数及完整误差范围以总报告为准。[逐窗数据](../../calibration/reference/windows.jsonl) · [模型实现](../../src/operator_model/operators/mhc/history.py) · [总建模方法](../modeling-strategy.md) · [误差分析](../error-analysis.md)
