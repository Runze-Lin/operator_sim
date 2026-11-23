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

**输入、输出和向量运算。** 记 `K=ND=4096,C=N²+2N=24`。输入 BF16 的 `x` 占 `2TK`，FP32 投影权重、alpha 和 bias 共 `4(CK+3+C)`，去重读 `2×4×4096+4(24×4096+3+24)=426092` B。普通输出为 BF16 的 `h_in` 与 FP32 的 `h_post,h_res`，本例还导出 inv_rms、h_mix、h_pre。写 `2TD+4T[N+N²+(1+C+N)]=8976` B。RMS 平方、求和、均值和 epsilon 共 `T(2K+1)=32772`，另有 `T=4` 次 rsqrt；投影后的缩放及仿射 `3TC=288`，sigmoid `2TN=32`，残差部分 `T[2N+ND+(N−1)D]=28704`（两项系数、逐元素乘、对 N 路求和），共计 61800。[算术及输出合同](../../src/operator_model/extensions/indexer_mhc.py)

下表取 950PR 的公开阶段分账。各段已处理模型可辨识的重叠，数值保留三位小数。

| 阶段字段 | 暴露时间 µs |
|---|---:|
| `pre` | 9.247 |
| **整窗** | **9.247** |

## 本例各阶段怎样定价

对每阶段 q 取目标成本 Tq=aq×fscalar + bq×Wq×fq。a 是在 950DT 拟合的固定阶段预算（µs），b 是每一工作单位增加的 µs，f 是目标的相对资源价。把 950DT 的 CA 分段拟合到 compute、vector、traffic、control 后，零斜率只说明在这些样本上没有分辨出独立增量。

本例 [4,4,1024] 需要 N²+2N=24 个投影，K=4×1024=4096。参考实现选 MK，核内 K=256，一次 16 行分块需 256/32=8 个 MMAD 单位。Vector 有 ⌈4/64⌉×⌈1024/256⌉=4 个通道片，traffic 为 (4×4096×24 + 2×4×4096)/1024=416 KiB，control 为 1 个 M 分块。代入 PR 的 traffic 倍率 1.02189、固定项倍率 1，得到 compute=0，vector=0.99802，traffic=2.31409+0.00231156×416×1.02189=3.297，control=3.69964+1.25261=4.95225 µs，合计 **9.24701 µs**。尽管 compute 的独占预算为零，实现仍需在 Cube 上发起投影。

[M/K 分块与各阶段单位](../../src/operator_model/operators/mhc/pre_dispatch.py) · [目标端定价](../../src/operator_model/scheduling/projection.py) · [分支系数与单位](../../src/operator_model/data/parameters/mhc_partition.json)

DT 校准使用同一库合同的临界核，从整窗共同原点开始逐个时间片按 Cube、SIMD、搬运（MTE/FIXP）、其余控制/等待的顺序唯一归属。将每段时长作为 y、该资源的工作单位作为 x，分别求非负直线 `y=a+bx`；如果所有样本该项只有一个 x，取该项实测中位数作 a、令 b=0。参与拟合的 DT Pre 开发窗中，最晚结束的核没有独占 Cube 区间。这里单位是关键路径上的暴露成本。

**核数和流量比例。** 这里的 MK 指沿 token 与投影归约维 K 作网格切分。参考分区以 32 个 AIC 为一组，`K≥512` 时取 `k_c=256×⌈K/(32×256)⌉=256`、`m_b=⌈T/16⌉=1`。用于跨芯片估计的任务数分别为 `m_b×⌈K/256⌉=16` 和 `T=4`，PR 的 28/56 核各只需一波。关键核 compute 单位为 `m_b×⌈k_c/32⌉=8`。对本例分支，预算采用 `(a,b)`：compute `(0,0)`、vector `(0.998017,0)`、traffic `(2.314089,0.00231156)`、control `(3.699636,1.252606)`；对应的 b 单位依次为 µs/计算块、µs/通道片、µs/KiB、µs/M 块。

Traffic 的 416 KiB 是权重与输入工作特征，另用全输入输出字节估算片外压力。参考 DT 的阶段和为 `t_0=9.225955` µs，`p=(426092+8976)/(4000×1000×t_0)=0.011789`。目标和参考未给独立 MTE2 峰值，单波且等频时本地比为 1。带宽比为 `4000/1400`，有效 traffic 倍率 `(1−p)×1+p×4000/1400=1.02189`。这个混合假设近似描述关键核可见的数据服务，不能从 `p` 得到实际 HBM miss 数。

## CA 标定样本与参数复算

校准取 950DT 拆分 mHC 调用中的第一个 kernel（Pre），与本页独立 Pre 的六输出合同一致。按每个 core type 选最晚退休核，再取其中整窗最长者；将共同原点到最后指令的时间片按 Cube、SIMD、搬运、其余控制的优先级唯一归属。以下 11 个窗口均为 N=4 的 MK 路线。核以数据库的 CoreTypeId/CoreId 标识。

| T,D | 核 | 共同原点→末 tick | Cube / Vector / 搬运 / 控制（周期） | 计算块 / 向量片 / 输入KiB / 控制块 |
| --- | --- | --- | --- | --- |
| (1, 128) | 2/0 | 5676→19005 | 0 / 1800 / 4262 / 7267 | 8 / 1 / 49 / 1 |
| (8, 128) | 3/0 | 5649→19271 | 0 / 1603 / 4637 / 7382 | 8 / 1 / 56 / 1 |
| (3, 128) | 3/0 | 5597→19531 | 0 / 1817 / 3764 / 8353 | 8 / 1 / 51 / 1 |
| (2, 128) | 2/0 | 5220→18429 | 0 / 1888 / 4649 / 6672 | 8 / 1 / 50 / 1 |
| (4, 256) | 3/0 | 5484→19378 | 0 / 1697 / 3998 / 8199 | 8 / 1 / 104 / 1 |
| (16, 128) | 3/3 | 5515→20873 | 0 / 794 / 3501 / 11063 | 8 / 1 / 64 / 1 |
| (1, 512) | 2/0 | 5738→19836 | 0 / 1855 / 4313 / 7930 | 8 / 2 / 196 / 1 |
| (4, 1024) | 2/0 | 5119→19523 | 0 / 1847 / 6240 / 6317 | 8 / 4 / 416 / 1 |
| (9, 1024) | 2/1 | 5437→20327 | 0 / 1320 / 4103 / 9467 | 8 / 4 / 456 / 1 |
| (16, 512) | 3/2 | 5286→20772 | 0 / 1900 / 4524 / 9062 | 8 / 2 / 256 / 1 |
| (32, 1024) | 3/4 | 5626→24384 | 0 / 1593 / 6927 / 10238 | 16 / 4 / 640 / 2 |

每段取该资源的工作单位作 x、互斥 tick/1650 作 y。11 个窗的独占 Cube 均为 0。Vector 原始斜率为 −0.002660 µs/片，按非负约束取 0，截距为时长平均 0.998017 µs。Traffic 均点是 (212.54545 KiB,2.805399 µs)，中心化求斜率 `Σ[(x−x̄)(y−ȳ)]/Σ(x−x̄)²=0.002311555` µs/KiB，截距 `2.805399−212.54545×0.002311555=2.314089` µs。Control 以 1 或 2 个 M 块为横轴，同理得斜率 1.252606 µs/块，截距 `5.066116−1.090909×1.252606=3.699636` µs。PR 的 1.02189 只作用在流量斜率项，由前文的逻辑字节压力和两芯片 HBM 峰值给出。原 DB 摘要列在 `v0.9/mhc/freeze.json → training_db.pre_mk`，上表已列出复算所需的全部点。[MK 价格](../../src/operator_model/data/parameters/mhc_partition.json) · [分块实现](../../src/operator_model/operators/mhc/pre_dispatch.py) [拆分 mHC 的分支与训练库指纹](https://github.com/Runze-Lin/operator_sim/blob/2059edce197af60ddee6cfbc7c78d40c715ed76f/results/v0.9/mhc/freeze.json) · [分支的训练程序](https://github.com/Runze-Lin/operator_sim/blob/2059edce197af60ddee6cfbc7c78d40c715ed76f/scripts/mhc-v9-freeze.py)

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
