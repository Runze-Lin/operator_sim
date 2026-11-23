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

在合作核的分 K 分支，参考模型为每个 MMAD 发起约 76.52 cycles，额外每 1024 个填充后的 MAC 约 0.497 cycles。阶段二每个有效行约 3299 cycles，由 950DT 的末次 VF 观测约束。HF32 在目标上的相对吞吐用 16 位 Cube 比例近似。

**逻辑工作量。** 记 `T=80,N=4,D=4096,K=ND=16384,C=N²+2N=24,I=20`。输入 `2TK+4(CK+3+C)=4194412` B（BF16 的 x、FP32 的权重、3 个 alpha 和 bias），输出 `2TD+4T(N+N²)=661760` B（h_in、h_post、h_res）。Pre 的 RMS 基本运算、rsqrt、仿射、sigmoid 和残差运算分别为 `T(2K+1)=2621520`、`T=80`、`3TC=5760`、`2TN=640`、`T[2N+ND+(N−1)D]=2294400`。Sinkhorn 再做 `TN(N−1)=960` 次行最大比较、`TN²=1280` 次减与同量 exp、`2ITN(N−1)=38400` 次归约加、`TN²+(2I−1)TN=13760` 次加 epsilon、`2ITN²=51200` 次除。两部分共 `4922400+106880=5029280` 个向量等效操作。[张量与算术来源](../../src/operator_model/extensions/indexer_mhc.py)

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

上述价格分别为 950DT 观测的 cycles/元素、cycles/MMAD、cycles/1024 MAC 和 cycles/行。目标每核 FP32 Vector 比为 1.027174；HF32 投影使用 16-bit Cube 每核峰值比例代理。整窗取 max(分段和, HBM 3.46405)=**14.61554 µs**。128 token 起的独立 Sinkhorn 团队，或 AIV:AIC 与 2:1 不同的芯片，改走重叠模板；该分支的独立阶段仍缺同等证据。

[联合流水与单行锚点](../../src/operator_model/operators/mhc/pre_sinkhorn.py) · [分核与重叠模板](../../src/operator_model/operators/mhc/sink_resources.py) · [锚点队列的来源](../../src/operator_model/data/parameters/fusion_overlap.json)

**先求本地块，再求每一段的价格。** 只有一个 M 分块（`⌈80/256⌉=1`），PR 在 K 向可用 28 个 AIC，`k=256×⌈⌈16384/28⌉/256⌉=768`，实际 `⌈16384/768⌉=22` 块。PR 的两 AIV 配一 AIC，第二段按每核 `⌈80/56⌉=2` 行，总共 `⌈80/2⌉=40` 个行任务，因此启动 `max(22,⌈40/2⌉)=22` 个核组。AIC 对齐 `m=16×⌈80/16⌉=80,n=16×⌈24/16⌉=32`；当前 kernel 的容量规则为 `k_L1=128×⌊min(⌊32768/m⌋,1024)/128⌋=384`，`k_L0=32×⌊256/max(m,n)⌋=96`。PR 的独立 MTE2 带宽未配置，cast 公式的额外连续读下界采用 0；HBM 下界只对三个主张量取 `[2TK+4KC+2TD]/(1400×1000)=3.46405` µs。

**参考价格的测法。** 在 DT 的 `T=1,20,33,56,80` 五个开发窗中读 Cast 首末 VF 的 cycles 跨度、首发至末退 MMAD 的跨度，以及末段 VF 首发到整窗退休。Cast 每元素价取这五点的两两斜率中位数；投影用 MMAD 数、补齐后 MAC/1024 为两列，将局部 MMAD 跨度作无截距最小二乘，得 76.5241、0.4968364；末段取每个窗的“末段跨度/每核行数”中位数 `med(3553,3299,3267,3275,6803/2)=3299`。向量倍率 `(27/64)/(23/56)=1.027174`，投影吞吐倍率 `(432/32)/(378/28)=1` 是 HF32 相对价格的代理。这里 µs 为每个实际阶段在关键核上的成本。

**如何复算 7.63307 µs。** 在 `T=1` 的 PR 锚点，分块为 `[16,768]`、3 条 MMAD、`16×32×768=393216` 补齐 MAC、1 个有效后处理行。代入上表三式，三段依次是 `768×0.1007028×1.027174/1650=0.048146`、`(3×76.5241+384×0.4968364)/1650=0.254762`、`3299×1.027174/1650=2.053725` µs，合计 2.356634。锚点整窗由 DT 的 `T=1,4,2` 三个队列模板投影：对每类核将各互斥区间时长乘该区间活动资源倍率的最大值后求和，再取三类核的最长值，各模板给出 9.758412、9.989707、10.698518 µs，中位数为 9.989707。中间模板的末核中，含 SIMD 区间累计 1.851515 µs，其余区间 8.087879 µs，`8.087879+1.851515×1.027174=9.989707`。从中扣回三段服务得到合并固定项，同一部分派发与跨核等待只计一次。

## CA 标定样本与参数复算

本页的五个 DT 开发点固定 N=4、D=4096、20 轮、同一推理 ELF。每次选最晚返回的 AIC 的首个 MMAD 发起至末个 MMAD 退休作为投影跨度；在最晚返回的 AIV 上，以末段 PC 尾号 14cb0 的 VF 起点为分界，之前第一条 VF 发起至最后一条 VF 退休的跨度计入 cast，分界至全核退休计入行处理。所有时刻使用共同的设备原点。

| T / 最晚 AIC / 最晚 AIV | Cast 元素 / cycles | MMAD 数 / 补齐 kMAC / cycles | 后段每核行数 / cycles | trace SHA-256 前缀 |
| --- | --- | --- | --- | --- |
| 1 / core0.cubecore0 / core0.veccore0 | 512 / 309 | 2 / 256 / 357 | 1 / 3553 | ce23a734892b |
| 20 / core19.cubecore0 / core8.veccore0 | 5120 / 689 | 2 / 512 / 517 | 1 / 3299 | 602d76748943 |
| 33 / core19.cubecore0 / core15.veccore1 | 8704 / 1171 | 4 / 768 / 632 | 1 / 3267 | a3224db8614b |
| 56 / core8.cubecore0 / core4.veccore0 | 14336 / 1623 | 4 / 1024 / 809 | 1 / 3275 | c8b446f27bb6 |
| 80 / core0.cubecore0 / core5.veccore1 | 20480 / 2307 | 6 / 1280 / 1074 | 2 / 6803 | ad282eb1e6db |

Cast 每元素价是上述十个不同点对 `Δ跨度/Δ元素` 的中位数，即 0.1007028 cycles/元素。投影用 `X=[MMAD发起数,补齐kMAC]`，在五个局部 MMAD 跨度上做无截距最小二乘。由表得到 `XᵀX=[[76,16384],[16384,3604480]]`、`Xᵀy≈[13956,3044608]`；代入原始未舍入时标，求得每 MMAD 发起 76.5241 cycles、每 1024 MAC 0.4968364 cycles。行价对五个 `末段跨度/每核行数` 取中位数，即 3299。单行锚点另外使用 DT 三个整窗 T=1、4、2 各核类上的互斥队列区间：用目标倍率逐区间计价、每窗取最长核，再跨窗取中位数，计算展开见前面的 9.989707 µs。取 BF16 作 HF32 的相对吞吐代理、L1/L0 几何和 MTE 缺省处理均为另一类可替换假设。原档案 `v0.26/presink/archive.json` 的 `cube/vec.milestones` 和 `prices.json → anchors` 保存未舍入时标。[融合实现](../../src/operator_model/operators/mhc/pre_sinkhorn.py) · [单行队列锚点](../../src/operator_model/data/parameters/fusion_overlap.json) [开发形状与事件索引](https://github.com/Runze-Lin/operator_sim/blob/2059edce197af60ddee6cfbc7c78d40c715ed76f/results/v0.26/presink/archive.json) · [五组时序观测](https://github.com/Runze-Lin/operator_sim/blob/2059edce197af60ddee6cfbc7c78d40c715ed76f/results/v0.26/presink/prices.json)

## 与完整 CA 核对

| 目标 | 公开预测 µs | 完整 CA µs | 有符号误差 | 预测有效 AIC / AIV |
|---|---:|---:|---:|---:|
| Ascend950DT_9582 | 13.651 | 13.782 | -0.95% | 32 / 64 |
| Ascend950PR_957b | 14.616 | 13.653 | +7.05% | 22 / 44 |
| Ascend970DT（外推） | 14.378 | N/A | N/A | 120 / 120 |

FIXP 描述符可能在同组 MMAD 退休前已发起。公开估计中启动/交接和第二向量阶段占较大份额，实际最后退休的也是第二阶段 AIV。此例仅对应 4096 宽融合路线。

示例收据的调用为 `aclnnMhcPreSinkhorn`，provider 为 CANN 9.2.1 ops-transformer，实际记录的 kernel 为 `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic`。

复核时，在逐窗记录中筛选 `contract=mhc_pre_sinkhorn`、此页 shape 和属性。有多个前序的样本逐窗保留。示例用于解释该次计算。校准统计、冻结后独立样本数及完整误差范围以总报告为准。[逐窗数据](../../calibration/reference/windows.jsonl) · [模型实现](../../src/operator_model/operators/mhc/pre_sinkhorn.py) · [总建模方法](../modeling-strategy.md) · [误差分析](../error-analysis.md)
