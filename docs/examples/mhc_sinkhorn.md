# mHC Sinkhorn · 4097 张四阶矩阵

4097 张 4×4 的 FP32 矩阵，4 次行列交替迭代，逻辑读写各 4097×16×4=262,208 B，标量除法等效工作 2×4×4097×16=524,416 次。

## 单卡输入与调用

```python
from operator_model import estimate

shape = [4097, 4]
attrs = {'num_iters': 4}
result = estimate('mhc_sinkhorn', shape, attrs, chip='Ascend950PR_957b')
print(result.latency_us, result.active_aic, result.active_aiv)
print(result.work["logical_read_bytes"], result.work["logical_write_bytes"])
```

## 从工作量到设备时间

PR 每核最多 ⌈4097/56⌉=74 张，至多分成 10 个寄存器组，每个迭代函数内部遍历这些组。首个 softmax、每轮两次归一化与最后输出依次处理。对同一张矩阵不可把四次迭代并行。

公开工作量统计为 Cube **0**、Vector **1,278,264** 个有用标量等效操作。逻辑读 **262,208 B**、写 **262,208 B**。片外实际流量取决于重用与缓存。

按首次、扩核和重用三种前序分别记录从 tiling 发起到输入、从 softmax 到每次行列归一以及写回的参考 tick。FP32 的单核基准为 255.68 ops/tick，用来缩放能归属向量服务的部分。发起控制另计。 [参数及单位](../../src/operator_model/data/parameters/sinkhorn_history.json)

**按算法数操作。** 令 `T=4097,N=4,I=4`。初次行 softmax 的比较、平移、指数分别为 `TN(N−1)=49164`、`TN²=65552`、`TN²=65552`。把初次 softmax 的归一化也算一趟，四轮共有 `2I=8` 趟归约和除法，分别需 `2ITN(N−1)=393312` 次加、`2ITN²=524416` 次除；epsilon 共加 `TN²+(2I−1)TN=180268` 次。总和 1278264。关键核需 UB `832+896×74=67136` B，其中 832 B 是固定区，896 B/矩阵是 host tiler 对双槽及临时缓冲的预留，小于 PR 配置 253952 B；只调度一块 UB。

下表取 950PR 的公开阶段分账。各段已处理模型可辨识的重叠，数值保留三位小数。

| 阶段字段 | 暴露时间 µs |
|---|---:|
| `tiling_and_setup` | 1.833 |
| `input_and_softmax_issue` | 0.412 |
| `first_column` | 2.502 |
| `normalization_loop` | 3.418 |
| `scatter_to_output_issue` | 0.400 |
| `output_and_retirement` | 0.250 |
| **整窗** | **8.815** |

## 本例各阶段怎样定价

4097 个 4×4 矩阵交给 56 个有效 AIV，关键核做 ⌈4097/56⌉=74 个矩阵，按最多 8 个矩阵一组做 10 次寄存器处理。模型按同一核心跟踪 setup、softmax 发起、首列归约、首行归约、scatter、输出发起、退休。循环外推时，除最初列/行以外的工作按首对、稳态对的发起间隔递推；本例四次迭代的首对约 1.032 µs，稳态对约 1.035 µs。固定项与斜率来自 DT 上这些 VF 和 DMA 的起止，负载按目标每核 FP32 吞吐和 MTE 服务换算。

| 历史情形 | setup VF | softmax VF | 首列 | 首行 | scatter | 写发起 | 退休（µs） |
|---|---:|---:|---:|---:|---:|---:|---:|
| 首次 | 2.456 | 3.062 | 5.564 | 5.880 | 8.982 | 9.382 | 9.666 |
| 扩核 | 2.140 | 2.511 | 5.013 | 5.328 | 8.431 | 8.831 | 9.096 |
| 已驻留 | 0.901 | 1.161 | 3.663 | 3.978 | 7.081 | 7.481 | 7.682 |

每种情形相邻标记之差就是公共表中的阶段预算；其中 normalization_loop 从首列跨到 scatter。调用方未传前序状态时，这个算子对三种情形等权，每一列阶段分别平均。例如首段=(2.456+2.140+0.901)/3≈1.833 µs，整窗=(9.66631+9.09611+7.68154)/3=**8.81465 µs**。状态平均是模型的默认约定。

[标记间隔](../../src/operator_model/operators/mhc/sink_pipeline.py) · [未知历史组合](../../src/operator_model/operators/mhc/sink_unknown.py) · [寄存器与 DMA 周期](../../src/operator_model/data/parameters/sinkhorn_pipeline.json) · [历史状态价格](../../src/operator_model/data/parameters/sinkhorn_history.json)

**从周期数复算七个标记。** 下表中三列依次为首次、扩核、已驻留。`A=⌈4097/74⌉=56` 为参与核数，`g=⌈74/8⌉=10` 为寄存器组数，`s=(27/64)/(23/56)=1.027174` 为同频向量周期倍率。单组固定项、与核数有关的项目 `a+bA` 来自 DT 对相同历史分类的 DMA/VF 起止差；保留非负的两两斜率中位数，并对残差取中位数作截距。多组的软启动和每组附加周期来自同一 DT kernel 不同寄存器遍数的发起间隔。输入（或写）共享下界 `F=262208×1.65/1400=309.031` 周期；本例无单核 MTE2/3 带宽项。

| 达到的时刻 | 从前一时刻开始如何累计（PR 周期，三元组次序同上） |
|---|---|
| setup | tiling 发起 `(710.038+13.2264A,1294.438+0.4375A,406)` 加 tiling 等待 `(442.645+5.139A,340.65+0.3292A,264.5)` 加 setup `(1872,1853.5,816.5)`，得 `(4053.15,3531.52,1487.00)`。 |
| softmax | 从 setup 加读发起 97，加 `max(F,input_wait)`，其中 input_wait 为 `(897.5,509.5,327)`，最后加 `(5,5,4.5)`；得 `(5052.65,4143.02,1915.50)`。 |
| 首列 VF | 取两个就绪时刻的较大值：softmax `+s×max(231,176+427(g−1))`，或 setup `+(1392.522+13.6405A,863.688+0.8854A,659.5)`。本例均由 softmax 后的 4128.21 周期决定。 |
| 首行 VF | 首列 `+s[443+8(g−2)]`，即加 520.78 周期。 |
| scatter | 首行加首对周期 `s[248+156.5625(g−1)]=1702.09`，再加 `(I−2)=2` 个稳态周期 `s[254+156.5625(g−1)]=1708.25`。 |
| 写发起 | scatter 加 `s[(202,202,201.5)+49(g−1)]`。 |
| 退休 | 三态的写等待分别为 `w = (313.369 + 2.5952×A, 430.5, 324.5)` 周期。第 j 态退休时刻为该态写发起时刻加 `max(F, w[j]) + e[j]`，其中 `e = (10, 7, 7)`。所得三个结束周期 `(15949.41,15008.59,12674.55)` 各自除 1650 得上表三窗。 |

多组系数还可从 DT 开发窗 `T=449,513,4097` 的 1、2、9 组对照核验。2 和 9 组 softmax 跨度分别为 1022、4011 周期，新增一组为 `(4011−1022)/7=427` 周期；2 组时发起距离为 `2893−2290=603`，故线性截距 `603−427=176`。首列到首行由 443 增至 499 周期，每组加 `(499−443)/7=8`。三窗稳态对周期为 254、411、1503，新增组价取 `med(411−254,(1503−254)/8)=156.5625`。首对与稳态对的单组基价各取 248、254 周期。末尾的 49 则是同一 PC 发出的相邻寄存器组 scatter 事件的稳态周期间隔。PR 本例多一组，按同一斜率延伸到 10 组。

309.031 周期只代表这轮所有输入或输出矩阵占用的 HBM 服务下界；各状态的描述符响应更慢。切换芯片时，共享带宽、频率、UB 可装矩阵数和每核 FP32 周期都重新计算。

## CA 标定样本与参数复算

**历史响应的标定。** 950DT 开发数据按实际前序分类为首次 16 窗、扩核 16 窗、驻留 2 窗，定位每窗最晚 AIV 的有效 DMA 与 VF。tiling_issue 是全核起点至 96 B tiling 读发起，tiling_wait 是该读跨度，setup_from_tile 是读退休到第一个 VF 发起。input_issue 是首 VF 到 payload 读发起，input_wait 是 payload 读跨度，soft_after_input 是数据退休到 softmax VF 发起。first_col_from_setup 取首 VF 到首列 VF 的发起差；write_gap 取末个 VF 到首次输出写的发起差；write_wait 是该写跨度，end 是输出写退休到整窗结束。

只为 tiling_issue、tiling_wait、first_col_from_setup、write_wait 拟合随活动核数增长的斜率。将不同活动核数的窗口两两取 `Δ时差/Δ核数`，其中位数截为非负即 b；截距 `a=max(0,median(时差−b×核数))`。同一活动核数的样本没有独立斜率，令 b=0。首次组覆盖 1 至 64 核，扩核组覆盖 4 至 64 核。例如首次 `[113,4],I=20` 的末核 `core16.veccore1`（57 核，SHA `fbc5dbe5c14c`）的 tiling 读为 1142→1944、首 VF 在 3607，提供了 1142、802、1663 三个周期观测。

两条驻留窗 `[65,4],I=10` 与 `[66,4],I=10` 各有 33 个活动核，最晚为 DB 核 `23.3`，SHA-256 前缀 `003f4278fdfa` 与 `1782e698a7e3`。tiling 读分别为 412→754、400→587，首 VF 发起 1525、1449；payload 为 1622→1950、1546→1872，softmax VF 发起 1955、1876。故 tiling_issue 为 `(412+400)/2=406`，tiling_wait 为 `(342+187)/2=264.5`，setup_from_tile 为 `[(1525−754)+(1449−587)]/2=816.5`，input_issue 为 97，input_wait 为 `(328+326)/2=327`，soft_after_input 为 4.5。输出写 5046→5369、4967→5293，整窗在 5376、5300 结束，因此 write_wait 为 324.5、end 为 7 周期。

**寄存器扩展价格。** 对照 DT `T=449,I=8` 的 `core14.veccore0`（SHA `9acafbf917ad`）、`T=513,I=6` 的 `core12.veccore1`（`32a1a58622ba`）、`T=4097,I=4` 的 `core31.veccore0`（`f817cf8c787b`）。单组的相邻行归约发起间隔序列为 [248,254,254,254,254,254,254]，得到首对 248、稳态 254。多组价格 427、8、156.5625 的代入见上节。末 PC `0x9000d10d38` 每八条 scatter 取一个组时标；稳定段 88220、88269、88318、88367… 的相邻差为 49 周期。核查原档案 `v0.26/sink/trace-evidence.json` 与 `v0.28/fusion/landmarks.json`。[历史分类价格](../../src/operator_model/data/parameters/sinkhorn_history.json) · [寄存器价格](../../src/operator_model/data/parameters/sinkhorn_pipeline.json) [单组与多组 trace 观测](https://github.com/Runze-Lin/operator_sim/blob/2059edce197af60ddee6cfbc7c78d40c715ed76f/results/v0.26/sink/trace-evidence.json) · [按历史分组的首尾时刻](https://github.com/Runze-Lin/operator_sim/blob/2059edce197af60ddee6cfbc7c78d40c715ed76f/results/v0.28/fusion/landmarks.json)

## 与完整 CA 核对

| 目标 | 公开预测 µs | 完整 CA µs | 有符号误差 | 预测有效 AIC / AIV |
|---|---:|---:|---:|---:|
| Ascend950DT_9582 | 8.105 | 8.964 | -9.58% | 0 / 64 |
| Ascend950PR_957b | 8.815 | 9.112 | -3.26% | 0 / 56 |
| Ascend970DT（外推） | 5.546 | N/A | N/A | 0 / 118 |

归档的设备调用此前有 DT57、PR52 个 AIV 的同族高水位。上表模型按调用方省略 resident_aiv 计算。生产常用的 20 次迭代是另一条输入合同。

示例收据的调用为 `aclnnMhcSinkhorn`，provider 为 CANN 9.2.1 ops-transformer，实际记录的 kernel 为 `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0`。

复核时，在逐窗记录中筛选 `contract=mhc_sinkhorn`、此页 shape 和属性。有多个前序的样本逐窗保留。示例用于解释该次计算。校准统计、冻结后独立样本数及完整误差范围以总报告为准。[逐窗数据](../../calibration/reference/windows.jsonl) · [模型实现](../../src/operator_model/operators/mhc/sink_pipeline.py) · [总建模方法](../modeling-strategy.md) · [误差分析](../error-analysis.md)
