# LightningIndexer · 一个 query 扫描 8192 个 key

Q[1,1,8,128]、K[1,8192,1,128]、权重[1,1,8] 均为 BF16，key 占 2 MiB。每个 key 有八个点积，Cube 工作为 2×8192×8×128=16,777,216 ops。输出 top512 写 2048 B。

## 单卡输入与调用

```python
from operator_model import estimate

shape = [1, 1, 8192, 8, 128]
attrs = {'dtype': 'bf16',
 'layout': 'BSND',
 'return_values': False,
 'sparse_count': 512,
 'sparse_mode': 3}
result = estimate('lightning_indexer', shape, attrs, chip='Ascend950DT_9582')
print(result.latency_us, result.active_aic, result.active_aiv)
print(result.work["logical_read_bytes"], result.work["logical_write_bytes"])
```

## 从工作量到设备时间

一个四 query 分组只装入一行。8192/128=64 页，单 AIC 共发 64 条 MMAD。排序 AIV 执行 256 条正时长直方图指令。DT 库启动 32 个 AIC 和 64 个 AIV，真正做这两种有效工作的各为一个。

公开工作量统计为 Cube **16,777,216**、Vector **188,416** 个有用标量等效操作。逻辑读 **2,099,216 B**、写 **2,048 B**。片外实际流量取决于重用与缓存。

参考首次 MMAD 为 4587 cycles，页到页的读发起间隔为 416 cycles，score 到 histogram 接力为 1160 cycles，三者分别由首次、稳态和交接事件得到。目标若有已知 MTE2 带宽，用每页 32768 B 的实际端口服务与补齐乘加服务重新比较。 [参数及单位](../../src/operator_model/data/parameters/indexer.json)

**从张量到账面工作。** 稀疏掩码在 `Q=1` 时保留全部 `P=8192` 对。每对有 `G=8` 次 ReLU 比较、8 次乘权重和 `G−1=7` 次加法，Vector 有用工作 `P(2G+G−1)=188416`，数据相关的 top-k 比较次数未计入。逻辑读 `2(QGD+LD+QG)=2(1024+1048576+8)=2099216` B，逻辑写 `4Q×512=2048` B。物理 Cube 运算以 16 行对齐，单页发出的 `2×16×128×128` ops 是有用工作的两倍。最多四个 query 分一组，由一个 AIC 和最多两个排序 AIV 协作；本例一个排序 AIV 有有效行。每核 BF16 的 8181.818 ops/cycle 来自 `432×1000/(32×1.65)`，DT 每核 FP32 的 255.682 来自 `27×1000/(64×1.65)`。

下表取 950DT 的公开阶段分账。各段已处理模型可辨识的重叠，数值保留三位小数。

| 阶段字段 | 暴露时间 µs |
|---|---:|
| `first_mmad` | 2.780 |
| `page_intervals` | 15.884 |
| `last_mmad` | 0.079 |
| `score_to_histogram` | 0.703 |
| `histogram_and_output` | 2.425 |
| `additional_shared_service` | 0.000 |
| **整窗** | **21.871** |

## 本例各阶段怎样定价

本例为 DT、单 query、8192 个可见 key、8 个头、depth=128。8 行头对齐到 M=16，8192/128=64 个键页。一个完整键页为 128×128×2=32768 B。整个 CV 窗分为下面五项，单位为 DT cycles。

| 阶段 | 公式与本例 |
|---|---|
| 首 MMAD | 参考关键核的第一次 MMAD 在 4587 cycles。 |
| 相邻页 | (64−1)×max(单页读发起间隔416, 2×16×128×128/8181.818)=26208。计算项使用每核 BF16 ops/cycle。 |
| 尾页退休 | 66.88053 cycles 响应 + 2×16×128×128/8181.818 = 130.96。 |
| 结果可排序 | Score 到首个 histogram 的等待为 1160。 |
| histogram 与输出 | 每 256 keys 一组，共 32 组、8 次 histogram/组。一个扫描区间需 540+48.75×32=2100 cycles，输出/退休需 635.857+39.5357×32≈1901 cycles。 |

合计 **(4587+26208+130.96+1160+4001)/1650=21.87088 µs**；HBM 下界约 0.525 µs 已被流水覆盖。页间隔包含高速本地读取与发起节奏，端口明确时以目标服务率重算；首发、尾部和 histogram 响应来自 950DT 分段 CA。超过 8192 key 时需要在同一行重复扫描，目前按行间交接作代理。

[键页与 histogram 计数](../../src/operator_model/operators/attention/indexer.py) · [页和扫描边界价格](../../src/operator_model/data/parameters/indexer.json)

**把观测换成阶段系数。** 采用 DT 的 `(Q,L)=(1,1024),(1,8192),(4,1024),(4,4096)` 四窗。首发取第一次 MMAD 相对整窗起点的中位数 `med(4608,4566,4490,4679)=4587`；末 MMAD 到首 histogram 发起的中位数为 `med(1236,931,1214,1106)=1160`。416 是在三页循环缓冲的稳态范围取 `median((t[i+3] − t[i])/3)` 得到的每页发起 cycles；66.88053 是各窗 MMAD 时长中位数扣除对应 `2MNK/r_cube` 后再跨窗取中位数的响应 cycles。每个 256-key 块对应 8 条 histogram 指令，`8192/256×8=256`。单行 4 块和 32 块的 histogram 首发至末次退休跨度为 735、2100 cycles，故拟合斜率 `(2100−735)/(32−4)=48.75`、截距 `735−4×48.75=540`；末次 histogram 到整窗结束为 794、1901 cycles，同理得斜率 39.5357、截距 635.857。这些是算子库路径的有效价格。DT 配置无独立 MTE2/L2 带宽，页间隔使用 416，L2 带宽下界不激活；2 MiB 的 key 可容入 128 MiB L2，冷流量下界 `(2099216+2048)/(4000×1000)=0.525316` µs。[分组与资源下界实现](../../src/operator_model/operators/attention/indexer.py)

## CA 标定样本与参数复算

四个 DT 数据库把全核第一条指令设为 0，分别取最晚退休 MMAD 的 AIC 和最后完成 histogram 的 AIV；下表以归档的 `CoreTypeId-CoreId` 标识。histogram 只计正时长事件。

| Q / L | AIC / AIV | 首 MMAD 发起→退休 | 末 MMAD 退休 | 首 histogram 发起→末 histogram 退休 | 整窗末 cycles | DB SHA-256 前缀 |
| --- | --- | --- | --- | --- | --- | --- |
| [1, 1024] | 1-0 / 2-0 | [4608, 4739] | 8358 | [9594, 10329] | 11123 | eb58dd4e7557 |
| [4, 1024] | 1-0 / 2-0 | [4490, 4685] | 8682 | [9896, 12479] | 13271 | a5ed89bdcfdd |
| [4, 4096] | 1-0 / 3-0 | [4679, 4874] | 19168 | [20274, 24594] | 25862 | 32233930737e |
| [1, 8192] | 1-0 / 2-0 | [4566, 4697] | 32538 | [33469, 35569] | 37470 | f09ed67f4921 |

例如 Q=1、L=8192 的交接是 `33469−32538=931` cycles，末 histogram 到全核退休是 `37470−35569=1901`。单发 MMAD 时长的中位数扣除对齐计算服务后，两个 Q=1 窗各给 66.92036 cycles，两个 Q=4 窗各给 66.84071，合取中位数为 66.88053。稳态页间隔在两份长窗（8192、4096）从第 4 条开始每三条 MMAD 取一份三发起差再除以 3，共 29 项，中位数 416。Q=4 两份窗的相邻行 histogram 交接在 4、16 块分别为 1113、1676 cycles，故另一路行间定价每块 `(1676−1113)/12=46.9167`、固定 `1113−4×46.9167=925.333`。本单行单扫描例无该间隔。原档案 `v0.34/indexer/physical.json` 的 `critical_mmad_ticks`、`sort_ticks` 与 `duration_ticks` 可重算。[阶段参数](../../src/operator_model/data/parameters/indexer.json) · [逐窗索引](../../calibration/reference/windows.jsonl) [Indexer 校准的事件与时刻](https://github.com/Runze-Lin/operator_sim/blob/2059edce197af60ddee6cfbc7c78d40c715ed76f/results/v0.34/indexer/physical.json) · [事件提取与定价代码](https://github.com/Runze-Lin/operator_sim/blob/2059edce197af60ddee6cfbc7c78d40c715ed76f/development/validation/indexer34/prices.py)

## 与完整 CA 核对

| 目标 | 公开预测 µs | 完整 CA µs | 有符号误差 | 预测有效 AIC / AIV |
|---|---:|---:|---:|---:|
| Ascend950DT_9582 | 21.871 | 22.709 | -3.69% | 1 / 1 |
| Ascend950PR_957b | 21.926 | N/A | N/A | 1 / 1 |
| Ascend970DT（外推） | 20.046 | N/A | N/A | 1 / 1 |

DT 原始数据库有 120,521 条记录，96 个参与核各有唯一物理 END，设备跨度 (43,410−5940)/1650=22.709 µs，完整 512 个索引集合与 CPU 一致。Q1 的 DT 样本属于开发回放。其他 Q 或候选数的 PR 不替代此形状。

示例收据的调用为 `aclnnLightningIndexer`，provider 为 CANN 9.2.1 ops-transformer，实际记录的 kernel 为 `entry_kernel_31b1b`。

复核时，在逐窗记录中筛选 `contract=lightning_indexer`、此页 shape 和属性。有多个前序的样本逐窗保留。示例用于解释该次计算。校准统计、冻结后独立样本数及完整误差范围以总报告为准。[逐窗数据](../../calibration/reference/windows.jsonl) · [模型实现](../../src/operator_model/operators/attention/indexer.py) · [总建模方法](../modeling-strategy.md) · [误差分析](../error-analysis.md)
