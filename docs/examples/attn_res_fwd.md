# AttnResFwd · 带 bank 写回的两行

本例 [T,D,S,N]=[2,4096,8,1]。对两行 4096 维向量评分并混合，S=8 是 bank 容量，实际仅 N=1 份缓存残差有效。materialize=True、write_idx=1 产生独立输出并写入指定 bank。

## 单卡输入与调用

```python
from operator_model import estimate

shape = [2, 4096, 8, 1]
attrs = {'addend': True, 'materialize': True, 'write_idx': 1}
result = estimate('attn_res_fwd', shape, attrs, chip='Ascend950PR_957b')
print(result.latency_us, result.active_aic, result.active_aiv)
print(result.work["logical_read_bytes"], result.work["logical_write_bytes"])
```

## 从工作量到设备时间

同一轮活跃 2 个 AIV，各做一行。每行 4096/64=64 个向量组。调度含公共权重读取、前缀加与存储、bank 评分及加权归一化，实际返回 Y、prefix 和 materialized 合同决定读写次数。

公开工作量统计为 Cube **0**、Vector **163,866** 个有用标量等效操作。逻辑读 **73,728 B**、写 **65,536 B**。片外实际流量取决于重用与缓存。

校准把读权重和计算得分时段展开成固定项加 64 通道向量组项。例如 bank_scores 的参考价格为 918.05 cycles 加每组 5.84 cycles。本例的 resident 路线使用下文按归一化组数和混合组数分别计价的 weighted_norm_write。有效残差数和输出合同决定参与各段的组数。 [参数及单位](../../src/operator_model/data/parameters/attention_residual.json)

**逐项核算。** 一行 BF16 占 `R=2D=8192` B。PR 有 56 个 AIV，启动 `A=min(T,56)=2` 个，忙核轮数 `⌈T/A⌉=1`。按实现预留固定工作区和 prefix 行，UB 可驻留的行数为 `⌊(253952−22D−32768)/(2D+4)⌋=15`，本例需 `N+1=2` 行，选择 key 40020。读去重后为 `R[2(1+1+1)+(2+1)]=73728` B，括号内依次是两行 prefix、addend、有效 bank 及共享的两行评分权重和一行归一化权重。DMA 量按发起核展开为 `R[2(1+1)+2+2A+2]=98304` B。每 token 写输出、更新 prefix、写 bank、另存 materialized 张量各一行，`4TR=65536` B。

四行参与点积（`T(N+1)=4`），连同输出共有六行做 RMS。FP32 基本操作依次为 RMS `6(2D+1)`、归一化点积 `4(3D−1)`、加权累加 `4(2D)`、softmax 的比较/移位/求和 `2(3×2−2)` 及缩放 `2×2`、输出归一化 `2DT`、按核处理权重 `AD`，合计 155662。再计入 `TD=8192` 次 BF16 加、4 次 exp、6 次 rsqrt、2 次倒数，得到 163866。[工作量与 UB 判定](../../src/operator_model/operators/attention/residual_work.py)

下表取 950PR 的公开阶段分账。各段已处理模型可辨识的重叠，数值保留三位小数。

| 阶段字段 | 暴露时间 µs |
|---|---:|
| `dispatch` | 2.744 |
| `score_weights` | 1.436 |
| `prefix_add_write` | 1.070 |
| `bank_scores` | 0.892 |
| `weighted_norm_write` | 2.502 |
| `exposed_transfer_floor` | 0.000 |
| **整窗** | **8.643** |

## 本例各阶段怎样定价

令 L=⌈4096/64⌉=64 个 FP32 向量块，s=255.682/248.918=1.02717 为参考/目标每核峰值比，r=48.363 B/cycle 是本核读端口。一份权重的字节服务 u=8192/r=169.39 cycles。该 resident 分支从 950 分段 CA 提取固定 cycles 与每块 cycles。以 cycles 代入后，各段再除 1650。

| 阶段 | 固定项 + 向量块 + 搬运 |
|---|---|
| dispatch | 4527 |
| score_weights | 1746.59 + 4.30855×64s + 2u |
| prefix_add_write | 1249.04 + 2.69742×64s + 2u |
| bank_scores | 918.05 + 5.83947×64s + u，本例有效 bank 数为 1 |
| weighted_norm_write | 2347.99 + (20.30835×64+2.09691×128)s + u；归一化一次，混合两组块 |

全片冷输入公共带宽下界为 0.053 µs；额外暴露 max(0,下界−上述阶段之和)=0。逐段得到 2.744+1.436+1.070+0.892+2.502 ≈ **8.643 µs**，保留全精度求和。write_idx/materialize 的改动影响写回和调度分支。

[resident 阶段公式](../../src/operator_model/operators/attention/residual_timing.py) · [打分与写回价格](../../src/operator_model/data/parameters/attention_residual.json)

**价格怎样取得。** 64 来自一个 256 B 向量寄存器容纳的 64 个 FP32。DT 每核峰值为 `27×1000/(64×1.65)=255.682` ops/cycle，PR 为 `23×1000/(56×1.65)=248.918`，所以只对向量块乘 `s=1.027174`。两端本例未配置 MTE2 端口带宽，使用 DT 已校准的 `r=48.362956` B/cycle。首发、评分、前缀、bank 的界标依次是全核起点、首个权重读、前缀读、首个 bank 读和 gamma 读发起。用相邻发起差减去对应的 `n_readR/r`，再以每重复的向量块数作横轴求斜率和固定项。例如 DT 的 `D=256,N=1` 与 `D=7168,N=2` 两次评分窗口分别为 1785、2822 cycles；取 `x=⌈D/64⌉=4,112`，扣除两行 BF16 的读取后，`y=1785−1024/r, 2822−28672/r`，斜率 `(y_2−y_1)/(112−4)=4.30855`，截距 `y_1−4×4.30855=1746.59`。前缀窗口为 1281、2144 cycles，横轴和两次读同上；bank 窗口为 952、3737 cycles，每个样本先除以 N 再扣除一行读。较宽样本只提供已发起的 DMA 边界间隔。最后一段在当前实现中单独拆成固定、一次归一化和随混合深度增长的三项，上表使用实现中的三个有效系数；发布参数 JSON 还保留旧的单斜率尾段。下节列出重算当前三个系数所用的九个 DT 实测窗口。共享带宽项本例取去重输入 `73728/(1400×1000)=0.05266` µs。

## CA 标定样本与参数复算

**读端口价。** 每核读价借用 950DT 的 DynamicMxQuant。先在 13 个 shape 的首块上，以有效读退休减发起作 y、请求字节作 x，非负直线拟合的固定响应为 972.527405 cycles。再选四个 ≥16384 B 的首块，取 `字节/(读跨度−固定响应)` 的最大值。`[576,4096]` 的请求为 73728 B、跨度 2496.999979 cycles，给出 `73728/(2496.999979−972.527405)=48.362956 B/cycle`。这是一项跨算子的端口近似。

**派发和数据读。** DT `[1,256,8,1]` 在 `core0.veccore0` 从整窗起点到首笔有效读为 4527 cycles，随后权重、prefix、bank 三段相邻读边界间隔为 1785、1281、952 cycles（SHA-256 前缀 `87a1753436b5`）。DT `[1,7168,8,2]` 同核的 popped log 中，三段分别为 2822、2144、3737 cycles（`ecf3fe17d667`，仅有发起边界）。权重与 prefix 都扣两行读；bank 的两次观测先分别除以有效 bank 数，再扣一行读。bank 的纵坐标为 `952−512/r` 与 `3737/2−14336/r`，横轴为 4、112 个向量组，得到斜率 5.83947、截距 918.05 cycles。

**末段三个价格。** 各窗选最后完成的 AIV，取其最后一笔 gamma 行读发起至全核末指令的跨度。令 `L=ceil(D/64)`，一行观测构成 `X=[1,L,L×(N+1)]` 和 `y=跨度−2D/r`，按不加权最小二乘求三个未知数。下表全为 950DT。

| [T,D,S,N] | 最后 AIV | gamma 发起至全核退休（cycles） | 合并 trace SHA-256 前缀 |
| --- | --- | --- | --- |
| [1, 7168, 8, 1] | core0.veccore0 | 5228.00053 | 25fdc2468337 |
| [1, 7168, 8, 3] | core0.veccore0 | 5744.99925 | 8d23be57c918 |
| [2, 4096, 8, 2] | core16.veccore0 | 4648.00108 | 663de3da21e8 |
| [5, 2048, 8, 1] | core0.veccore1 | 3305.99921 | d213fe02f38c |
| [2, 1024, 8, 4] | core16.veccore0 | 2857.99964 | 96c2f8b09c6c |
| [8, 512, 8, 2] | core0.veccore0 | 2868.99884 | 505d2ff7e2ad |
| [3, 512, 8, 1] | core0.veccore0 | 2703.00057 | 4e185e687860 |
| [4, 256, 8, 3] | core16.veccore1 | 2339.00092 | c318313b0c6d |
| [1, 256, 8, 2] | core0.veccore0 | 1956.00050 | 8c961e5cfb0d |

第一行 `X=[1,112,224]`，`y=5228.00053−14336/48.362956=4931.57531`。九行的矩阵秩为 3，解出 2347.99163 cycles 的固定成本、20.3083544 cycles/输出归一化块和 2.09690792 cycles/混合块。PR 的 `[1,512,8,2]` 只参与旧单斜率尾段。原档案 `v0.19/workloads/attn-res-extension-evaluation.json` 的 `ca_phases_us[-1]` 与 `trace_sha256` 可按表定位。[现用尾段公式](../../src/operator_model/operators/attention/residual_timing.py) · [逐窗索引](../../calibration/reference/windows.jsonl) [9 个 DT 开发窗的阶段观测](https://github.com/Runze-Lin/operator_sim/blob/2059edce197af60ddee6cfbc7c78d40c715ed76f/results/v0.19/workloads/attn-res-extension-evaluation.json) · [尾段回归代码](https://github.com/Runze-Lin/operator_sim/blob/2059edce197af60ddee6cfbc7c78d40c715ed76f/development/validation/fusion21/prices.py)

## 与完整 CA 核对

| 目标 | 公开预测 µs | 完整 CA µs | 有符号误差 | 预测有效 AIC / AIV |
|---|---:|---:|---:|---:|
| Ascend950DT_9582 | 8.603 | 8.113 | +6.04% | 0 / 2 |
| Ascend950PR_957b | 8.643 | 7.984 | +8.24% | 0 / 2 |
| Ascend970DT（外推） | 7.885 | N/A | N/A | 0 / 2 |

本例的实际 tiling key 是 40020。Kimi 驻留 prefill 合同 materialize=False、write_idx=−1 改变输出别名、写量和分核，应另查该合同的直接 CA 资格。

示例收据的调用为 `aclnnAttnResFwd`，provider 为 vLLM Ascend custom OPP，实际记录的 kernel 为 `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020`。

复核时，在逐窗记录中筛选 `contract=attn_res_fwd`、此页 shape 和属性。有多个前序的样本逐窗保留。示例用于解释该次计算。校准统计、冻结后独立样本数及完整误差范围以总报告为准。[逐窗数据](../../calibration/reference/windows.jsonl) · [模型实现](../../src/operator_model/operators/attention/residual_timing.py) · [总建模方法](../modeling-strategy.md) · [误差分析](../error-analysis.md)
