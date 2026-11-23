# FA · 128Ki KV 的九行主模型校验

一次主模型校验 Q=9 行、L=131072 个 KV、D=64、B=H=1。K、V 各 16 MiB。QK 和 PV 的有效算术各有 2×9×131072×64=150,994,944 ops。

## 单卡输入与调用

```python
from operator_model import estimate

shape = [9, 131072, 64, 1, 1]
attrs = {}
result = estimate('fa', shape, attrs, chip='Ascend950PR_957b')
print(result.latency_us, result.active_aic, result.active_aiv)
print(result.work["logical_read_bytes"], result.work["logical_write_bytes"])
```

## 从工作量到设备时间

Q 仅一个任务，在两个 950 上各为 1 AIC+2 AIV。每页 256 个 key，512 页。每页 QK/PV 各两段 128 个 key，因此全核共 2048 条 MMAD。首轮在 K/V 发起后等待数据，之后只对每页付一次稳态 II，末页再排空。

有效配对数为 $P=9×131072$。QK 和 PV 合计 $4PD=301{,}989{,}888$ 个 Cube 操作。公开的 softmax 算法工作量按 scale、max、减、exp、sum、除六类计，max 与 sum 每行各少一次，共 $6P-2×9=7{,}077{,}870$ 个 Vector 等效操作。边界逻辑读为 $2D(Q+2L)=33{,}555{,}584$ B，逻辑写为 $2QD+2×4×8Q=1728$ B，含两份 FP32 统计量。下面为流水定价还需考虑对齐、核间分工，以及在线 softmax 按页更新的工作。[边界工作量定义](../../src/operator_model/workload.py)

下表取 950PR 的公开阶段分账。各段已处理模型可辨识的重叠，数值保留三位小数。

| 阶段字段 | 暴露时间 µs |
|---|---:|
| `launch` | 6.479 |
| `first_tile` | 3.877 |
| `steady` | 264.297 |
| `drain` | 0.335 |
| **整窗** | **274.988** |

## 本例各阶段怎样定价

### 一页里有多少工作

按当前 host 策略，$BH\lceil Q/128\rceil=1<28$，选 64 行的 query 块。长 KV 选每页 $n=256$ 个位置。实际行数 $m=9$，头维 $d=64$，共 $p=\lceil131072/256\rceil=512$ 张满页。一个 AIC 做两次矩阵乘，配两个 AIV 分行，因此最忙 AIV 有 $r=\lceil9/2\rceil=5$ 行。记 $\hat x=16\lceil x/16\rceil$，Cube 按 $(\hat m,\hat n,\hat d)=(16,256,64)$ 计。每种矩阵乘又按 128 个 KV 位置切开，片数 $u=\lceil n/128\rceil=2$。16 对齐对应矩阵乘的分形布局，128 个 KV 位置一片则取自本库路线的 MMAD 记录。以下时间均为设备周期，950PR 的 1 µs = 1650 周期。[Mmad 的矩阵布局与分形](https://gitcode.com/cann/asc-devkit/blob/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/mmad_compute/Mmad.md) · [同形 DT 逐指令图区间](assets/fa.json)

| 每满页的约束 | 工作量怎样得到 | 除以什么速率、怎样代入 | 周期/页 |
|---|---|---|---:|
| Cube | QK 做 $[m,d]×[d,n]$，PV 做 $[m,n]×[n,d]$。每种乘法有 $F=2\hat m\hat n\hat d=524288$ 个填充后操作，乘加计两次。两种工作共享一颗 Cube。 | 每 AIC 峰值 $P_C=378×1000/(28×1.65)=8181.818$ op/周期。单种乘法取 $F/P_C+u h$。QK 为 $64.0796+2×73.4602=211$，PV 为 $64.0796+2×40.4602=145$。 | **356.00** |
| Vector | 记分处理为每个 score 五份等价工作（缩放、最大值、指数/差值、求和与归一相关工作），输出累积更新按每行每通道三份计。按最忙 AIV 得 $5rn=6400$、$3rd=960$。5 和 3 是此资源模型的工作量权重。 | $P_V=19.65112$ 等价 op/(AIV·周期)。两段服务 $s_1=6400/P_V=325.681$，$s_2=960/P_V=48.852$，同一 AIV 上相加。 | **374.53** |
| K/V 读 | K 和 V 各 $n×d$ 个 BF16，均为 $2nd=32768$ B。Q 的 1152 B 在本波首发时只读一次。 | 每核有效读速率 $R=\min(80.06842,1400/1.65)=80.06842$ B/周期。此波只有一个工作 AIC，PR 配置未另限 MTE2 或 L2。$(32768+32768)/R$。 | **818.50** |
| MTE1，L1→L0 | QK 的两个操作数占 $2(\hat m d+\hat n d)$ B。PV 的占 $2(\hat m\hat n+\hat n d)$ B，其中 $\hat m\hat n$ 是进入第二次乘法的 P。合计 $B_1=2[2\hat n d+\hat m(d+\hat n)]=75776$ B。 | $B_1/R_1=75776/163.42591$。$R_1$ 为该路片上装载的有效 B/周期。 | **463.67** |
| FIX，Cube 结果搬出 | QK 输出 score 有 $\hat m\hat n$ 个槽，PV 输出有 $\hat m d$ 个槽，按 FP32 中间结果定价。$B_F=4\hat m(\hat n+d)=20480$ B。 | $B_F/R_F=20480/115.03598$。$R_F$ 为结果搬出的有效 B/周期，最终 BF16 写回另在排空段计。 | **178.03** |
| 页发起控制 | 开发 trace 给出一个 256 位置、两个片段的持续发起间隔 $J$。对一页 $u$ 个片段，模型按 $u/2$ 缩放这项节拍。 | $J u/2=853.4047135×2/2$。它含同步、缓冲槽复用和回压的共同影响。 | **853.40** |

流水装满后，每推出一页，各资源都要完成表中对应的工作。共享同一资源的服务先相加（例如 QK+PV），可与相邻页并行的资源取最慢者。模型假设未分辨的阻塞由发起节拍覆盖。因此本例的页间隔为

$$I=\max(356.00,\ 374.53,\ 818.50,\ 463.67,\ 178.03,\ 853.4047)=853.4047\ \text{周期/页}.$$

首张页有独立的依赖链。余下 511 页增加 **$511I/1650=264.297$ µs**。这个最大值也能解释瓶颈何时改变：若保持其余五项不变，本核可用读速率低于 $65536/853.4047=76.794$ B/周期，K/V 读就会超过发起控制。若 KV 尾页只有 $n_t$ 个位置，则以 $n_t$ 重算字节、对齐量和片数，续发成本为 $(p-2)I(256)+I(n_t)$（$p\ge2$）。[单页公式与调度实现](../../src/operator_model/operators/attention/resources.py)

### 这些速率和固定周期从哪里来

- **Cube 与 Vector。** Cube 使用 PR 整片 BF16 峰值 378 TOPS、28 AIC 与 1.65 GHz。每片开销 $h$ 用 DT 同路开发窗中从该页首个 MMAD 发起到末个 MMAD 退休的跨度 $S$，扣掉理想运算周期再均摊，即 $h=\operatorname{median}(\max(0,S-F/P_{C,DT})/u)$。宽页 QK、PV 的跨度分别为 211、145 周期。Vector 取 DT 32Ki 开发窗较忙 AIV 的 SIMD 忙区间并集除以页数，得每页 0.220985 µs。基准有效速率 $P_{V,DT}=7360/(0.220985×1650)=20.18512$。在 PR 上乘以每 AIV 的 FP32 峰值比，$P_V=P_{V,DT}×[23000/(56×1.65)]/[27000/(64×1.65)]=19.65112$。这个校准包含归约、特殊函数和发起开销。[PR 芯片配置](../../src/operator_model/data/chips/Ascend950PR_957b.json) · [DT 芯片配置](../../src/operator_model/data/chips/Ascend950DT_9582.json)
- **三个搬运价格。** 对 DT 32Ki 开发窗，找 K、V 都能连续推进的最快八页窗口。对每个起点取两条流耗时之大者，再在起点中取最小，为 3.968485 µs。局部读速率 $R_{\rm local}=8×65536/(3.968485×1650)=80.06842$ B/周期。MTE1、FIX 分别将对应最忙核该队列的有效指令时间区间取并集，再除 128 页，得到 0.281013、0.107898 µs/页。因此 $R_1=75776/(0.281013×1650)$，$R_F=20480/(0.107898×1650)$。目标端另有端口配置时按实现覆盖或限速，共享 L2/外存带宽还需按本波工作核数均摊。这些 950 有效价格含软件和等待的影响。[校准提取程序](https://github.com/Runze-Lin/operator_sim/blob/e99f2eb6e645de0e6e187379ba7f9d89d05d3297/development/validation/sku16/fa_dt_calibrate.py)
- **为什么是 853.4047。** 在同一 DT 32Ki 窗中，按地址分开 K、V 两条读流。以零起算页号 $i=8,\ldots,119$ 和实际**发起**时间 $t_i$（µs）做直线拟合，斜率 $b=\sum(i-\bar i)(t_i-\bar t)/\sum(i-\bar i)^2$ 表示每新发一页需要多少 µs。K 得 0.51710899，V 得 0.51721498。两条流都要推进，故 $J=1650\max(b_K,b_V)=853.4047135$ 周期。最快八页窗口描述局部搬运能力，这里较长中段的斜率描述可持续节拍，两者有不同的测量依据。PR 沿用此 DT 周期先验，单次读响应或目标端独立控制成本不能由这条斜率分离出来。[32Ki 发起时间戳与分路斜率](https://github.com/Runze-Lin/operator_sim/blob/e99f2eb6e645de0e6e187379ba7f9d89d05d3297/results/v0.5/trace-analysis.json) · [取样范围与拟合实现](https://github.com/Runze-Lin/operator_sim/blob/e99f2eb6e645de0e6e187379ba7f9d89d05d3297/scripts/fit-v5.py)

### 首张页和末端怎样接回整窗

本例使用 DT 宽页路线的两个开发窗（KV=4096、32768）提取首尾价格。启动 $L_s$ 取整窗开始到 Q 首次读取发起的中位数，$(10648+10732)/2=10690$。同理，Q→K、K→V 首发差 $g_K,g_V$ 为 92、1992.5；Q/K 均读完到 QK 首发的间隔 $g_0=174$。把 Q、K、V 首笔描述符的持续周期分别扣除其字节服务，取非负残差的中位数，六个非负残差居中的两项为 992.750 和 1219.612，故首读响应 $L_r=(992.750+1219.612)/2=1106.181$ 周期。[宽页开发窗逐项观测](https://github.com/Runze-Lin/operator_sim/blob/e99f2eb6e645de0e6e187379ba7f9d89d05d3297/results/v0.16/fa-dt-evidence.json) · [实际参数](../../src/operator_model/data/parameters/attention_resources.json)

1. **启动与操作数就绪。** 启动为 $L_s/1650=6.479$ µs。以 Q 首发为时间零点，$t_Q=L_r+1152/R=1120.569$，$t_K=g_K+L_r+32768/R=1607.431$，$t_V=g_K+g_V+L_r+32768/R=3599.931$ 周期。读请求可重叠，同一读口还需完成 Q 与 K 的合计字节。
2. **首轮依赖。** $t_{QK}=\max(t_Q,t_K,(1152+32768)/R)+g_0=1781.431$。PV 同时依赖 V 和第一段向量结果，$t_{PV}=\max(t_V,t_{QK}+211+s_1)+h=6252.431$。$h=2652.5$ 周期取自开发窗：从 QK 与 V 都就绪到首个 PV 发起的实际间隙，扣去还能暴露的 Vec1 工作，取非负残差中位数。两个开发窗中这段 Vec1 都已被等待 V 的时间覆盖，所以 $h=(2516+2789)/2$。本例 V 较晚到，前面的 Vec1 已被覆盖。首轮完成为 $(t_{PV}+145)/1650=3.877$ µs。
3. **最后退休。** 输出共 $b_o=2md+2×4×8m=1728$ B。开发窗的末次 PV 退休到全核结束跨度，扣去 $3rd/P_{V,DT}$ 与输出服务后的非负残差中位数给出 $h_D=(530+573)/2-47.560-21.582=482.359$ 周期（用未舍入值计算）。PR 本例写速率 $R_W=80.06842$ B/周期。排空为 $(s_2+h_D+b_o/R_W)/1650=0.335$ µs。合计设备时间按未舍入值算为 **274.988 µs**。

用开头的公开调用即可逐项复核。下列两项依次给出六项页需求和整个设备窗的四段时间。

```python
tile = result.projection["timing_diagnostics"]["wave_types"][0]["first"]
print(tile["demands_cycles"], tile["steady_bottleneck"])
print(result.projection["stages_us"])
```

## 与完整 CA 核对

| 目标 | 公开预测 µs | 完整 CA µs | 有符号误差 | 预测有效 AIC / AIV |
|---|---:|---:|---:|---:|
| Ascend950DT_9582 | 274.987 | 274.821 | +0.06% | 1 / 2 |
| Ascend950PR_957b | 274.988 | 265.893 | +3.42% | 1 / 2 |
| Ascend970DT（外推） | 252.093 | N/A | N/A | 1 / 1 |

DT 首个 QK 模型 7.558 µs、CA 7.315 µs。末页模型 271.855 µs、CA 273.460 µs。全窗很近仍包含相互抵消的阶段差别。970DT 的 1:1 配置给该任务 1 个 AIV。

示例收据的调用为 `aclnnFlashAttentionScoreV2`，provider 为 CANN 9.2.1 ops-transformer，实际记录的 kernel 为 `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic`。

![长 KV 的 QK 和 PV](assets/fa.png)

图选首、中、尾页对比，同一小窗内两栏采用同一尺度，未画出的中间页保留在数据中。 [图上使用的数值](assets/fa.json)

复核时，在逐窗记录中筛选 `contract=fa`、此页 shape 和属性。有多个前序的样本逐窗保留。示例用于解释该次计算。校准统计、冻结后独立样本数及完整误差范围以总报告为准。[逐窗数据](../../calibration/reference/windows.jsonl) · [模型实现](../../src/operator_model/operators/attention/resources.py) · [总建模方法](../modeling-strategy.md) · [误差分析](../error-analysis.md)
