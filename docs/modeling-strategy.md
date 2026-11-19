# 算子建模

系统仿真器输入每卡 shape、dtype、布局和算子属性，预测一次设备指令窗口的微秒数。后端按 **Vector、Cube、Cube + Vector（CV）** 三类选择方法。算法给出工作量与依赖，库实现决定分块、分核和缓冲复用，硬件配置与校准给出阶段服务时间。在线调用只做分块计算和流水调度。单次进程内查询的验收目标为 **100 ms 以内**，预测出来的设备时间和离线 CA 采集成本分别记录。

**950DT 是校准参考，系统目标由调用方选择。** 同一算子分为四层。算法层由 shape、dtype、布局及专家分布确定有效工作和张量字节。实现层按库路线产生可分配的 tile 与依赖。芯片层从独立 JSON 提供峰值、核数、频率和存储参数。调度层将工作映射到该芯片，重新计算每核服务、可重叠区段、尾波和净设备时间。对 960、970、980，调用方可以假设沿用 950 的实现族，并分别提供自己的配置。更换 dtype 仍须有对应的 kernel 合同。下面三类方法决定哪些工作能随芯片资源扩展。

### Vector 算子怎样建模

1. 先找可以并行的维度。逐行归约可以把行分给多个 AIV，KDA recurrent 的同一条状态链则要依次处理 token。根据库的分核规则算出最忙核承担的行数、列片数和尾块，再用 UB 容量决定每轮读入多少数据。
2. 写出一轮的读、算、写。读写以实际 dtype 字节和对齐块计，向量计算以有效向量分组和归约轮次计。对已知双缓冲实现，让下一轮预读与本轮计算重叠，同时约束同一个缓冲槽在上一次使用完成后才能写入。源码无法还原完整指令依赖的路径，采用临界 AIV 的分段时间模板，按每核工作量缩放。
3. 校准时分别读取 MTE2、SIMD、MTE3 和控制流水的指令区间。流水可同时繁忙。对已知调度用开始和完成时刻求关键路径，对时间模板把每个重叠区间归属一次。两种实施都要检查最慢核、尾轮和首尾开销。

以 dynamicMXquant 的 BF16 输入 `[1024,7168]` 为例，分核规则给出 64 个 AIV，最忙核处理 16 行，UB 每轮处理 4 行，共 4 轮。每轮读取 \(4×7168×2=57344\) 字节，按每 32 个元素一组得到 896 个量化组。读第一块要等完整读延迟，后续块可按发起间隔推进，第三块还要等第一块释放输入槽。因此把单块时间直接乘 4 会丢掉双缓冲效果。RMSNorm 同属 Vector，单行数据要先完成平方和归约才能得到缩放因子，其依赖和动态量化不同。第 3 节给出量化的逐轮递推，第 6、8、9 及 11 节给出各 Vector 路由的可执行公式。

### Cube 算子怎样建模

1. 按安装包的矩阵布局、dtype 和分支选择输出 tile，再把每个 tile 的 K 方向分成归约段。输出块之间可以并行，同一输出块的累加需要按归约段推进。M/N 尾块、K 补齐、专家之间不能拼接的边界都改变任务数。
2. 将输出块映射到 AIC，计算每核最多几轮。对代表块估算 A/B 读入量、矩阵乘加工作和输出字节，分别求读、Cube 和写出阶段服务时间。校准时检查 MMAD 指令数、块数、活跃 AIC 和阶段区间。
3. 对规则整块流水可用 \(T=H+R+C+F+(w-1)\max(R,C,F)\)。这里 \(R,C,F\) 分别为一轮读取、矩阵计算和写出的微秒数，\(H\) 为未被阶段覆盖的控制时间，\(w\) 为最忙核的轮数。首轮先填满流水，后续轮次受最慢阶段限制。“MatMulV3”一节给出目标芯片的跨波调度，第 4 节另述 DT 参考校准。当第一波内部已经存在明显重叠时，Addmm 用最长队列时长加上次要队列暴露的时间，具体公式见第 6.2 节。

例如 BF16 Addmm 的 `[M,N,K]=[9,128,256]` 配合 `self=[128]` 与单位系数。按 16×64 候选输出块，列方向有两块，两核一轮可以完成。每块逻辑上需要 9×256 个 A 元素、256×64 个 B 元素和 64 个 bias 元素，算出 9×64 个输出。两块共用同一片 A，实际搬运量还取决于缓存复用。K 增大主要延长归约，M 或 N 增大可能增加并行块或尾波。同样的总 FLOPs 可以得到不同耗时。专家 GMM2 还要对每个专家单独向上取整，再合并专家块调度。

### CV 融合算子怎样建模

1. 从实现列出 Cube、Vector、搬运和同步的生产者、消费者。例如 FA 的一块 K/V 先做 QK，交给 Vector 做 softmax 更新，随后 PV 再使用 Cube。核内缓冲深度决定下一块最早何时可以开始。
2. 在每个阶段使用 \(s=\max(e_{dep},a_{res},a_{buf})\)、\(e=s+t\)。\(s,e,t\) 为阶段开始、结束、服务时间，单位都是微秒。\(e_{dep}\) 是最晚前驱完成时刻，\(a_{res}\) 是设备可再次接受工作的时刻，\(a_{buf}\) 是缓冲槽可用时刻。长循环压缩为首轮填充、稳态间隔和尾部排空。运行时难以完整恢复调度的融合路径，保留开发 trace 中各流水同时活跃的区间模板，按目标 tile 的 Cube、Vector、搬运工作分别缩放，在每段重叠区间取最长服务。
3. 用独立 shape 检查瓶颈是否真的随工作量移动，包括长 KV、尾块、专家偏斜及不同核数。多 kernel 串行链分别估每段，再求净窗口之和。外部通信、host launch 和两次 launch 之间的空闲需在系统层计入。

例如 950DT 参考校准模式下，普通 FA 的 `Q=9,L=32768,D=64,B=H=1`。该路径预测 1 个 AIC、2 个 AIV，每个 KV tile 256 token，共 128 轮。已校准的首轮及首尾固定部分合计 10.498 µs，每增加一轮约 0.5172 µs，故预测约 \(10.498+127×0.5172=76.18\) µs。这个例子中延迟和发起间隔分别由 32k 的开发 trace 识别，其他布局和因果属性需要各自的分支。MoE 的 GMM1–SwiGLU–量化同样采用 CV 方法，利用本 rank 专家 token 直方图计算各资源的块数。

### 逐算子算例

每页从一个具体的本卡 shape 和输入属性出发，给出工作量、分块、耗时的数字推导，以及相应库窗口的校验范围。

| 路线 | 算例 |
|---|---|
| Vector | [quant](examples/quant.md) · [rmsnorm](examples/rmsnorm.md) · [mhc_sinkhorn](examples/mhc_sinkhorn.md) · [mhc_post](examples/mhc_post.md) · [kda_recurrent](examples/kda_recurrent.md) |
| Cube | [matmul](examples/matmul.md) · [addmm](examples/addmm.md) · [gmm2](examples/gmm2.md) |
| CV 与组合 | [fa](examples/fa.md) · [lightning_indexer](examples/lightning_indexer.md) · [mhc_pre](examples/mhc_pre.md) · [mhc_pre_sinkhorn](examples/mhc_pre_sinkhorn.md) · [mhc](examples/mhc.md) · [grouped_matmul_swiglu_quant_v2](examples/grouped_matmul_swiglu_quant_v2.md) · [moe_ffn](examples/moe_ffn.md) |
| 结构可用、时延待校准 | [kda_chunk](examples/kda_chunk.md) |

## 目标芯片怎样决定一个 tile 的时间

**新芯片的默认路径。** 对 Quant、Matmul、FA，给定同一 shape，目标核数和缓冲容量先决定分核和 tile。各 tile 产生按精度区分的运算数、各级传输字节和需要先后执行的操作。目标芯片对这些需求计价。

| 核心量 | 目标上的计算方式 | 换芯片时什么会变 |
|---|---|---|
| 计算 | 对齐后的工作 F 除以每核吞吐 p，再加指令发起和流水填充周期。p=1000×整片 TOP/s÷核数÷GHz | 核算力、每波可并行 tile、尾波占用 |
| 访存 | 按层级区分字节 D，每个端口的服务为 D/b，b 为目标 B/cycle；并行端口的竞争与总带宽约束取最大，加首读延迟 | 同一 tile 的首次就绪时间、续发间隔、带宽瓶颈 |
| 流水 | 前驱完成、资源可用、缓冲槽释放三者约束下取最早开始。长循环合并相同块，最后按 1000×GHz 将总周期换为微秒 | 哪个阶段能隐藏、哪一段成为关键路径 |

例如保持时钟和单核算力，增加核数只对可分给更多核的任务有用。单头 Q=9 的长 FA 仍是单个 Q 任务；额外 Cube 核不会平分串行 KV 循环。提高全片带宽也只加速带宽真正限制的阶段。可以用同一参数文件显式改发起周期、MTE 端口或数组峰值，检查总时间是否应随该瓶颈变化。[资源敏感性与历史回放](../results/v0.16/resource-replay.md)

**硬件输入及其来源。** 全片算力采用同一调度域的稠密峰值（FMA 算两次）。32/16/8/4 位是四个依次排列的字段；同位宽的数据类型也可能使用不同指令。DT、PR 的位宽表据 950 架构白皮书表 3-1 对应核数档位，频率、存储容量和核数来自 CANN 平台 INI，PR 片外带宽按 112 GiB 档推定。970DT 是调用方给出的架构情景。

| 参数 | 950DT_9582 | 950PR_957b | 970DT 情景 |
|---|---:|---:|---:|
| AIC / AIV；GHz | 32 / 64；1.65 | 28 / 56；1.65 | 120 / 120；1.8 |
| Cube 32/16/8/4 位 TOP/s | 未知 / 432 / 865 / 1730 | 未知 / 378 / 756 / 1513 | 未知 / 1770 / 3540 / 14160 |
| Vector 32/16/8/4 位 TOP/s | 27 / 54 / 54 / 未知 | 23 / 47 / 47 / 未知 | 55.296 / 110.592 / 110.592 / 未知 |
| 外存容量（十进制 GB）/ GB/s | 103.08 / 4000 | 120.26 / 1400 | 288 / 14400 |
| L2 容量 / 整片带宽 | 128 MiB / 未配置 | 112 MiB / 未配置 | 200 MB / 30000 GB/s |
| 每核 UB / L1 / L0A / L0B / L0C | 248 / 512 / 64 / 64 / 256 KiB | 同 DT | 继承 DT |

970 的 Cube 比例取 16:8:4 位为 1:2:8。Vector 采用保持每核数据通路的假设：每 AIV 每拍每路 128 个 16 位乘加，双发射，每次乘加计两次运算，故 $120×1.8×128×2×2/1000=110.592$ TOP/s。DT 的同一估法给出 54.0672，与配置的 54 整数峰值接近。970 的 FP32 和 INT8 用 DT 的相对位宽比推得。双发射已经包含在整片峰值中，将总峰值换算为每核吞吐时不再乘二。归约、特殊函数、依赖及带宽仍约束实际执行。[DT 配置与来源](../configs/chips/Ascend950DT_9582.json) · [PR 配置与来源](../configs/chips/Ascend950PR_957b.json) · [970 配置与假设](../configs/chips/Ascend970DT.json)

**怎样读带宽利用率。** 某路径在一段 $t$ µs 窗口中搬了 $X$ 个 payload 字节，相对其名义 $B$ GB/s 的比例是 $X/(1000 t B)$。核内 MTE 的分母按参与核数乘每核端口带宽，整片 HBM 或 L2 的分母只用一次整片带宽。启动、尾块、每核短传输、描述符发起间隔都会压低有效比例。同一个算子有三种不同的量：接口字节只计张量读写；模型按复用假设还原的各级 payload 用于定价；trace 指令忙区并集衡量流水有事件的时长。后两者也不能互换。Matmul 直接返回整窗 HBM、L2 的 payload 比例；FA 返回各波满块稳态的 K/V 读占整片 HBM 比例；Quant 给冷读写字节与共享带宽下界。真实 MTE/HBM 利用率仍需硬件计数或精确路径字节。

同一 970DT 配置在分别改变 UB 容量、每核写端口、共享带宽、窗内回写和 FA 核绑定时，小输入和大输入的反应不同。逐项替换结果见[未知参数灵敏度](../results/v0.17/sensitivity.md)。

**校准的边界。** 新路径的时序参数只来自 DT 的阶段观测。常量包括有效带宽、特殊向量指令等效成本和未拆开的等待周期，其中一部分会随微架构和编译器变化。整片峰值不能推导这些常量。缺少目标实测时继承已说明的 DT 假设，可用已知目标细项替换。历史 PR 证据用于检查分块、末波调度和写回重叠，Quant 读写系数仅从 DT 描述符识别；结构选择与 DT 阶段参数的数值来源分别记录。新的 PR shape 在冻结后采集，检验整套方法对同代另一 SKU 的预测能力；960/970/980 只有带条件的架构情景估计。

**选择计时路径。** `timing_mode="resource"` 对以下三项显式选择目标资源调度，芯片名不参与计算。`auto` 在原 DT 配置使用长期校准路线，其他配置上的这三项走资源调度。其余算子现有目标投影仍依赖 DT 阶段模板，显式要求资源法会返回 `unsupported`。系统里可用 `projection.method` 区分。已登记 PR 需要复现旧版专属估计时使用 `calibrated`。[输入和返回值](integration.md)

### DynamicMXQuant 的目标资源模型

固定合同为 ops-nn 的 `aclnnDynamicMxQuant`，BF16 输入沿末轴每 32 个元素一组，输出 E4M3FN 数据和 E8M0 scale。目标单卡输入为行数 R、列数 C。第 3.1、3.2 节的行列分核和 UB 规则使用目标 AIV 总数和 UB 容量，给出实际分配核数 A、最忙核行列尺寸和本地 tile 的行列 h、w。实现假设来自 950 的 tail-axis 路线。

**每个 tile 要做多少工作。** 分组数为 `g=h⌈w/32⌉`，BF16 输入读 `x=2hw` B，数据和 scale 合计写 `y=hw+g` B。主向量循环和 scale 整理循环合计 `n=⌈g/8⌉+⌈g/128⌉` 轮。模型将每轮换成 DT 观测的等效工作 κ=3469.52 峰值操作单位，代表指令混合与占槽效率。该数不能解读为 3470 条 ISA 指令。尾块按实际 h、w 重算。

**按目标硬件计价。** 下式所有时长均为目标周期。f 为 GHz，P 为全芯片 16 位 Vector TOP/s，V 为全部 AIV 数，`p=1000P/(Vf)` 为每核每周期的峰值操作数。缺失 P 时使用 DT 每核每周期能力。令 `B=min(B_HBM,B_L2)/f` 为全片可用 B/cycle，未给 L2 吞吐时只使用已知的片外限额。以 `a=RC/(r_core c_core)` 个等效满载核估计共享带宽分摊，分母是最忙核行列尺寸。局部读、发起、写数据率分别为 bᵣ、bᵢ、b𝓌，均为单核 B/cycle。取局部与共享限额中较慢者。

| 每 tile 服务 | 公式与含义 |
|---|---|
| 读就绪延迟 | `L=max(L₀+x/bᵣ, x/(B/a))`，L₀ 是描述符发起到首批数据就绪的固定部分。 |
| 读发起端占用 | `I=x/min(bᵢ,B/a)`，前一次数据仍在途时，端口可以发起下一块。 |
| 向量服务 | `S=H_v+H_dep+nκ/p`，H_v 是 VF 发射/固定指令与未分离的队列开销，H_dep 是输入就绪到发起 VF 的间隙。 |
| 输出数据 | `D=L𝓌+hw/b𝓌` 为数据描述符服务。两条描述符分别计价，合并公式见下。 |

每个 scale 占 1 B。将各核所写的 scale 行列映到连续地址，计算同一 32 B 粒度中最多的不同写核数 $k$。多个生产者共享同一粒度时，开销用 $q=130.867(k-1)\log_2 A$ 周期近似。单写者时此项为零。scale 描述符服务 $S_w=\max(307+g/b_w,D+81)+q$；输出窗口 $W=\max(D,1+S_w,(hw+g)/b_w)$ 周期。307 和 81 是 scale 自身固定服务和数据写后的暴露尾部，1 是两个描述符的典型发起差。32 B 表达描述符相邻写者几何；实际仲裁单元、缓存与冲突因果尚待额外证据。

**如何把服务合成时间。** 设 Uᵢ 为第 i 块读发起，Vᵢ 为向量结束，Eᵢ 为写结束，首块从周期 0 开始，尚不存在的前驱约束忽略。则
`Uᵢ=max(Uᵢ₋₁+Iᵢ₋₁,Vᵢ₋₂)`，
`Vᵢ=max(Uᵢ+Lᵢ,Vᵢ₋₁,Eᵢ₋₂)+Sᵢ`，
`Eᵢ=max(Vᵢ,Eᵢ₋₁)+Wᵢ`。
第三次预读要等第一块用完其输入槽，第三次计算也要等第一块输出写完。总耗时为
`[H₀+h₀ log₂A + max(E_last,(X+Y)/B) + H_end]/(1000f)` µs。
X、Y 为全张量的读写字节，按冷输入、输出均写至片外的情景给出共享总量下界。H₀ 是首次发数据读之前的启动周期，h₀ 是活跃核数每翻倍的额外到达周期，H_end 是最后写完成至 kernel 指令结束。一次 UB 直接闭式计算。多轮利用五个状态的最大加法递推，重复等尺寸块用矩阵快速幂，复杂度随轮数对数增长。

**校准参数。** 启动、计算、端口发起项沿用 950DT 的分段观察。新增的两描述符固定项、每核端口斜率和 scale 同粒度写者项，从旧 DT DMA 边界提取，未用 PR 总时间拟合。这些值是带有本实现等待的有效成本，尤其多个 AIV 争用的机制仍是假设。[参数](../results/v0.17/quant.json) · [DT 写描述符证据](../results/v0.17/quant-dt-evidence.json) · [提取脚本](../development/validation/sku17/quant_dma_calibrate.py)

| 细项 | DT 缺省值 | 可替换的目标字段 |
|---|---:|---|
| 启动 H₀、每次核数翻倍延迟 h₀、排空 H_end | 4746.54、139.47、16 周期 | `quant_launch_cycles`、`quant_arrival_log2_cycles`、`quant_end_cycles` |
| 读、数据写固定延迟 L₀、L𝓌 | 972.53、242.51 周期 | `quant_read_latency_cycles`、`quant_write_latency_cycles` |
| VF 固定与交接 H_v、H_dep | 850.52、5 周期 | `quant_vf_dispatch_cycles`、`quant_vf_dependency_cycles` |
| 本地读 bᵣ、读发起 bᵢ、写 b𝓌 | 48.36、68.91、92.03 B/cycle/核 | `mte2_bandwidth_gbs` 同时覆盖两种本地读率，`mte3_bandwidth_gbs` 覆盖本地写率，输入单位为 GB/s/核 |

**算例。** BF16 [768,4096] 读 6,291,456 B，写 3,244,032 B。DT 有 64 个 AIV、6×4096 的 UB 块，首块读 49,152 B、102 次向量循环，同粒度写者 1。启动、UB 流水、排空分别为 3.384、3.440、0.010 µs，合计 6.833 µs。PR 最忙核使用 55×512 块，首块 117 次向量循环，2 个写者，估计 759.99 周期的 scale 竞争。三段 3.368、6.811、0.010 µs，合计 10.188 µs。970 配置得 6.069 µs。只有分给额外核的并行工作能缩短；额外写者可能延长尾部。[逐步算例](examples/quant.md) · [两条写描述符的测量与限制](../results/v0.17/quant-method.md)

### MatMulV3：从目标资源独立计算 BF16 稠密 GEMM

输入 `[M,N,K]` 和目标芯片配置后，[模型](../operator_model/matmul_resource.py)直接输出时延、活跃核和各阶段耗时；同一组资源换成任意芯片名，结果相同。计算中既不读取另一颗芯片的总预测，也不加载 SKU 配对时间、按形状查表或以总时延回归。适用对象为已支持的安装包 `aclnnMatmul → MatMulV3`、BF16、无转置的连续 ND 格式、无 bias。对未来芯片的假设是库仍使用类似的分块、DMA 和 MMAD 指令；算法或指令选择改变时仅是情景预测。

**输出块与工作量。** 以目标 AIC 数、L0/L1 容量和算存比搜索主块 $m×n$、L0 的 MMAD 深度 $k_0$ 和 L1 搬运页深 $p$。共有 $J=⌈M/m⌉⌈N/n⌉$ 个父块。前面已有整波、末波仅余 $t=J\bmod C>0$ 个父块时，库可以用空闲核继续切分。按 ASW 的 M 优先策略增加行、列切数 $s_m,s_n$，使 $s_m s_n t≤C$；N 方向额外受 128 B 条件约束。运行时按实际尾边裁剪每片。每个输出片仍需完成整个 K，MMAD 总数是实际输出片数乘 $⌈K/k_0⌉$。每条片内指令的阵列工作约为 $2\hat a\hat b\hat z$，其中 a、b、z 分别为其行、列、K 深度，帽号表示向上对齐 16。输出的有用运算是 $2MNK$。

**传输与缓存假设。** 对一波实际输出片 $(a,b)$ 逐片求和，深度 $d$ 的 L2→L1 流量 $X_d=2dΣ(a+b)$ B，写 $Y=2Σab$ B。L1→L0 使用相同操作数流量代理。输入页可容纳于 L2 时，假设整算子的 HBM 冷读去重为 $2K(M+N)$ B，超过容量则按 L2 重复流量计价。整输出 $Y_o=2MN$ B；$Y_h=Y_o-ρ\min(Y_o,\max(0,S-P))$ 是本 kernel 指令窗内暴露的片外写。$S$ 表示 L2 容量，$P=2(M+N)\min(p,K)$ 是唯一输入页字节，$ρ$ 是允许留脏比例。默认 $ρ=1$；设 0 则要求全量落片外。所有情景都保留最终完整写出字节，容量只能约束可能驻留量。

**资源服务（单位为目标周期）。** 每核读、L0、写带宽为 $β_2,β_1,β_w$，单位 B/cycle；整片有效 L2 和 HBM 带宽 $B_L,B_H$ 分别是配置 GB/s 除以 GHz 后再乘可用系数 $η_L,η_H$（默认 1）。令 $γ$ 为算子 HBM 输入/L2 输入的字节比例。对每波的最长描述符，读 payload 服务 $D(d)=\max(2d\max(a,b)/β_2,X_d/B_L,γX_d/B_H)$，首读完成 $R=L_r+D(d)$，$L_r$ 是首次读固定延迟。波内 L0 首片服务 $L=L_0+2z\max(a,b)/β_1$，$z=\min(k_0,d)$。每条 MMAD 的峰值部分为对齐运算/单核算力 $π$。若一页 $n_f$ 片，$C(d)=(n_f-1)\max(I_c+F(z)/π,2z\max(a,b)/β_1)+L_c+F(z_t)/π$，$F$ 取波内最大对齐工作，$z_t$ 是末片深度，$I_c,L_c$ 是续发与完成固定周期。页间隔 $I_p=\max(2I_r,D(d),R/2,C(d))$，其中 $I_r$ 是一个读描述符的发起间隔，分母 2 对应两个预取槽。共 $n_p$ 页时，前置页占 $(n_p-1)I_p$，最后一页按真实深度重算。写 payload $W_i=\max(2\max(ab)/β_w,Y/B_L,Y·Y_h/(Y_o B_H))$，写完成还需加固定延迟 $L_w$。

**跨波调度。** 下一波的读可以在上一波计算或写回尚未结束时发起。每到新波，计算开始取“该波预读和 L0 就绪、上一波 Cube 发起端释放”较晚者；写开始再取“本波计算完成、上一波的写 payload 服务完毕”较晚者。前置同质波通过发起间隔闭式累计，至多枚举最后一波的重切片，最后按最晚写完成排空。全体 L2 读写/$B_L$、本窗 HBM 读写/$B_H$ 同为整窗下界，只对超出流水预测的差额补时。发起成本 $H_0+h\log_2 A$ 整次调用付一次，$A$ 为活跃 AIC 数。总周期除以 $1000f$ 得 µs。返回的 `shared_traffic` 分别给出全片 HBM/L2 payload GB/s 与名义峰值比例。[重切几何和跨波递推细节](../results/v0.17/matmul-method.md)

**校准证据。** [导出脚本](../development/validation/sku16/matmul_dt_calibrate.py)仅枚举 v0.2 的 DT 捕获清单与 `runs/v8cube/matmul_*` 的成功 DT 记录，对照 SoC、原始 trace 或 DB 的 SHA 后提取。17 个不同旧 DT shape：10 份 msOpProf，7 份 npusim。逐个选取尾核；对首对读、L0 提前量、最后 MMAD 后的写回分别按该步的字节量解非负直线，截距是周期，斜率倒数是有效 bytes/cycle；首发 / 到达按核数对数校准，MMAD 首延迟和 II 取扣去对齐 ops/π 后的中位数。总实测时间只在参数冻结后的回放中使用；不作为任何参数的拟合因变量。[参数](../results/v0.16/matmul.json)与[来源清单、原始阶段观察和 SHA](../results/v0.16/matmul-dt-provenance.json)均可重建。17 个 DT 旧 shape 在当前调度下的开发回放 WAPE 为 4.12%。

| 参数 | DT 来源/默认值 | 配置覆盖（可选） |
|---|---|---|
| H₀、h | 3381.3 cycles；91.25 cycles/核数翻倍 | `matmul_launch_cycles` 修改 H₀；h 暂继承 |
| Lr、Ir、β₂ | 1173.5、15 cycles，47.15 B/cycle；Ir 指两条输入描述符之间，不是整页 completion 的 II | `matmul_read_latency_cycles`, `matmul_read_ii_cycles`, `mte2_bandwidth_gbs` |
| L₀、β₁ | 33.69 cycles、180.65 B/cycle | `matmul_l0_latency_cycles`, `mte1_bandwidth_gbs` |
| Lc、Ic | 99.92、97.92 cycles；按平均有效成本继承，不等同单元硬性流水深度 | `matmul_mmad_latency_cycles`, `matmul_mmad_ii_cycles` |
| Lw、βw | 381.81 cycles、158.30 B/cycle；βw 是暴露写回代理，不是独立测得的 HBM 吞吐 | `matmul_drain_cycles`, `mte3_bandwidth_gbs` |
| βL、βH | L2 缺省沿用 100 B/核/周期的库调度提示；HBM 来自目标必填配置。没有单独的命中率、争用或 DRAM 首延迟校准 | `l2_bandwidth_gbs`（片级），`hbm_bandwidth_gbs`（片级） |

**算例。** DT [256,512,256] 有 32 个 64×64×256 输出 tile，各一条 MMAD。有效工作 67,108,864 ops、冷输入 393,216 B、L2 重复输入 2,097,152 B、输出 262,144 B。启动、首读、转 L0、MMAD、写回为 2.326、1.132、0.130、0.216、0.281 µs，合计 4.086 µs；DT CA 为 4.192 µs。PR [1024,2048,32] 的 4 个末波父块各拆成 3×2，合计 28+24=52 条 MMAD，预计 4.823 µs；旧 PR CA 实测 4.464 µs。后者用于判定跨波机制，属开发回放。参数只定价库已支持的 BF16 ND 路线。输入复用、写回驻留及未来库分块仍是可替换情景。[逐步算例](examples/matmul.md)

### BF16 稠密 FA 的绝对资源模型

接口输入每卡 `[Q,KV,D,B,H]`。实际校准对象是 CANN 9.2.1 ops-transformer 的 `aclnnFlashAttentionScoreV2`，采用 SBH、非因果、无 mask/dropout、D=64/128、Q/K/V 头数相同。预测按目标芯片重新决定分核、每个阶段的服务和暴露时间。参数集的十个开发数据库均来自 950DT。所有设备走相同计算图，在线不读取 CANN、trace 或其他 SKU 的时间记录。[资源模型](../operator_model/fa_resource.py) · [DT 输入文件、哈希与提取的阶段](../results/v0.16/fa-dt-evidence.json) · [校准脚本](../development/validation/sku16/fa_dt_calibrate.py) · [冻结参数](../results/v0.16/fa.json)

#### Tile 和波次

设 $C,V$ 为整卡 Cube 和 Vector 核数，初始任务数 $G_0=BH\lceil Q/128\rceil$。对于小于 $C$ 的初始网格，基本 Q 块 $M=64$，其余 $M=128$。满足前述小网格且 $KV>1024$，或满足 $D=64$、$Q$ 是 128 的倍数、$KV$ 是 256 的倍数时，基本 KV 块 $N=256$，其余 $N=128$。后一个条件用模运算表示为 $Q\bmod128=KV\bmod256=0$。这是 950 host regular basic 路由在所列输入子域上的规则。工作单元数 $G=BH\lceil Q/M\rceil$，同时占用 $A=\min(C,V,G)$ 个 AIC 和 $\min(λA,V)$ 个 AIV，需 $\lceil G/A\rceil$ 波。相同的满波合并计价，每波用剩余真实 Q 行的最长任务计时。λ 是每个忙碌 AIC 最多能使用的 AIV 数，由实现绑定决定。配置项 `fa_aiv_per_aic` 缺省为 2，970DT 配置取 1；可设 2 研究允许借用空闲 AIV 的情景。

#### 工作与资源

以下计算单位是目标时钟周期，最后除以 $1000 f$ 得到微秒，$f$ 是 GHz。令 $m,n$ 是当前 tile 的实际 Q/KV 长度，$d=D$，带帽表示向上对齐 16。此处 $u=\lceil n/128\rceil$ 是一个 KV tile 的 MMAD 分段数，两个乘法各需要 $u$ 段。$r=\lceil m/\min(λ,V/a)\rceil$ 是本波每个繁忙 AIV 的保守行数，$a$ 为本波实际 AIC 数。

| 服务 | 每 tile 的工作与价格 |
|---|---|
| QK 与 PV | 各 $2mnd$ 有效 op，各 $F=2\hat m\hat n\hat d$ 填充后算术 op，每次乘加计两次。$t_Q=F/P_C+u h_Q$，$t_P=F/P_C+u h_P$。$P_C$ 为整芯片 BF16 TOPS 除以核数和时钟换算成 op/cycle/core。$h_Q,h_P$ 来自第一块 QK、PV 各自 MMAD 起止跨度，扣去该块峰值服务再除指令分段数。|
| score 与 PV 更新 | 每块 score 暂计缩放、求最大、指数差、累加、归一相关的五项等价工作，输出累计暂计每行每通道三项，暴露服务 $s_1=5rn/P_V$、$s_2=3rd/P_V$。$P_V$ 由目标每核 FP32 峰值乘以 DT 校准的效率。DT 记录的 SIMD 忙时含特殊函数、归约及发射成本，等价工作量不等于实际 ISA 条数。|
| GM/L2 到核内 | Q 每 Q 任务只读 $2md$ B，K、V 每 KV tile 各读 $b=2nd$ B。单核流式速度 $R=\min(R_{local},B_{MTE2},B_{L2}/a,B_{HBM}/a)$，单位 B/cycle。未知 L2 或显式 MTE2 端口限额时，继承 $R_{local}$（80.07 B/cycle）的上限。HBM 使用目标显存介质的整卡规格。在多个 Q 任务间默认 K/V 各自冷读，跨核命中会减小真实外存流量。|
| L1→L0 与 FIX | 每块近似传 $b_1=2[2\hat n d+\hat m(d+\hat n)]$ B 的操作数，包括 Q/K 与 P/V。FIX 近似处理 $b_F=4\hat m(\hat n+d)$ B 的 FP32 结果。带宽 $R_1,R_F$ 来自 DT 长循环相应流水的非排除指令忙时。可用每核 MTE1 GB/s 显式覆盖 $R_1$。|
| 输出写回 | 最后一块写出 BF16 输出 $2md$ B，两份 FP32 [m,8] 统计量共 $64m$ B。$R_W$ 计可用 MTE3 每核及 HBM/L2 全卡带宽，未知 MTE3 时使用 DT 局部流速。 |

#### 首轮、稳态与尾块

首轮先等待 Q 与 K，随后做 QK、Vec1，最后等待 V 后执行 PV。以 Q 首次发起为起点，$L$ 表示首读固定延迟，$g_K,g_V$ 表示 Q→K、K→V 首发差值，$g_0$ 表示读就绪至首 MMAD 开始的间隙，$h$ 表示 QK/Vec1 与 V 都就绪后仍暴露的控制及交接周期。定义

$$t_{QK,start}=\max(L+2md/R,\ g_K+L+b/R,\ (2md+b)/R)+g_0,$$
$$t_{PV,start}=\max(g_K+g_V+L+b/R,\ t_{QK,start}+t_Q+s_1)+h,\qquad T_{fill}=t_{PV,start}+t_P.$$

两个初始请求可以重叠，同一端口的累计 payload 仍受带宽约束。填充时的等待和后续发起两个 tile 的间隔可以不同。稳态使用

$$I(n)=\max(t_Q+t_P,\ s_1+s_2,\ 2b/R,\ b_1/R_1,\ b_F/R_F,\ Ju/2).$$

$J$ 是 DT 32k 的两条 256-token K/V 流中段较大间隔，约 853.40 周期。物理上它含当时的发起、缓冲复用和同步节拍，这三个份额不能由同一条 trace 唯一确定。外推采用继承该有效间隔的假设，128-token 分段按一半计价。对有目标证据的芯片，可分别设置 `details.fa_issue_cycles`、`details.fa_read_latency_cycles`、`details.fa_mte_bytes_per_cycle`。这些字段覆盖相应的绝对单位，当一个带宽或算力项成为更慢的资源时，$I$ 会按该资源重新定价。

令 $K=\lceil KV/N\rceil$，末轮有效长度 $n_t=1+(KV-1)\bmod N$。第一块用 $n=\min(KV,N)$，当 $K>1$ 时加上 $(K-2)I(N)+I(n_t)$。排空取 $s_2+h_D+(2md+64m)/R_W$，$h_D$ 为末条 PV 完成到全核结束的跨度扣除上述向量和写回后的剩余周期。所有波累加，首次 Q 发起前的 launch 周期整次调用只加一次。读延迟、发起间隔、计算服务和暴露控制残差在资源模型中分别可见。预测只处理有限种首尾 tile 和工作波，复杂度不随 KV 或头数线性增长。

#### 算例与证据边界

DT 的 [9,32768,64,1,1] 使用 1 AIC、2 AIV，Q 只搬 1152 B，128 轮 K/V 各搬 32768 B，共 512 条 MMAD。每个满块两次乘法各有 294912 有效 op；按 16 行填充后，各定价 524288 op。预测分为启动 6.479 µs、首轮 3.877 µs、127 次续发 65.686 µs、排空 0.334 µs，合计 76.377 µs。该 DT 开发库实测 75.980 µs。若改为 KV=16385，模型计 64 个满块和 1 个单 token 块，末次 K/V 各搬 128 B，末块每种乘法各一段。在 [64,2048,128,1,64] 上分别降低 Vector 峰值、Cube 峰值和显存带宽，独立测试检查瓶颈资源变化确实提高预测时长。[资源敏感性与尾块测试](../tests/test_fa_resource.py)

校准分支由 host 的 KV 块模板（128 或 256）选择。首尾参数来自相应 DT 模板下各条指令边界的中位数；256 模板只有 4k 与 32k 两条开发记录，不能推断控制周期在其他微架构完全相同。由短期最快 8 回合 K/V 发起窗口得到的局部流速包含实现影响，它只代表当前代码观察到的有效速度。多个 wave 间同一 kernel 的 overlap、不同 head 的 cache 命中、GQA 共享 KV、因果裁剪、专用 FIA/KDA、同一 wave 不均匀向量绑定及模型外的 KV 存储格式均未在本校准中测定。若目标缺 BF16 Cube 或 FP32 Vector TOPS，沿用 DT 的每周期每核能力假设。

**换芯片后哪段变慢。** 950DT 与 970 情景均用资源模式，单头 Q9、32k 分别为 76.377 和 70.033 µs，两者稳态均受 K/V 发起间隔限制。本卡 B4、H8、Q17、32k 时，DT 的 32 AIC/64 AIV 主要受 K/V 读取限制；970 的 32 AIC/32 AIV 转为 Vector 限制，预计 97.270 µs，允许 2 个 AIV 池化后为 70.079 µs。核绑定会改变每个 Vector worker 的行数，即使全片峰值提高，特定 shape 也可能变慢。本卡 B128、H8、Q9、32k 则从 32 波降到 9 波，两端均为 K/V 读取限制，预测 2218.584→642.574 µs。以上是资源预测；970 缺少目标 CA。各波的瓶颈、L/K/Q/B/H、起止延迟、配置假设一并列出。[场景比较](../results/v0.17/fa-scenarios.md) · [逐步算例](examples/fa.md)

历史实现级开销测量中，复用已解析的 ChipConfig，通过 resource_fa 直接查询各一千次。[9,32768,64,1,1] 中位数 0.034 ms、P95 0.039 ms、最大 0.337 ms；[2048,524288,128,1,144] 中位数 0.049 ms、P95 0.055 ms、最大 0.175 ms。此口径不含公共 API 的输入校验、工作量统计与返回对象构建；当前部署接口成本以[总误差报告](error-analysis.md)为准。

## DT 参考校准、PR 专属校准与其余算子的阶段投影

本节介绍其余路线所用的阶段投影和已登记 SKU 的显式校准。Quant、Matmul、FA 的新目标默认路径见上节。设备 0 是有 CA 数据的 950DT，设备 1 是调用方的目标。输入为已经完成模型并行切分的单卡 shape、dtype、布局和路由属性。输出保留这次数学工作的 Cube/Vector 操作数以及逻辑字节，重新给执行这些工作的队列、波次和依赖定价。可以将 960、970、980 设成不同配置做情景仿真。尚无这些芯片的实测误差。

### 从队列到时间

对单元 $e$（Cube 或 Vector）、精度 $b$，令 $C_{i,e}$ 是芯片核数，$P_{i,e,b}$ 是该芯片该精度的总峰值（十进制 TOP/s）。$J_e$ 是按 950DT 核函数 tile 粒度能独立分配的工作数。它来自算子分块，例如矩阵乘的 M/N 输出格、FA 的 batch × head × Q 分块、RMSNorm 的行。KV 上的串行循环不增加 FA 独立头数。目标每核服务时间相对参考值的倍率为

$$
r_{e,b}=\frac{P_{0,e,b}/C_{0,e}}{P_{1,e,b}/C_{1,e}},\qquad
w_e=\frac{\lceil J_e/C_{1,e}\rceil}{\lceil J_e/C_{0,e}\rceil}.
$$

保留 DT 分块的路线将 DT 计算队列服务微秒数乘以 $r_{e,b}w_e$。混合精度按可统计操作数对 $r$ 加权，隐含“每个已统计操作对参考时间等权”的近似。标量启动、同步和未识别控制按 $f_0/f_1$ 调整，$f_i$ 为 GHz。缺少同精度可比峰值时，计算也暂用时钟比，返回备注说明这一假设。算力已经包含运行频率，因此已使用算力比的计算阶段不再乘频率比。核数增长只在能减少波数的方向上产生收益。单条指令是否能与另一条配对仍由历史阶段效率近似；供电、降频和多算子并发争用由系统情景另行考虑。

用逻辑 I/O 给访存阶段建立带宽敏感度。令 $L$ 为本 kernel 接口一次读取各输入和写出各输出的字节之和，$T_0$ 为参考设备估计微秒数，$B_i$ 为整片外存带宽（十进制 GB/s）。估计的带宽敏感比例和 MTE2/MTE3 时间倍率为

$$
p=\min(1,L/(1000 B_0 T_0)),\qquad
r_{\rm MTE}=(1-p)r_{\rm local}+p B_0/B_1.
$$

1000 把 GB/s 换算成 B/µs。$r_{\rm local}$ 默认取时钟比乘所属计算域波数比；如两个配置都有可比的对应 MTE 带宽，则该项改用两带宽之比。主导域对 CV 暂取 Cube 域。$p$ 是同一 kernel 的逻辑流量/参考容量代理。缓存命中、同一权重多次读取、跨核重放、首访延迟均能造成偏差。它没有被拟合为可观测的 HBM 时间比例。HBM 容量用于系统级驻留和放置；修改容量本身不改变这个 kernel 的时间。Quant 按 UB 容量重新划分行列。Matmul 使用 L0、L1 和 L2 容量选择块形状与 K 片。其余路线保留各自的参考缓冲假设。

### 其余 13 条路线怎样得到目标时间

- **Vector。** RMSNorm 重新计算目标 SIMD 能覆盖的写回。设参考写服务为 $U$，已覆盖时间为 $H$，目标 SIMD 服务为 $V'$，则目标暴露写回为 $U'-\min(U',H r_{SIMD},V')$，其中 $U'=U r_{MTE3}$。mHC Post、Sinkhorn 和 KDA recurrent 将各自已经互斥的阶段按所属队列倍率迁移。参考 tile 和单位时间中的等待均可能随未来库变化。
- **Cube。** Addmm 对已观察到的最长队列、其余队列的暴露比例、控制和到达分别计价。GMM2 按各本地专家的 token 数决定独立 output tile，再对队列模板中的并发区间重新计价。独立的数学派生工作仍由 shape 和 dtype 决定。
- **CV 与组合。** Indexer、融合 PreSinkhorn、GMM1 按每个队列同时活跃的观测时间段计价。对活动集合 S 和参考模板时间 $t_S$，目标时间为 $t_S\max_{q∈S}[(W_q/W^0_q)r_q]$，其中 $W_q,W^0_q$ 是本次与参考的分块工作，$r_q$ 是该队列的目标时间倍率。mHC 拆分链 $T≤256$ 用每个阶段的截距、单位斜率和工作量迁移，详细步骤见第 8 节。多 kernel 分别重算后相加。KDA chunk 可以报工作和任务图，缺少合格耗时。

### 已登记 950PR 的可选校准

显式 `timing_mode="calibrated"` 时，Quant、Matmul、FA 在完整匹配 PR 配置的情况下可复现原专属阶段估计。Quant 的单 UB 高并发路径将启动、DMA 和 VF 分开，按已校准条件调整 DT 阶段；其余输入按原分支处理。Matmul 以首对读、L0、逐页 MMAD 发起/完成和最后写回分段；FA 以第一条 MMAD 之前的首发、QK/softmax/PV 重叠窗口、排空三段定价。它们保留原先覆盖的 shape 和误差。未知目标应使用上面的资源法或相应算子的参考阶段投影。历史参数与显式校准的结果可分别复现。[Quant 校准](../operator_model/quant_transfer_v14.py) · [Matmul 配对阶段](../results/v0.14/cube-phases.json) · [FA 配对阶段](../results/v0.15/fa-diagnosis.json)

### Matmul 的主块选择与 K 片深度

给定 BF16、连续 ND、不转置、无 bias 的 `[M,N,K]` 和芯片 JSON，先选择安装包 `aclnnMatmul → MatMulV3` 的主块，再计算 L0 的归约深度与 L1 的搬运深度。以下依据本地开源 [MatMulV3TilingHelper](https://gitcode.com/cann/ops-nn/blob/602e4ad985da00aad28c16432841e9c5069e10de/matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_helper.cpp)（ops-nn `602e4ad985da` 的 `GetRebalanceBlock / GetBaseK / CalL1TilingDefault`）重建普通 BF16 路径。开源参考与实际安装包不等价；是否选中同一划分由旧配对的**完整 MMAD 维度直方图**检验。

1. **筛选候选。** 枚举 M 方向 16 元素粒度和 N 方向 32/64/128 元素粒度的输出块 `(a,b)`，上限受输入形状、L0A/B/C 和 L1 容量约束。源码按计算、读入、写出的相对压力选 N 粒度。对应估计量为 `E=B₂/(1000P) + max(1,2(M+N)K/S₂)(1−B₂/Bh)(1/M+1/N)−(1+B₂/Bh)/K`，其中 `P` 为全芯片 BF16 Cube TFLOP/s，`B₂,Bh` 为 L2/HBM GB/s，`S₂` 为 L2 字节数。与 L0C 能容纳的参考大块的 `1/a+1/b` 比较：内存压力高时取 N 粒度 64，否则 32；若 `K<MN/(M+N)×Bh/B₂` 则以写回对齐为先，取 128。L2 带宽未知时，取源码默认提示的 100 字节/核/周期乘频率和核数；JSON 的 HBM、Cube 标称值近似代替源码私有 tiling rate，不是实测速率。芯片名不参与决策。
2. **平衡访存和核间负载。** 用 `J=ceil(M/a)ceil(N/b)` 表示输出块数，`C` 为 Cube 核数。单波的负载均衡率 `U≈MN/(C a b)`，若有多波则除以轮数（最后一波可分的情况下按源码调整评分）；极小维按实际尺寸算有效面积。按源码的比较顺序，用 `(1/a+1/b)/U` 较小的块，并在达到 Cube 压力边界时优先较均衡的块。直观上，块太大则多个核闲置，块太小则重复搬另一侧输入。输出活跃核为 `min(C,J)`，波数为 `ceil(J/C)`。
3. **确定 K 粒度，保留真实尾块。** 对 BF16，若 L0A/B 各可用 `S_A,S_B` 字节，每份操作数 2 字节，双缓冲，则一次 MMAD 可容纳的 K 上限为 `kmax=min(floor(S_A/(4a)),floor(S_B/(4b)))`。整段 K 对齐 16 后能装下就用整段；否则选不超过该上限的对齐片，优先 128 个元素的整数倍，再退到 64/32/16。尾 MMAD 用真实剩余 K，填充未计入数学 FLOP。L1 每页可合并 1 至 8 个计算片，需满足 `4(a+b)d≤S₁`、`8 max(a,b)d≤S₁`，其中 d 是候选页深、`S₁` 为 L1 字节数；按源码 256/512 字节对齐偏好和 48 KiB 单描述符目标挑选。核内 read、MTE1、MMAD 和末次写回仍沿用 v0.14 的阶段时间模型，但首页、跨页间隔和 MTE1 的片深现在依据实际选择的 d、baseK，而非固定 512/256。
4. **算例。** `[256,512,256]` 在 28 AIC 的 PR 选择 `a=96,b=64`，共有 `3×8=24` 个输出块，最后一行块仅 64 行；64 KiB L0A/B 双缓冲得到 `kmax=floor(65536/(4×96))=170`，每条 MMAD 取 K=128。一波 24 核、每核两条：32 条 `[96,64,128]`，16 条 `[64,64,128]`，与现有 trace 一致。DT 的 32 核则选择 `[64,64,256]`，32 核各一条。`[512,512,256]` 在 PR 调为 `[80,128,128]`，4 个 N 块 ×7 个 M 块恰用 28 核。直接替换 JSON 核数、资源、吞吐就可重算，但未来芯片仍是假设沿用同类实现。

### 从工作量到有用吞吐

设 $W_{e,b}$ 为资源 $e$ 在精度 $b$ 上累计的标量等价操作数，$P_{e,b}$ 为该资源在同一精度的整片峰值（TOP/s），$T$ 为预测微秒数。API 返回整片名义峰值占比

$$
u_e=\frac{1}{T}\sum_b\frac{W_{e,b}}{10^6 P_{e,b}} .
$$

分子是完成这些已统计运算所需的理想微秒数。对缺少峰值的有效精度返回空值。Vector 中一次特殊函数与一次简单加法的真实服务时间可以不同，混合操作的这个比值只表征有效工作。实际忙碌周期、物理外存事务需要单独的硬件或 CA 证据。逻辑接口吞吐用 $(L_r+L_w)/(1000T)$ GB/s，其中 $L_r,L_w$ 为以字节计的逻辑读和写。

## 每条路线返回哪些工作量


输入是完成并行切分后该卡的 shape 与张量属性。每个矩阵乘加计 2 次标量运算。Vector 按元素计数，一次 exp、比较、除法或 rsqrt 各计 1 次。不同类别每周期可完成的数量不同。字段 `cube_ops_by_dtype` 按矩阵输入精度归类，`vec_ops_by_dtype` 按所列代数步骤的计算精度或显式注明的源精度归类，`vec_ops_by_class` 进一步区分算术与特殊函数。部分可辨识的路线只累计已列步骤。

`logical_read_bytes`、`logical_write_bytes` 单位为字节，每个 kernel 的每个有效输入或输出张量按一次完整读或写计。组合算子的生产者写和消费者读分别计入。实际缓存复用、重复搬运、对齐填充和 workspace 需要各层事务证据，所以全部路线返回 `physical_hbm_bytes=null`。逻辑字节也不等于程序峰值显存占用。

| 路线 | Cube、Vector 工作量口径 | 读写范围与遗漏 |
|---|---|---|
| DynamicMXQuant | BF16 输入上逐元素取绝对值、每 32 元素取最大，后者每组 31 次比较 | 读 BF16 输入，写 FP8 数据及 E8M0 尺度。指数生成、缩放和编码工作未展开 |
| RMSNorm | FP32 平方、求和、乘均值系数、加 epsilon、rsqrt、归一化和乘 gamma | x 与 gamma 各读一次，写 BF16 y 和逐行 FP32 rstd。类型转换未计 |
| Matmul | BF16 Cube 为 2MNK | 读 A、B，写 Y。无分核间重复读 |
| Addmm | BF16 Cube 为 2MNK，融合 bias 的 MN 次加法单独记 `fused_bias_additions` | 增加 N 个 BF16 bias 输入 |
| FA | 稠密非因果 QK 和 PV 的 Cube 共 4BHQ·KV·D。FP32 逻辑 softmax 的缩放、比较、减、exp、求和、除法 | 读 BF16 Q/K/V，写 BF16 O 和两路 [B,H,Q,8] 的 FP32 统计量。在线分块归并未展开 |
| Lightning Indexer | 按各 query 可见 key 数累计 BF16 点积、FP32 ReLU、逐头加权与归约 | 读 query、key、weights，写每行 top-512 的 int32 索引。TopK 比较数随实现变化，未计 |
| mHC Pre | 宽度 W=N²+2N 的投影为 2TNDW。HF32 按 32-bit 输入算力代理。Vector 含 RMS、仿射、sigmoid 和 residual 混合 | 读 BF16 x 与 FP32 phi/alpha/bias，写 BF16 h_in 和五个 FP32 输出（h_post、h_res、inv_rms、h_mix、h_pre） |
| mHC Sinkhorn | 按实际 I 次迭代计 FP32 最大值、减法、exp、归约、epsilon 加法及 2I 轮逐元素除法 | 读、写各一个 FP32 [T,N,N]。推理合同没有辅助输出 |
| mHC Post | FP32 residual 矩阵混合、post 支路乘加，另按 FP32 源元素数计输出 cast | 读 x/h_out/h_post/h_res，写 BF16 [T,N,D]。切核和驻留状态不改变逻辑张量 |
| mHC PreSinkhorn | 融合 Pre 与 Sinkhorn 的代数步骤 | 写三个有效推理张量。need_backward=false 的辅助描述符不计有效写，片内中间结果不重复计 |
| mHC | 实际 split Pre、Sinkhorn、Post 的工作之和 | 分别累计三次 kernel 输入和输出，外部 attention/MLP 单独调度 |
| KDA recurrent | FP32 核心递推为 B·S·Hv·(7KV+2V)。门处理与缩放未完全展开 | 含 Q/K/V、gate、beta、初始状态的读以及输出、最终状态的写 |
| KDA chunk | 按分块累加可见三角点积、稠密等价矩阵项和 state decay。三角求解等项未知 | 按选择的导出计状态、两路注意力和 gate 前缀。原始 gate 或未知附加输出会令对应方向的总字节为 null。计时未验证 |
| GMM1–SwiGLU–MX | 活跃 token 数 M 的 MXFP8 Cube 为 4MKF，FP32 SwiGLU 的五个标量步骤各 MF | 读活跃专家权重、输入及两者 MX 尺度、int64 专家直方图，写 FP8 与尺度。重定标与转码工作未展开 |
| GMM2 | MXFP8 Cube 为 2MFK | 同样计有效专家、尺度及直方图。输出 BF16 [M,K]，MX 缩放并入矩阵乘 |
| MoE FFN | 上述 GMM1 与 GMM2 合计 | 组件间 FP8 激活及尺度均计一次写和一次读，通信及路由由系统层处理 |

### 可手算核对的例子

- RMSNorm [2,6144]，令元素数 X=12288。平方、乘逆均方根、乘 gamma 共 3X 次乘法，每行另乘一次 1/D。每行求和用 D−1 次加法，加 epsilon 一次，总加法数为 X，再求两次 rsqrt。共 49,156 次 FP32 标量工作，读 BF16 x 与 gamma 共 36,864 B，写 24,576 B 的 y 和 8 B 的 rstd。
- mHC Sinkhorn [1,4]、I=1。row softmax 取最大值比较 12 次、减法 16 次、exp 16 次，行、列各归约 12 次并对每个元素做除法 16 次。行 softmax 后每元素加 epsilon，列归约分母每列加 epsilon，合计 20 次 epsilon 加法。因此 Vector 共 120 次，FP32 输入、输出各 64 B。
- BF16 FA [Q=9,KV=512,D=64,B=2,H=4]。两个矩阵乘合计 9,437,184 次 Cube 运算。读 1,057,792 B，写 13,824 B，其中统计输出 4,608 B。二维得分矩阵在 kernel 内部计算，无对外完整张量读写。
- mHC 的 T=257、N=4、D=16384 仍可按 split 三段计算工作量。该形状的耗时随原有超出 256-token 范围的估时路线标为外推，张量公式自身与 950DT 核数无关。

## 1. 目标、输入和实际库算子

各算子参考公式使用 **CANN 9.2.1 / Ascend950DT_9582**。DT 包含 32 AIC、64 AIV，频率 1.65 GHz，单核 UB 253952 B。第二个校准平台 **Ascend950PR_957b** 有 28 AIC、56 AIV。两者的分核与片外存储证据见[双 SKU 校准](../results/v0.13/sku/README.md)。960、970、980 当前只提供基于配置的外推。

| 算子族 | 模型输入 | 实际/候选库接口 | 当前能力 |
|---|---|---|---|
| dynamicMXquant | `(R,C)`，BF16→E4M3FN，block32 | ops-nn / `aclnnDynamicMxQuant` | 时延、分核、UB循环、选定指令计数 |
| matmul | `(M,N,K)`，BF16 ND，不转置 | ops-nn / `aclnnMatmul`，已测 MatMulV3 | 时延、启发式tile/分核、MMAD计数 |
| FA | `(Q,L,D,B,H)`，BF16 SBH，非因果，无mask/dropout | ops-transformer / `aclnnFlashAttentionScoreV2` | 时延、分核、流水；资源路径返回 MMAD 计数 |
| RMSNorm | `(R,C)`、dtype、gamma_dtype、epsilon | ops-nn / `aclnnRmsNorm` | 实测校准阶段成本、独立 shape 验证 |
| Addmm | `(M,N,K)`、self_shape、alpha/beta | ops-nn / `aclnnAddmm` | 实测校准阶段成本、独立 shape 验证 |
| LightningIndexer | `(B,Q,L,G,D)`、sparse_count/mode | ops-transformer / `aclnnLightningIndexer` | 打分/TopK 工作量与流水重叠模板 |
| mHC | `(T,N,D)`；Sinkhorn为`(T,N)` | ops-transformer / `aclnnMhcPre`、`aclnnMhcSinkhorn`、`aclnnMhcPreSinkhorn`、`aclnnMhcPost` | 各组件与拆分链分别采集、受限校准 |
| KDA | `(B,S,Hq,Hv,K,V)`、recurrent/chunk | ops-transformer / `aclnnRecurrentKda`、`aclnnChunkKdaFwd` | recurrent 预测与独立验证，chunk 的 CA 同步资格待解决 |
| MoE融合链 | `(M,K,F,E)`及本rank专家token直方图 | ops-transformer / `aclnnGroupedMatmulSwigluQuantV2`，后续`aclnnGroupedMatmulV5` | 真实GMM1→MX→GMM2链；组件与链分别预测 |

CA 调用器使用安装包的真实 ACLNN 接口。参考源码来自固定版本的 ops-nn/ops-transformer，用来解释候选分支；它们不被宣称与安装包逐字同源。库文件存在只是“候选实现”，只有实际调用的 entry、tiling、核数和指令证据才能确定某个 shape 的执行路径。compact 数据库没有导出 ELF 时，运行时二进制 SHA 保持空值，不用安装文件 SHA 冒充。

## 2. 统一时延、流水和利用率口径

本节及第3—5节的约定：`⌈x⌉`/`⌊x⌋` 分别向上/向下取整，`align(x,a)=a⌈x/a⌉` 为向上补齐到 a 的倍数；所有 shape 维度、核数和循环数均为无量纲的整数计数。`KiB=1024 B`；除 tick 明示项外，所有时刻、阶段延迟和总耗时都以 µs 为单位。带帽的量是预测，未带帽的 CA 时间是观测。每一节的符号就近定义，例如 matmul 的 F 表示工作量或 FIXP 特征下标，不是后文 MoE 的 FFN 通道数。

目标是所有执行核在共同时间轴上的指令窗口：

\[
T_{CA}=\frac{\max_j tick^{end}_j-\min_j tick^{start}_j}{1650}\quad[\mu s].
\]

这里 j 遍历同一个 kernel、同一时间基准上的全部核指令记录，`tick_start/end` 是各记录的开始/结束 tick；`max−min` 从最早开始一直量到最晚完成。1650 是 1.65 GHz 对应的每微秒指令时钟 tick 数。这个窗口包含内部等待、同步和空隙，不包含 host enqueue 和窗外设备调度；不能直接替代框架 API 总延迟。最大单核 span 不能替代共同时间轴上的跨核 makespan。若一次 API 含多个 kernels，保存完整调用链。本轮统一数据库不证明跨DB时钟已对齐，因此串行多kernel采用各自跨核窗口之和，明确称为“净串行指令时间”；不包含kernel间空闲/launch开销，也不冒称API端到端延迟。

总体抽象是：**shape/属性 → 实现路径与分块 → 核内资源/依赖 → 最慢核和尾轮 → 固定控制开销**。对资源 p 的第 j 个任务，可写为：

\[
s_{p,j}=\max\{a_{p,j},s_{p,j-1}+II_p\},\qquad e_{p,j}=s_{p,j}+L_p.
\]

p 表示一种可并行工作的硬件资源（如读通路、Cube 或 Vector），j 表示分配给它的第 j 个任务；\(s_{p,j},e_{p,j}\) 是开始/完成时刻，\(s_{p,j-1}\) 是上一个任务的开始时刻。\(a_{p,j}\) 是输入和缓冲可用时刻，\(L_p\) 是本次任务从开始到结果可用的延迟，\(II_p\) 是相邻任务允许的最小开始间隔，以上均为 µs。没有前驱时省略其约束。直观地说，一项任务既要等数据和空缓冲，也要等该资源允许再次发起，然后才开始计自己的完成延迟。两者必须分开：硬件可能在前一个任务尚未完成时发起后一个。这个式子是共同建模语言；下面各族使用的具体简化式与已校准参数才是当前实现。

同时区分三种量：

- **核覆盖率**：\(AIC_{active}/32\)、\(AIV_{active}/64\)，分子分别是观测到的活跃 Cube 核/Vector 核数量，分母是目标平台可用核数量；结果是无量纲比例，表示实际用了多少核。
- **流水占用率**：去掉不应计入的等待/同步后，某流水指令时间区间的并集长度除以 \(T_{CA}\)。重叠指令 duration 不能直接累加；Vector family 包含load/store/control，不是纯算术利用率。
- **理论算力利用率**：应由独立 ISA 发射/吞吐模型得到 \(T_{ideal}/T_{CA}\)。\(T_{ideal}\) 是在明确核数、有效工作量和峰值能力下执行该工作需要的理想时间（µs），比值无量纲。目前独立峰值模型未校准，故此项为 `null`。不能用已经从 CA 拟合出来的阶段系数反过来证明硬件峰值效率。

L2及以内的影响当前隐含在阶段成本和有效II里，尚未被唯一分离。历史配置文件中的 `cache_enable=0` 不能单独证明本次全SoC backend采用了该开关；不据此声称所有采集都关闭了L2。

## 3. dynamicMXquant：二维分核、UB和双缓冲

### 3.1 范围与分核

**算例 · dynamicMXquant。** 一行 192 个 BF16 数占 384 字节。按 32 个数共享一个 MX scale，输出有 192 个 FP8 数和 6 个逻辑 scale。小输入仍要经历发起、读入、量化和写回；[1,192] 的模型时间为 4.23 µs，完整 CA 窗口为 4.13 µs。

当前路径固定 BF16 输入、E4M3FN 输出、E8M0尺度、尾轴block32、rint、scale_alg=0；C为64的倍数且≤65536。API接受更大的R不表示这些shape都已实测。

R、C 分别是输入的行数和每行元素数；一行通常对应一个 token 的特征。先选择列切片宽度：

\[
w=\begin{cases}256,&R\lceil C/512\rceil\le48,\\512,&\text{其他},\end{cases}\qquad
C_s=\lceil C/w\rceil,\quad P=\min(64,RC_s).
\]

其中 w 是一次列切片覆盖的元素数；\(C_s\) 是一行被切成的列片数；P 是候选并行核数上限，受 64 个 AIV 与总行片任务数共同限制。48=64×0.75 是当前分核规则的固定阈值，256/512 是列片粒度，不是带宽或峰值吞吐。

在有限的行/列分核候选 \((p_r,p_c)\) 中，令 \(p_c=\max(1,\lfloor P/p_r\rfloor)\)，要求 \(p_r\le R,p_c\le C_s\)。每核覆盖

\[
r_c=\lceil R/p_r\rceil,\quad c_s=\lceil C_s/p_c\rceil,\quad
c_c=\min(C,c_sw),\quad
AIV=\lceil R/r_c\rceil\lceil C_s/c_s\rceil.
\]

\(p_r,p_c\) 分别是行方向/列片方向的候选分区数；\(r_c\) 是代表性最忙核覆盖的行数，\(c_s\) 是它覆盖的列片数，\(c_c\) 是对应列元素数。最后的 AIV 是按实际分区回算的活跃核数，不保证等于预算 P。通俗地说，既可以让不同核处理不同 token，也可以把很宽的一行拆给多个核；尾部没填满的分区影响负载和利用率。

候选集合来自≤64核的整除/商划分；采用源码参考的尾洞评分，优先减少未填满的行列尾块，再按列分核数、行分核数排序。完整评分在 `quant_partition()` 中固定，不能把 `min(64,R)` 当作分核。该规则已由部分实际核数校验，但不是对所有安装包分支的完整移植。

### 3.2 UB容量与动态指令工作量

将32元素作为一个量化组。当前容量公式为：

\[
G_{max}=8\left\lfloor\frac{253952-2048}{256\cdot6+8\cdot2\cdot5}\right\rfloor=1240.
\]

\(G_{max}\) 的单位是量化组/核，每组32元素；253952是单核 UB 字节容量，2048是该路径固定保留字节数。容量预算以每8组为一个分配单元：256=8×32 个数据元素，6 B/元素包含 BF16输入与FP8输出的双缓冲预算，`8×2×5` 是模型沿用的 scale/临时空间预算（B/分配单元）；不能把这些预算系数当成实测搬运字节或单条指令开销。分子除以分母先得到可驻留分配单元数，再乘8得到组数。设每核列组数 \(g_c=\lceil c_c/32\rceil\)，则

\[
n_c=\lceil g_c/G_{max}\rceil,\quad
 g_t=\operatorname{align}(\lceil g_c/n_c\rceil,2),
\]
\[
n_r=\left\lceil\frac{r_c}{\lfloor G_{max}/g_t\rfloor}\right\rceil,\quad
r_t=\lceil r_c/n_r\rceil,\qquad tile=(r_t,32g_t).
\]

\(g_c\) 是每核一行的量化组数；\(n_c\) 是列方向需要的 UB 分批数；\(g_t\) 是每批一行可容纳的组数，补齐到偶数组以满足尺度打包。\(n_r\) 是行方向分批数，\(r_t\) 是每批行数，`tile=(r_t,32g_t)` 的两个分量均为元素维度。直观地说，核拿到的大矩形不一定能一次装进 UB，需要先沿列、再沿行切成能够放入双缓冲的小矩形。

在线逐UB tile保留真实尾行/尾列，而不是让最后一轮也使用满tile。对第i个tile的 \((r_i,c_i)\)，

\[
g_i=r_i\lceil c_i/32\rceil,\quad
m_i=\lceil g_i/8\rceil,\quad z_i=\lceil g_i/128\rceil,
\]
\[
x_i^{read}=2r_ic_i/1024,\quad x_i^{write}=(r_ic_i+g_i)/1024\quad[KiB].
\]

i 是同一代表核上按行优先遍历的 UB tile 编号，\(r_i,c_i\) 是该批真实行数/列数，\(g_i\) 是该批32元素量化组总数。\(m_i,z_i\) 分别是主VF和尺度VF循环代理（轮），分母8/128是对应代理的每轮组数。读特征 \(x_i^{read}\) 按 BF16 的2 B/元素计算；写特征 \(x_i^{write}\) 包括 FP8 的1 B/元素及每组1 B 的 E8M0 scale；两者以 KiB 表示，仅是逻辑 tile 工作量。当前写阶段实际使用固定排空项，而没有把该写字节特征代入带宽公式。选定 `RV_*` 动态指令的最大核计数通过离线非负拟合得到，当前接近

\[
\widehat N_{RV}=60n_{tile}+44\sum_i m_i+32\sum_i z_i.
\]

\(n_{tile}\) 是该核实际 UB tile 数，求和遍历它的全部 tile；\(\widehat N_{RV}\) 的单位是 RV 日志事件条数，60/44/32 分别是每 tile 固定事件、每主VF轮事件、每scale-VF轮事件的拟合权重（这里取近似整数用于说明）。因此这不是把元素数直接除以硬件 SIMD 宽度。

**计数口径修正说明**：这里预测的是历史msopprof CSV的RV日志事件数，并非已统一的唯一执行指令数。本轮抽查旧raw日志，发现相同PC/ID/opcode的成对记录（RV_SEND甚至四条）；compact数据库的ExecutedInstructions采用不同口径。新点保留原始执行记录数，跨口径指令APE为N/A，不对新计数乘2。此差异不改变各流水区间并集与指令makespan的时间定义，但限制了利用率推导；原始证据见 [RV计数审计](../results/v0.6/rv-accounting.json)。

它包含load/store/control，不能当作纯Vector算术指令数，也不是理论峰值周期。

### 3.3 保留的基础流水与小shape回退

\(R_i,V_i,W_i\) 分别为第i轮读就绪、Vector完成、写完成。令 `line(a,b,x)=a+bx`，则

\[
P_q=a_P+b_P\log_2(1+AIV),\quad R_0=P_q+L_{read}(x_0),
\]
\[
R_i=\max\{R_{i-1}+J_{read}(x_i),V_{i-2}+L_{read}(x_i)\},
\]
\[
V_i=\max\{R_i,V_{i-1},W_{i-2}\}+S_{vec}(m_i+z_i),\quad
W_i=\max\{V_i,W_{i-1}\}+D_{write}.
\]

上述所有 R/V/W、\(P_q\)、D 项均为 µs；i 从0开始，`last=n_tile−1`。\(P_q\) 是首个读之前的控制前缀；\(a_P,b_P\) 是 prefix 表中的截距和核数对数系数。\(x_i=x_i^{read}\) 是第i批读 KiB；\(L_{read}\) 使用 first_read 的截距/斜率，\(J_{read}\) 使用 read_interval 的截距/斜率，\(S_{vec}\) 使用 vector 行的截距/斜率。通用 `line(a,b,x)` 中 a 为 µs、b 为 µs/特征单位、x 为该阶段工作特征。\(D_{write}\) 是一批输出的固定写排空时长，\(D_{suffix}\) 是最终写完成后的尾开销。\(J_{read}\) 在当前 quant 实现中由相邻读完成间隔拟合并用于就绪时刻递推，是含排队/反压的有效间隔；不要把它误读为独立测得的 DMA 发射周期。

不存在的前驱项省略。两轮后复用输入/输出槽，分别受前一使用者消费完毕/写回完毕约束；这正是UB双缓冲带来的依赖。首读延迟只影响当前就绪，后续读还受发起间隔与反压限制。

基础模型对小shape的控制路径采用饱和回退（单 UB 宽行分支见下节）：

\[
T_{pipe}=W_{last}+D_{suffix},\quad
T_{ctrl}=a_c+b_c\min(\log_2(1+AIV),4),\quad
\widehat T_{q,base}=\max(T_{pipe},T_{ctrl}).
\]

\(T_{pipe}\) 是流水递推的最终完成时间，\(T_{ctrl}\) 是小 shape 经验控制路径的总时间，\(\widehat T_{q,base}\) 取两者较大值，避免把本应重叠或已包含的控制时间再重复相加。\(a_c,b_c\) 分别是 control_floor 的截距及对数特征系数（µs），`min(...,4)` 将该特征封顶在4。直观地说，极小张量尚未耗尽计算能力，准备/控制流程就可能占据大部分耗时。

回退项是小工作量数据中的经验总时延路径，不是物理硬件下界，也不能随核数无上限增长。基础阶段参数来自 13 份量化 trace，其中 3 个 shape 涉及多个 UB tile；小控制路径使用9个每核工作量≤4096元素的校准点。

| 阶段 | 截距 µs | 斜率 µs/单位 | 单位/含义 |
|---|---:|---:|---|
| prefix | 2.784570 | 0.101181 | \(\log_2(1+AIV)\) |
| first_read | 0.589411 | 0.017218 | 每tile读KiB |
| read_interval | 0.345656 | 0 | 每tile读KiB；现拟合为常数 |
| vector | 0.065425 | 0.007099 | \(m_i+z_i\) |
| control_floor | 3.830949 | 0.399708 | 饱和后的核数对数 |
| write_drain / suffix | 0.344848 / 0.009697 | — | 写排空 / 最后尾开销 |

这些是带该CA路径调度影响的统计参数。当前主要风险为每核工作不均、短列分支、UB边界、读间隔与写排空只有少量shape证据；小shape回退也可能遮盖错误的流水缩放。

### 3.4 单 UB 宽行片段的 VF 调度残差

基础Vector成本采用SIMD指令区间并集，未必覆盖整个VF执行窗口里的调度间隙。校准使用 19 个 quant trace，对每个shape先选择用于阶段分析的同一关键核，再计算

\[
g_j^{obs}=\max\{0,\sum_i VFSpan_{j,i}-UnionSIMD_j\}.
\]

这里 j 索引校准 shape，i 索引该 shape 同一关键核的 VF 批次；\(g_j^{obs}\) 是观测残差（µs）。每轮VFSpan为该轮VF窗口长度（µs），UnionSIMD为**同一核**的Vector-family指令区间并集；不能用一个核的VF包络减另一个核的最大忙碌时间。这个差值表示包络内未由SIMD并集解释的调度残差，可能含等待、队列、控制和反压，不是新增算术服务或独立硬件峰值。

19条旧记录拟合同一个非负二参数模型：

\[
x_j=\begin{cases}AIV_j/64,&n_{tile,j}=1,\\0,&n_{tile,j}>1,\end{cases}
\qquad \hat g_j=a_g+b_gx_j,
\]
\[
(a_g,b_g)=\arg\min_{a,b\ge0}\sum_j(a+bx_j-g_j^{obs})^2
=(0.11987486887649926,\ 0.6213462046069332)\ [\mu s].
\]

\(x_j\) 是无量纲特征：单 UB shape 取活跃 AIV 占64核的比例，多 UB shape 取0；\(n_{tile,j}\) 是 shape j 的每核 UB 批数。\(a_g\) 是残差固定项，\(b_g\) 是核覆盖比例每增加1的残差增量，单位均为 µs；\(\hat g_j\) 是预测残差。`argmin` 表示在 a/b 非负的限制下，选择让所有校准点残差平方和最小的两个系数。该限制避免负的额外时间，但不保证系数就是某个独立硬件延迟。

多UB点仅以截距特征参与残差拟合，并没有被赋予同一个多UB修正公式。实际预测仅在 **`iterations==1 && cols_per_core>=1024`** 的受限分支使用新增项：

\[
\boxed{\widehat T_q=\begin{cases}
T_{pipe}+a_g+b_g AIV/64,&n_{tile}=1\land c_c\ge1024,\\
\max(T_{pipe},T_{ctrl}),&\text{其他}.
\end{cases}}
\]

最终 \(\widehat T_q\) 是 quant 的当前时延预测（µs）；分支中的 \(c_c\) 是前述每核列元素数，并非整个输入 C。通俗地说，只给已由数据覆盖的“单次 UB、较宽行片段”补上 VF 窗口内的调度空隙；其他区域仍走原流水和小 shape 回退。

`cols_per_core>=1024` 表示block32语义下每行片段至少产生32个scale字节，是当前保守的模型分支限制；**不能由此证明cacheline对齐、没有写冲突或真实库必然切换实现**。

### 3.5 标量预测开销

对单UB点，递推严格退化为

\[
T_{pipe}=P_q+L_{read}(x_0)+S_{vec}(m_0+z_0)+D_{write}+D_{suffix}.
\]

统一查询入口用这条等价式计算单 UB 点，并按第 3.3 和 3.4 节的规则选择回退或残差。多 UB 点只保存推进所需的前两轮状态。对相同 tile 的长循环，用第 11 节的矩阵合成把遍历次数从 O(n) 降到 O(log n)。这里 n 指最忙核上重复的 UB tile 数；阶段服务和资源约束均保留。

## 4. Matmul 的 950DT 参考时延：输出 tile、核轮次和阶段成本

本节用于参考设备的时延与 DT 复用基线。其他芯片执行本页开头的候选输出块、K 分片和阶段时间公式。

**算例 · Matmul。** 输入 A[9,256] 和 B[256,128] 得到 Y[9,128]，做 9×128×256 次乘加。按 16×64 输出块切成两个有效的 9×64 块，可在一波内并行。模型估 3.60 µs，950DT CA 为 4.02 µs。

M 是输出行数，N 是输出列数，K 是两矩阵相乘的归约长度（乘积累加的项数）；三者单位都是元素数。输入 A[M,K]、B[K,N]、输出[M,N]，固定BF16 ND、无转置、cubeMathType=0，M/N/K各≤4096。数学工作 \(F=2MNK\) 的单位是 FLOPs，将一次乘法和一次加法分别计1；逻辑字节 \(2(MK+KN+MN)\)；后者不含cache重用、补齐和重复读。

从输出tile \((t_m,t_n)=(16,64)\) 开始。若 \(\lceil M/t_m\rceil\lceil N/t_n\rceil>32\)，比较将M或N tile翻倍的候选（单边最多256），依次选择“tile数接近32”“补齐后输出面积小”“tile边长和小”的候选。令

\[
N_t=\lceil M/t_m\rceil\lceil N/t_n\rceil,\quad
AIC=\min(32,N_t),\quad W=\lceil N_t/AIC\rceil.
\]

\(t_m,t_n\) 是一块输出的行/列容量（元素），\(N_t\) 是输出块总数，AIC 是参与工作的 Cube 核数，W是最忙核的输出tile轮数。通俗地说，先把输出矩阵切块，块多于32核就要分轮；增大块可以减少轮数，却会增加每块成本与尾块浪费。该映射是受源码和小规模实测约束的启发式，不是完整的CANN tiler。`gemv`仅为M=1的模型分组，不能据此声称实际改成Vector kernel。

令 \(e_m=\min(M,t_m),e_n=\min(N,t_n),e_k=align(K,16)\)，阶段特征为

\[
x_R=2(e_me_k+e_ne_k)/1024,\quad
x_C=2e_me_ne_k/131072,\quad x_F=2e_me_n/1024.
\]

\(e_m,e_n\) 是代表块的有效行/列数，\(e_k\) 是补齐到16的归约长度。\(x_R\) 是该块A/B的逻辑读 KiB，\(x_C\) 是每131072 FLOPs为一单位的 Cube 工作特征，\(x_F\) 是输出 KiB。字节因子2来自 BF16；这些是拟合代理，不是所有 cache/尾块的实测流量。

阶段下标 s 分别取 R（MTE2读）、C（Cube乘加）、F（FIXP写出/格式处理）。每阶段服务成本 \(S_s=a_s+b_sx_s\) 由校准trace的区间并集拟合；\(S_s,a_s\) 单位为 µs，\(b_s\) 单位为 µs/对应特征单位。\(x_C\) 的131072仅为特征缩放常数，不是峰值吞吐。

| 阶段 | 截距 µs | 斜率 µs/单位 |
|---|---:|---:|
| MTE2读 R | 0.794096 | 0.010016 |
| Cube C | 0.130526 | 0.005558 |
| FIXP F | 1.522516 | 0.046052 |

当前Vector和MTE3阶段为0；完整时延为

\[
B_m=S_R+S_C+S_F+(W-1)\max(S_R,S_C,S_F),
\]
\[
\widehat T_m=\max\{0.001,0.628759+1.013249B_m+0.041410\log_2(1+AIC)+\delta_b\}.
\]

\(B_m\) 是未加总时延校正的流水完成时间（µs），\(\widehat T_m\) 是最终预测（µs）；0.628759 是校正截距（µs），1.013249 是无量纲缩放系数，0.041410 是核数对数项系数（µs），0.001 µs 是数值下界。\(\delta_b\) 是实现分组 b 的经验残差（µs），与下一段列出的分组逐一对应。

首轮填充按顺序相加，其后按最慢资源推进；总时延校正承担未分离的控制/同步和路径偏差。分支优先级为M=1→`gemv`，否则K%16≠0→`k_tail`，否则tile≠16×64→`coarse_tile`，其余`dense`。对应残差 \(\delta_b\) 分别0.024992、0.041869、−0.062131、−0.006530µs；残差用 \(\sum residual/(n_b+2)\) 收缩，避免一两个样本单独拟合完整模型。其中 residual 是各样本 CA 时间减未加该分组修正的预测（µs），\(n_b\) 是分组 b 的校准样本数，分母额外加2相当于把小样本残差向0收缩；它不是额外两个真实样本。

每个最忙核的选定MMAD计数近似 \(W\lceil K/256\rceil\)，现计数系数≈1。K尾块和不同tile可能改变实际指令形状，需对比实际MMAD维度与数量。分支校准仅2/1/1/4点，未见组合仍为外推；不能将已校准BF16 matmul直接用于MX GMM、HF32 mHC或Addmm总调用。

## 5. 普通FA：通用路径和长循环路径

Q 表示本次 query token 数，L 表示可见的 KV 上下文长度，D 表示每 head 的通道宽度，B 表示本次单卡 batch 请求数，H 表示此调用实际承载的 attention head 数（已考虑运行时切分/复制）；这些均为计数。输入张量名称 Q 与其长度符号相同，按上下文区分。输入Q为[Q,B,H×D]，K/V为[L,B,H×D]；BF16、SBH、无mask/dropout、非因果，scale=1/√D。它是普通FA代理，不等于三个模型实际MLA、压缩注意力、稀疏注意力、KDA或decode专用FIA。输入guard允许Q≤2048、D∈{64,128}、B≤256、H≤32；这只是可计算范围，不是完整验证范围。

### 5.1 通用tiling与耗时

**算例 · FA。** 单 batch、单 head 校验 9 个 token，KV 长度 131072，head 宽 64。沿 KV 每 256 个 token 分一块，共 512 块；第一个块建立流水，后面再推进 511 次。该非因果 SBH 路径预测 274.79 µs，独立 CA 为 274.82 µs。

Q容量和KV容量由当前启发式选择：

\[
q_b=\begin{cases}64,&BH\lceil Q/128\rceil<32,\\128,&\text{其他};\end{cases}
\]
\[
k_b=\begin{cases}256,&(D=64\land Q\bmod128=0\land L\bmod128=0)\ \lor L>1024,\\128,&\text{其他}.\end{cases}
\]

\(q_b,k_b\) 分别是模型选择的 query/KV tile 容量，单位为 token；选择条件用整数余数判断整齐边界，32对应平台 AIC 上限。它们是当前受限启发式，不代表所有真实库 shape 都按同一规则分块。

实际特征tile为 \(t_q=16\)（Q=1时）或 \(\min(Q,q_b)\)，\(t_k=\min(align(L,16),k_b)\)。令

\[
N_q=BH\lceil Q/q_b\rceil,\ AIC=\min(32,N_q),\ AIV=2AIC,
\ W=\lceil N_q/32\rceil,\ I=\lceil L/k_b\rceil.
\]

\(t_q,t_k\) 是用于估算阶段工作量的代表 tile 长度（token），Q=1时用16的特征宽度反映模型保留的最小工作粒度。\(N_q\) 是 batch×head×query块的任务数，AIC/AIV分别是预测活跃核数，W是最忙核需要处理的query任务轮数，I是每个query任务遍历KV所需的轮数。这里沿Q/head分核，KV在每个任务内循环，没有额外假设split-KV。阶段特征：

\[
x_R=2(t_qD+2t_kD)/1024,\quad x_C=4t_qt_kD/131072,
\]
\[
x_V=t_qt_k/512,\quad x_F=4t_qt_k/1024,\quad x_W=2t_qD/1024.
\]

\(x_R\) 是一个tile的Q/K/V逻辑读 KiB，系数2来自BF16；\(x_C\) 同时覆盖QK与PV两次乘加，单位为131072 FLOPs；\(x_V\) 是以512个query-key对为单位的Vector工作代理，含两AIV分担的归约/Softmax比例；\(x_F\) 是以KiB为单位的FP32中间结果FIXP代理，\(x_W\) 是最终BF16输出KiB。读特征包含Q不能解释为每个真实KV轮次都重新从GM读Q；这是通用模型的平均阶段代理，长循环路径另按trace拆解。以上常数只是模型特征定义，不是理论带宽或吞吐。

各服务 \(S_s=a_s+b_sx_s\)：s取R/C/V/F/W，对应read/cube/vector/fix/write；\(a_s\) 为固定服务项（µs），\(b_s\) 为每单位特征的增量（µs/单位），\(S_s\) 为该阶段预测时间（µs）。

| 阶段 | 截距 µs | 斜率 µs/单位 |
|---|---:|---:|
| read | 0.953363 | 0.002909 |
| cube | 0.093779 | 0.008431 |
| vector | 0.251097 | 0.004456 |
| fix | 0.065210 | 0.003613 |
| write | 0.405945 | 0.009818 |

QK和PV共享Cube，Softmax和在线累积引入循环依赖，因此通用模型使用

\[
II_{base}=\max(S_R,S_C+S_F,S_V+S_C/2),
\]
\[
B_{FA}=W\{S_R+S_C+S_V+S_F+(I-1)II_{base}+S_W\},
\]
\[
\widehat T_{base}=\max\{0.001,7.173997+0.709561B_{FA}+0.103476\log_2(1+AIC+AIV)+\delta_b\}.
\]

\(II_{base}\) 是相邻KV循环的有效推进间隔（µs），\(B_{FA}\) 是所有W轮query任务的流水时间代理（µs）；\(S_C/2\) 将两次矩阵乘总服务的一半作为PV依赖近似，尚非分别测得的两段耗时。\(\widehat T_{base}\) 是最终通用预测（µs），7.173997和0.103476分别是校正截距与活跃核总数对数系数（µs），0.709561是无量纲缩放，0.001 µs是数值下界；\(\delta_b\) 是分组b的时间残差（µs）。直观地说，一个query块需要反复读KV、算分数、做Softmax再累积结果；后续轮可以重叠，但共享Cube和在线Softmax依赖让它不能无限流水。

`d64_kv256`、`d64_kv128`、`d128_kv128`的残差分别为0.161074、−0.059147、−0.057041µs；未见分支不添加残差。校准数量分别2/6/1，模型保留“未见/少样本/包围盒外”的覆盖标签，标签不构成准确率保证。

Q容量为64的路径使用结构MMAD计数

\[
N_{MMAD,max}=2\lceil L/128\rceil W,
\]

\(N_{MMAD,max}\) 是最忙AIC上选定MMAD指令次数，128是该已验证路径的KV指令分段长度；前面的2表示QK、PV各一组指令，W是该核承担的query任务轮数。两项分别是QK沿N与PV沿K分段。Q容量128的路线尚未验证该公式，计数返回空。部分trace缺少MMAD维度，不能由有维度的子集倒推完整执行FLOPs。

### 5.2 Q9/D64/B1/H1的长KV专用路径

仅当 **Q=9、D=64、B=H=1、L≥4096** 时采用此路径。活跃核为1AIC+2AIV，\(I=\lceil L/256\rceil\)，MMAD仍为 \(2\lceil L/128\rceil\)。参数来自旧32k完整DB，不消耗目标shape的trace。

32k中Q只读一次，K/V各128次32768B读取。按源地址恢复两个流，再以已知MMAD维度对应的PC区分QK与PV；不能把按时间混排的每4条MMAD都当作一次源码循环。K/V首轮读完成延迟与后续发起间隔不同，模型为

\[
\boxed{\widehat T_{long}=P+\max(L_K,\Delta_V+L_V)+F+(I-1)II+D_{drain}}.
\]

这里 \(\widehat T_{long}\) 是整次kernel时延预测；P、\(L_K,L_V,\Delta_V\)、F、II及\(D_{drain}\) 的单位均为 µs。I仍是KV轮次数，256是每轮覆盖的token数，因此I−1表示扣除已在填充项单独计算的首轮。

P为第一K读之前的前缀；\(L_K,L_V\) 为第一K/V读取本身的延迟；\(\Delta_V\) 为V比K晚开始的时间；F为首轮K/V均就绪到首轮PV结束的剩余路径；\(D_{drain}\) 为最后PV结束后的排空。两条读流可重叠，所以取完成时刻最大值。F包含尚未隐藏的软最大值、调度/同步等，不是纯Cube算术服务。

稳态从32k同一DB的中段i=8..119拟合：

\[
s_i\simeq a+Ji,\qquad
J=\frac{\sum(i-\bar i)(s_i-\bar s)}{\sum(i-\bar i)^2},
\]
\[
II=\max(J_K,J_V,S_C,S_V,S_{MTE1},S_{FIXP}).
\]

回归中的i是32k trace内的KV循环序号（无量纲），\(s_i\) 是该读流第i轮开始时刻（µs），\(\bar i,\bar s\) 是选中中段样本的平均序号/平均开始时刻，a是回归截距（µs），J是回归斜率（µs/轮）；\(J_K,J_V\) 分别拟合K/V读流。\(S_C,S_V,S_{MTE1},S_{FIXP}\) 是每KV轮的Cube、Vector、内部搬运、FIXP有效资源服务时间（µs/轮），这里V下标表示Vector，与\(J_V,L_V\) 中表示value张量的V不同。

J是当前路径的有效发起间隔，包含buffer复用、同步和反压，**不是独立HBM带宽**。服务项由各资源指令区间并集/128轮得到，作为资源需求下限；两个AIV取最大。当前J主导公式，仍不能据此证明纯访存bound。

| 参数 | µs | 参数 | µs |
|---|---:|---|---:|
| P | 6.560000 | F | 1.778182 |
| \(L_K\) | 0.849697 | \(L_V\) | 0.745455 |
| \(\Delta_V\) | 1.066667 | \(D_{drain}\) | 0.347273 |
| \(J_K\) | 0.517109 | \(J_V\) | 0.517215 |
| \(S_C\) | 0.215933 | \(S_V\) | 0.220985 |
| \(S_{MTE1}\) | 0.281013 | \(S_{FIXP}\) | 0.107898 |

通俗地说，读取一块数据要等多久，与多久可以继续读取下一块，是两件事。首轮把读取延迟和计算填充付一次；长上下文后续只按最慢的持续推进间隔增加成本，所以不会把每轮的完整读延迟重复串加。这里用同一32k数据库恢复每轮流起点，比额外采集128k完整trace拟合全部参数更节省仿真成本。

对应数值简式为 \(10.497575758+(\lceil L/256\rceil-1)\cdot0.517214978\) µs。尾块暂按完整256处理，MMAD计数仍按128分段。该简式是阶段模型结果，不是从新留出总时间反拟合直线。32k已经作为开发数据；128k和16k+1是冻结后的独立验证。256k/512k仍未实测，也不能把这组参数缩放到其他Q/D、B/H、因果mask或A7/A8。

## 6. RMSNorm和Addmm 的算法与调用边界

### 6.1 RMSNorm

**算例 · RMSNorm。** 对 [65,128] 的每一行，先求 128 个数的平方均值、加 ε，再开方取倒数，再用它和 128 个 gamma 权重缩放该行。若每核最多负责两行，32 核各处理两行，最后一核处理一行。这个 shape 的 trace 实际启动 33 个 AIV，默认预测 3.82 µs，CA 为 3.08 µs。少 token 宽通道的另一例 [1,5120] 只有一核、每核 40 组，默认预测 2.33 µs，已知前序同类调用覆盖该核时预测 2.17 µs，开发 CA 为 1.93 µs。

对x[R,C]、gamma[C]：

\[
r_r=(C^{-1}\sum_jx_{rj}^2+\epsilon)^{-1/2},\qquad y_{rj}=x_{rj}r_r\gamma_j.
\]

数学乘法 \(R(3C+1)\)、加法 \(RC\)、rsqrt R次；不含rsqrt的FLOPs为 \(R(4C+1)\)。这些是标量数学量，归约树和SIMD/VF指令仍需额外证据。逻辑字节 \(2RCb_x+Cb_g+4R\)，分别是x读、y写、一次gamma读、FP32 rstd写，不是实测HBM流量。

NORMAL候选分核令 \(f=\lceil R/64\rceil\)，活跃AIV \(\lceil R/f\rceil\)。\(C_a=align(C,32/b_x)\)，\(F=2^{\lfloor\log_2(C_a-1)\rfloor}\)，\(L=align(\lceil F/64\rceil,32)\)，则

\[
u=\left\lfloor\frac{253952-1024-2048-C_ab_g}{4C_ab_x+12+L}\right\rfloor.
\]

当 \(C_a\le16384,u\ge1\) 且不进入短列transpose候选时，tile行为 \(\min(u,f)\)，最忙核轮数 \(\lceil f/\min(u,f)\rceil\)。1024/2048分别为host与arch35保留空间，不能按常量名称误读。长归约Split/SplitAR和短列高行数transpose未移植，返回未解析dispatch。实际 kernel 入口和 AIV 数见总误差表。

参数解释：R是行/token数，C是每行元素数；r、j是行列索引；x/y是输入/输出数值，gamma是逐列权重，epsilon是防止除零的小常数，r_r是FP32倒数RMS。b_x、b_g为输入和gamma每元素字节数。f是最忙核的行数，C_a是按32字节对齐的列数，F是归约所用的2次幂片段，L是归约辅助缓冲字节数，u是UB最多能容纳的tile行数；align表示向上对齐。253952等容量单位均为字节。数学FLOPs是逻辑运算数，不等于真实向量指令数。

通俗地说，RMSNorm先算一行的平均平方，再用同一个倒数根号系数缩放整行。多行可分给不同核；列更宽则单核归约与搬运更长，不能只看总元素数。

### 6.2 Addmm

**算例 · Addmm。** 令 A[9,256]、B[256,128]、bias[128] 且 α=β=1，输出每个元素等于一项长度 256 的点积再加所在列的 bias。一个有效 9×64 输出块需读 64 个 BF16 bias，共 128 字节；写回该块可以与其他流水部分重叠。该次库调用的 CA 为 3.76 µs。

\[
Y=\beta\operatorname{broadcast}(S)+\alpha AB,\qquad F_{mm}=2MNK.
\]

另列MN合并加法和alpha/beta非1时各MN级缩放；这是稠密代数工作上界，广播前缩放或零系数消除可能减少工作。逻辑字节 \(b(MK+KN+|S|+MN)\)，不含中间tensor、copy/cast和多kernel开销。

参数解释：A[M,K]、B[K,N]是矩阵，S是可广播的self/bias，Y[M,N]是输出；M/N/K分别为输出行、输出列、归约长度；alpha/beta是无量纲缩放系数，F_mm单位为FLOPs；b为每元素字节数，|S|是self元素数。通俗地说，Addmm在矩阵乘结果上再加一个缩放后的bias；bias形状和系数能改变库选择的路径，不能只用M/N/K套Matmul时间。

self可为标量、[N]、[1,N]、[M,1]、[M,N]等合法广播。源码候选按空tensor/零系数、bias转换、inplace FP32 GemmV3、DAV_3510 FusedMatMul(add)、一般Matmul+Muls/Add/Axpy分支处理。FusedMatMul(add)要求完整矩阵self、alpha/beta≈1和其他dtype/layout条件；[N] bias不满足此特定分支。FP32融合候选还要求cubeMathType=3。

Addmm不能直接继承Matmul的总耗时或分核。已测 `[2,16,16]`、self `[16]` 仅观察到一个kernel，其他属性可能产生多kernel。Matmul 的时延见第 4 节，Addmm 的队列重叠公式见本节；APE使用同属性校准/冻结验证分别报告，理论峰值利用率仍未校准。

**Addmm 临界核的队列重叠。**

保持 BF16 连续 ND、`self=[N]`、α=β=1。每个任务产生一个输出块。沿用矩阵切块规则得到最忙核的波数 w、有效输出块 m×n、A 个活跃 AIC 以及按 16 对齐的归约长度 k。本核搬入工作 R=2[(m+n)k+n]/1024 KiB（含列偏置），计算工作 C=2mnk/131072（以 131072 FLOPs 为单位），输出 F=2mn/1024 KiB。

对 MTE2、MTE1、Cube、FIXP 分别令服务时间 s=a+b·x，其中 x 依次为 R、R、C、F，a 是固定微秒成本，b 是微秒/相应工作单位。MTE1 的 R 是片内读工作代理。各系数用 12 个旧完整 DB 最后完成核的各队列忙碌区间并集分别校准。两种搬运可与 Cube 和 FIXP 部分重叠。记最长队列服务为 J=max(s)，非最长队列可见比例为 ρ，预测 T=H+g·log₂(A)+wJ+ρ(Σs−J)。H=2.1024 µs 是核内未被这些队列覆盖的控制/等待时间中位数，g=0.0460 µs 是临界核相对首个发起核的偏移除以 log₂(A) 后的多核样本中位数。ρ 对每个旧 trace 取（所有数据队列的区间并长−最长单队列并长）/（各队列并长之和−最长单队列并长），再取中位数，结果为 0.10858。多数次要流水被最长活动覆盖，只把暴露段加到首轮关键路径上；多波预测沿用最长队列作为推进间隔。这些量包含该库调度的影响，多波、其他 dtype 和跨代成本需额外证据。

例：M=9、N=128、K=256，输出按列分为两个 9×64 块，A=2，w=1。每核 R=36.625 KiB，C=2.25，F=1.125 KiB。MTE2/MTE1/Cube/FIXP 服务分别约 1.096/0.115/0.091/1.354 µs。其余队列仅暴露约 0.141 µs，首核错峰约 0.046 µs，预测 2.102+0.046+1.354+0.141=3.644 µs，开发样例 CA 为 3.76 µs。

该分块规则为启发式，输出尾部和非对齐 K、不同库入口可能产生新偏差。知识卡 `l1_l0_storage_structure_and_data_paths` 支持输入、Cube 与 L0C 搬出分阶段建模；具体重叠比例和服务成本来自该安装包数据库。每个合格验证点必须通过安装包 aclnnAddmm 的完整数值检查与可解析的单 kernel CA 数据库。

对应的 (a,b) 数值为 MTE2 (0.83794,0.0070473)，MTE1 (0.02527,0.0024447)，Cube (0.06927,0.0095394)，FIXP (1.29390,0.0533037)。所有截距以 µs 计；MTE2、MTE1、FIXP 的斜率单位为 µs/KiB，Cube 的斜率单位为 µs/131072 FLOPs。每条直线都是独立拟合的队列时间，参见 [参数与逐 trace 统计](../results/v0.9/addmm/profile.json)。

## 7. LightningIndexer：打分、head归约和TopK

**算例 · Lightning Indexer。** B=1、Q=1、L=640、G=8、D=128，选择 top-512。对 640 个候选 key，要先算 640×8 个长度为 128 的点积，再跨 head 合并、选出 512 个下标。按每段 128 个 key 共五段点积，模型估 6.65 µs，CA 为 6.20 µs。

普通950接口固定D=128、key head数1，G∈{8,16,24,32,64}；Q/K/weights同为BF16或FP16。当前结构输入是BSND，不覆盖所有paged/TND路线。对每个query：

\[
a_{qgk}=\sum_dQ_{qgd}K_{kd},\quad
s_{qk}=\sum_gW_{qg}\max(0,a_{qgk}),\quad indices_q=TopK_k(s_{qk}).
\]

令每batch的合法query-key对数为

\[
v=\begin{cases}QL,&mode=0,\\Q(L-Q)+Q(Q+1)/2,&mode=3,Q\le L,\\L(L+1)/2,&mode=3,Q>L.\end{cases}
\]

mode3为右下对齐因果，\(P=Bv\)。工作量分别为点积 \(2PGD\)、ReLU PG次、weight乘法PG次、head归约 \(P(G-1)\) 次。实际padding/mask可能执行更多，另保留稠密 \(2BQLGD\) 作为参照。TopK比较、分块选取与合并的完整安装包算法尚未移植；时延先采用第11节融合流水模板，不宣称已恢复逐指令成本。不能用topk/L减少前面必须完成的打分工作。

输入输出唯一字节下界为

\[
2(BQGD+BLD+BQG)+BQk(4+2\mathbf1_{returnValues}).
\]

不包括score中间量、重读和workspace。结构计算为O(1)，不会分配BQGL规模张量。实际采集保留entry、核数和MMAD/RV次数，但不能把本地工作量代理当作完整安装包tiling。比如模型config提供32个index heads，直接TP8切成G4会不满足此普通接口，必须确认运行时是否复制heads或选择其他量化接口，而非静默改成G8。

参数解释：B是batch，Q是query长度，L是key长度，G是query/index head数，D是每head宽度；q/g/k/d是对应索引，Q/K/W在打分式中分别是query张量、key张量和head权重（不要把张量Q与长度Q混淆）。a是点积，s是head加权后的分数，indices是最终整数下标。mode为稀疏mask模式，v是单batch有效配对数，P是所有batch有效配对总数；字节式中的k是top-k个数，不是前面的key索引。returnValues指是否额外写回分数；指示函数成立时为1，否则0；2和4分别对应BF16与int32字节数。

通俗地说，Indexer先对候选key打分，再选最高的若干项。选取数量少并不意味着打分工作少；小query可能仍启动大量核处理长key和归并结果。

## 8. mHC：Pre、Sinkhorn和Post的组成

下面五个例子分别对应实际可调用的路径，右列为完整 CA 窗口。已用于本版校准的例子用于说明怎么算，准确率取另行冻结的独立 shape。

| 算子 | 输入和一步可以核算的工作 | CA 净时间 |
|---|---|---:|
| mHC Pre | [4,4,1024]、三项 α。每个 token 展平为 4096 个元素，投影得到 4²+2×4=24 个系数。四个 token 的投影共有 2×4×4096×24=786432 FLOPs。在该例中 K=4096 可由多个核分摊；模型估 9.23 µs。 | 8.73 µs |
| mHC Sinkhorn | [4,4] 表示 4 个 token，每个有一张 4×4 矩阵。5 次迭代下，按下式共有 2×5×4×4²=640 次标量除法，后一次归一化要等前一次结果。预测 3.58 µs。 | 2.95 µs |
| mHC Post | [32,4,512]、带残差矩阵。对每个 token 的每一输出通道，四路残差乘各自权重求和，再加入外部层贡献。仅残差矩阵的混合就要 32×4²×512=262144 次乘法。默认预测 2.28 µs。 | 2.39 µs |
| 融合 mHC PreSinkhorn | [3,4,4096]，20 次迭代，关闭反向辅助输出。每个 token 有 16384 个投影输入，归一化对象为 4×4；计入一次融合 API 的净设备窗口，预测为 9.94 µs。 | 10.37 µs |
| 拆分 mHC 链 | [4,4,1024]。Pre、Sinkhorn、Post 的窗口分别为 8.730、6.016、3.475 µs。本版三段估 9.226、5.578、3.322 µs，合计 18.13 µs。外部 Attention/MLP 另计。 | 18.22 µs |

输入X[T,N,D]，N是残差流数，T为本次实际token总数。普通完整分支 \(M=N^2+2N\)，x为ND维展平行：

\[
r_t=(mean(x_t^2)+\epsilon_n)^{-1/2},\quad z_t=(x_t\odot\gamma)\Phi^T,\quad w_t=r_tz_t,
\]
\[
h^{pre}=\sigma(\alpha_0w_{:N}+b_{:N})+\epsilon_h,\quad
h^{post}=2\sigma(\alpha_1w_{N:2N}+b_{N:2N}),
\]
\[
H'=reshape(\alpha_2w_{2N:}+b_{2N:},N,N),\quad
h^{in}_{td}=\sum_nh^{pre}_{tn}X_{tnd}.
\]

无gamma时省略该乘法。Pre投影为 \(2T(ND)M\) FLOPs；归约、rsqrt、sigmoid与混合分别计数，不全部视为同吞吐FMA。alpha只有两项的路线无residual矩阵分支，不能仍按M=N²+2N计。

Sinkhorn先 \(A_0=rowSoftmax(H')+\epsilon\)，列归一化一次，然后I−1轮行/列交替归一化；每次分母为对应轴和加epsilon。包含softmax本身后，标量计数为：

\[
N_{exp}=TN^2,\quad N_{div}=2ITN^2,\quad N_{reduceAdd}=2ITN(N-1).
\]

divide可能编译为reciprocal+multiply，不能将该数量直接作为ISA次数。Post为

\[
Y_{tid}=h^{out}_{td}h^{post}_{ti}+\sum_jX_{tjd}H_{tji}.
\]

有residual矩阵时混合乘法TN²D次、归约加法TN(N−1)D次，再加各TND次post乘法与合并加法；无H时直接残差X。包内Post、Sinkhorn标记VectorCore，不能因为公式可写成矩阵乘就除以Cube峰值。

拆分链是Pre→Sinkhorn及外部Attention/MLP→Post；融合链以PreSinkhorn替换前两项，两条路线互斥，不能重复计数。外部Attention/MLP主体不包含在mHC wrapper工作里。单独Pre支持N=4/6/8、D≤16384且16对齐；融合 PreSinkhorn 的源码候选为 N4、I20、D4096/7168、epsilon1e−6，本版时延入口仅校准 D4096；独立Sinkhorn为FP32、N4/6/8、I1..100。融合投影涉及HF32，不能复用BF16 Matmul参数。split链保留Pre辅助输出并用恒等外部层连接hIn到hOut，测得的链不包含Attention/MLP；与无辅助输出的推理模式不是相同合同。时延估计见第11节。

参数解释：T是token数，N是残差流数量，D是每流隐藏宽度，M=N²+2N是投影输出宽度；t/n/d/i/j为索引。x是展平后的输入，gamma是可选逐元素权重，Phi是投影矩阵，z是未归一化投影，r_t是倒数RMS，w是归一化投影；alpha_0/1/2和b是三段缩放与bias，sigma是sigmoid，epsilon_n/h和epsilon是对应数值稳定常数。h_pre/h_post是混合权重，H′是未归一化残差混合矩阵，H是Sinkhorn结果，h_in/h_out是外部层输入/输出。I为Sinkhorn迭代数；N_exp/N_div/N_reduceAdd是标量exp/div/归约加法次数，不是硬件指令次数。

通俗地说，Pre为多路残差生成混合权重并合成外部层输入；Sinkhorn把残差矩阵反复按行/列归一化；Post再把外部层结果分配回各残差流。融合实现可能共享数据和减少同步，不能把融合PreSinkhorn与独立Pre/Sinkhorn同时相加。

### 8.1 Pre 与拆分链的阶段成本

Pre 接受 BF16 输入 `[T,N,D]`，本版数值路径限定 N=4、T≤256、D 为 16 的倍数且 ≤16384，返回六个输出。T 是 token 数，N 是残差流数，D 是每条流的通道数。拆分链先执行同一 Pre，再调用 20 轮 Sinkhorn 和 Post，中间接恒等外部层；对应三次真实库 API 的净指令窗口和。单测的独立 Sinkhorn/Post 具有自己的调用属性和缓存状态，链内模型使用同一链上测得的阶段。

**分块。** 设投影归约宽度 K=ND。参考 arch35 MhcPre 的 CanUseMk 和 CalculateMkKSplit，K≥512 时走 M-K 候选分支；T≤256 时分成单组 M 块，沿 K 最多切 32 份，临界 K 片宽为 Kc=256⌈K/(32×256)⌉。单核投影工作的代理为 ⌈T/16⌉⌈Kc/32⌉。16 为矩阵行粒度，256 为 K 切片对齐，32 为相对于观测到的每个对齐片 8 条 MMAD 的计数粒度。K<512 时使用 ND 候选，工作代理为 ⌈T/32⌉⌈K/256⌉。公开源码是固定快照，库二进制的 PC 对应关系尚未验证，尤其新架构或较长 K 的内部部分和策略可能变化。在冻结前的开发 trace 里，D128–1024 的多个低 T 样例虽然投影总 FLOPs 变大，最忙核仍是 8 条 MMAD。该现象符合沿 K 分配给更多核的解释。这里的单核工作代理也包含 M 方向块数，不保证在所有 T 上精确还原动态指令。

**资源校准。** 对同一库入口合同，取最晚结束核，从全核共同起点开始按真实 tick 扫描。每个时间片依 Cube、SIMD、搬运（MTE/FIXP）、其余控制与等待的优先级，归入唯一资源 j。分别以非负直线近似这些互斥时间片。每段净时延为 Σⱼ(aⱼ+bⱼxⱼ)，a 是可见固定成本（µs），b 是每工作单位的可见服务时间（µs/单位），x 是下表对应工作量。某项若只有一种开发输入，使用该项实测中位数作截距，斜率未识别并设零。Pre 各开发形状最后完成核上的独占 Cube 时间均为零，依赖于 Cube 的等待归在控制项。该分解无法识别独立 Cube 吞吐或隐藏的重叠时间。系数及旧 DB 哈希分别保存在[模型参数](../results/v0.9/mhc/model.json)和[冻结记录](../results/v0.9/mhc/freeze.json)中。

|阶段|Vector 工作（逻辑组）|搬运（逻辑 KiB）|控制（块或波次）|
|---|---|---|---|
|Pre|⌈T/64⌉⌈D/256⌉|[4K(N²+2N)+2TK]/1024，来自 FP32 投影权重及 BF16 输入|⌈T/16⌉|
|链内 Sinkhorn|⌈T/64⌉·20N²|4TN²/1024|⌈T/64⌉|
|链内 Post|w⌈D/(64J)⌉N(N+1)|4TND/1024|w|

Post 的 J 是对每个 token 进一步沿 D 切出的片数。初值为 1；当 T<32、片宽大于 256 且当前任务数 TJ≤32 时，J 翻倍。最忙核的波次 w=⌈TJ/64⌉。T≥32 的逐行候选单列模板；开发集该分支仅一个点，未识别独立斜率。各阶段的逻辑字节作为共享流量压力代理，不等于真实总线传输量。独立输出的 Pre 和链内 Pre 共享相同实现和输入属性；这里固定的输出合同不能代替无辅助输出的推理模式。

**例，单独 Pre。** [4,4,1024] 中 K=4096，Kc=256，每核投影工作代理为 8，逻辑读需求为 416 KiB。代入 M-K 模板可见计算/向量/搬运/控制分别约 0/1.00/3.28/4.95 µs，总计 9.23 µs；旧开发 trace 为 8.73 µs。

**例，拆分链。** 同一 [4,4,1024]，每个 token 的 Post 沿 D 切 4 片，16 个任务在一波完成。Pre、固定 20 轮 Sinkhorn、Post 分别预测 9.23、5.58、3.32 µs，链共 18.13 µs。采集时核对三次 API 同步、9 组全部输出、顺序 launch ID 与三个 DB 的实际入口；外部注意力、MLP 和 launch 间空闲由系统层计时。独立验证范围以逐例表为准；超出已测量范围的数字属外推。冻结模板在 950DT 上校准，更换代际需重新确认分核、库分支及阶段成本。

实现依据为固定版本 ops-transformer 14d022b1 的 `mhc/mhc_pre/op_host/op_tiling/arch35/mhc_pre_tiling.cpp`。CANNBot 的 `sgl_mhc_pre_post` 卡说明不同输出合同的边界；本轮 CA 用的是 CANN 9.2.1 安装包的 ACLNN 入口。

单次打包 T 超过 256、其余属性成立且 D≤16384 时，系统入口保留原有整段流水重叠估计，并标记 `extrapolation`。本轮独立误差只适用于新测范围。

## 9. KDA：状态递推与chunk前向

### 9.1 Recurrent

**算例 · KDA recurrent。** [1,3,1,1,128,128] 只有一条 128×128 状态链。第 2 个 token 读取第 1 个更新后的状态，第 3 个同理。因此模型取三步串行，预测 22.02 µs，CA 为 21.93 µs。

shape=(B,S,Hq,Hv,K,V)，Hv须为Hq倍数。每个sequence/value-head的state记为 \(S_t\in\mathbb R^{V\times K}\)，则

\[
\bar S_t=S_{t-1}\odot d_t,\quad
\delta_t=\beta_t(v_t-\bar S_tk_t),\quad
S_t=\bar S_t+\delta_tk_t^T,\quad o_t=S_tq_t.
\]

\(d_t\) 是gate衰减；raw gate前处理和q/k归一化另有工作，不包含在下面的代理中：

\[
F_{rec}=BSH_v(7KV+2V).
\]

7KV分别来自衰减、预测matvec、rank-one更新和输出matvec，2V来自残差/beta。过去上下文压缩在state内，不能追加FA式的上下文L工作。源码任务数BHv、按block步进分配，每任务S个token串行；真实blockDim未知，核数不填猜测。

950当前recurrent支持BF16、K128、V128/256、S≤8、state为BF16/FP32。因此推测8对应主模型Q9不能直接调用此路线，模型返回unsupported；也不能未经状态槽/接受token规则验证而静默拆成8+1。

### 9.2 Chunk

**算例 · KDA chunk。** 单头单序列 S=128、K=V=128，块长 64。共有两个块，每块有 64×65/2=2080 个有效因果位置，两块合计 4160 个；第二块的状态更新依赖第一块的末状态。该例可返回工作量和依赖。目前完整库调用在 CA 遇到同步错误，耗时字段返回 null。

每个chunk长度 \(c_j\le C\)，C为64或128。核心关系为

\[
q_g=q\exp(\gamma),\quad k_g=k\exp(\gamma_{last}-\gamma),\quad
A_{kk}=(I+L_{tri})^{-1},\quad W=A_{kk}W_0,\quad U=A_{kk}U_0,
\]
\[
V_{new}=U-Wh_{prev},\quad
h_{next}=\exp(\gamma_{last})\odot h_{prev}+k_g^TV_{new},\quad
O=scale\cdot q_gh_{prev}+A_{qk}V_{new}.
\]

\(L_{tri}\) 为带gate/beta的严格下三角KK项。定义 \(P=\sum c_j(c_j+1)/2\)、\(P_-=P-S\)、\(D_c=\sum c_j^2\)，用满chunk数和尾长O(1)计算。每sequence/value-head的阶段工作为：

| 阶段 | 工作量 | 口径 |
|---|---:|---|
| Aqk | \(2KP\) | 有效因果点积 |
| Akk | \(2KP_-\) | 不含三角逆/solve |
| Post-WU | \(2(K+V)D_c\) | 稠密GEMM等价量 |
| FwdH | \(4SKV\)，另有\(n_{chunk}KV\)乘法 | matmul及state衰减 |
| Finalize | \(2SKV+2VD_c\) | 稠密等价量 |

整体乘BHv。三角求逆、gate、padding/转换尚未纳入，`total_executed_flops=null`。FwdH每条state chain沿chunk串行，其他阶段可按chunk/head并行；这五段是主kernel内部阶段，不能加五次launch；本轮真实API另派发一个辅助kernel，必须把辅助kernel及主kernel均计入调用链，也不能无条件取Cube/Vector总时长的max。

当前chunk支持BF16/FP16、K/V为16倍数且在[16,256]、Hv≤128、FP32 state。公开Aqk/Akk输出 `[B,Hv,S,C]` 必选；关掉其他中间输出不意味着可以省掉它们。packed/varlen、投机state-slot与真实fusion flags未建模。KdaInputProj是另外的已识别库候选，包含输入MX quant，当前没有其完整结构/时延入口，不能再无条件重复加独立quant。

参数解释：B为batch，S为序列长度，Hq/Hv分别为query/value head数，K/V为key/value通道数。递推式S_t是第t步的状态矩阵（与长度S不是同一量）；q_t/k_t/v_t是当前token向量，d_t是逐通道衰减，beta_t为更新系数，delta_t为预测残差，o_t为输出；F_rec是代理FLOPs。实际接口还将q乘scale，当前scale=1/sqrt(K)。数学写成V×K或存成K×V是转置约定，caller按state_v_first=false检查实际布局。

chunk式中C为最大chunk长度，c_j为第j块有效长度，gamma为累计gate，gamma_last是该块末尾累计gate；q_g/k_g是gate调整后的向量，L_tri为严格下三角关联项，I在矩阵逆式中是单位矩阵；A_kk为三角逆，A_qk为因果QK关联，W_0/U_0为逆变换前的key/value项，W/U为变换后项。h_prev/h_next是chunk前后的状态，V_new是更新残差，O是输出，scale是query缩放；n_chunk为chunk数。P是有效因果token对数，P_-去掉对角，D_c是各块完整平方尺寸之和，均为计数。阶段表为数学工作量，不是安装包五个独立kernel。

通俗地说，recurrent每一步都依赖上一步状态，所以短序列也有不能并行的链；chunk把块内运算改为矩阵计算，但跨chunk的状态仍需顺序传递。recurrent 把状态大小和序列长度代入各阶段，chunk 保留分块工作图与 CA 同步问题记录，不能从S≤8递推公式猜出Q9实现。

## 10. MoE：专家负载与GMM–SwiGLU–GMM组成

以下三例使用同一次真实链采集。输入 [M,K,F,E]=[32,256,128,2]，两个本地专家各接收 1、31 行。

| 算子 | 工作与计时 | 模型 / CA |
|---|---|---:|
| GMM1–SwiGLU–MX 量化 | 各专家从 K=256 投影到 2F=256。SwiGLU 合并两半后剩 32×128 个值，再按每 32 个值量化。按 16 行切块时两位专家分别占 1 和 2 个行块。 | 9.94 / 8.21 µs |
| GMM2 | 读取第一段生成的量化值及 scale，各专家把 F=128 投影回 K=256，矩阵部分共 2×32×128×256=2097152 FLOPs。 | 3.70 / 3.90 µs |
| MoE FFN 组合 | 同一批专家行先执行融合 GMM1，再执行 GMM2；用未舍入的两段净窗口相加。本例不计跨卡 dispatch/combine。 | 13.64 / 12.12 µs |

当前shape=(M,K,F,E)，M为本rank dispatch后行容量，F为SwiGLU后宽度，E是本rank专家数，**不是EP数**。必须输入 \(m_e\) 直方图或带明确语义的group_list，结构描述允许空专家与行填充，\(T=\sum_e m_e\le M\)。当前时延入口要求所有行均有效，即 T=M。EP只约束全局到rank的映射，不能唯一确定本地直方图。

\[
C_e=dequant(X_eW_{1,e}),\quad Z_e=SiLU(A_e)\odot G_e,
\]
\[
(\hat Z_e,s_e)=MXQuant(Z_e),\quad Y_e=GMM2(\hat Z_e,W_{2,e};s_e).
\]

第一权重逻辑为[E,K,2F]，因此

\[
F_{GMM1}=4TKF,\quad N_{SwiGLU}=TF,\quad F_{GMM2}=2TFO,
\]

O是最终输出宽度。相同T但不同 \(m_e\) 的FLOPs一致，实际tile尾部和调度并不相同；另记录活跃专家数与 \(\max m_e/(T/E)\)，但该比值不是时延公式。模型复杂度O(E)，不为每token创建任务图。

初版仅ND MXFP8（E4M3FN/E5M2、E8M0尺度、普通V2分支），不混合FP4/NZ/per-token路径。MX逻辑每32元素一尺度，物理以两个尺度打包：输入scale `[M,ceil(K/64),2]`，输出scale `[M,ceil(F/64),2]`。实际总流量还受padding、中间区、重读和缓存影响。

`aclnnGroupedMatmulSwigluQuantV2`仅包含GMM1→dequant→SwiGLU→MX quant；`groupedmatmulswiglugmm`/`moe_ffn`是组成链别名，不是已确认的同名单kernel。本轮caller将GMM1实际device output和E8M0 scale直接传到GroupedMatmulV5；是否通过完整CPU golden、具体shape和每kernel身份以总误差表为准。router、输入量化（若未融合）、dispatch/combine通信及TP collective单独存在，当前不输出其耗时，不能视为零。AIC/AIV交接和组尾调度必须从真实tiling/trace建立，不假设一个专家一个核。

参数解释：M为本rank输入行容量，K为输入隐藏宽度，F为激活后中间宽度，E为本rank专家数，m_e为专家e的有效token数，T为其总和，O为第二层输出宽度（本轮O=K）。X_e是专家输入，W_1,e/W_2,e是两层权重，C_e为解量化后的第一层输出；A_e/G_e为其激活半部和门控半部，Z_e为激活相乘结果，Z_hat_e和s_e为MX输出及scale，Y_e为第二层输出。SiLU(x)=x/(1+exp(-x))；⊙表示逐元素乘。F_GMM1/F_GMM2以FLOPs计，N_SwiGLU是激活元素数。每32元素一scale，物理最后维度2打包；每个E8M0字节表达共享指数而非浮点乘数本身。

通俗地说，同样多的token如果挤到少数专家，会改变尾块和最慢核工作量。需要本rank的专家直方图；EP大小本身不能替代它。当前直接串联两次真实API，且按同步边界把两个DB归属到组件，计算总时间时只相加一次。

## 11. 校准、独立验证和仿真效率

### RMSNorm 与 mHC Post 的临界核时间

在 950DT 的已测路径上，BF16 RMSNorm 调用 ops-nn 的 `aclnnRmsNorm`，实测入口为 `entry_kernel_1388`。四流 BF16 mHC Post（含残差映射）调用 ops-transformer 的 `aclnnMhcPost`，少 token 分支对应 `entry_kernel_5`，按 token 分核分支对应 `entry_kernel_6`。校准目标是全芯片跨核净窗口。选取最后完成的 AIV，把它开始前的时间计作 arrival，从开始到结束按 SIMD、MTE3、MTE2、SCALAR 的优先顺序归属重叠区间，其余计入 other。六项相加恢复 CA 时间。这个约定衡量流水实际暴露了多久，不直接给出每条队列的物理带宽。

**分给最忙核多少工作？** 令 $T$ 为 token 数，$D$ 为 BF16 通道宽度，$W=\lceil T/64\rceil$ 为最忙核的行数。每行分 $J$ 个通道片。RMSNorm 总取 $J=1$。Post 在 $T\ge32$ 时也取 1，少于 32 时从 1 开始，只要 $\lceil D/J\rceil>256$ 且 $TJ\le32$ 就将 $J$ 翻倍。活跃核数为 $A=J\lceil T/W\rceil$，每片宽度 $d=\lceil D/J\rceil$。RMSNorm 的 SIMD 工作量为 $U=W\lceil D/128\rceil$ 个向量组。Post 对 $N=4$ 路残差做加权混合，使用 $U=W\lceil d/64\rceil N(N+1)$ 个向量工作单位。

**启动上下文。** 调度器可显式提供 `resident_aiv`，即隔离同类序列的前序最大活跃核数 $H$。$H=0$ 表示首次发射，$0<H<A$ 表示覆盖扩展，$H\ge A$ 表示已覆盖，记三种状态为 $z$。前序高水位只能描述这类受控调用的历史。混合算子、并发、跨流或缓存驱逐的行为尚无标定。调用方省略 $H$ 即可获得无状态默认预测。

**RMSNorm 每核如何分轮。** 最忙核负责 $W$ 行，第 6.1 节按 UB 容量算出一轮可容纳的最大行数 $u$。从 $W$ 中逐次取最多 $u$ 行，得到每轮行数 $n_i$。例如宽度 4096 时 $u=7$，512 个 token 在 64 核上每核 8 行，计划为 $[7,1]$。每轮写两种输出：$4n_i$ B 的 FP32 行统计量 `rstd`，以及 $2Dn_i$ B 的 BF16 归一化结果。在开发例 `[512,4096]` 的关键核上，对应的四笔传输依次为 28、57344、4、8192 B。首轮主输出尚未写完时，尾轮已经开始。

**把写入在途时间与实际暴露时间分开。** 以下量均为 950DT 的 CA tick（1650 tick/µs）。第 $i$ 轮的向量工作单位为 $x_i=n_i\lceil D/128\rceil$。令 $r_i$ 为统计量写出的发起时刻，以首笔统计量写的开始为零；$y_i$ 为同轮主输出的写发起时刻。对于降平方和、归一化两个片段，各用一个截距与单位工作系数：

$$
r_1=0,\qquad r_i=y_{i-1}+a_R+b_R x_i\ (i>1),\qquad
y_i=r_i+a_N+b_N x_i.
$$

$a_R,a_N$ 表示片段固定 tick，$b_R,b_N$ 表示每个向量工作单位增加的 tick。第一轮规约发生在参考零点之前，已包括在整次 SIMD 估计中。设 $K=L/4$，其中 $L$ 是相邻统计量写入的有效地址窗口（字节），每个统计量占 4 B。$P=\max(0,\min(A,K/W)-1)\min(1,A/K)$ 是邻近写者的有效数量。两笔写入的结束时刻取

$$
e_{r,i}=r_i+a_s+b_s P,\qquad
e_{y,i}=\max\{e_{r,i},\ y_i+a_y+b_y(2Dn_i/1024)\}.
$$

$a_s,a_y$ 是固定写延迟，单位 tick；$b_s$ 是每个有效邻近写者的 tick，$b_y$ 是每 KiB 本核本轮主输出的 tick。43 个旧 shape 中共有 44 对关键核记录，每一对的主输出完成时间都不早于统计量。这个局部完成顺序约束可以从数据库核对，具体缓存或总线仲裁仍未知。用每轮**本核**字节估计写服务，保留核间小写入的竞争项。

令 $I$ 为所有 $[r_i,e_{r,i})$、$[y_i,e_{y,i})$ 区间的并集，$|I|$ 表示至少有一笔写在途的 tick 数。令 $S$ 为整个关键核的 SIMD 预测 tick，按启动状态校准截距，加每个向量工作单位的服务时间。末轮主输出发起时刻为 $y_m$，$m$ 是 UB 轮数。估计

$$
O=\min\{S,\ |I\cap[0,y_m)|\},\qquad
t_{\rm MTE3}=(|I|-O)/1650,\qquad t_{\rm SIMD}=S/1650.
$$

$O$ 是与向量工作重叠的写回时间。该式把首笔统计量发起至末轮主输出发起之间近似为可覆盖窗口，实际 SIMD 在窗口内可以有空隙。因此真实覆盖量由独立 trace 检验。末轮写入在向量窗口关闭后的尾部全额计入。另加 arrival、SCALAR、MTE2、other 的暴露时长；这些阶段使用已有状态相关截距。未知 $H$ 时对三个状态按既有高水位分布加权，已知 $H$ 时选择对应状态。在线只合并每轮两笔写的区间，无需枚举执行指令或读取数据库。[本轮阶段系数](../results/v0.12/rms/profile.json) · [基础启动参数](../results/v0.10/wide/profile.json)

下表给出冻结的 950DT 有效系数。截距单位为 tick，服务斜率的分母按工作单位列出。用非负拟合和转换尺度 80 tick 的稳健损失，目标取自关键核的指令区间。

| 项 | 截距 | 斜率 | 开发观测及可识别范围 |
|---|---:|---:|---|
| SIMD 总时间 | 首次 515.38、扩核 324.39、热态 302.77 | 11.288 tick/向量组 | 43 个 shape，按各状态权重合成截距 |
| 统计量写服务 | 335.89 | 117.018 tick/有效邻近写者 | 44 笔 FP32 写；有效窗口取 128 B |
| 主输出写服务 | 302.50 | 6.610 tick/KiB | 17 笔可辨识写，单笔 0.25–16 KiB；主输出晚于统计量收尾超过 16 tick，期间无后续统计量写干扰 |
| 归一化片段 | 75.46 | 6.049 tick/向量组 | 44 对统计量和主输出开始时刻之差 |
| 后续规约片段 | 270.87 | 4.491 tick/向量组 | 42 个单轮样本，以 SIMD 活跃时间减去同轮归一化时间作代理；开发集仅一个多轮 shape |

**算例。** `[512,4096]` 中每核首轮 7 行、尾轮 1 行。模型估计写在途并集 1788 tick、被 SIMD 覆盖 1371 tick，写回暴露 **0.253 µs**，CA 对应 2230 tick、1057 tick 和 **0.711 µs**。模型整例为 4.969 µs，CA 为 4.851 µs；写回仍低估 0.458 µs，部分被 SCALAR 和发射偏移的高估抵消。首轮主输出 56 KiB 已超出可独立辨识的 16 KiB 写服务范围，逐轮归因应结合新的多轮样本检查。

**mHC Post 的三个条件时间。** 先按上述分核计算工作量，再分别估算首次、扩核和热态。对状态 $z$，阶段 $j$ 的暴露时间为 $t_j(z)=b_{j,z}+s_j X_j+e_j(W-1)$，六段之和记作 $t(z)$。$b$ 是固定微秒数，$s$ 是每单位工作的微秒数，$e$ 是最忙核多处理一行的附加微秒数。SIMD 取 $X=U$。按 token 分核时，MTE2 取 $X=2W dN/1024$ KiB 作为每核搬运的规模代理，SCALAR 和 MTE3 各有额外行成本。其余服务斜率固定为零。两条路线分别用 43 个已合格的开发 shape 校准，每个阶段使用非负系数和尺度为 0.1 µs 的稳健损失。现有 trace 无法单独识别缓存状态或多流之间的干扰。

**不传前序覆盖时怎样兼顾三种状态？** 通道切片分支（$J>1$）在同一 shape 上可能有显著不同的启动时间。令 $l=\min_z t(z)$、$h=\max_z t(z)$，单位为 µs，默认返回

$$
\widehat t=\frac{2lh}{l+h}.
$$

这个选择使对最快状态的相对高估 $(\widehat t-l)/l$ 等于对最慢状态的相对低估 $(h-\widehat t)/h$，其余状态介于两端。两端来自条件模型，所以真实 CA 仍可能超出这个范围。阶段明细按相同权重混合两端时间，最快状态权重为 $h/(l+h)$，最慢状态权重为 $l/(l+h)$。这样六段仍能加回总时间。该默认值适合关注未知历史下单次调用误差的场景；连续热态负载可能持续偏高。对于未切通道或按 token 分核的分支，按组内开发状态次数取三个条件时间的加权中位数。提供 $H$ 时所有分支都直接返回对应的 $t(z)$。

例子 `[1,4,4096]` 切成 16 片，每个 AIV 做 256 列，$U=80$。首次、扩核、热态分别估 4.788、4.211、2.313 µs，默认估 **3.120 µs**。该开发 shape 的 CA 为 2.256 µs，属于热态。因此调用方已知 $H\ge16$ 时可得到更贴近该观测的 2.313 µs。模型给出的三个条件时间仅用于执行历史敏感性分析。[分组与阶段系数](../results/v0.12/post/profile.json)

**适用范围与成本。** 当前 RMSNorm 接受 $T\le512$，Post 接受 $T\le256,N=4$，两者均要求 $D\le8192$ 且为 16 的倍数。被接受的 shape 仍可能处于稀疏区。在线只计算分核、固定数量的阶段和状态，无需逐行访问张量。尺寸放大时，同一 kernel、同宽度、同活跃核数仍可能经历首次发射、覆盖扩展或 UB 分块改变。后端的有效阶段成本当前只在 950DT CA 上校准。[新 shape 分组与分块检验](../results/v0.12/audit/summary.md) · [固定宽度放大实验](../results/v0.11/scaling/summary.md)

### 独立 mHC Sinkhorn 的临界核时间

N=4，每个 token 对 16 个 FP32 元素做 I 轮行列归一化。T 是 token 数，每核最多 W=ceil(T/64) 行，A=ceil(T/W) 个 AIV 有任务，I 是调用方传入的迭代数。模型分六段估时，单段时间为 a+bX，a 是固定微秒开销，b 是每个工作单位所需的微秒数。SIMD 的工作量为 I·ceil(16W/64)，其中 64 是向量组的候选元素数。MTE2 用每核读 64W/1024 KiB，MTE3 用全片写 64T/1024 KiB，作为竞争代理。arrival 用 log₂(A)，SCALAR 用 W−1，other 取常数。工作量不变的阶段使用可见耗时的中位数。各段只计独占的暴露时间，再相加。T=9 和 128、I=20 的实测最忙核均有 1903 条 RV 事件，支持在小行组内共享向量循环的解释。[逐阶段校准参数](../results/v0.9/vector/profile.json)

### KDA recurrent 的临界向量核时间

在本版已校准的 B=Hq=Hv=1、K=128、V=128/256、S≤8 路径，只有一条状态链，向量工作单位为 U=S·K·V/16384。分母 16384 表示一个 128×128 的状态块。对最后结束的 AIV，依 SIMD、MTE3、MTE2、SCALAR 的优先级将忙区唯一归属，剩余区间作为控制外间隙。各阶段的有效时延用 a+bU 表示，a 以 µs 计，b 以 µs/状态工作单位计。单链发起偏移取零；读写的多核竞争因子 log₂(1+A) 在 A=1 时等于 1。[冻结参数与开发 trace](../results/v0.8/vector/profile.json)

| 阶段 | a | b |
|---|---:|---:|
| SIMD | 4.30242 | 2.93740 |
| MTE3 写 | 0.24121 | 0.44823 |
| MTE2 读 | 1.41485 | 0.02602 |
| SCALAR | 4.11182 | 0.06377 |
| 剩余间隙 | 1.52586 | 0 |

将五项相加得到设备时间。第 9.1 节算例有三个状态工作单位，代入即得 22.02 µs。读、算、写之间隐藏的重叠已经包含在可见阶段成本里，跨代需重新校准。

### Cube 计时校准

Matmul 的输出 tile、MMAD 和阶段成本见第 4 节。Addmm 在相同的 BF16 ND 合同下单独提取临界核的忙时、控制间隙与队列重叠，第 6.2 节列出对应公式和校准值。

KDA chunk的可执行结构入口为`operator_model.cube_v8_chunk.predict('kda_chunk',shape,attrs)`。设单个序列切成长度为c_i的块，块上限C，块数n=ceil(S/C)，并行状态链数U=B×Hv。对单条链，有效因果元素对数P=Σc_i(c_i+1)/2，块矩阵稠密大小D=Σc_i²。QK与严格下三角KK工作分别为2KP与2K(P-S)，Post-WU为2(K+V)D，FwdH为4SKV次FLOPs和nKV次状态衰减乘法，Finalize为2SKV+2VD；全算子乘U。K/V是每头key/value宽度。FwdH在同一链内按块顺序推进，QK/KK/Finalize可按不同块和头分核。该入口返回各阶段工作、依赖、分块、适用属性与尚缺的三角求解/资源成本。上述两次v0.8实测均遇同步错误，故可执行结构模型的时延字段为null。

### Chunk KDA 的十组独立结构核对

从两份已冻结清单选取十个不同 shape，B=1，S 从 1 到 256，Hq=Hv 取 1、2、4、8，K=V 取 16 或 128，C=64。CPU 核查器枚举每个头、块、有效 token 和有序 token 对，检查六个阶段的逻辑工作量、工作为零的阶段、DAG 顺序、任务数、串行步数和状态字节。S=1 时严格下三角为空，KK 工作为零，该节点在 DAG 中省略。用非零初始状态和每头一个活跃 K/V 通道，进一步对照块内三角求解和逐 token 递推的输出、各块末状态；已激活 gate 和 raw-safe gate 均覆盖。十组均通过，FP64 运算的最大差值为 9.72×10⁻¹⁷。该结果限于逻辑计数与稀疏数学恒等式。安装包 CA 耗时仍为 0/10，BF16 全输出、内部融合与物理吞吐待验证。逐项结果见 [chunk-structure-validation.json](../results/v0.8/cube/chunk-structure-validation.json)，可运行 `python3 scripts/cube-v8-chunk-structure.py` 复现。

### 融合算子的流水区间模型

1. **还原工作。** 按代码中的tile与归约方向，把每次shape换成各队列的工作单位。Indexer对长KV按128个key为一段执行点积，按512个key为一段估计TopK归并；mHC按并行token波次计算投影、读写与归约；MoE从本rank的专家token直方图分别划分各专家的输出矩阵，只对非空专家分配任务。输出列和归约维交换时改变GMM2的工作单位。上述分块是950DT的实现代理，执行分支和真实库tiling仍需按shape核对。
2. **从开发trace提取流水重叠。** 每个kernel、每种核类型取最晚完成的核，按指令开始/结束tick把时间轴切成若干段。每段记录同时工作的流水集合，例如SCALAR+MTE2和SIMD+MTE3。等待、启动和首核错峰归在空闲段，计入真实关键路径。每种集合只累计一次时长，解决同一时间读、算、写重复相加的问题。
3. **在新shape重算。** 设参考shape为r，新shape为s，每个资源q的工作量为W(q,s)。流水集合A的时间估为其参考占用时长t(A,r)乘以集合中最大的工作量比例。最忙核的总时间和多个模板的稳健估计为

   $T(s)=\mathrm{median}_{r}\left[\max_{h}\sum_A t_{h,A,r}\max_{q\in A}\{W(q,s)/W(q,r)\}\right]$。

   h是核类型，A是同时占用的资源，t用微秒，W是下面的逻辑块数，比例无单位。对空闲区间采用波次比例。该式假设同一实现路径能够保留参考trace的重叠顺序；跨算法切换、KV尾部归约方式、在用核数变化或专家碎片会破坏这一假设。库端同步出现多个kernel时，分别应用公式，按实际串行依赖相加。组合mHC只含Pre、Sinkhorn及Post，MoE只含两次GMM链，不包含路由通信或host开销。

| 路径 | 波次与工作单位 |
|---|---|
| Indexer (B=1,G=8,D=128) | 波次p=ceil(Q/32)；CUBE/MTE1工作=p·ceil(L/128)·ceil(G/16)，MTE2=p·ceil(L/128)，SIMD/MTE3=p·ceil(L/512)。Q为查询长度，L为key长度，G为head数，D为head宽度。 |
| 融合 PreSinkhorn | T 为 token 数，N=4，D=4096。投影波次 p=ceil(T/32)，CUBE/MTE1 为 p·ceil(ND/128)，向量和读写为 p·ceil(D/128)。I=20。普通 Pre 与拆分链使用第 8.1 节的分核方法。 |
| MoE | 记每专家有效行为m_e，输出宽度o，归约宽度k，p=ceil[(Σ_e ceil(m_e/16))·ceil(o/128)/32]。Cube和读工作=p·ceil(k/64)，向量工作=ceil[Σ_e ceil(m_e·F/256)/64]。第一层o=2F、k=K，第二层o=K、k=F。F/K分别为激活后宽度和输入隐藏宽度。SCALAR、FIXP和等待项以p计。 |

例如MoE输入有1行与31行分给两个专家时，先产生1+2个行块，再与输出列块相乘。即使总行数相同，不同专家分布也可能改变波次。对Indexer的 Q=1、L从1024增至2048的同一路径，点积段数8增至16、TopK段数2增至4；控制部分仍是一波。重叠的读/算按较慢队列完成时间伸缩，整个kernel不直接乘2。

参考区间描述的是950DT上给定库路线的有效服务时间。更换SoC需替换核数、单资源服务成本和tiling，并重新验证区间重叠；库专有的控制/等待尚未独立识别为硬件参数。在线查询仅巡访开发模板的有限集合，MoE另需一次遍历专家直方图。输入dtype、输出与布局合同见model.json的contracts；分项实现与冻结清单见本目录model.json及freeze.json。

**采集和验证。** Indexer、融合 PreSinkhorn 和 MoE 使用 v0.8 的独立 CA 观察。MoE 第一段的设备输出与 scale 直接送入第二段，两个组件和整链共享一次采集。每个 DB 按 API 同步和 launch ID 归属。其余路径的开发与冻结验证数量、实际 kernel 和成本见总误差报告。

### 查询复杂度与误差定义

默认入口在进程内复用参数，每次从请求的 shape 和属性重新计算时间。quant 对少于 256 个 UB tile 的请求使用原递推，对更长的循环保存五个时刻（当前读、算、写完成，以及上一个 tile 的算、写完成）。每个依赖是“某个前驱的完成时间加阶段成本”，多个前驱取最大值，因此相同 tile 的更新可用 max-plus 矩阵的重复平方合并。完整行组与尾行/尾列分别处理，参数与逐 tile 算法相同。重复次数为 n 时只需 O(log n) 次固定大小的矩阵合成。FA 长 KV 采用首轮加稳态间隔，MoE 只遍历专家直方图。100 ms 验收按统一入口单次墙钟计算。v0.17 从源码目录外调用已安装 wheel，在 DT、PR、970 的共 453 个冻结请求上测得中位 0.183 ms、P95 0.569 ms、最大 3.632 ms；单列导入 39.20 ms，含 3 次 KDA chunk 的结构查询。Matmul 末波最多枚举本波实际子块数；FA 合并相同任务波，时长计算不逐项访问 KV 元素。[逐算子查询开销](../results/v0.17/wheel-smoke.json)

误差统一使用

\[
APE_i=|\widehat T_i-T_i|/T_i,\quad
WAPE=\sum_i|\widehat T_i-T_i|/\sum_iT_i.
\]

其中i为成功实测shape索引，T_i是CA参考微秒值，T_hat_i是冻结预测微秒值；APE/WAPE均为无量纲比率，报告时乘100显示百分比。通俗地说，APE看单例相差多少，WAPE看同族总绝对偏差占总参考时间的比例，较慢样本权重更大。

每个算子报告开发与独立验证数量、独立验证 WAPE 与最大误差、覆盖 shape 和缺口。未校准族写N/A而不是0%。不同算子/机制混合后的低WAPE可能被大耗时FA主导，不能代替逐算子表。当前总表按算子提供一套模型的独立验证误差。历史对照可从版本归档查阅。

## 12. 场景映射与跨代边界

用户场景包括prefill单卡batch=1/4/16，decode=1/4/16/32/128/256；输入32k/128k/256k/512k、输出1k；MTP草拟0..16、主模型校验1..17，常用草拟8对应Q9。prefill的R/T/Q由当次chunk和实际token打包决定，不能把总KV长度直接当作一次norm或projection的行数。decode矩阵行数通常与batch×本次query相关，但实际flatten、padding、DP/TP及实现分支仍须核实。

模型config足以给候选hidden/head/expert尺寸，无需先下载权重；但不足以单独确定每卡shape和dispatch。attention TP需确认head切分/复制，attention DP决定请求归属；MoE EP、TP+EP和纯TP需结合本rank专家数及token路由直方图。重复计量化、融合组件或通信都会改变系统估时。

对未发布的目标采用本页开头的跨芯片迁移公式。950 校准的有效发起间隔包含当时的反压。若新库引入新的融合、指令或不同缓冲策略，需要补充对应分支和测量。现有验证范围见[逐 SKU 误差](error-analysis.md)。

## 13. 系统接入与校验

系统框架使用 `operator_model.estimate(op, shape, attrs)` 取得设备微秒数和适用状态。当前首先覆盖 CANN 9.2.1 的 Ascend950DT_9582。可用输入、可选上下文、缺少合格时延时的调度器回退见[系统接入指南](integration.md)。各条路径的独立样本、实际库入口、误差及 CPU 端所需查询时间见[当前 CA 校验](error-analysis.md)。采集、冻结和复现实验的流程见[开发维护指南](../development/README.md)。
