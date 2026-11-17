# 当前整套算子建模方法

本文是项目持续维护的**完整方法说明**，覆盖所有已接入的算子族；不是某一轮的修改记录。逐算子准确率、每个 shape 的实际库调用和仿真成本由配套总误差报告维护，版本目录保存冻结参数和历史验收结果。本文说明当前实际执行的公式，不能将“计划建立”的时延模型误作已经可用的能力。

当前 v0.7 保留 v0.6 的三条耗时主模型：dynamicMXquant 的分核/UB 双缓冲及受限VF调度残差模型、matmul 的分块流水模型、普通 FA 的通用路径与 Q9 长循环专用路径。RMSNorm、Addmm、LightningIndexer、mHC、KDA 和 MoE 增加真实库调用、完整数值检查、通用机制图接口以及受限的小 shape 时延对照。新族的参数只由校准点拟合，冻结后留出点不参与调参；属性不匹配、超出本地支持域或数值检查未通过时返回 `predicted_us=null`。结构工作量不是指令数，两三个校准点也不足以代表全部算法/tiling分支；第11节给出实际执行的局部公式和边界。

## 0. 通用建模思想：算法与硬件分离，不拟合950DT答案表

**本项目要建立可复用的算子执行模型：从shape、算法、实现策略和硬件约束推导时间，再用CA检查推导哪里不对。目标不是记住950DT上各个shape的耗时。** 同一种方法可以用于其他shape和代际，但硬件参数、实际库算法和tiling必须按目标平台重新实例化；“方法可迁移”不等于“950DT参数直接缩放即可准确预测A7/A8”。

建模分为四层，参数不能相互冒充：

1. **算法层**：输入shape/dtype/属性决定数学工作、数据依赖和必须读写的数据。描述归约、矩阵乘、非线性、TopK、状态更新等基本过程。此层不填950DT实测总时延。
2. **实现策略层**：明确库采用哪个算法分支，怎样切tile、分核、处理尾块、复用UB/L1、安排双缓冲及同步。策略是受容量约束的规则；已确认源码规则、候选规则和未知dispatch分开标记，不能用目标shape的实测核数作为预测输入。
3. **硬件成本层**：按资源/指令类别描述每周期吞吐、结果延迟、发起间隔、带宽、核数和缓冲容量。优先用小微基准或可隔离的trace片段校准；每个参数保留单位、来源和适用条件。不同指令类别成本不同，不把所有Vector FLOPs除以同一个峰值。
4. **调度层**：把分块后的工作组成有依赖的任务图，计算资源竞争、缓冲占用和最慢核尾轮；以关键路径完成时间预测算子耗时。迭代次数很大时压缩重复子图，用稳态间隔计算，不逐token、逐条指令在线仿真。

统一形式为

\[
G=\mathcal L(\text{shape},\text{attrs};\pi,\mathcal H),\qquad
\widehat T=\operatorname{Schedule}(G,\mathcal H,\theta).
\]

G是带资源与依赖的任务图；L是从算子算法展开任务图的过程；shape是张量维度，attrs是dtype/mask/layout等属性，pi是实现策略，H是硬件结构配置，theta是资源成本参数；Schedule表示调度计算，结果T_hat以微秒表示。算法工作和实现策略给出“要做什么、怎么分”，硬件参数给出“每一步多快、哪些步骤能重叠”。任何无法区分的成本都要报告不可辨识，不能靠添加shape专属系数吸收。

不同类算子使用适合的机制：

| 类别 | 重点建模机制 | 典型算子 |
|---|---|---|
| 逐元素/归约Vector | 指令类别、向量有效宽度、归约树、UB容量、读算写与双缓冲 | dynamicMXquant、RMSNorm、mHC Post/Sinkhorn |
| Cube矩阵计算 | 输出tile、K循环、搬运与矩阵指令、尾块、跨核轮次 | Matmul、Addmm、专家GMM |
| Cube+Vector融合 | 两类任务间依赖、交接、共享缓冲与反压；不能只取两段总忙时max | FA、Indexer、mHC Pre/PreSinkhorn、GMM1–SwiGLU–Quant |
| 递推/分块算法 | 并行任务与串行状态链分开；算法切换建立独立图 | KDA recurrent/chunk |
| 多kernel组成 | 每个子图及实际调用边界；避免融合项重复计算 | mHC split、MoE FFN |

**校准不是任意拟合总耗时。** 用于校准的shape应覆盖机制变化（循环数、尾块、核数、算法分界），不是只增加相邻样本。共享资源成本尽量跨shape复用；实现专有的指令模板可以存在，但必须能解释循环和操作次数。先冻结代码/参数，再在未见shape上验证，并比较简单固定耗时基线；若简单基线更好，检查是否启动开销主导或机制尚未辨识，而不是自动加高次拟合项。

**当前成熟度也必须如实区分。** quant/FA等已经实现部分分块与流水机制，但仍有950DT局部校准残差和分支限制，并非完成了跨代验证。v0.7新增族的少量CA用于建立调用/正确性/指令证据，`a+bW`仅保留为**局部诊断对照**，不是通用模型的验收依据，不能拿它在两个校准点上的低误差证明泛化。通用资源图引擎与局部对照分别输出，缺少指令展开、共享资源成本或完整依赖时，通用预测标未校准/不完整，不把局部拟合值填入通用准确率栏。后文同时说明当前已经执行的机制与仍需补齐的证据。

通俗地说，我们要知道一条生产线每个工序做多少次、需要哪些设备、能否同时做，再换设备参数估时间；不能只记下几批产品的交付时间画一条曲线。这也是后续外推A7/A8能够成立的前提。

## 1. 目标、输入和实际库算子

目标平台固定为 **CANN 9.2.1 / Ascend950DT_9582**：平台配置为 32 AIC、64 AIV、1.65 GHz，单核 UB 253952 B。这里的时延预测以 CA Model 为参考，不是实际板卡实测；A7/A8 尚未校准。

| 算子族 | 模型输入 | 实际/候选库接口 | 当前能力 |
|---|---|---|---|
| dynamicMXquant | `(R,C)`，BF16→E4M3FN，block32 | ops-nn / `aclnnDynamicMxQuant` | 时延、分核、UB循环、选定指令计数 |
| matmul | `(M,N,K)`，BF16 ND，不转置 | ops-nn / `aclnnMatmul`，已测 MatMulV3 | 时延、启发式tile/分核、MMAD计数 |
| FA | `(Q,L,D,B,H)`，BF16 SBH，非因果，无mask/dropout | ops-transformer / `aclnnFlashAttentionScoreV2` | 时延、分核、流水；Q64路径MMAD计数 |
| RMSNorm | `(R,C)`、dtype、gamma_dtype、epsilon | ops-nn / `aclnnRmsNorm` | 结构模型＋局部时延校准/独立验证 |
| Addmm | `(M,N,K)`、self_shape、alpha/beta | ops-nn / `aclnnAddmm` | 结构模型＋局部时延校准/独立验证 |
| LightningIndexer | `(B,Q,L,G,D)`、sparse_count/mode | ops-transformer / `aclnnLightningIndexer` | 打分/TopK结构＋局部时延模型 |
| mHC | `(T,N,D)`；Sinkhorn为`(T,N)` | ops-transformer / `aclnnMhcPre`、`aclnnMhcSinkhorn`、`aclnnMhcPreSinkhorn`、`aclnnMhcPost` | 各组件与拆分链分别采集、受限校准 |
| KDA | `(B,S,Hq,Hv,K,V)`、recurrent/chunk | ops-transformer / `aclnnRecurrentKda`、`aclnnChunkKdaFwd` | recurrent/chunk分别校准和验证 |
| MoE融合链 | `(M,K,F,E)`及本rank专家token直方图 | ops-transformer / `aclnnGroupedMatmulSwigluQuantV2`，后续`aclnnGroupedMatmulV5` | 真实GMM1→MX→GMM2链；组件与链分别预测 |

CA 调用器使用安装包的真实 ACLNN 接口。参考源码来自固定版本的 ops-nn/ops-transformer，用来解释候选分支；它们不被宣称与安装包逐字同源。库文件存在只是“候选实现”，只有实际调用的 entry、tiling、核数和指令证据才能确定某个 shape 的执行路径。compact 数据库没有导出 ELF 时，运行时二进制 SHA 保持空值，不用安装文件 SHA 冒充。

之前使用的“桥接”指同 shape 的**采集方式一致性校验**：比较 msopprof 全 trace 与 npusim 压缩 DB 的平台、entry、核数、指令、正确性和计时窗口。它不是一种算子，也不算新的独立 shape。

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

基础模型对小shape的控制路径采用饱和回退（v0.6在下一节限定分支替换该回退）：

\[
T_{pipe}=W_{last}+D_{suffix},\quad
T_{ctrl}=a_c+b_c\min(\log_2(1+AIV),4),\quad
\widehat T_{q,base}=\max(T_{pipe},T_{ctrl}).
\]

\(T_{pipe}\) 是流水递推的最终完成时间，\(T_{ctrl}\) 是小 shape 经验控制路径的总时间，\(\widehat T_{q,base}\) 取两者较大值，避免把本应重叠或已包含的控制时间再重复相加。\(a_c,b_c\) 分别是 control_floor 的截距及对数特征系数（µs），`min(...,4)` 将该特征封顶在4。直观地说，极小张量尚未耗尽计算能力，准备/控制流程就可能占据大部分耗时。

回退项是小工作量数据中的经验总时延路径，不是物理硬件下界，也不能随核数无上限增长。以下基础阶段参数仍取13个旧quant trace，未随v0.6重拟合，已观察到3个多UB校准shape；小控制路径使用9个每核工作量≤4096元素的校准点。

| 阶段 | 截距 µs | 斜率 µs/单位 | 单位/含义 |
|---|---:|---:|---|
| prefix | 2.784570 | 0.101181 | \(\log_2(1+AIV)\) |
| first_read | 0.589411 | 0.017218 | 每tile读KiB |
| read_interval | 0.345656 | 0 | 每tile读KiB；现拟合为常数 |
| vector | 0.065425 | 0.007099 | \(m_i+z_i\) |
| control_floor | 3.830949 | 0.399708 | 饱和后的核数对数 |
| write_drain / suffix | 0.344848 / 0.009697 | — | 写排空 / 最后尾开销 |

这些是带该CA路径调度影响的统计参数。当前主要风险为每核工作不均、短列分支、UB边界、读间隔与写排空只有少量shape证据；小shape回退也可能遮盖错误的流水缩放。

### 3.4 v0.6：单UB宽行片段的VF调度残差

基础Vector成本采用SIMD指令区间并集，未必覆盖整个VF执行窗口里的调度间隙。v0.6复用19个已有quant trace，对每个shape先选择用于阶段分析的同一关键核，再计算

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

首行分支以pipeline+gap替换旧control floor，不再取两者max；否则会把正在修正的经验回退重新加回结果。其余短行片段和全部多UB路径保持原式，读/Vector/写阶段系数也完全不变。

`cols_per_core>=1024` 表示block32语义下每行片段至少产生32个scale字节，是当前保守的模型分支限制；**不能由此证明cacheline对齐、没有写冲突或真实库必然切换实现**。19点中的旧留出数据现已用于开发，角色已重分类。leave-one-out诊断只重拟合新增残差，而冻结的base已经见过部分点，所以它仅是**开发选择诊断**，不是独立交叉验证准确率。旧 `[576,4096]`、`[64,1024]` 两点误差反而增大，保留在总误差报告，不从汇总中删去。

当前版本先冻结公式/参数，再新增 `[36,5120]`、`[288,7168]`、`[1024,7168]` 三个未见shape验证；总CA预算450秒，分别检查宽行片段、decode候选聚合行数和多UB不变路径。成功/失败、正确性、实际dispatch、误差和成本以配套误差报告为准，不把冻结预测写成实测结果。

### 3.5 标量预测开销

对单UB点，递推严格退化为

\[
T_{pipe}=P_q+L_{read}(x_0)+S_{vec}(m_0+z_0)+D_{write}+D_{suffix}.
\]

v0.6的 `predict_us(case,use_cache=False)` 直接计算这条等价式，再应用同一guard和残差/回退；无需UB tile列表、逐阶段数组、递归冻结或缓存插入。它仍使用相同分核规则和系数，减少的是Python对象构造，不是省略硬件成本。公开标量查询入口 `operator_model.fast_query.Predictor` 对多UB也保持原递推，仅保存前两轮状态，临时内存从O(tile数)降为O(1)，并复用一次分核规划；完整 `predict()` 仍保留诊断字段，**不承诺完整诊断接口也更快**。标量/完整接口数值一致性由测试约束；性能应按两种接口分别报告。

## 4. matmul：输出tile、核轮次和阶段成本

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

## 6. RMSNorm和Addmm：结构模型与调用边界

### 6.1 RMSNorm

对x[R,C]、gamma[C]：

\[
r_r=(C^{-1}\sum_jx_{rj}^2+\epsilon)^{-1/2},\qquad y_{rj}=x_{rj}r_r\gamma_j.
\]

数学乘法 \(R(3C+1)\)、加法 \(RC\)、rsqrt R次；不含rsqrt的FLOPs为 \(R(4C+1)\)。这些是标量数学量，归约树和SIMD/VF指令仍需额外证据。逻辑字节 \(2RCb_x+Cb_g+4R\)，分别是x读、y写、一次gamma读、FP32 rstd写，不是实测HBM流量。

NORMAL候选分核令 \(f=\lceil R/64\rceil\)，活跃AIV \(\lceil R/f\rceil\)。\(C_a=align(C,32/b_x)\)，\(F=2^{\lfloor\log_2(C_a-1)\rfloor}\)，\(L=align(\lceil F/64\rceil,32)\)，则

\[
u=\left\lfloor\frac{253952-1024-2048-C_ab_g}{4C_ab_x+12+L}\right\rfloor.
\]

当 \(C_a\le16384,u\ge1\) 且不进入短列transpose候选时，tile行为 \(\min(u,f)\)，最忙核轮数 \(\lceil f/\min(u,f)\rceil\)。1024/2048分别为host与arch35保留空间，不能按常量名称误读。长归约Split/SplitAR和短列高行数transpose未移植，返回未解析dispatch。v0.7复用 `[1,128]` 并新增宽行及batch样本；实际entry和AIV见总误差表。仍不能据此确认所有输入。

参数解释：R是行/token数，C是每行元素数；r、j是行列索引；x/y是输入/输出数值，gamma是逐列权重，epsilon是防止除零的小常数，r_r是FP32倒数RMS。b_x、b_g为输入和gamma每元素字节数。f是最忙核的行数，C_a是按32字节对齐的列数，F是归约所用的2次幂片段，L是归约辅助缓冲字节数，u是UB最多能容纳的tile行数；align表示向上对齐。253952等容量单位均为字节。数学FLOPs是逻辑运算数，不等于真实向量指令数。

通俗地说，RMSNorm先算一行的平均平方，再用同一个倒数根号系数缩放整行。多行可分给不同核；列更宽则单核归约与搬运更长，不能只看总元素数。

### 6.2 Addmm

\[
Y=\beta\operatorname{broadcast}(S)+\alpha AB,\qquad F_{mm}=2MNK.
\]

另列MN合并加法和alpha/beta非1时各MN级缩放；这是稠密代数工作上界，广播前缩放或零系数消除可能减少工作。逻辑字节 \(b(MK+KN+|S|+MN)\)，不含中间tensor、copy/cast和多kernel开销。

参数解释：A[M,K]、B[K,N]是矩阵，S是可广播的self/bias，Y[M,N]是输出；M/N/K分别为输出行、输出列、归约长度；alpha/beta是无量纲缩放系数，F_mm单位为FLOPs；b为每元素字节数，|S|是self元素数。通俗地说，Addmm在矩阵乘结果上再加一个缩放后的bias；bias形状和系数能改变库选择的路径，不能只用M/N/K套Matmul时间。

self可为标量、[N]、[1,N]、[M,1]、[M,N]等合法广播。源码候选按空tensor/零系数、bias转换、inplace FP32 GemmV3、DAV_3510 FusedMatMul(add)、一般Matmul+Muls/Add/Axpy分支处理。FusedMatMul(add)要求完整矩阵self、alpha/beta≈1和其他dtype/layout条件；[N] bias不满足此特定分支。FP32融合候选还要求cubeMathType=3。

Addmm不能直接继承Matmul的总耗时或分核。已测 `[2,16,16]`、self `[16]` 仅观察到一个kernel，其他属性可能产生多kernel。两族局部时延模型见第11节；APE使用同属性校准/冻结验证分别报告，理论峰值利用率仍未校准。

## 7. LightningIndexer：打分、head归约和TopK

普通950接口固定D=128、key head数1，G∈{8,16,24,32,64}；Q/K/weights同为BF16或FP16。当前结构输入是BSND，不覆盖所有paged/TND路线。对每个query：

\[
a_{qgk}=\sum_dQ_{qgd}K_{kd},\quad
s_{qk}=\sum_gW_{qg}\max(0,a_{qgk}),\quad indices_q=TopK_k(s_{qk}).
\]

令每batch的合法query-key对数为

\[
v=\begin{cases}QL,&mode=0,\\Q(L-Q)+Q(Q+1)/2,&mode=3,Q\le L,\\L(L+1)/2,&mode=3,Q>L.\end{cases}
\]

mode3为右下对齐因果，\(P=Bv\)。工作量分别为点积 \(2PGD\)、ReLU PG次、weight乘法PG次、head归约 \(P(G-1)\) 次。实际padding/mask可能执行更多，另保留稠密 \(2BQLGD\) 作为参照。TopK比较、分块选取与合并的完整安装包算法尚未移植；时延先采用第11节局部代理，不宣称已恢复逐指令成本。不能用topk/L减少前面必须完成的打分工作。

输入输出唯一字节下界为

\[
2(BQGD+BLD+BQG)+BQk(4+2\mathbf1_{returnValues}).
\]

不包括score中间量、重读和workspace。结构计算为O(1)，不会分配BQGL规模张量。实际采集保留entry、核数和MMAD/RV次数，但不能把本地工作量代理当作完整安装包tiling。比如模型config提供32个index heads，直接TP8切成G4会不满足此普通接口，必须确认运行时是否复制heads或选择其他量化接口，而非静默改成G8。

参数解释：B是batch，Q是query长度，L是key长度，G是query/index head数，D是每head宽度；q/g/k/d是对应索引，Q/K/W在打分式中分别是query张量、key张量和head权重（不要把张量Q与长度Q混淆）。a是点积，s是head加权后的分数，indices是最终整数下标。mode为稀疏mask模式，v是单batch有效配对数，P是所有batch有效配对总数；字节式中的k是top-k个数，不是前面的key索引。returnValues指是否额外写回分数；指示函数成立时为1，否则0；2和4分别对应BF16与int32字节数。

通俗地说，Indexer先对候选key打分，再选最高的若干项。选取数量少并不意味着打分工作少；小query可能仍启动大量核处理长key和归并结果。

## 8. mHC：Pre、Sinkhorn和Post的组成

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

拆分链是Pre→Sinkhorn及外部Attention/MLP→Post；融合链以PreSinkhorn替换前两项，两条路线互斥，不能重复计数。外部Attention/MLP主体不包含在mHC wrapper工作里。单独Pre支持N=4/6/8、D≤16384且16对齐；融合PreSinkhorn当前仅N4、I20、D4096/7168、epsilon1e−6；独立Sinkhorn为FP32、N4/6/8、I1..100。融合投影涉及HF32，不能复用BF16 Matmul参数。v0.7分别测Pre、Sinkhorn、PreSinkhorn、Post和split链。split链保留Pre辅助输出并用恒等外部层连接hIn到hOut，测得的链不包含Attention/MLP；与无辅助输出的推理模式不是相同合同。时延代理见第11节。

参数解释：T是token数，N是残差流数量，D是每流隐藏宽度，M=N²+2N是投影输出宽度；t/n/d/i/j为索引。x是展平后的输入，gamma是可选逐元素权重，Phi是投影矩阵，z是未归一化投影，r_t是倒数RMS，w是归一化投影；alpha_0/1/2和b是三段缩放与bias，sigma是sigmoid，epsilon_n/h和epsilon是对应数值稳定常数。h_pre/h_post是混合权重，H′是未归一化残差混合矩阵，H是Sinkhorn结果，h_in/h_out是外部层输入/输出。I为Sinkhorn迭代数；N_exp/N_div/N_reduceAdd是标量exp/div/归约加法次数，不是硬件指令次数。

通俗地说，Pre为多路残差生成混合权重并合成外部层输入；Sinkhorn把残差矩阵反复按行/列归一化；Post再把外部层结果分配回各残差流。融合实现可能共享数据和减少同步，不能把融合PreSinkhorn与独立Pre/Sinkhorn同时相加。

## 9. KDA：状态递推与chunk前向

### 9.1 Recurrent

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

通俗地说，recurrent每一步都依赖上一步状态，所以短序列也有不能并行的链；chunk把块内运算改为矩阵计算，但跨chunk的状态仍需顺序传递。v0.7耗时分别按固定状态尺寸下的序列长度校准，不能从S≤8递推公式猜出Q9实现。

## 10. MoE：专家负载与GMM–SwiGLU–GMM组成

当前shape=(M,K,F,E)，M为本rank dispatch后行容量，F为SwiGLU后宽度，E是本rank专家数，**不是EP数**。必须输入 \(m_e\) 直方图或带明确语义的group_list，允许空专家，\(T=\sum_e m_e\le M\)。EP只约束全局到rank的映射，不能唯一确定本地直方图。

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

### 11.1 v0.7局部诊断对照公式（不作为通用模型）

原有quant/matmul/FA使用第3—5节；新增族的局部对照使用

\[
\widehat T_o=a_o+b_o W_o,\quad a_o\ge0,\ b_o\ge0.
\]

其中o为算子及固定属性路径；T_hat是预测净指令窗口（微秒），a是拟合启动/控制截距（微秒），b是每单位代理工作的有效成本（微秒/工作单位），W为下面的无量纲工作量。a/b只用该族校准样本做非负最小二乘，**不是已独立识别的硬件启动或ISA吞吐参数**。设校准样本的W完全相同，b不可辨识，固定b=0、a=校准耗时均值；它就是明确标记的常量回退。若不同W的拟合斜率为负，比较非负解边界后取最小平方误差解。固定耗时对照也只用校准均值，不看留出。

| 路径 | 当前W代理 | 解释与限制 |
|---|---|---|
| RMSNorm | ceil(R/64)×ceil(C/128) | 行波次数×128列工作片；64是AIV总数，128只是代理分组，不宣称真实VF宽度 |
| Addmm | ceil(ceil(M/16)ceil(N/64)/32)×ceil(K/16) | 输出tile波数×归约片数；16/64是本地代理分块，32是AIC总数，不作为预测MMAD次数 |
| Indexer | ceil(BQ/32)×ceil(L/512)×G×ceil(D/128) | query波数、key分片与head工作；不能从该代理推断实际只启动BQ个核 |
| mHC Pre | ceil(T/64)×ND(N²+2N)/128 | 每波投影工作；小T增加核数时W不增加 |
| mHC Sinkhorn | ceil(T/64)×I×N² | 每波矩阵归一化工作；并未恢复VF中多矩阵打包细节 |
| mHC PreSinkhorn | ceil(T/32)×[ND(N²+2N)/128+IN²] | 融合路径单独校准，不复用拆分链参数 |
| mHC Post | ceil(T/64)×N(N+1)×ceil(D/64) | 每波残差混合工作 |
| mHC split | ceil(T/64)×[ND(N²+2N)/128+IN²] | 直接对真实链的净时间校准；外部层恒等，Pre写辅助输出 |
| KDA recurrent / chunk | S | 固定B/H/K/V/C/flags的各自局部模型，分别拟合；不能跨两种算法共享系数 |
| GMM1融合 | ceil(J×ceil(2F/128)/32)×ceil(K/64) | J=sum_e ceil(m_e/16)，输出任务波数×归约片数；参数在真实MX路径拟合 |
| GMM2 | ceil(J×ceil(K/128)/32)×ceil(F/64) | 第二层归约长度是F，输出宽度是K，不能写反 |
| MoE FFN | T_hat_GMM1+T_hat_GMM2 | 对已同步串联链求净时间；排除router、dispatch/combine及TP通信 |

对mHC Sinkhorn/Post，若校准DB确认每token一个AIV且小T只改变活跃核，则局部对照另用 W=log2(1+min(T,64)) 描述启动/核间尾差；a、b仍仅是局部统计系数，不能解释成单核服务吞吐。该选择和校准参数在freeze前固定；负斜率依然退回非负边界，不靠加项拟合下降趋势。

表中ceil为向上取整；R/C、M/N/K、B/Q/L/G/D、T/N/D、S、m_e/E/F已在对应章节逐一定义。J是专家输出行tile的逻辑数量，单位为任务；表中32/64是当前平台核心数，其他分组常数是本地特征定义，并非已经证实的全部库tiling。在线模型不使用待预测shape实测核数或trace，避免把答案作为输入。实际entry/核数用于离线诊断；之后扩大规模需移植更准确调度或新增少量边界点。

通俗地说，先估算“最忙的一轮要做多少活”，再用少量测量确定启动和每单位工作成本。若两个小shape每核都只做一行，两点不能告诉我们多做一行要花多久，常量回退比强行画一条斜线诚实。该对照用于检验新增数据和识别缺失机制，不代表通用建模闭环已验收；核心模型必须采用第0节机制分层，并分别报告覆盖缺口。

支持域保存在model.json的routes.domain，属性合同保存在routes.attrs。域包围预先选定的校准和留出shape，不意味着包围盒内所有形状已测。可变维度越界或dtype/layout/辅助输出不匹配，预测为null；expert_counts必须完整且总和等于当前有效行数。代码会返回work_identifiable及特征范围外推标记。失败算子不会用其他算子的时间填补。

### 11.2 通用资源图引擎的当前实现与证据缺口

`operator_model/mechanism.py`独立于局部对照。每个宏任务包含资源类别、总工作量、并行任务数、串行步数和前驱；硬件profile按资源给核数、时钟、每周期服务量、结果延迟和发起间隔，不接受op/shape专属系数。当前聚合调度用

\[
q=\frac{W}{PSvf},\quad L=\max(\ell/f,q),\quad
J=\begin{cases}L,&\text{依赖前步状态},\\\max(i/f,q),&\text{可流水发起},\end{cases}
\]
\[
d=\lceil P/C\rceil[L+(S-1)J],\quad
s=\max(e_{pred},e_{resource}),\quad e=s+d.
\]

W为此宏任务跨所有并行任务/串行步骤的工作总量，P为独立任务数，S为每任务串行步数；v是每资源单元每周期的工作量，f是每微秒周期数（GHz×1000），ell和i分别为结果延迟/发起间隔的周期数，C是该资源可用单元数。q/L/J/d/s/e单位均为微秒，分别表示每步服务时间、首步时间、后续间隔、整个宏任务跨度、开始与结束时刻；e_pred为所有前驱最晚完成时刻，e_resource为共享资源可用时刻。相同资源池的阶段互斥，不同资源仅在依赖允许时并行。这个聚合式不是完整逐tile实现，任务负载不均匀时必须显式拆分（例如按专家），不能只用平均值掩盖尾部。

通俗地说，一段工作要同时满足“输入已准备好”和“设备空闲”才能开始；递推步骤还必须等上一步状态。只计算Cube/Vector忙时的最大值会漏掉这些等待。引擎支持把逻辑工作按有来源的模板展开为ISA计数；未展开的数学量不会冒充实际指令数。

当前`configs/hardware-950dt-v7.json`**没有填造出来的资源速率**。复用已有完整数据库得到opcode执行duration分布（`results/v0.7/shared-opcode-evidence.json`），但duration可能包含操作数长度、排队和依赖，不能直接倒数当吞吐或II。因此新扩展族的通用`predicted_us`仍会返回null并列出缺项；可插入具有独立证据的resource profile进行调度试验，合成测试参数不属于950DT实测。下一步需要按实际实现模板补指令/搬运展开，再用资源可分离的片段或小微基准识别L/II，沿用现有CA数据检验跨shape机制。这是尚未完成的校准工作，不能通过更多总耗时拟合掩盖。

### 11.3 数据、冻结和成本规则

所有耗时系数离线提取；在线不调用CANN、不读取目标shape trace、不构造token级事件。校准和验证按以下规则执行：

1. 先复用现有源码、配置、entry、核数和指令/流水证据；只在证据不足时添加可区分机制的小shape。UB分界、尾块、分核饱和、算法切换比密集笛卡尔积更有价值。
2. 原始trace按环境/属性/shape/调用器身份缓存。已有数据用于拟合后，角色转为开发，不能继续作为该版独立留出。相同trace的112个tile也不是112个独立shape。
3. 新模型、代码、环境身份、待测shape与预测先冻结并提交，再采集新留出；不看结果回调该版参数。失败/超时也记录成本，不自动重试，不把部分trace纳入误差。
4. compact完整指令DB优先用SQL归约，避免生成巨大的重复JSON。减少产物不保证减少CA执行计算量，所以同时限制每case时间、总墙钟、CPU和磁盘，而非只延长超时。
5. 完整输出与CPU golden做数值检查后，才接纳周期；同时比对实际entry、核数和选定指令。性能模型只能解释已正确执行的路线。

低成本查询实现使用一次加载的冻结模型、不可变输出、最多4096项LRU。长FA、通用FA、matmul为有界算术/有限tiling搜索；quant完整诊断为O(UB tile数)，单UB未缓存标量路径采用上面的等价求和，多UB标量路径仍为O(tile数)时间、O(1)临时空间；Indexer/KDA为O(1)，MoE为O(E)。原quant/matmul/FA只需要时延的调用可从 `operator_model.fast_query` 导入 `Predictor` 并使用 `predict_us(case,use_cache=False)`，避免构造完整诊断对象；新增族通用资源图请使用 `operator_model.mechanism.predict_case`，局部对照才使用 `operator_model.model_v7.Predictor`。缓存命中、未缓存完整预测、标量预测、模型加载/Python启动应分别计时，不能混成一个“仿真开销”。

作为已记录的成本参照，v0.5两点长FA共15.71分钟CA墙钟、8.04CPU小时、约469.48MiB保留数据；额外RMSNorm/Addmm两点调用共79.54秒。模型已加载后的长FA完整预测中位开销约28.81µs，缓存约1.59µs，取决于评估主机和负载；不是整网系统仿真吞吐。后续迭代应报告当前结果并保留同口径旧版对照。

误差统一使用

\[
APE_i=|\widehat T_i-T_i|/T_i,\quad
WAPE=\sum_i|\widehat T_i-T_i|/\sum_iT_i.
\]

其中i为成功实测shape索引，T_i是CA参考微秒值，T_hat_i是冻结预测微秒值；APE/WAPE均为无量纲比率，报告时乘100显示百分比。通俗地说，APE看单例相差多少，WAPE看同族总绝对偏差占总参考时间的比例，较慢样本权重更大。

每个算子分别报告开发/历史验证/新独立验证数量、WAPE、最大误差、覆盖shape和缺口；未校准族写N/A而不是0%。不同算子/机制混合后的低WAPE可能被大耗时FA主导，不能代替逐算子表。固定耗时均值保持冻结，用同一组成功新留出比较；小shape固定控制开销主导时它可能优于复杂模型，长单核循环增长时则无法缩放。

## 12. 场景映射与跨代边界

用户场景包括prefill单卡batch=1/4/16，decode=1/4/16/32/128/256；输入32k/128k/256k/512k、输出1k；MTP草拟0..16、主模型校验1..17，常用草拟8对应Q9。prefill的R/T/Q由当次chunk和实际token打包决定，不能把总KV长度直接当作一次norm或projection的行数。decode矩阵行数通常与batch×本次query相关，但实际flatten、padding、DP/TP及实现分支仍须核实。

模型config足以给候选hidden/head/expert尺寸，无需先下载权重；但不足以单独确定每卡shape和dispatch。attention TP需确认head切分/复制，attention DP决定请求归属；MoE EP、TP+EP和纯TP需结合本rank专家数及token路由直方图。重复计量化、融合组件或通信都会改变系统估时。

A7/A8目前只保留架构假设，不能直接把A5总时延乘频率比或CV比。跨代需要重新求tile/分核和尾波，分别修改资源服务/发起约束、缓冲容量和同步，然后重算依赖关键路径。A5中有效II已经包含反压；仅凭Cube/Vector忙碌时间不能证明其他代际保持同一瓶颈。本文暂不输出跨代已验证利用率。

## 13. 实现与证据入口

当前通用资源图入口为 `operator_model/mechanism.py`；`operator_model/model_v7.py`嵌入冻结v0.6模型并为新增族保留局部对照；原核心入口为 `operator_model/model_v6.py`（quant受限VF残差及单UB标量化简），继承 `model_v3.py`（quant UB流水）、`model_v4.py`（小shape回退）、`model_v2.py`（基础分块与校准）、`model_v5.py`（长FA）；基础分核/调度在 `model.py`。扩展入口是 `operator_model/extensions/`。参数取自冻结版本文件，本文数值仅便于审阅，完整精度以JSON为准。

- [当前v0.7参数](../results/v0.7/model.json)、[冻结预测](../results/v0.7/frozen-predictions.json)、[全族评估](../results/v0.7/evaluation.json)。
- [保留的v0.6参数](../results/v0.6/model.json)、[quant残差trace归约](../results/v0.6/trace-analysis.json)、[开发诊断](../results/v0.6/calibration-diagnostics.json)。
- [保留的v0.5参数](../results/v0.5/model.json)、[32k原始归约证据](../results/v0.5/trace-analysis.json)、[实际库身份](../results/v0.5/evidence.json)。
- [RMSNorm/Addmm详细证据](extensions/norm-addmm.md)、[最小调用记录](extensions/smoke-950dt.json)。
- [Indexer/mHC接口与来源](extensions/indexer-mhc.md)、[KDA/MoE接口与来源](extensions/kda-moe.md)。
- [采集方法复现](fa-compact-capture.md)、[真实工作负载候选映射](../results/v0.4/workload-coverage.json)。

本总文档随当前实现更新；每轮的模型变化和同集误差变化另写短记录，历史模型和验收报告不覆盖。
