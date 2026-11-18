# 算子建模

系统仿真器输入每卡 shape、dtype、布局和算子属性，预测一次设备指令窗口的微秒数。v0.9 按 **Vector、Cube、Cube + Vector（CV）** 三类选择方法。算法给出工作量与依赖，库实现决定分块、分核和缓冲复用，硬件配置与校准给出阶段服务时间。在线调用只做分块计算和流水调度。单次模型查询墙钟预算为 **1 秒**，预测出来的设备时间和离线 CA 采集成本分别记录。

### Vector 算子怎样建模

1. 先找可以并行的维度。逐行归约可以把行分给多个 AIV，KDA recurrent 的同一条状态链则要依次处理 token。根据库的分核规则算出最忙核承担的行数、列片数和尾块，再用 UB 容量决定每轮读入多少数据。
2. 写出一轮的读、算、写。读写以实际 dtype 字节和对齐块计，向量计算以有效向量分组和归约轮次计。对已知双缓冲实现，让下一轮预读与本轮计算重叠，同时约束同一个缓冲槽在上一次使用完成后才能写入。源码无法还原完整指令依赖的路径，采用临界 AIV 的分段时间模板，按每核工作量缩放。
3. 校准时分别读取 MTE2、SIMD、MTE3 和控制流水的指令区间。流水可同时繁忙。对已知调度用开始和完成时刻求关键路径，对时间模板把每个重叠区间归属一次。两种实施都要检查最慢核、尾轮和首尾开销。

以 dynamicMXquant 的 BF16 输入 `[1024,7168]` 为例，分核规则给出 64 个 AIV，最忙核处理 16 行，UB 每轮处理 4 行，共 4 轮。每轮读取 \(4×7168×2=57344\) 字节，按每 32 个元素一组得到 896 个量化组。读第一块要等完整读延迟，后续块可按发起间隔推进，第三块还要等第一块释放输入槽。因此把单块时间直接乘 4 会丢掉双缓冲效果。RMSNorm 同属 Vector，单行数据要先完成平方和归约才能得到缩放因子，其依赖和动态量化不同。第 3 节给出量化的逐轮递推，第 6、8、9 及 11 节给出各 Vector 路由的可执行公式。

### Cube 算子怎样建模

1. 按安装包的矩阵布局、dtype 和分支选择输出 tile，再把每个 tile 的 K 方向分成归约段。输出块之间可以并行，同一输出块的累加需要按归约段推进。M/N 尾块、K 补齐、专家之间不能拼接的边界都改变任务数。
2. 将输出块映射到 AIC，计算每核最多几轮。对代表块估算 A/B 读入量、矩阵乘加工作和输出字节，分别求读、Cube 和写出阶段服务时间。校准时检查 MMAD 指令数、块数、活跃 AIC 和阶段区间。
3. 对规则整块流水可用 \(T=H+R+C+F+(w-1)\max(R,C,F)\)。这里 \(R,C,F\) 分别为一轮读取、矩阵计算和写出的微秒数，\(H\) 为未被阶段覆盖的控制时间，\(w\) 为最忙核的轮数。首轮先填满流水，后续轮次受最慢阶段限制。Matmul 的校正见第 4 节。当第一波内部已经存在明显重叠时，Addmm 用最长队列时长加上次要队列暴露的时间，具体公式见第 6.2 节。

例如 BF16 Addmm 的 `[M,N,K]=[9,128,256]` 配合 `self=[128]` 与单位系数。按 16×64 候选输出块，列方向有两块，两核一轮可以完成。每块逻辑上需要 9×256 个 A 元素、256×64 个 B 元素和 64 个 bias 元素，算出 9×64 个输出。两块共用同一片 A，实际搬运量还取决于缓存复用。K 增大主要延长归约，M 或 N 增大可能增加并行块或尾波。同样的总 FLOPs 可以得到不同耗时。专家 GMM2 还要对每个专家单独向上取整，再合并专家块调度。

### CV 融合算子怎样建模

1. 从实现列出 Cube、Vector、搬运和同步的生产者、消费者。例如 FA 的一块 K/V 先做 QK，交给 Vector 做 softmax 更新，随后 PV 再使用 Cube。核内缓冲深度决定下一块最早何时可以开始。
2. 在每个阶段使用 \(s=\max(e_{dep},a_{res},a_{buf})\)、\(e=s+t\)。\(s,e,t\) 为阶段开始、结束、服务时间，单位都是微秒。\(e_{dep}\) 是最晚前驱完成时刻，\(a_{res}\) 是设备可再次接受工作的时刻，\(a_{buf}\) 是缓冲槽可用时刻。长循环压缩为首轮填充、稳态间隔和尾部排空。运行时难以完整恢复调度的融合路径，保留开发 trace 中各流水同时活跃的区间模板，按目标 tile 的 Cube、Vector、搬运工作分别缩放，在每段重叠区间取最长服务。
3. 用独立 shape 检查瓶颈是否真的随工作量移动，包括长 KV、尾块、专家偏斜及不同核数。多 kernel 串行链分别估每段，再求净窗口之和。外部通信、host launch 和两次 launch 之间的空闲需在系统层计入。

例如普通 FA 的 `Q=9,L=32768,D=64,B=H=1`。该路径预测 1 个 AIC、2 个 AIV，每个 KV tile 256 token，共 128 轮。已校准的首轮及首尾固定部分合计 10.498 µs，每增加一轮约 0.5172 µs，故预测约 \(10.498+127×0.5172=76.18\) µs。这个例子中延迟和发起间隔分别由 32k 的开发 trace 识别，其他布局和因果属性需要各自的分支。MoE 的 GMM1–SwiGLU–量化同样采用 CV 方法，利用本 rank 专家 token 直方图计算各资源的块数。

硬件改变时重新计算可用核数、分块、带宽、各阶段服务和重叠条件。950DT 实测得到的有效服务成本包含当时的资源竞争。当前独立验证只针对 950DT，A7/A8 的参数与库策略仍待校准。

## 1. 目标、输入和实际库算子

目标平台固定为 **CANN 9.2.1 / Ascend950DT_9582**：平台配置为 32 AIC、64 AIV、1.65 GHz，单核 UB 253952 B。这里的时延预测以 CA Model 为参考，不是实际板卡实测；A7/A8 尚未校准。

| 算子族 | 模型输入 | 实际/候选库接口 | 当前能力 |
|---|---|---|---|
| dynamicMXquant | `(R,C)`，BF16→E4M3FN，block32 | ops-nn / `aclnnDynamicMxQuant` | 时延、分核、UB循环、选定指令计数 |
| matmul | `(M,N,K)`，BF16 ND，不转置 | ops-nn / `aclnnMatmul`，已测 MatMulV3 | 时延、启发式tile/分核、MMAD计数 |
| FA | `(Q,L,D,B,H)`，BF16 SBH，非因果，无mask/dropout | ops-transformer / `aclnnFlashAttentionScoreV2` | 时延、分核、流水；Q64路径MMAD计数 |
| RMSNorm | `(R,C)`、dtype、gamma_dtype、epsilon | ops-nn / `aclnnRmsNorm` | 实测校准阶段成本、独立 shape 验证 |
| Addmm | `(M,N,K)`、self_shape、alpha/beta | ops-nn / `aclnnAddmm` | 实测校准阶段成本、独立 shape 验证 |
| LightningIndexer | `(B,Q,L,G,D)`、sparse_count/mode | ops-transformer / `aclnnLightningIndexer` | 打分/TopK 工作量与流水重叠模板 |
| mHC | `(T,N,D)`；Sinkhorn为`(T,N)` | ops-transformer / `aclnnMhcPre`、`aclnnMhcSinkhorn`、`aclnnMhcPreSinkhorn`、`aclnnMhcPost` | 各组件与拆分链分别采集、受限校准 |
| KDA | `(B,S,Hq,Hv,K,V)`、recurrent/chunk | ops-transformer / `aclnnRecurrentKda`、`aclnnChunkKdaFwd` | recurrent 预测与独立验证，chunk 的 CA 同步资格待解决 |
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

v0.6 先冻结公式/参数，再新增 `[36,5120]`、`[288,7168]`、`[1024,7168]` 三个未见shape验证；总CA预算450秒，分别检查宽行片段、decode候选聚合行数和多UB不变路径。成功/失败、正确性、实际dispatch、误差和成本以配套误差报告为准，不把冻结预测写成实测结果。

### 3.5 标量预测开销

对单UB点，递推严格退化为

\[
T_{pipe}=P_q+L_{read}(x_0)+S_{vec}(m_0+z_0)+D_{write}+D_{suffix}.
\]

统一查询入口用这条等价式计算单 UB 点，并按第 3.3 和 3.4 节的规则选择回退或残差。多 UB 点只保存推进所需的前两轮状态。对相同 tile 的长循环，v0.8 用第 11 节的矩阵合成把遍历次数从 O(n) 降到 O(log n)。这里 n 指最忙核上重复的 UB tile 数；阶段服务和资源约束均保留。

## 4. matmul：输出tile、核轮次和阶段成本

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

**算例 · RMSNorm。** 对 [65,128] 的每一行，先求 128 个数的平方均值、加 ε，再开方取倒数，再用它和 128 个 gamma 权重缩放该行。若每核最多负责两行，32 核各处理两行，最后一核处理一行。这个 shape 的 trace 实际启动 33 个 AIV，预测 3.12 µs，CA 为 3.08 µs。

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

v0.8 的分块仍为启发式，输出尾部和非对齐 K、不同库入口可能产生新偏差。知识卡 `l1_l0_storage_structure_and_data_paths` 支持输入、Cube 与 L0C 搬出分阶段建模；具体重叠比例和服务成本来自该安装包数据库。所有旧实测点转为开发回放，新的 10 个独立 shape 在相同冻结合同下同时比较 v0.8 与 v0.9。每个合格验证点必须通过安装包 aclnnAddmm 的完整数值检查与可解析的单 kernel CA 数据库。

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
| mHC Post | [32,4,512]、带残差矩阵。对每个 token 的每一输出通道，四路残差乘各自权重求和，再加入外部层贡献。仅残差矩阵的混合就要 32×4²×512=262144 次乘法。预测 2.49 µs。 | 2.39 µs |
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

拆分链是Pre→Sinkhorn及外部Attention/MLP→Post；融合链以PreSinkhorn替换前两项，两条路线互斥，不能重复计数。外部Attention/MLP主体不包含在mHC wrapper工作里。单独Pre支持N=4/6/8、D≤16384且16对齐；融合 PreSinkhorn 的源码候选为 N4、I20、D4096/7168、epsilon1e−6，本版时延入口仅校准 D4096；独立Sinkhorn为FP32、N4/6/8、I1..100。融合投影涉及HF32，不能复用BF16 Matmul参数。v0.7分别测Pre、Sinkhorn、PreSinkhorn、Post和split链。split链保留Pre辅助输出并用恒等外部层连接hIn到hOut，测得的链不包含Attention/MLP；与无辅助输出的推理模式不是相同合同。时延估计见第11节。

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

通俗地说，recurrent每一步都依赖上一步状态，所以短序列也有不能并行的链；chunk把块内运算改为矩阵计算，但跨chunk的状态仍需顺序传递。v0.8 recurrent 把状态大小和序列长度代入各阶段，chunk 保留分块工作图与 CA 同步问题记录，不能从S≤8递推公式猜出Q9实现。

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

### RMSNorm、独立 Sinkhorn 和 Post 的临界向量核

选最后结束的 AIV，用 SIMD、MTE3、MTE2、SCALAR 的优先级把每段非等待指令占用时间唯一归属。other 表示余下的核内时间，arrival 是临界核相对最早发起核的起始偏移。它们相加正好是跨核指令窗口。每条路线的 13 个旧 shape 均转入开发。用每段实测时间拟合非负的固定成本与单位服务成本，估计式为  $\widehat{t}=\sum_p(a_p+b_p x_p)$。这里的时延估计以 µs 计，p 是上面六段之一，a 的单位为 µs，b 的单位为 µs/相应工作单位。每段使用 soft-L1 损失（尺度 0.15 µs），降低到达时间抖动对结果的影响。若该工作单位在开发集中完全不变，采用该阶段的中位观测并令斜率为零。全部成本见[逐阶段校准参数](../results/v0.9/vector/profile.json)。共享写流量的斜率包含竞争、重复指令和实际重叠，仅在这批数据覆盖的库路线上可解释。

设 T 为行/token 数，w=ceil(T/64) 为每个 token 分组容纳的行数，A=ceil(T/w) 为有工作行组数。arrival 的工作量取 log₂(活跃核数)，SCALAR 用 w−1，other 为固定项。MTE2 用每个活跃核的一份局部数据 KiB，MTE3 用全芯片主要输出 KiB，作为核间竞争代理。SIMD 的工作量随算子改变。

- **RMSNorm** 输入 BF16 矩阵 [T,C]，每行归约 C 列。每核计算单位为 w·ceil(C/128)，读工作为 2wC/1024，写工作为 2TC/1024；常量 2 是 BF16 字节数，128 为开发 trace 所见列分组。gamma、rstd 等固定读写计入有效成本，未来需按 dtype/输出合同区分。例如 [17,7168] 分给 17 核，每核处理一行、56 个计算分组，读 14 KiB，全芯片主输出 238 KiB；预测 4.416 µs。
- **mHC Sinkhorn** 的 N=4，每个 token 对 16 个 FP32 元素做 I 轮行列归一化。目标实现把多个小矩阵填入一个向量组，计算单位为 I·ceil(16w/64)，64 为一组 FP32 元素的假设宽度，现有样本只辨识到小行组范围。每核读 64w/1024 KiB，全芯片写 64T/1024 KiB。曾用 w·I，造成跨 64 行后把迭代成本翻倍。旧 DB 的 T=9 和 128 在 I=20 时，最忙核均有 1903 条 RV 事件。例如 [11,4]、20 轮，11 核各一组，计算段 3.655 µs，合计 5.932 µs。
- **mHC Post** 的 BF16 输入 [T,N,D] 固定 N=4。T≥32 选按 token 分配的入口候选，每核 w 行。较小 T 选择按列切片候选，初始 J=1，若 ceil(D/J)>256 且 TJ≤32 就把 J 翻倍。总活跃核为 A·J，每片宽 d=ceil(D/J)。计算单位为 w·ceil(d/64)·N(N+1)，读工作取 2wdN/1024，写工作取 2TDN/1024。两条入口分别校准。T=32 的分界来自相邻旧样本的入口与核数证据，属于待进一步核对的分支假设。例如 [6,4,512] 将列切成两片，12 核各处理 256 列，80 个计算单位，预测 3.868 µs。

RMSNorm 的两个输出、Post 输出和 Sinkhorn 输出都在真实 950DT 安装包的 ACLNN 接口同步后逐元素比对。每族 10 个独立 shape 在一批次中依次调用，首例及后续例的缓存状态可能不同。单次输入的启动与缓存状态无法从 shape 唯一推出。以上估计仅适用于已列明的类型、形状和推理属性。

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

KDA chunk另测具有不同源码执行条件的两条路径。普通整块样例为 `[1,64,1,1,128,128]`，在预激活gate下复现CA重复set_flag错误。随后用独立调用器验证 `[1,64,2,2,128,128]`。BF16、FP32 raw gate、A_log、safeGate及useGateInKernel为真、偶数value heads、S按64对齐且关闭qg/v_new/h导出，对应固定参考源码的内联Gate和Post-WU/FwdH融合候选。CPU按每个head单独计算 `gamma=lowerBound/(1+exp(-exp(A_log)*g))`，并以gamma驱动串行状态递推和chunk内三角矩阵求逆，检查全部请求的O、最终状态、gk、Aqk、Akk与输入状态完整性。此次CA在117.8秒发出三个重复set_flag错误，主kernel没有完整时间线，因此缺少该实现的可信时延。两个合同各预设10形状，先行样例的失败保留，其余未运行。冻结样例及接口参数见`safe-chunk-freeze.json`。实际已编译分支的源码映射仍待tid/tiling/PC证据核实。

KDA chunk的可执行结构入口为`operator_model.cube_v8_chunk.predict('kda_chunk',shape,attrs)`。设单个序列切成长度为c_i的块，块上限C，块数n=ceil(S/C)，并行状态链数U=B×Hv。对单条链，有效因果元素对数P=Σc_i(c_i+1)/2，块矩阵稠密大小D=Σc_i²。QK与严格下三角KK工作分别为2KP与2K(P-S)，Post-WU为2(K+V)D，FwdH为4SKV次FLOPs和nKV次状态衰减乘法，Finalize为2SKV+2VD；全算子乘U。K/V是每头key/value宽度。FwdH在同一链内按块顺序推进，QK/KK/Finalize可按不同块和头分核。该入口返回各阶段工作、依赖、分块、适用属性与尚缺的三角求解/资源成本。上述两次v0.8实测均遇同步错误，故可执行结构模型的时延字段为null。

### Chunk KDA 的十组独立结构核对

从两份已冻结清单选取十个不同 shape，B=1，S 从 1 到 256，Hq=Hv 取 1、2、4、8，K=V 取 16 或 128，C=64。CPU 核查器枚举每个头、块、有效 token 和有序 token 对，检查六个阶段的逻辑工作量、工作为零的阶段、DAG 顺序、任务数、串行步数和状态字节。S=1 时严格下三角为空，KK 工作为零，该节点在 DAG 中省略。用非零初始状态和每头一个活跃 K/V 通道，进一步对照块内三角求解和逐 token 递推的输出、各块末状态；已激活 gate 和 raw-safe gate 均覆盖。十组均通过，FP64 运算的最大差值为 9.72×10⁻¹⁷。该结果限于逻辑计数与稀疏数学恒等式。安装包 CA 耗时仍为 0/10，BF16 全输出、内部融合与物理吞吐待验证。逐项结果见 [chunk-structure-validation.json](../results/v0.8/cube/chunk-structure-validation.json)，可运行 `python3 scripts/cube-v8-chunk-structure.py` 复现。

### 融合算子的流水区间模型

1. **还原工作。** 按代码中的tile与归约方向，把每次shape换成各队列的工作单位。Indexer对长KV按128个key为一段执行点积，按512个key为一段估计TopK归并；mHC按并行token波次计算投影、读写与归约；MoE从本rank的专家token直方图分别划分各专家的输出矩阵，只对非空专家分配任务。输出列和归约维交换时改变GMM2的工作单位。上述分块是950DT的实现代理，执行分支和真实库tiling仍需按shape核对。
2. **从开发trace提取流水重叠。** 每个kernel、每种核类型取最晚完成的核，按指令开始/结束tick把时间轴切成若干段。每段记录同时工作的流水集合，例如SCALAR+MTE2和SIMD+MTE3。等待、启动和首核错峰归在空闲段，计入真实关键路径。每种集合只累计一次时长，解决同一时间读、算、写重复相加的问题。所有v0.7有效shape在本版转入开发集。
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

### 数据、冻结和成本规则

所有耗时系数离线提取；在线不调用CANN、不读取目标shape trace、不构造token级事件。校准和验证按以下规则执行：

1. 先复用现有源码、配置、entry、核数和指令/流水证据；只在证据不足时添加可区分机制的小shape。UB分界、尾块、分核饱和、算法切换比密集笛卡尔积更有价值。
2. 原始trace按环境/属性/shape/调用器身份缓存。已有数据用于拟合后，角色转为开发，不能继续作为该版独立留出。相同trace的112个tile也不是112个独立shape。
3. 新模型、代码、环境身份、待测shape与预测先生成带时间与哈希的冻结记录，再采集新留出；不看结果回调该版参数。失败/超时也记录成本，合格重试保持原模型及预测，部分 trace 另计。
4. compact完整指令DB优先用SQL归约，避免生成巨大的重复JSON。减少产物不保证减少CA执行计算量，所以同时限制每case时间、总墙钟、CPU和磁盘，而非只延长超时。
5. 完整输出与CPU golden做数值检查后，才接纳周期；同时比对实际entry、核数和选定指令。性能模型只能解释已正确执行的路线。

默认入口在进程内复用参数，每次从请求的 shape 和属性重新计算时间。quant 对少于 256 个 UB tile 的请求使用原递推，对更长的循环保存五个时刻（当前读、算、写完成，以及上一个 tile 的算、写完成）。每个依赖是“某个前驱的完成时间加阶段成本”，多个前驱取最大值，因此相同 tile 的更新可用 max-plus 矩阵的重复平方合并。完整行组与尾行/尾列分别处理，参数与逐 tile 算法相同。重复次数为 n 时只需 O(log n) 次固定大小的矩阵合成。FA 长 KV 采用首轮加稳态间隔，MoE 只遍历专家直方图。1 秒验收以统一入口单次墙钟计算，默认报告每个验证 shape 的多次查询最大值。

作为已记录的成本参照，v0.5两点长FA共15.71分钟CA墙钟、8.04CPU小时、约469.48MiB保留数据；额外RMSNorm/Addmm两点调用共79.54秒。模型已加载后的长FA完整预测中位开销约28.81µs，缓存约1.59µs，取决于评估主机和负载；不是整网系统仿真吞吐。后续迭代应报告当前结果并保留同口径旧版对照。

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

A7/A8目前只保留架构假设，不能直接把A5总时延乘频率比或CV比。跨代需要重新求tile/分核和尾波，分别修改资源服务/发起约束、缓冲容量和同步，然后重算依赖关键路径。A5中有效II已经包含反压；仅凭Cube/Vector忙碌时间不能证明其他代际保持同一瓶颈。本文暂不输出跨代已验证利用率。

## 13. 实现与证据入口

统一入口为 `operator_model.predictor.predict(op, shape, attrs)`。默认入口返回当前版本的设备时间估计。Vector 阶段参数、Cube 控制开销、CV 时间模板与各自的校准和冻结记录见 [v0.9 改动路径](../results/v0.9) 与 [沿用的 v0.8 路径](../results/v0.8)。动态量化、Matmul 与普通 FA 沿用已验证分块与流水参数。完整的逐算子样例、误差与测得的单次最大预测墙钟见[当前验证报告](error-analysis.md)。
