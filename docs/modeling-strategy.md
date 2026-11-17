# 当前整套算子建模方法

本文是项目持续维护的**完整方法说明**，覆盖所有已接入的算子族；不是某一轮的修改记录。逐算子准确率、每个 shape 的实际库调用和仿真成本由配套总误差报告维护，版本目录保存冻结参数和历史验收结果。本文说明当前实际执行的公式，不能将“计划建立”的时延模型误作已经可用的能力。

当前 v0.6 耗时预测由三部分组成：dynamicMXquant 的分核/UB 双缓冲及受限VF调度残差模型、matmul 的分块流水模型、普通 FA 的通用路径与 Q9 长循环专用路径。RMSNorm、Addmm、LightningIndexer、mHC、KDA 和 MoE 目前提供结构工作量、依赖和约束，**没有通用耗时校准时返回 `predicted_us=null`**。一个小调用验证成功不等于已完成该族的时延建模。

## 1. 目标、输入和实际库算子

目标平台固定为 **CANN 9.2.1 / Ascend950DT_9582**：平台配置为 32 AIC、64 AIV、1.65 GHz，单核 UB 253952 B。这里的时延预测以 CA Model 为参考，不是实际板卡实测；A7/A8 尚未校准。

| 算子族 | 模型输入 | 实际/候选库接口 | 当前能力 |
|---|---|---|---|
| dynamicMXquant | `(R,C)`，BF16→E4M3FN，block32 | ops-nn / `aclnnDynamicMxQuant` | 时延、分核、UB循环、选定指令计数 |
| matmul | `(M,N,K)`，BF16 ND，不转置 | ops-nn / `aclnnMatmul`，已测 MatMulV3 | 时延、启发式tile/分核、MMAD计数 |
| FA | `(Q,L,D,B,H)`，BF16 SBH，非因果，无mask/dropout | ops-transformer / `aclnnFlashAttentionScoreV2` | 时延、分核、流水；Q64路径MMAD计数 |
| RMSNorm | `(R,C)`、dtype、gamma_dtype、epsilon | ops-nn / `aclnnRmsNorm` | 结构模型；已完成一个最小CA调用验证 |
| Addmm | `(M,N,K)`、self_shape、alpha/beta | ops-nn / `aclnnAddmm` | 结构模型；已完成一个最小CA调用验证 |
| LightningIndexer | `(B,Q,L,G,D)`、sparse_count/mode | ops-transformer / `aclnnLightningIndexer` | 打分/TopK结构；未校准 |
| mHC | `(T,N,D)`；Sinkhorn为`(T,N)` | ops-transformer / `aclnnMhcPre`、`aclnnMhcSinkhorn`、`aclnnMhcPreSinkhorn`、`aclnnMhcPost` | 组件及融合/拆分链结构；未校准 |
| KDA | `(B,S,Hq,Hv,K,V)`、recurrent/chunk | ops-transformer / `aclnnRecurrentKda`、`aclnnChunkKdaFwd` | 递推/chunk工作与状态依赖；未校准 |
| MoE融合链 | `(M,K,F,E)`及本rank专家token直方图 | ops-transformer / `aclnnGroupedMatmulSwigluQuantV2`，后续`aclnnGroupedMatmulV5`候选 | GMM1→SwiGLU→量化及GMM2结构；串联ABI未验证 |

CA 调用器使用安装包的真实 ACLNN 接口。参考源码来自固定版本的 ops-nn/ops-transformer，用来解释候选分支；它们不被宣称与安装包逐字同源。库文件存在只是“候选实现”，只有实际调用的 entry、tiling、核数和指令证据才能确定某个 shape 的执行路径。compact 数据库没有导出 ELF 时，运行时二进制 SHA 保持空值，不用安装文件 SHA 冒充。

之前使用的“桥接”指同 shape 的**采集方式一致性校验**：比较 msopprof 全 trace 与 npusim 压缩 DB 的平台、entry、核数、指令、正确性和计时窗口。它不是一种算子，也不算新的独立 shape。

## 2. 统一时延、流水和利用率口径

目标是所有执行核在共同时间轴上的指令窗口：

\[
T_{CA}=\frac{\max_j tick^{end}_j-\min_j tick^{start}_j}{1650}\quad[\mu s].
\]

1650 是每微秒的指令时钟 tick 数。这个窗口包含内部等待、同步和空隙，不包含 host enqueue 和窗外设备调度；不能直接替代框架 API 总延迟。最大单核 span 不能替代共同时间轴上的跨核 makespan。若一次 API 含多个 kernels，必须保存完整调用链并对齐时间，不能只取第一个 kernel。

总体抽象是：**shape/属性 → 实现路径与分块 → 核内资源/依赖 → 最慢核和尾轮 → 固定控制开销**。对资源 p 的第 j 个任务，可写为：

\[
s_{p,j}=\max\{a_{p,j},s_{p,j-1}+II_p\},\qquad e_{p,j}=s_{p,j}+L_p.
\]

\(a_{p,j}\) 是输入和缓冲可用时刻，\(L_p\) 是本次任务结果延迟，\(II_p\) 是连续发起间隔。两者必须分开：硬件可能在前一个任务尚未完成时发起后一个。这个式子是共同建模语言；下面各族使用的具体简化式与已校准参数才是当前实现。

同时区分三种量：

- **核覆盖率**：\(AIC_{active}/32\)、\(AIV_{active}/64\)，表示实际用了多少核。
- **流水占用率**：去掉不应计入的等待/同步后，某流水指令时间区间的并集长度除以 \(T_{CA}\)。重叠指令 duration 不能直接累加；Vector family 包含load/store/control，不是纯算术利用率。
- **理论算力利用率**：应由独立 ISA 发射/吞吐模型得到 \(T_{ideal}/T_{CA}\)。目前独立峰值模型未校准，故此项为 `null`。不能用已经从 CA 拟合出来的阶段系数反过来证明硬件峰值效率。

L2及以内的影响当前隐含在阶段成本和有效II里，尚未被唯一分离。历史配置文件中的 `cache_enable=0` 不能单独证明本次全SoC backend采用了该开关；不据此声称所有采集都关闭了L2。

## 3. dynamicMXquant：二维分核、UB和双缓冲

### 3.1 范围与分核

当前路径固定 BF16 输入、E4M3FN 输出、E8M0尺度、尾轴block32、rint、scale_alg=0；C为64的倍数且≤65536。API接受更大的R不表示这些shape都已实测。

先选择列切片宽度：

\[
w=\begin{cases}256,&R\lceil C/512\rceil\le48,\\512,&\text{其他},\end{cases}\qquad
C_s=\lceil C/w\rceil,\quad P=\min(64,RC_s).
\]

在有限的行/列分核候选 \((p_r,p_c)\) 中，令 \(p_c=\max(1,\lfloor P/p_r\rfloor)\)，要求 \(p_r\le R,p_c\le C_s\)。每核覆盖

\[
r_c=\lceil R/p_r\rceil,\quad c_s=\lceil C_s/p_c\rceil,\quad
c_c=\min(C,c_sw),\quad
AIV=\lceil R/r_c\rceil\lceil C_s/c_s\rceil.
\]

候选集合来自≤64核的整除/商划分；采用源码参考的尾洞评分，优先减少未填满的行列尾块，再按列分核数、行分核数排序。完整评分在 `quant_partition()` 中固定，不能把 `min(64,R)` 当作分核。该规则已由部分实际核数校验，但不是对所有安装包分支的完整移植。

### 3.2 UB容量与动态指令工作量

将32元素作为一个量化组。当前容量公式为：

\[
G_{max}=8\left\lfloor\frac{253952-2048}{256\cdot6+8\cdot2\cdot5}\right\rfloor=1240.
\]

分子扣除保留空间；分母来自输入/输出双缓冲与scale/临时空间。设每核列组数 \(g_c=\lceil c_c/32\rceil\)，则

\[
n_c=\lceil g_c/G_{max}\rceil,\quad
 g_t=\operatorname{align}(\lceil g_c/n_c\rceil,2),
\]
\[
n_r=\left\lceil\frac{r_c}{\lfloor G_{max}/g_t\rfloor}\right\rceil,\quad
r_t=\lceil r_c/n_r\rceil,\qquad tile=(r_t,32g_t).
\]

在线逐UB tile保留真实尾行/尾列，而不是让最后一轮也使用满tile。对第i个tile的 \((r_i,c_i)\)，

\[
g_i=r_i\lceil c_i/32\rceil,\quad
m_i=\lceil g_i/8\rceil,\quad z_i=\lceil g_i/128\rceil,
\]
\[
x_i^{read}=2r_ic_i/1024,\quad x_i^{write}=(r_ic_i+g_i)/1024\quad[KiB].
\]

\(m_i,z_i\) 分别是主VF和尺度VF循环代理。选定 `RV_*` 动态指令的最大核计数通过离线非负拟合得到，当前接近

\[
\widehat N_{RV}=60n_{tile}+44\sum_i m_i+32\sum_i z_i.
\]

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

不存在的前驱项省略。两轮后复用输入/输出槽，分别受前一使用者消费完毕/写回完毕约束；这正是UB双缓冲带来的依赖。首读延迟只影响当前就绪，后续读还受发起间隔与反压限制。

基础模型对小shape的控制路径采用饱和回退（v0.6在下一节限定分支替换该回退）：

\[
T_{pipe}=W_{last}+D_{suffix},\quad
T_{ctrl}=a_c+b_c\min(\log_2(1+AIV),4),\quad
\widehat T_{q,base}=\max(T_{pipe},T_{ctrl}).
\]

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

每轮VFSpan为该轮VF窗口长度，UnionSIMD为**同一核**的Vector-family指令区间并集；不能用一个核的VF包络减另一个核的最大忙碌时间。这个差值表示包络内未由SIMD并集解释的调度残差，可能含等待、队列、控制和反压，不是新增算术服务或独立硬件峰值。

19条旧记录拟合同一个非负二参数模型：

\[
x_j=\begin{cases}AIV_j/64,&n_{tile,j}=1,\\0,&n_{tile,j}>1,\end{cases}
\qquad \hat g_j=a_g+b_gx_j,
\]
\[
(a_g,b_g)=\arg\min_{a,b\ge0}\sum_j(a+bx_j-g_j^{obs})^2
=(0.11987486887649926,\ 0.6213462046069332)\ [\mu s].
\]

多UB点仅以截距特征参与残差拟合，并没有被赋予同一个多UB修正公式。实际预测仅在 **`iterations==1 && cols_per_core>=1024`** 的受限分支使用新增项：

\[
\boxed{\widehat T_q=\begin{cases}
T_{pipe}+a_g+b_g AIV/64,&n_{tile}=1\land c_c\ge1024,\\
\max(T_{pipe},T_{ctrl}),&\text{其他}.
\end{cases}}
\]

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

输入 A[M,K]、B[K,N]、输出[M,N]，固定BF16 ND、无转置、cubeMathType=0，M/N/K各≤4096。数学工作 \(F=2MNK\)，逻辑字节 \(2(MK+KN+MN)\)；后者不含cache重用、补齐和重复读。

从输出tile \((t_m,t_n)=(16,64)\) 开始。若 \(\lceil M/t_m\rceil\lceil N/t_n\rceil>32\)，比较将M或N tile翻倍的候选（单边最多256），依次选择“tile数接近32”“补齐后输出面积小”“tile边长和小”的候选。令

\[
N_t=\lceil M/t_m\rceil\lceil N/t_n\rceil,\quad
AIC=\min(32,N_t),\quad W=\lceil N_t/AIC\rceil.
\]

W是最忙核的输出tile轮数。该映射是受源码和小规模实测约束的启发式，不是完整的CANN tiler。`gemv`仅为M=1的模型分组，不能据此声称实际改成Vector kernel。

令 \(e_m=\min(M,t_m),e_n=\min(N,t_n),e_k=align(K,16)\)，阶段特征为

\[
x_R=2(e_me_k+e_ne_k)/1024,\quad
x_C=2e_me_ne_k/131072,\quad x_F=2e_me_n/1024.
\]

每阶段服务成本 \(S_s=a_s+b_sx_s\) 由校准trace的区间并集拟合。\(x_C\) 的131072仅为特征缩放常数，不是峰值吞吐。

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

首轮填充按顺序相加，其后按最慢资源推进；总时延校正承担未分离的控制/同步和路径偏差。分支优先级为M=1→`gemv`，否则K%16≠0→`k_tail`，否则tile≠16×64→`coarse_tile`，其余`dense`。对应残差 \(\delta_b\) 分别0.024992、0.041869、−0.062131、−0.006530µs；残差用 \(\sum residual/(n_b+2)\) 收缩，避免一两个样本单独拟合完整模型。

每个最忙核的选定MMAD计数近似 \(W\lceil K/256\rceil\)，现计数系数≈1。K尾块和不同tile可能改变实际指令形状，需对比实际MMAD维度与数量。分支校准仅2/1/1/4点，未见组合仍为外推；不能将已校准BF16 matmul直接用于MX GMM、HF32 mHC或Addmm总调用。

## 5. 普通FA：通用路径和长循环路径

输入Q为[Q,B,H×D]，K/V为[L,B,H×D]；BF16、SBH、无mask/dropout、非因果，scale=1/√D。它是普通FA代理，不等于三个模型实际MLA、压缩注意力、稀疏注意力、KDA或decode专用FIA。输入guard允许Q≤2048、D∈{64,128}、B≤256、H≤32；这只是可计算范围，不是完整验证范围。

### 5.1 通用tiling与耗时

Q容量和KV容量由当前启发式选择：

\[
q_b=\begin{cases}64,&BH\lceil Q/128\rceil<32,\\128,&\text{其他};\end{cases}
\]
\[
k_b=\begin{cases}256,&(D=64\land Q\bmod128=0\land L\bmod128=0)\ \lor L>1024,\\128,&\text{其他}.\end{cases}
\]

实际特征tile为 \(t_q=16\)（Q=1时）或 \(\min(Q,q_b)\)，\(t_k=\min(align(L,16),k_b)\)。令

\[
N_q=BH\lceil Q/q_b\rceil,\ AIC=\min(32,N_q),\ AIV=2AIC,
\ W=\lceil N_q/32\rceil,\ I=\lceil L/k_b\rceil.
\]

这里沿Q/head分核，KV在每个任务内循环，没有额外假设split-KV。阶段特征：

\[
x_R=2(t_qD+2t_kD)/1024,\quad x_C=4t_qt_kD/131072,
\]
\[
x_V=t_qt_k/512,\quad x_F=4t_qt_k/1024,\quad x_W=2t_qD/1024.
\]

各服务 \(S_s=a_s+b_sx_s\)：

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

`d64_kv256`、`d64_kv128`、`d128_kv128`的残差分别为0.161074、−0.059147、−0.057041µs；未见分支不添加残差。校准数量分别2/6/1，模型保留“未见/少样本/包围盒外”的覆盖标签，标签不构成准确率保证。

Q容量为64的路径使用结构MMAD计数

\[
N_{MMAD,max}=2\lceil L/128\rceil W,
\]

两项分别是QK沿N与PV沿K分段。Q容量128的路线尚未验证该公式，计数返回空。部分trace缺少MMAD维度，不能由有维度的子集倒推完整执行FLOPs。

### 5.2 Q9/D64/B1/H1的长KV专用路径

仅当 **Q=9、D=64、B=H=1、L≥4096** 时采用此路径。活跃核为1AIC+2AIV，\(I=\lceil L/256\rceil\)，MMAD仍为 \(2\lceil L/128\rceil\)。参数来自旧32k完整DB，不消耗目标shape的trace。

32k中Q只读一次，K/V各128次32768B读取。按源地址恢复两个流，再以已知MMAD维度对应的PC区分QK与PV；不能把按时间混排的每4条MMAD都当作一次源码循环。K/V首轮读完成延迟与后续发起间隔不同，模型为

\[
\boxed{\widehat T_{long}=P+\max(L_K,\Delta_V+L_V)+F+(I-1)II+D_{drain}}.
\]

P为第一K读之前的前缀；\(L_K,L_V\) 为第一K/V读取本身的延迟；\(\Delta_V\) 为V比K晚开始的时间；F为首轮K/V均就绪到首轮PV结束的剩余路径；\(D_{drain}\) 为最后PV结束后的排空。两条读流可重叠，所以取完成时刻最大值。F包含尚未隐藏的软最大值、调度/同步等，不是纯Cube算术服务。

稳态从32k同一DB的中段i=8..119拟合：

\[
s_i\simeq a+Ji,\qquad
J=\frac{\sum(i-\bar i)(s_i-\bar s)}{\sum(i-\bar i)^2},
\]
\[
II=\max(J_K,J_V,S_C,S_V,S_{MTE1},S_{FIXP}).
\]

J是当前路径的有效发起间隔，包含buffer复用、同步和反压，**不是独立HBM带宽**。服务项由各资源指令区间并集/128轮得到，作为资源需求下限；两个AIV取最大。当前J主导公式，仍不能据此证明纯访存bound。

| 参数 | µs | 参数 | µs |
|---|---:|---|---:|
| P | 6.560000 | F | 1.778182 |
| \(L_K\) | 0.849697 | \(L_V\) | 0.745455 |
| \(\Delta_V\) | 1.066667 | \(D_{drain}\) | 0.347273 |
| \(J_K\) | 0.517109 | \(J_V\) | 0.517215 |
| \(S_C\) | 0.215933 | \(S_V\) | 0.220985 |
| \(S_{MTE1}\) | 0.281013 | \(S_{FIXP}\) | 0.107898 |

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

当 \(C_a\le16384,u\ge1\) 且不进入短列transpose候选时，tile行为 \(\min(u,f)\)，最忙核轮数 \(\lceil f/\min(u,f)\rceil\)。1024/2048分别为host与arch35保留空间，不能按常量名称误读。长归约Split/SplitAR和短列高行数transpose未移植，返回未解析dispatch。实际仅 `[1,128]` 的最小CA调用得到与NORMAL key一致的entry；不能据此确认所有输入。

### 6.2 Addmm

\[
Y=\beta\operatorname{broadcast}(S)+\alpha AB,\qquad F_{mm}=2MNK.
\]

另列MN合并加法和alpha/beta非1时各MN级缩放；这是稠密代数工作上界，广播前缩放或零系数消除可能减少工作。逻辑字节 \(b(MK+KN+|S|+MN)\)，不含中间tensor、copy/cast和多kernel开销。

self可为标量、[N]、[1,N]、[M,1]、[M,N]等合法广播。源码候选按空tensor/零系数、bias转换、inplace FP32 GemmV3、DAV_3510 FusedMatMul(add)、一般Matmul+Muls/Add/Axpy分支处理。FusedMatMul(add)要求完整矩阵self、alpha/beta≈1和其他dtype/layout条件；[N] bias不满足此特定分支。FP32融合候选还要求cubeMathType=3。

Addmm不能直接继承Matmul的总耗时或分核。已测 `[2,16,16]`、self `[16]` 仅观察到一个kernel，其他属性可能产生多kernel。两族均尚无可用于泛化的时延参数，因此本节不输出APE或利用率。

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

mode3为右下对齐因果，\(P=Bv\)。工作量分别为点积 \(2PGD\)、ReLU PG次、weight乘法PG次、head归约 \(P(G-1)\) 次。实际padding/mask可能执行更多，另保留稠密 \(2BQLGD\) 作为参照。TopK比较、分块选取与合并算法未知，当前没有指令/时延公式；不能用topk/L减少前面必须完成的打分工作。

输入输出唯一字节下界为

\[
2(BQGD+BLD+BQG)+BQk(4+2\mathbf1_{returnValues}).
\]

不包括score中间量、重读和workspace。结构计算为O(1)，不会分配BQGL规模张量。实际tiling/分核未知。比如模型config提供32个index heads，直接TP8切成G4会不满足此普通接口，必须确认运行时是否复制heads或选择其他量化接口，而非静默改成G8。

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

拆分链是Pre→Sinkhorn及外部Attention/MLP→Post；融合链以PreSinkhorn替换前两项，两条路线互斥，不能重复计数。外部Attention/MLP主体不包含在mHC wrapper工作里。单独Pre支持N=4/6/8、D≤16384且16对齐；融合PreSinkhorn当前仅N4、I20、D4096/7168、epsilon1e−6；独立Sinkhorn为FP32、N4/6/8、I1..100。融合投影涉及HF32，不能复用BF16 Matmul参数。当前分核、tile、服务时间均未校准。

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

整体乘BHv。三角求逆、gate、padding/转换尚未纳入，`total_executed_flops=null`。FwdH每条state chain沿chunk串行，其他阶段可按chunk/head并行；这五段在同一kernel内，不能加五次launch，也不能无条件取Cube/Vector总时长的max。

当前chunk支持BF16/FP16、K/V为16倍数且在[16,256]、Hv≤128、FP32 state。公开Aqk/Akk输出 `[B,Hv,S,C]` 必选；关掉其他中间输出不意味着可以省掉它们。packed/varlen、投机state-slot与真实fusion flags未建模。KdaInputProj是另外的已识别库候选，包含输入MX quant，当前没有其完整结构/时延入口，不能再无条件重复加独立quant。

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

`aclnnGroupedMatmulSwigluQuantV2`仅包含GMM1→dequant→SwiGLU→MX quant；`groupedmatmulswiglugmm`/`moe_ffn`是组成链别名，不是已确认的同名单kernel。后级GroupedMatmulV5的scale ABI和串联尚未验证。router、输入量化（若未融合）、dispatch/combine通信及TP collective单独存在，当前不输出其耗时，不能视为零。AIC/AIV交接和组尾调度必须从真实tiling/trace建立，不假设一个专家一个核。

## 11. 校准、独立验证和仿真效率

所有耗时系数离线提取；在线不调用CANN、不读取目标shape trace、不构造token级事件。校准和验证按以下规则执行：

1. 先复用现有源码、配置、entry、核数和指令/流水证据；只在证据不足时添加可区分机制的小shape。UB分界、尾块、分核饱和、算法切换比密集笛卡尔积更有价值。
2. 原始trace按环境/属性/shape/调用器身份缓存。已有数据用于拟合后，角色转为开发，不能继续作为该版独立留出。相同trace的112个tile也不是112个独立shape。
3. 新模型、代码、环境身份、待测shape与预测先冻结并提交，再采集新留出；不看结果回调该版参数。失败/超时也记录成本，不自动重试，不把部分trace纳入误差。
4. compact完整指令DB优先用SQL归约，避免生成巨大的重复JSON。减少产物不保证减少CA执行计算量，所以同时限制每case时间、总墙钟、CPU和磁盘，而非只延长超时。
5. 完整输出与CPU golden做数值检查后，才接纳周期；同时比对实际entry、核数和选定指令。性能模型只能解释已正确执行的路线。

低成本查询实现使用一次加载的冻结模型、不可变输出、最多4096项LRU。长FA、通用FA、matmul为有界算术/有限tiling搜索；quant完整诊断为O(UB tile数)，单UB未缓存标量路径采用上面的等价求和，多UB标量路径仍为O(tile数)时间、O(1)临时空间；Indexer/KDA为O(1)，MoE为O(E)。只需要时延的调用可从 `operator_model.fast_query` 导入 `Predictor` 并使用 `predict_us(case,use_cache=False)`，避免构造完整诊断对象。缓存命中、未缓存完整预测、标量预测、模型加载/Python启动应分别计时，不能混成一个“仿真开销”。

作为已记录的成本参照，v0.5两点长FA共15.71分钟CA墙钟、8.04CPU小时、约469.48MiB保留数据；额外RMSNorm/Addmm两点调用共79.54秒。模型已加载后的长FA完整预测中位开销约28.81µs，缓存约1.59µs，取决于评估主机和负载；不是整网系统仿真吞吐。后续迭代应报告当前结果并保留同口径旧版对照。

误差统一使用

\[
APE_i=|\widehat T_i-T_i|/T_i,\quad
WAPE=\sum_i|\widehat T_i-T_i|/\sum_iT_i.
\]

每个算子分别报告开发/历史验证/新独立验证数量、WAPE、最大误差、覆盖shape和缺口；未校准族写N/A而不是0%。不同算子/机制混合后的低WAPE可能被大耗时FA主导，不能代替逐算子表。固定耗时均值保持冻结，用同一组成功新留出比较；小shape固定控制开销主导时它可能优于复杂模型，长单核循环增长时则无法缩放。

## 12. 场景映射与跨代边界

用户场景包括prefill单卡batch=1/4/16，decode=1/4/16/32/128/256；输入32k/128k/256k/512k、输出1k；MTP草拟0..16、主模型校验1..17，常用草拟8对应Q9。prefill的R/T/Q由当次chunk和实际token打包决定，不能把总KV长度直接当作一次norm或projection的行数。decode矩阵行数通常与batch×本次query相关，但实际flatten、padding、DP/TP及实现分支仍须核实。

模型config足以给候选hidden/head/expert尺寸，无需先下载权重；但不足以单独确定每卡shape和dispatch。attention TP需确认head切分/复制，attention DP决定请求归属；MoE EP、TP+EP和纯TP需结合本rank专家数及token路由直方图。重复计量化、融合组件或通信都会改变系统估时。

A7/A8目前只保留架构假设，不能直接把A5总时延乘频率比或CV比。跨代需要重新求tile/分核和尾波，分别修改资源服务/发起约束、缓冲容量和同步，然后重算依赖关键路径。A5中有效II已经包含反压；仅凭Cube/Vector忙碌时间不能证明其他代际保持同一瓶颈。本文暂不输出跨代已验证利用率。

## 13. 实现与证据入口

当前可执行入口为 `operator_model/model_v6.py`（quant受限VF残差及单UB标量化简），继承 `model_v3.py`（quant UB流水）、`model_v4.py`（小shape回退）、`model_v2.py`（基础分块与校准）、`model_v5.py`（长FA）；基础分核/调度在 `model.py`。扩展入口是 `operator_model/extensions/`。参数取自冻结版本文件，本文数值仅便于审阅，完整精度以JSON为准。

- [当前v0.6参数](../results/v0.6/model.json)、[quant残差trace归约](../results/v0.6/trace-analysis.json)、[开发诊断](../results/v0.6/calibration-diagnostics.json)。
- [保留的v0.5参数](../results/v0.5/model.json)、[32k原始归约证据](../results/v0.5/trace-analysis.json)、[实际库身份](../results/v0.5/evidence.json)。
- [RMSNorm/Addmm详细证据](extensions/norm-addmm.md)、[最小调用记录](extensions/smoke-950dt.json)。
- [Indexer/mHC接口与来源](extensions/indexer-mhc.md)、[KDA/MoE接口与来源](extensions/kda-moe.md)。
- [采集方法复现](fa-compact-capture.md)、[真实工作负载候选映射](../results/v0.4/workload-coverage.json)。

本总文档随当前实现更新；每轮的模型变化和同集误差变化另写短记录，历史模型和验收报告不覆盖。
