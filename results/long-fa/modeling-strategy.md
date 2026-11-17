# 当前建模策略：v0.4 冻结模型与长 FA 采集迭代

本轮优化 CA 验证与采集，**预测公式、参数和内核实现均不变**。当前模型是 [v0.4/model.json](../v0.4/model.json)，SHA256 `1b7a116c234d07fed008634e7874b41f73e930426a42833944769367f03c02d0`。本轮长 shape 的测量不参与重新拟合。配套的逐点误差、实际库内核、失败记录和图表见 [error-analysis.md](error-analysis.md)。

## 1. 目标量、平台与校准

平台固定为 CANN 9.2.1 / `Ascend950DT_9582`，32 AIC、64 AIV，指令数据库报告频率 1.65 GHz。预测目标是核间共同时间轴上，从第一条指令开始到最后一条指令结束的窗口，单位 µs：

\[
T_{CA}=\max_c t^{end}_c-\min_c t^{start}_c,
\qquad T_{DB}=\frac{\max_j tick^{end}_j-\min_j tick^{start}_j}{1650}.
\]

这不是主机 ACLNN 调用墙钟，也不是 CPU 模拟速度。不同核上的时间不能直接相加；单核最长 span 也不一定等于核间 makespan。流水忙碌率、使用核数比例和独立 ISA 峰值利用率分别定义；目前未校准最后一项，不输出伪精确的算力利用率。

成本函数统一记为 \(L_s(x)=a_s+b_sx\)，参数通过旧 CA trace 非负拟合。阶段服务和整体修正均是当前 CA 环境下的代理，不是跨代硬件常数。v0.4 共复用38个历史输入，quant 成本来自13个历史 quant trace；matmul/FA 时延沿用 v0.2 冻结拟合，FA 指令公式采用 v0.3 结构修正。

## 2. dynamicMXquant：分核、UB 双缓冲和小 shape 回退

输入 \((R,C)\)：BF16 → E4M3FN，尾轴 block32，rounding=`rint`；输出 scale 为 E8M0。实际调用 `aclnnDynamicMxQuant`，不是自行实现的简化 kernel。

**分核。** 列分片宽度

\[
s=\begin{cases}256,&R\lceil C/512\rceil\le48\\512,&\text{其他}\end{cases},\quad
C_s=\lceil C/s\rceil,\quad A_0=\min(64,RC_s).
\]

枚举行列划分 \((p_r,p_c)\)，候选、尾块空洞评分及平局规则逐字对应当前 [quant_partition](../../operator_model/model.py) 实现；该规则来自固定版本源码，但只有已有 CA shape 验证过。令 \(r=\lceil R/p_r\rceil,c_s=\lceil C_s/p_c\rceil,c=\min(C,c_ss)\)，实际预测活跃 AIV 数 \(A=\lceil R/r\rceil\lceil C_s/c_s\rceil\)。不是强制填满64核。

**UB 计划。** 当前双缓冲容量公式：

\[
G_{max}=8\left\lfloor\frac{253952-2048}{256\times6+8\times2\times5}\right\rfloor,\quad
G_c=\lceil c/32\rceil,\quad I_c=\lceil G_c/G_{max}\rceil,
\]
\[
g_c=\operatorname{align}_2(\lceil G_c/I_c\rceil),\quad
I_r=\left\lceil r/\lfloor G_{max}/g_c\rfloor\right\rceil,\quad
t_r=\lceil r/I_r\rceil,\ t_c=32g_c.
\]

按 \((t_r,t_c)\) 列举单核实际 UB tile，尾块不补成整块。tile \(i\) 的元素数 \(r_ic_i\)，组数 \(g_i=r_i\lceil c_i/32\rceil\)，两类循环数 \(u_i=\lceil g_i/8\rceil,v_i=\lceil g_i/128\rceil\)，读量 \(x_i=2r_ic_i/1024\) KiB。

**max-plus 流水。** 设读完成 \(D_i\)、Vector 完成 \(V_i\)、写完成 \(W_i\)，不存在的前驱为0：

\[
P=L_{prefix}(\log_2(1+A)),\quad D_0=P+L_{read}(x_0),
\]
\[
D_i=\max\{D_{i-1}+L_{interval}(x_i),\ V_{i-2}+L_{read}(x_i)\}\quad(i\ge2),
\]

\(i=1\) 仅取第一个分支。输入槽需等 Vector 消费后复用；输出槽需等写回后复用：

\[
V_i=\max(D_i,V_{i-1},W_{i-2})+L_{vec}(u_i+v_i),\quad
W_i=\max(V_i,W_{i-1})+d_{write}.
\]
\[
\widehat T_q=\max\left(W_{last}+d_{suffix},\ a_f+b_f\min(\log_2(1+A),4)\right).
\]

后一项是经验控制路径回退，不是证明过的硬件延迟下界。cap=4 由历史数据诊断选择；大工作量仍随 UB 流水增长。

| 成本 | 截距 µs | 斜率 µs/输入单位 |
|---|---:|---:|
| prefix（log2核数） | 2.784570091 | 0.101181332 |
| 首次读（KiB） | 0.589410549 | 0.017217547 |
| Vector（u+v） | 0.065425023 | 0.007099291 |
| 后续读完成间隔（KiB） | 0.345656256 | 0 |
| 饱和控制回退 | 3.830949329 | 0.399708419 |

\(d_{write}=0.344848007\), \(d_{suffix}=0.009696963\) µs。完整精度参数以 JSON 为准。多 UB 成本仅有3个旧 shape 支持，不覆盖任意竞争模式。

动态 RV 指令计数代理：
\[
\widehat N_{RV}=59.999773475 I+43.999984348\sum_i u_i+32.000283293\sum_i v_i.
\]

该统计包含 RV load/store/control，不等于纯算术指令次数。

## 3. Matmul：tile 计划与有限流水

输入 \((M,N,K)\)，BF16 ND，调用 `aclnnMatmul`，库实际派发 `MatMulV3`。

从输出 tile \((t_m,t_n)=(16,64)\) 开始。当网格大于32且允许增大时，分别尝试某一维翻倍（各≤256），按“与32核距离、填充面积、tile边长和、维度”的字典序选取，直至网格≤32或容量到顶。这是有限源码启发式，不是所有 CANN 分支的完整重实现。

\[
G=\lceil M/t_m\rceil\lceil N/t_n\rceil,\quad A=\min(32,G),\quad W=\lceil G/A\rceil.
\]

令 \(m=\min(M,t_m),n=\min(N,t_n),k=\operatorname{align}_{16}(K)\)。read/cube/fix 的特征分别为 \(2(mk+nk)/1024,2mnk/131072,2mn/1024\)。每项服务 \(S_s=L_s(x_s)\)，Vector/write为0：

\[
B_m=\sum_s S_s+(W-1)\max_s S_s,\quad
\widehat T_m=\max(0.001,w_0+w_1B_m+w_2\log_2(1+A)+\delta_{branch}).
\]

branch 优先级：M=1 → gemv；K非16倍数 → k_tail；输出tile非(16,64) → coarse_tile；否则 dense。

| 服务 | 截距 µs | 斜率 µs/单位 |
|---|---:|---:|
| read | 0.794096073 | 0.010015807 |
| cube | 0.130526025 | 0.005557719 |
| fix | 1.522515835 | 0.046051710 |

\((w_0,w_1,w_2)=(0.628759231,1.013248530,0.041409906)\)。分支残差 gemv/k_tail/coarse_tile/dense = 0.024991658/0.041868983/−0.062130971/−0.006530111 µs。MMAD 计数约为 \(W\lceil K/256\rceil\)，仍需核验实际 matmul 分支。

## 4. 普通 FA：QK → Softmax → PV

输入 \((Q,L,D,B,H)\)，L为KV长度；BF16 SBH、非因果、无 mask/dropout，scale=\(1/\sqrt D\)。调用 `aclnnFlashAttentionScoreV2`，不是 decode 专用 FIA、MLA 或压缩注意力。

\[
q_b=\begin{cases}64,&BH\lceil Q/128\rceil<32\\128,&\text{其他}\end{cases},\quad
k_b=\begin{cases}256,&(D=64\land Q\bmod128=0\land L\bmod128=0)\lor L>1024\\128,&\text{其他}\end{cases}.
\]
\[
A=\min(32,BH\lceil Q/q_b\rceil),\quad A_V=2A,\quad
W=\lceil BH\lceil Q/q_b\rceil/32\rceil,\quad I=\lceil L/k_b\rceil.
\]

\(t_q=16\) 若Q=1，否则 \(\min(Q,q_b)\)；\(t_k=\min(\operatorname{align}_{16}(L),k_b)\)。

| 服务 | 特征 x | 截距 µs | 斜率 µs/单位 |
|---|---|---:|---:|
| R/read | \(2(t_qD+2t_kD)/1024\) | 0.953363430 | 0.002908672 |
| C/cube | \(4t_qt_kD/131072\) | 0.093779333 | 0.008430680 |
| V/vector | \(t_qt_k/512\) | 0.251097269 | 0.004456407 |
| F/fix | \(4t_qt_k/1024\) | 0.065210442 | 0.003613195 |
| O/write | \(2t_qD/1024\) | 0.405944790 | 0.009818075 |

\[
II=\max(R,C+F,V+C/2),\quad
B_{FA}=W[R+C+V+F+(I-1)II+O],
\]
\[
\widehat T_{FA}=\max(0.001,7.173997224+0.709561102B_{FA}+0.103475679\log_2(1+3A)+\delta_{branch}).
\]

分支 d64_kv256/d64_kv128/d128_kv128 的残差为 0.161074210/−0.059146776/−0.057040876 µs；不存在的分支取0并标记未见。当前模型以全 tile 推进，未显式处理每轮 softmax 依赖和访存争用的长程变化；\(II\) 只是旧小 shape 拟合出来的重叠假设，长 KV 的独立结果必须检验它。

当 \(q_b=64\) 时，QK沿N、PV沿K各按128拆分：
\[
N_{MMAD,max}=2\lceil L/128\rceil W.
\]

Q128 路径可能换算法，计数返回空值，不强套此式。耗时准确不代表次数准确，反之亦然。

## 5. 采集方案、误差指标和成本

旧参考：msopprof 全指令 JSON/CSV。新候选：官方 npusim helper 0.8.0 / contract7 的 SQLite，直接 SQL 归约，不生成大型 JSON 时间线；仍采完整指令 DB，并非只仿真部分核或部分 KV。`--core-id` 仅限制解析；`--dump=false` 不会关闭950实时解析，因此不能当作计算量缩减。

npusim 默认运行时返回 PR_9589，本轮增加显式、仅仿真环境生效的 SKU 适配器，使 `rtGetSocVersion/halGetSocVersion` 返回 DT_9582；沿用底层错误码，不替换物理模型、不改 kernel。它与标准 msopprof 在旧 case 上对比 kernel entry、分核、MMAD、正确性和计时，但不宣称一个桥接点证明全部 dispatch 等价。缺少导出 kernel 二进制的 DB 必须保留身份缺口。

新 SQL 流水量是排除 helper 标记的非计时指令后、每个 queue 的执行区间并集。该口径和旧 trace 的 VECTOR_FAMILY 分组不直接混作同一个拟合样本。原生 core summary 的 tick 窗口还未证明可替换指令窗，只作诊断。

\[
e_i=\widehat T_i-T_i,\quad APE_i=|e_i|/T_i,\quad
WAPE=\frac{\sum_i|e_i|}{\sum_iT_i},\quad MAE=\frac{1}{n}\sum_i|e_i|.
\]

完整 CA 通过才有误差；超时、部分 trace、只有正确性、错 SKU 以及未运行分别列示，不拿部分时间充当总耗时。重复旧 shape 是采集桥接，不计作新独立留出。固定耗时基线仍是旧数据均值：quant 5.801072344、matmul 3.923394018、FA 9.538626485 µs。

仿真成本分别报告墙钟、累计 CPU、保留原始字节；失败探索也计入。npusim 超时杀掉前端时 getrusage 可能漏掉后代进程，其 CPU 只可记作下界。降低保留文件量不等于同幅降低峰值临时空间或 CA 执行时间。

## 6. 在线开销、场景与外推边界

在线不读 trace、不加载 CANN。quant复杂度 O(UB tile数)，matmul/FA 为有限结构计算；LRU最多4096条，不可变结果可复用。v0.4 已测 quant 完整未缓存 P95约66 µs、缓存约1 µs，本轮不把采集优化算成在线模型加速。

用户场景：输入32k/128k/256k/512k、输出1k；推测8个时主模型校验Q=9；prefill batch=1/4/16，decode batch=1/4/16/32/128/256。TP8、chunk512/2048仍是规划假设。上下文与输出端点分别用 L和L+1024表示。B=H=1的长FA探针只验证一个低并行组件，不代表上述所有 batch、TP 或模型专用注意力。

A7/A8 尚无对应实现与独立校准。可在研究时把 \(S_s=a_s+work_s/rate_s\) 分解、重算核数/轮次并重新求流水关键路径，但不可将A5的总耗时直接乘核数比或Cube/Vector比。跨代访存、同步、固定开销和dispatch变化均未识别；本轮不提供具有已验证准确性含义的A7/A8利用率。
