# v0.5 当前建模策略：将FA读延迟与发起间隔分开

本轮保留quant、matmul和其他FA路径的v0.4模型，只修正已有32k数据支持的普通FA长循环路径。对应 [误差分析与Summary](error-analysis.md)、[冻结模型](model.json)、[采集前冻结预测](frozen-predictions.json)。新增算子族的结构模型另列于文末，未校准的算子不输出耗时。

## 1. 实际运行的库算子及“采集一致性校验”

实际运行的是 **CANN 9.2.1安装包的950算子**，通过C++ ACLNN调用。不是用Python公式替代算子执行，也不是把下载的源码重新编译后冒充安装库。

| 建模对象 | 库组件 | 实际接口 | 950库内实现目录 |
|---|---|---|---|
| dynamicMXquant | ops-nn | aclnnDynamicMxQuant | ops_nn/dynamic_mx_quant |
| matmul | ops-nn | aclnnMatmul，派发MatMulV3 | ops_nn/mat_mul_v3 |
| FA | ops-transformer | aclnnFlashAttentionScoreV2 | ops_transformer/flash_attention_score_apt |

目录相对于 `toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/`。安装包分别是 `cann-950-ops-nn_9.2.1_linux-aarch64.run`、`cann-950-ops-transformer_9.2.1_linux-aarch64.run`。FA候选文件 `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5.o` 的SHA与此前msopprof实际导出的设备对象一致。compact采集未重新导出ELF，所以本轮仍分别记录“安装包对象身份”与“实际DB entry”，不伪造新的运行时ELF哈希。

此前文档里的“桥接”是**采集方式一致性校验**：用相同旧shape分别运行msopprof全trace和npusim压缩DB，检查SoC、kernel entry、核数、指令数、正确性和计时窗口是否兼容。它不是算子，也不是一次独立shape验证。本轮直接复用已通过该校验的采集路径，不增加重复校验CA。

源码解释参考固定commit的ops-transformer/ops-nn；知识卡覆盖CANN9.1，实际安装包为9.2.1，二者不宣称完全同源。源码约束与实际指令证据一起限定模型适用域。

## 2. 预测目标与本轮数据划分

预测目标为全体实际执行核在共同时间轴上的指令窗口：

\[
T_{CA}=\frac{\max_j tick^{end}_j-\min_j tick^{start}_j}{1650}\quad(\mu s).
\]

1650来自helper报告的1.65GHz指令时钟。这个微秒量与“CA在主机上跑了多少秒”分开报告。多核阶段不能直接相加，也不能用最大单核span代替核间makespan。

- **开发/校准**：只复用旧 `[9,32768,64,1,1]` 完整DB，参考75.98µs。它在上一轮是独立验证，本轮用于建模后即转为开发样本。
- **新独立留出**：先 `[9,131072,64,1,1]`，再 `[9,16385,64,1,1]`，冻结后才采集；结果不再回调同一模型。
- **原模型对照**：v0.4参数不变；固定FA耗时9.538626485µs仍取旧数据均值，不能用本轮新数据重新算均值。

## 3. 为什么原来高估47.4%

旧模型的每轮稳态推进使用：

\[
II_{old}=\max(R,C+F,V+C/2),\qquad
T_{old}=w_0+w_1\{fill+(I-1)II_{old}+drain\}+\cdots.
\]

其中R来自旧小shape的MTE2聚合窗口，包含首轮/固定成本。一次读取需要经过多少时间才能完成（**latency**），和硬件流水每隔多久能接受下一轮工作（**initiation interval，II**）不是一个量。若下一次读可以在前一次完成前发起，把完整R重复加到每轮，就会高估长循环。

32k中实际只有1次1152B的Q搬入，随后K和V各有128次32768B搬入；Q驻留复用。源码 `IterateBmm1Nd` 仅在s2LoopCount=0搬Q，并分别使用K/V缓冲；实际DB证实这条路径。每4条按时间排序的MMAD不是严格源码循环：QK可以超前PV，因此本轮进一步按Q/K/V地址流和已知MMAD维度对应的PC划分，不继续把所有MMAD每4条机械归一组。

## 4. 新FA模型：填充一次，再按有效间隔推进

### 4.1 适用域及分核

本轮修正只用于普通BF16、SBH、非因果、无dropout/mask的：

\[
Q=9,\quad D=64,\quad B=H=1,\quad L=KV\ge4096.
\]

该路径Q tile容量64、KV tile容量256；预测1个AIC和2个AIV，只有一轮Q tile，不因KV增加而增加核数。循环数：

\[
I=\lceil L/256\rceil,\qquad N_{MMAD}=2\lceil L/128\rceil.
\]

MMAD公式分别计QK沿N和PV沿K的128分段。尾块时耗暂按完整256 tile计，是保守近似；指令次数仍按真实128分段，二者不能混用。其余Q、D、B/H的FA仍回退v0.4，尚未证明新参数可直接缩放过去。

### 4.2 首轮填充的定义

以第一条全局指令为零点，设：

- P：第一轮K读开始之前的时间，含初始化和Q准备的未隐藏部分。
- \(L_K,L_V\)：第一轮K/V读指令的完成延迟。
- \(\Delta_V\)：第一轮V读开始相对第一轮K读的偏移。
- \(F\)：第一轮K/V都读完后，到第一轮PV最后一条MMAD结束的剩余时间。
- D：最后一轮PV的MMAD完成后，到全局最后一条指令结束的排空时间。

首轮K/V读可以重叠，所以取完成时刻的最大值，不把两个读延迟相加：

\[
R_{fill}=\max(L_K,\Delta_V+L_V),\qquad
T_{fill}=P+R_{fill}+F.
\]

F是**剩余关键路径窗口**，包括首轮软最大值、同步、调度等，不是纯Cube算术服务时间；已经与读重叠的计算不会被再加一遍。P也不是单独测得的纯Scalar延迟。

### 4.3 后续发起间隔的估计

通过GM源地址还原K/V的第i个256-token tile，分别记录读开始时刻 \(s^K_i,s^V_i\)。采用i=8..119这112个中段tile，排除首尾填充/排空后，对每条流拟合：

\[
s_i\approx a+Ji,\qquad
J=\frac{\sum_i(i-\bar i)(s_i-\bar s)}{\sum_i(i-\bar i)^2}.
\]

这不是用总耗时除以长度直接调倍率。斜率J表示当前数据流下每轮工作推进的**有效发起间隔**；使用多点直线斜率可以保留持续吞吐，同时减小单次短暂抖动的影响。中段不同区间的斜率另存为稳定性诊断；同一DB内分段不是独立shape验证。

J包含缓冲槽复用、标量调度、同步和下游反压，不能解释为独立HBM峰值带宽。资源占用另提供保守服务需求 \(S_C,S_V,S_{M1},S_F\)：每种流水的非排除指令区间并集除以128轮，向量取两个AIV的最大值。取最大而非相加，表示这些资源能交叠；同一资源不能超出其服务能力：

\[
II=\max(J_K,J_V,S_C,S_V,S_{M1},S_F).
\]

当前这些服务项小于读流有效间隔，II由J确定。**这只说明当前路径的有效推进间隔主导公式，不等价于证明算子纯访存bound。** 例如J本身可能受Scalar/同步阻塞；跨架构修改带宽时不能只按比率缩J。

### 4.4 总耗时及参数

\[
\boxed{\widehat T_{FA}=P+\max(L_K,\Delta_V+L_V)+F+(I-1)II+D}
\]

读延迟只在填充项支付一次，后续按II推进；循环次数增加时线上计算仍为O(1)，不回放全部指令。

| 参数 | µs | 来源/含义 |
|---|---:|---|
| P | 6.560000 | 第一K读起点 |
| 首轮K延迟 \(L_K\) | 0.849697 | 第一K读结束减开始 |
| V启动偏移 \(\Delta_V\) | 1.066667 | 第一V读开始减第一K读开始 |
| 首轮V延迟 \(L_V\) | 0.745455 | 第一V读结束减开始 |
| F | 1.778182 | 首轮PV完成减首轮K/V全部就绪 |
| D | 0.347273 | 全局结束减最后PV完成 |
| \(J_K\) | 0.517109 | K流中段开始时刻的斜率 |
| \(J_V\) | 0.517215 | V流中段开始时刻的斜率 |
| \(S_C\) / \(S_V\) | 0.215933 / 0.220985 | 每轮Cube/最大AIV服务需求 |
| \(S_{M1}\) / \(S_F\) | 0.281013 / 0.107898 | 每轮MTE1/FIXP服务需求 |

所以当前域内简化为 \(10.497575758+(\lceil L/256\rceil-1)\times0.517214978\) µs。简式是上面阶段模型的数值结果，不是另一次总时延线性回归。32k的76.183878µs是校准回放结果，不能作为独立准确性宣称。

全部参数和128条K/V读事件见 [trace-analysis.json](trace-analysis.json)。128条PV的第二条MMAD缺少m/n/k字段；只借同DB内已知维度事件识别对应PC，分组计时，不补造缺失维度。该归因不跨二进制复用PC值。

## 5. 保留的quant、matmul和其他FA策略

以下公式和系数不因本轮FA调整而改变；完整系数、分核评分和shape边界见 [上轮完整策略](../long-fa/modeling-strategy.md) 与本轮model.json里的base_v4_model。

**quant**：先按源码启发式二维分核，计算每核实际UB tile和尾块。令读就绪、Vector完成、写完成分别为 \(R_i,V_i,W_i\)，输入/输出双缓冲槽每两轮复用：

\[
R_0=P_q+L_{read}(x_0),\quad
R_i=\max(R_{i-1}+J_{read}(x_i),V_{i-2}+L_{read}(x_i)),
\]
\[
V_i=\max(R_i,V_{i-1},W_{i-2})+S_{vec}(i),\quad
W_i=\max(V_i,W_{i-1})+D_{write},
\]
\[
\widehat T_q=\max\{W_{last}+D_{suffix},a+b\min(\log_2(1+AIV),4)\}.
\]

i=1时省略i−2项；不存在的前驱为0。max表示资源/数据同时可用才能开始，避免覆盖尚未消费的输入槽或尚未写回的输出槽。第二条统计路径处理小shape的控制开销饱和；它不是物理硬件下界。参数用13个旧quant trace拟合，本轮不增加quant CA。复杂度O(UB tile数)。

**matmul**：从16×64输出tile开始，在核网格超过32时按固定启发式增大tile；计算活跃核和最大核轮数W，read/cube/fix服务以仿真数据校准的非负仿射函数 \(S_s=a_s+b_sx_s\) 给出：

\[
B_m=\sum_sS_s+(W-1)\max_sS_s,\quad
\widehat T_m=\max(0.001,w_0+w_1B_m+w_2\log_2(1+AIC)+\delta_{branch}).
\]

第一项体现流水填充，后续各轮按最慢资源推进；branch区分gemv、K尾块、粗tile和dense，偏移是缩减后的历史残差。它不是ops-nn全部tiling算法的完整移植。MMAD计数约 \(W\lceil K/256\rceil\)。

**其他FA**：仍按旧Q/KV分块和 \(\max(R,C+F,V+C/2)\) 推进。新长循环参数不会自动套到Q17、D128、多head、多batch、prefill或MLA/KDA；这些路径的旧误差仍需单独解决。新计划中继承的旧units/opcode_features是结构兼容字段，不参与本轮长FA耗时公式；选定指令计数用单独的 \(2\lceil L/128\rceil\)。

## 6. 新增算子族：结构模型与待校准边界

新增RMSNorm、Addmm、LightningIndexer、mHC、KDA、MoE相关融合链的入口、准确API、公式和shape约束见 [扩展算子目录](../../docs/extensions/README.md)。每族单独区分“结构支持”“库调用已编译”“完整CA验证”“时延已校准”。没有成本证据的预测值返回null，不能将源码FLOPs直接除以峰值后宣称已校准。

RMSNorm不能只数乘加忽略归约；Addmm需包含alpha/beta和self广播；LightningIndexer不是普通FA；mHC分离和融合路径不能重复计启动；RecurrentKda单序列≤8与主模型Q=9存在约束冲突；GroupedMatmulSwigluQuantV2只覆盖GMM1→SwiGLU→量化，完整MoE还需GMM2与路由等。这些结构模型目前不影响冻结的FA验证。

## 7. 复现、误差和范围

运行示例：`python3 -m operator_model --model results/v0.5/model.json --op fa --shape 9 131072 64 1 1`。冻结与采集脚本分别为 `scripts/fit-v5.py`、`scripts/collect-v5.py`；已冻结模型禁止覆盖重拟合。FA新CA预算20分钟、最多两个shape；新增RMSNorm/Addmm调用验证另设每点120秒；每例启动独立进程，不自动重试。CA文件、时间、CPU消耗与在线预测开销分别报告。

\(APE_i=|\hat T_i-T_i|/T_i\)，\(WAPE=\sum_i|\hat T_i-T_i|/\sum_iT_i\)。新旧模型和常数必须在同一组成功留出点比较，失败点单列，不能用不同测试集误差相减。256k/512k若没有完整CA，仍标为长度外推；B/H/TP、因果mask、真实模型专用注意力、A7/A8均未因此得到验证。
