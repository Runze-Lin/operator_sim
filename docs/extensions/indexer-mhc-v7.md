# v0.7 LightningIndexer 与 mHC：真实库调用、完整输出校验与建模边界

本文说明当前新增验证资产及建议的局部耗时模型；CA 实测数值、冻结参数和独立误差由总报告记录。编译成功不等于 CA 已验证。当前 caller 使用本地 **CANN 9.2.1 / Ascend950DT_9582** 安装包中的公开 ACLNN，未构造替代 AscendC kernel。

## 调用与验证范围

| 算子 | 实际两段式 API（表列第二段名称） | 输入/属性 | 请求且逐元素检查的输出 | correctness 组数 |
|---|---|---|---|---:|
| LightningIndexer | `aclnnLightningIndexer` | Q `[1,Q,8,128]`、K `[1,L,1,128]`、W `[1,Q,8]`，BF16；BSND、右下因果、TopK=512 | `[1,Q,1,512]` INT32 indices；每个 Query 的完整集合、无漏项、无多项；不规定未承诺的输出顺序 | 1 |
| mHC Pre | `aclnnMhcPre` | X `[T,4,D]` BF16；phi `[24,4D]`、alpha `[3]`、bias `[24]` FP32，gamma=null | hin BF16；hpost、hres、invRms、hMix、hPre FP32，六个完整输出 | 6 |
| mHC Sinkhorn | `aclnnMhcSinkhorn` | `[T,4,4]` FP32，20 iterations、eps=1e-6 | 完整 `[T,4,4]` FP32 output | 1 |
| mHC PreSinkhorn | `aclnnMhcPreSinkhorn` | X `[T,4,4096]` BF16，phi FP32，N=4、I=20，两个eps=1e-6，needBackward=false | 完整 hin `[T,D]`、hpost `[T,N]`、hres **`[T,N*N]`** | 3 |
| mHC Post | `aclnnMhcPost` | X `[T,4,D]`、hOut `[T,D]` BF16，hRes `[T,4,4]`、hPost `[T,4]` FP32 | 完整 `[T,4,D]` BF16 out | 1 |
| mHC split 链路 | `aclnnMhcPre → aclnnMhcSinkhorn → aclnnMhcPost` | 同Pre；外部子层取恒等映射 hOut=hin | Pre六输出、Sinkhorn输出、Post输出，额外独立端到端输出 | 9 |

Sinkhorn 的 `normOut/sumOut` 未请求；PreSinkhorn 的五个反向辅助输出在 `needBackward=false` 时契约明确无效，但其Tensor描述符仍为required，因此提供契约shape/dtype的占位缓冲，不读取它们的无效值、不把它们假称已验证。Indexer `returnValues=false` 不产生 scores 输出。所有**本次模式下定义且请求**的输出均完整校验，不是只取前几个元素。训练反向暂存、FP16、其他N/头数、PA/TND、量化Indexer不在这份推理验证范围。

mHC split 链路在 GPU/NPU Device 缓冲区间直接传递 `Pre.hres → Sinkhorn.x → Post.hres`，以及 `Pre.hpost → Post.hpost`、`Pre.hin → Post.hOut`。它验证残差封装的真实可组合性，不包含 Attention/MLP 工作量。Pre 此处仍请求三个可选中间输出；该配置必须写入耗时模型的作用域，不能与不写aux的模型混用。链路总时间口径为 **serial_kernel_instruction_sum**，不是 Python、ACLNN host、Memcpy 或整网 wall time。

## 小样本冻结协议与成本

详细机器可读清单：`configs/extension-indexer-mhc-v7.json`。所有shape先登记，两个 calibration 用于拟合；冻结参数后再执行第三个 holdout。若观察 holdout 后调整模型，该点转为开发回放，不能继续称独立验证。

| 算子 | calibration 1 | calibration 2 | 待冻结后 holdout |
|---|---|---|---|
| LightningIndexer (B,Q,L,G,D) | (1,1,512,8,128) | (1,1,1024,8,128) | (1,2,768,8,128) |
| Pre (T,N,D) | (1,4,128) | (8,4,128) | (3,4,128) |
| Sinkhorn (T,N;I) | (1,4;20) | (64,4;20) | (17,4;20) |
| PreSinkhorn (T,N,D) | (1,4,4096) | (4,4,4096) | (2,4,4096) |
| Post (T,N,D) | (1,4,128) | (32,4,128) | (9,4,128) |
| split链路 (T,N,D) | (1,4,128) | (8,4,128) | (3,4,128) |

PreSinkhorn首次cal1在GetWorkspaceSize阶段以161002失败，未启动kernel。已证实caller的hRes形状不符合该接口契约：融合API要求`[T,N*N]`，独立Pre才是`[T,N,N]`。重试只修正Tensor视图rank，不改数值、golden或容差；新校准id追加`_r2`。原cal1保留为`failed_attempt`、尚未执行的旧cal2标`superseded`，二者不进入拟合/误差样本；holdout id保持不变。第二次`_r2`已越过该参数错误，但GetWorkspaceSize返回561103（内部nullptr）。逐项核对安装包`mhc_pre_sinkhorn_apt.json`可见八个output全部`paramType=required`；`needBackward=false`只表示五个辅助输出内容无效，不等于允许空描述符。`_r3`为hPre `[T,N]`、hcBeforeNorm `[T,N*N+2N]`、invRms `[T,1]`、sumOut `[40,T,N]`、normOut `[40,T,N,N]`提供FP32占位Tensor，仍保持needBackward=false与三个有效输出的完整校验。两个失败尝试均无kernel，不能给出0误差或算子耗时。

每case登记120秒CA预算；失败/超时保留状态而不自动拉长，完整数据库与部分trace分开。父线程统一串行采集，以便控制CA总CPU、磁盘和trace开销；本caller只执行一次目标路径，无重复warmup。复用同一批数据库完成指令、核数、流水、耗时分析，避免为每个指标重复CA。编译脚本 `bash scripts/build-extension-indexer-mhc.sh`；命令 `build/smoke/extension_indexer_mhc <op> <三个参数>`，Indexer参数Q/L/G，Sinkhorn参数T/N/I，其余T/N/D。

这些shape覆盖最小负载、token数变化和Indexer新Q，不证明跨dtype、长上下文或切换kernel算法后的泛化。特别是小T增加可能只增加活跃核，而非每核工作，因此耗时可能近似不变；拟合必须保留这一事实，不能用总FLOPs强行得到线性增长。

## 数值golden及其含义

BF16输入先进行round-to-nearest-even编码，然后CPU从实际BF16位模式解码，以FP64完成golden。phi选用有正有负的二进制分数、非零alpha和有变化的bias；避免全0或全1掩盖布局/转置错误。PreSinkhorn phi的绝对值不超过8/2048，满足D4096、N4下文档要求的 `1/sqrt(ND)` 幅值限制。Sinkhorn输入为非对称非恒定矩阵；Post hRes非对称，能检查矩阵转置方向。

逐元素通过条件为

\[
|y_i-\bar y_i|\le a+r|\bar y_i|.
\]

`y_i`是Device输出第i个元素，`bar y_i`是CPU golden；`a`为绝对容差，保护接近0的值；`r`为相对容差。BF16采用a=r=2^-6、绝对误差上限1；FP32采用a=2^-16、r=2^-10、绝对误差上限0.01。小case要求**所有元素**通过且有限，比技能99%通过率更严格。正确性误差与预测耗时误差是两类不同的指标。

Indexer CPU按实际BF16 Q/K/W计算全量有效候选分数，再比较完整Top512集合。键输入使用两个可精确表示的分量编码位置，形成非恒定、无TopK边界歧义的评分；不能把排序顺序未定义误判为错误。不同head的Query刻意混合正负符号，令部分点积触发ReLU清零，其余head保留正分数；因此不仅覆盖正值线性累加。该构造保障排名可解释，不等于覆盖所有随机分布或近似并列边界。

## 数学语义：每个参数的解释

### LightningIndexer

\[
s_{b,q,j}=\sum_{g=0}^{G-1}w_{b,q,g}\max\left(0,\sum_{d=0}^{D-1}Q_{b,q,g,d}K_{b,j,0,d}\right),\quad
\mathcal I_{b,q}=\operatorname{TopK}_{j\in\mathcal V_q}(s_{b,q,j},k).
\]

B是batch，q是Query位置，j是Key位置，G是Query头数，D是每头特征数，w是每Query每头的权重，k是输出索引数；Q/K分别是查询和键张量。集合 `V_q` 是mask允许的位置，本轮右下因果为 `0≤j<L-Q+q+1`。直观上，先为每个head算相似度，负相似度置0，按权重加总，再选最相关的k个历史位置。TopK省的是下游Attention的稀疏访问，并不消除本算子的全候选评分。

定义有效Query-Key对数

\[
P=B\sum_{q=0}^{Q-1}\max(0,\min(L,L-Q+q+1)),\quad F_C=2PGD.
\]

P单位是“对”；F_C单位是逻辑浮点运算数，一次乘加计2。G/D如上。该公式是语义工作量，不是已执行Cube指令数；真实tiling可能整块计算被mask的边缘。TopK的实际比较次数不能由k直接推导，需要对应kernel/trace证据，不在这里假设一定是完整排序。

### mHC Pre

令 `X_t` 表示token t的N路残差，每路D个值；展平宽度K=ND，投影宽度P=N²+2N。

\[
r_t=(K^{-1}\sum_{i=0}^{N-1}\sum_{d=0}^{D-1}X_{t,i,d}^2+\epsilon_n)^{-1/2},\quad
M_{t,j}=\sum_{i,d}X_{t,i,d}\Phi_{j,iD+d},\quad U_{t,j}=r_tM_{t,j}.
\]

r是倒均方根，`epsilon_n`是normEps防除零量；Phi是FP32权重；M是hMix原始投影，U是乘倒均方根后的投影。将U按N/N/N²拆成三段，分别记U_pre/U_post/U_res；bias按相同宽度拆为b_pre/b_post/b_res。

\[
a^{pre}_{t,i}=\sigma(\alpha_0U^{pre}_{t,i}+b^{pre}_i)+\epsilon_h,\quad
H^{post}_{t,i}=2\sigma(\alpha_1U^{post}_{t,i}+b^{post}_i),
\]
\[
Z^{res}_{t,i,j}=\alpha_2 U^{res}_{t,i,j}+b^{res}_{i,j},\quad
h^{in}_{t,d}=\sum_i a^{pre}_{t,i}X_{t,i,d}.
\]

sigma(z)=1/(1+exp(-z))是sigmoid；alpha0/1/2是三种连接的缩放参数，`epsilon_h`是hcEps；Z_res是尚未Sinkhorn归一化的hRes。本轮alpha长度3、gamma=null；带gamma及alpha长度2的公式不能无条件套用。通俗地说，Pre先归一化并生成连接权重，然后把N路残差汇合成一路给子层，同时保留之后把结果分回多路所需的权重。

逻辑投影工作量 `F_C=2T(ND)(N²+2N)`；T为token数。平方/归约和汇合随TND增长，sigmoid数为2TN。它们是不同指令类型，不能把exp/除法都当作一条普通乘加。

### mHC Sinkhorn

\[
A^{(0)}=\operatorname{softmax}_{row}(Z)+\epsilon_h,\quad
A^{(1)}_{ij}=A^{(0)}_{ij}/(\sum_a A^{(0)}_{aj}+\epsilon_h).
\]

此后I-1轮先按行和归一化，再按列和归一化。Z为每token的N×N连接logits；A^(s)为第s个归一化半步后的矩阵；i/j是行/列，a是被求和的行索引，I是迭代次数。本轮I=20。输出为最后的A^(2I-1)。直观上，通过反复让行和、列和接近1，避免某条残差通路被无约束放大。

softmax后共 `2I-1` 个归一化半步；除法的逻辑元素数为 `2ITN²`（包括softmax自身归一化）。exp有TN²个；归约加法为 `2ITN(N-1)`。T/N/I分别是token数、残差路数、完整迭代次数。递归依赖意味着下一半步必须看到前一半步结果，不能把20轮当作20个完全重叠任务。

### mHC PreSinkhorn

其数学结果为Pre的hin/hpost与 `Sinkhorn(Pre.hres)`。本轮只使用N4、I20、D4096和epsilon=1e-6的已列配置。它是**一个融合API**：在建立流水预测时合并Pre和Sinkhorn的依赖，不能把已测融合耗时再加一次独立Sinkhorn耗时。文档说明其投影使用HF32，所以FP64 golden并非逐指令bitwise参照，而是数值精度参照。

### mHC Post及组合

\[
Y_{t,i,d}=h^{out}_{t,d}H^{post}_{t,i}+\sum_{j=0}^{N-1}X_{t,j,d}H^{res}_{t,j,i}.
\]

Y是输出多路残差；hOut是外部子层的一路输出；Hpost是Pre产生的系数；Hres是Sinkhorn归一化后的连接矩阵。注意右侧取Hres[j,i]，相当于矩阵转置；X是原始多路残差。直观上，一部分输出来自子层结果按不同系数分配，另一部分来自旧残差通路的混合。

本轮在组合测试中令hOut=hin，避免引入未建模的Attention/MLP，但最终CPU golden仍从原始输入独立算完Pre、BF16中间边界、Sinkhorn和Post。独立Post的hOut使用另一组非恒定输入。分离实现耗时可按其实际串行kernel时间相加；融合PreSinkhorn路线则是 `tau_fused+tau_post`，不能加上单独Pre和Sinkhorn。

## 局部耗时对照：先限制可识别参数，不替代通用模型

本节的两点拟合用于局部对照基线，不作为通用建模方法。跨shape通用预测需按实际指令、tiling、分核、搬运和流水机制计算，并使用总建模文档描述的机制模型；不能用本节的经验斜率替代这些参数。

对同一已验证kernel/tiling分支，局部基线形式为

\[
\hat\tau(x)=\alpha_r+\beta_r W_r(x),\qquad \alpha_r\ge0,\ \beta_r\ge0.
\]

x为算子shape与dtype/属性组合；r为真实选择的执行分支；alpha_r单位us，表示启动、填充、收尾等固定成本；beta_r单位us/工作单位；W_r是**最繁忙核**的工作指标，不是无条件使用整算子元素数。每分支只有两个校准点时最多辨识这两个参数；不能同时自由拟合多种访存/Vector/Cube系数。若两个shape分核后W相同，就无法估计斜率，应冻结斜率的来源或回退为此分支常量均值并标识低可信度。

建议先从CA提取真实活跃核及每核tile数，再建立以下特征：

| 算子 | 工作特征与来源 | 直观解释/限制 |
|---|---|---|
| Indexer | 最重核Query tile数×Key tile数，以及当前分支TopK合并轮数 | 一个核需要扫多少候选块、合并多少轮；G、D固定的小域可压为一个工作指标，跨算法不能沿用 |
| Pre | 最重核投影tile数；固定N/D时每核token轮数 | token分散到更多核，单核工作可能不变，因此时间并非按T比例增长 |
| Sinkhorn | 最重核矩阵块数×I；初始softmax单列固定阶段 | 每个块的矩阵要反复行列归一化；向量lane并行使TN²总数不等于串行耗时 |
| PreSinkhorn | 同一融合分支的最长依赖路径/每核循环轮数 | 不能把Pre和Sinkhorn视为完全串行的两个单独启动，也不能假定完全重叠 |
| Post | 最重核item数×N×(N+1)×ceil(Dtile/V) | V是实际每条向量操作有效lane宽度，Dtile是一个D切片；N²混合与N广播相加工作 |
| split链路 | 实际三段预测的串行和，逐kernel保留明细 | 不加入host时间；保留Pre可选输出配置，不能与无auxPre混用 |

Post的可解释分核参考来自固定源码：令Dtile为D切片宽度，`J=ceil(D/Dtile)`、`U=TJ`，U是总item数；A为可用AIV核数，最重核item数约为 `ceil(U/min(U,A))`，实际应复现tiling中的尾核收紧规则。小D128时每个token为一个item，T≤A时每核最多一个item，固定耗时基线优于“总元素线性时间”是有机制依据的可能情况，而不是悖论。`8(N+2)Dtile≤UB`为固定950源码的常驻X路线条件；UB单位字节。该条件支持源码解释，安装包9.2.1的实际分支仍必须用采集结果核对。

在trace确认流水后，可将工作项扩展为

\[
\tau_{loop}=\ell_{fill}+(R-1)\cdot II+\ell_{drain}.
\]

R为最繁忙核循环轮数，ell_fill是首轮填充延迟，ell_drain是末轮排空延迟，II是稳态**发起间隔**，全部时间单位us。不能把一次读操作从发出到完成的读延迟当作每轮都要支付的II。两个校准点不足以独立确定fill、II、drain三项，可先将fill+drain合为alpha。此处是数据足够后采用的模型形式，不能把尚未辨识的参数写成已校准值。

## 来源与可外推边界

- SDK真实声明：`toolchains/cann/9.2.1/cann-9.2.1/include/aclnnop/aclnn_{lightning_indexer,mhc_pre,mhc_sinkhorn,mhc_pre_sinkhorn,mhc_post}.h`。
- 对应开源语义参考：`third_party/ops-transformer/attention/lightning_indexer/docs/aclnnLightningIndexer.md` 及 `third_party/ops-transformer/mhc/*/docs/aclnn*.md`，已登记commit及hash见 `docs/extensions/indexer-mhc-evidence.json`。
- 已复用且完整核读知识卡：`knowledge/model/inference/concepts/modules/mhc.md`；`knowledge/ops/ascendc/operators/aclnn/transformer/mhc/mhc_post_950.md`。Post卡对应CANN9.1.0，仅作分核/UB机制参考，不证明9.2.1二进制逐指令相同。当前SDK允许950 hRes缺省，但本次明确提供hRes。
- 实际运行所选kernel只能由安装包采集日志、kernel名/二进制hash和数据库证明，不能把开源文件路径冒充已运行的库实现。
- 本轮不把A5校准的固定启动时间或经验斜率直接搬到A7/A8；跨代需重新指定核数、频率、lane/指令吞吐、UB容量、发起间隔及算法分支。新硬件参数未经验证时只能称条件外推。

### PreSinkhorn r3 安装包输出契约证据

实际APT配置：`toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend950/ops_transformer/mhc_pre_sinkhorn_apt.json`。
SHA256：`fedc2cdd9418d6f040af4dc89cc52c877393327a48d33d412bea8e0188e52bde`。其中BF16/FP16两组binList都将下列八个输出标为`paramType=required`。shape来自同接口公开文档参数表，r3固定N=4、I=20。

| 输出参数 | caller Tensor shape | dtype | needBackward=false的值是否有效并校验 |
|---|---|---|---|
| hin | `[T,D]` | BF16，与x一致 | 是，完整检查 |
| hPost | `[T,N]` | FP32 | 是，完整检查 |
| hRes | `[T,N*N]` | FP32 | 是，完整检查 |
| hPre | `[T,N]` | FP32 | 否，required描述符占位 |
| hcBeforeNorm | `[T,N*N+2N]` | FP32 | 否，required描述符占位 |
| invRms | `[T,1]` | FP32 | 否，required描述符占位 |
| sumOut | `[2I,T,N]` | FP32 | 否，required描述符占位 |
| normOut | `[2I,T,N,N]` | FP32 | 否，required描述符占位 |

r3只在fused分支分配sumOut/normOut占位，独立Pre、Sinkhorn、Post和split调用路径不增加这两份分配。失败日志现在同时打印`aclGetRecentErrMsg()`，避免只保留数值状态码而丢失ACLNN诊断。此处记录的是配置与ABI修复证据；r3是否CA成功仍以其独立结果JSON为准。
