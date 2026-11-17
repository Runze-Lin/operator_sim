# MoE FFN 真实库调用、数值基准和受限建模（v0.7）

当前采用真实 CANN 9.2.1 两阶段完整链采集，一次CA复用GMM1、GMM2两份完整指令DB；首个r3校准点已完成全部输出/scale正确性验证并正常退出。manifest为3个校准shape加1个冻结后独立留出，后续成功、失败、耗时和kernel身份以 `results/v0.7/` 逐case记录为准。当前通用建模方向是独立硬件资源参数与源码/ISA展开依赖图；本文件保留的少样本线性式只作局部对照，不能当通用主模型。

## 范围与身份

本轮是两个真实库算子的链：`aclnnGroupedMatmulSwigluQuantV2`（GMM1 + SwiGLU + MX 输出量化）→ `aclnnGroupedMatmulV5`（GMM2）。不创建自定义 Kernel，不把整链描述成单个融合算子，不包括 MoE 路由、dispatch/combine、EP 通信和其他 dtype 路线。目标运行 SoC 为 `Ascend950DT_9582`；采集器的 `Ascend950DT` 名称与实际 Runtime 返回值分开记录。

实际链接的是 `toolchains/cann/9.2.1/cann-9.2.1/lib64/libopapi.so`；预期 kernel 目录为 `opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_transformer/grouped_matmul_swiglu_quant_v2_apt/` 与 `grouped_matmul_apt/`。此处仅是安装包目录，实际命中的 binary/hash/tiling 应读取各次捕获，不用预期目录代替观测身份。

证据来源：

- 安装包同名 `include/aclnnop/aclnn_grouped_matmul_swiglu_quant_v2.h`、`aclnn_grouped_matmul_v5.h` 确定真实 ABI。
- 开源仓 `third_party/ops-transformer/gmm/grouped_matmul_swiglu_quant_v2/docs/aclnnGroupedMatmulSwigluQuantV2.md` 与 `gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md` 确定 dtype/layout 与 group-list 约束。
- 安装包 `opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common/cgmct/kernel/kernel_gmm_swiglu_mxquant.h` 与 `epilogue/cgmct_block_epilogue_swiglu_mx_quant.h` 确定本轮 golden 中两半顺序、BF16 中间舍入、MX scale 指数规则。
- 全文读取平台 950 知识卡 `third_party/cannbot-knowledge/knowledge/ops/ascendc/operators/aclnn/transformer/gmm/grouped_matmul_swiglu_quant_950.md`；它标记 CANN 9.1.0，作实现机制参考，不能代替 9.2.1 安装包行为。尤其旧卡中的半宽顺序表述不能直接套入当前 golden。

## 数学语义与 CPU golden

设 `M` 是输入容量行数，`K` 是模型输入/最终输出通道数，`F` 是 FFN 中间通道数，`E` 是单卡本地专家数。`m_e` 是专家 e 的实际 token 数，`T=Σ_e m_e≤M` 是有效行数。E 不是全局专家数或 EP 度。`groupListType=1` 直接传 `[m_0,…,m_(E−1)]`。

对专家 e 的有效输入块：

\[
C_e=X_e W_{1,e},\quad A_e=C_e[:,0:F],\quad G_e=C_e[:,F:2F],
\]
\[
S_e=\operatorname{BF16}_{\rm RNE}\left(\frac{A_e}{1+\exp(-A_e)}\odot G_e\right).
\]

这里 `X_e` 和 `W_1,e` 均指先按 E8M0 scale 恢复的实数；矩阵乘由独立 CPU double 循环计算。`⊙` 表示对应位置相乘；RNE 是最近偶数舍入。通俗地说，先做两组专家线性变换，一半做 SiLU 激活，再乘另一半；安装包会把这一中间值舍入成 BF16 后量化，golden 明确复现这个可观测舍入。

每一行每连续 32 个中间值组成一个 MX 块 b：

\[
a_b=\max_{j\in b}|S_j|,\qquad p_b=\max(-127,\lfloor\log_2 a_b\rfloor-8),\qquad s_b=2^{p_b},
\]
\[
q_j=\operatorname{E4M3FN}_{\rm RNE,sat}(S_j/s_b).
\]

`a_b` 是块内最大绝对值；`p_b` 是共享 scale 的二进制指数；8 是 E4M3FN 最大正规指数；`sat` 表示饱和到有限值 ±448。全零块按本实现边界令 scale 编码为 0。本轮输入避免 NaN/Inf/极端下溢，因此不声称覆盖这些例外。E8M0 字节编码为 `p_b+127`，其实际值是 `2^(byte−127)`。代码以 BF16 指数字段对应的 floor 规则为准，而不是把文档中的近似舍入符号理解成 log2 四舍五入。

GMM2 输出：

\[
Y_e=\operatorname{BF16}_{\rm RNE}\bigl((Q_e\odot\operatorname{broadcast}(s_e))W_{2,e}\bigr).
\]

`Q_e` 是第一阶段 FP8 输出，`s_e` 沿每行 K/F 归约维每 32 值广播。CPU 链 golden 从自身计算出的 q/s 计算最终结果，不用待测最终输出构造参考。

输入不是全零、单位阵或常量：FP8 数据含正负、非均匀幅值；输入和权重 E8M0 scale 在 125/126/127 间变化，并跨行/专家/组变化。每个有效行的所有 FP8 输出逐元素检查（atol=1/16，rtol=1/4，最大绝对误差≤4）；scale 所有有效字节要求完全一致；GMM2 全部有效 BF16 输出检查 atol=rtol=1/64 且最大绝对误差≤1。小 case 要求 100% 满足，比通用标准 99% 匹配更严格。只检查 `T` 个逻辑有效行，不把 `M−T` 容量尾行当作定义输出。

## 两阶段 scale ABI 与采集

| 对象 | dtype | 物理 shape（不转置 ND） |
|---|---|---|
| X | E4M3FN | `[M,K]` |
| X scale | E8M0 | `[M,ceil(K/64),2]` |
| W1 | E4M3FN | `[E,K,2F]` |
| W1 scale | E8M0 | `[E,ceil(K/64),2F,2]` |
| GMM1 output / GMM2 input | E4M3FN | `[M,F]` |
| GMM1 outputScale / GMM2 perTokenScale | E8M0 | `[M,ceil(F/64),2]` |
| W2 | E4M3FN | `[E,F,K]` |
| W2 scale | E8M0 | `[E,ceil(F/64),K,2]` |
| GMM2 output | BF16 | `[M,K]` |

末维 2 是每 64 元素的两个 32 元素 scale，不能当成两个独立 scale tensor。链中直接传相同 device tensor，不进行重新量化、CPU回填或自定义排布变换。各个 tensor list 只有一个成员。GMM1 属性 `dequantMode=2,dequantDtype=0,quantMode=2,groupListType=1`；GMM2 属性 `splitItem=3,groupType=0,groupListType=1,actType=0`；可选 bias/offset/activation/tuning 均为空。

每个 stage 在 ACLNN 提交前打印 `stage=... api=...`，同步后打印 `stage_complete=...`。链统一记录能够复用两组件 DB；是否恰好两个 kernel 必须按实际导出数确认，额外 memcpy/辅助 kernel 不能悄悄丢弃。单 kernel 时间口径采用其内部完成的指令跨度；若组件有多 kernel，记录各跨度之和作为明确的净执行代理。跨 DB 时钟未校准时不取首末时间差冒充端到端，也不把两个组件时长之和称作真实 host latency。

## 少量 shape 与冻结划分

| case | `(M,K,F,E)` | expert_counts | 角色 |
|---|---|---|---|
| moe_cal_e1 | `(2,64,64,1)` | `[2]` | calibration |
| moe_cal_e2_skew | `(4,128,64,2)` | `[1,3]` | calibration |
| moe_cal_e1_f128 | `(2,64,128,1)` | `[2]` | calibration；独立变化GMM2归约维F |
| moe_holdout_e2_wide | `(6,128,128,2)` | `[1,5]` | 冻结后 independent holdout |

每例先以 180s 为 CA 超时预算；四例是四次链采集，不是十二次独立采集。每例三个 correctness groups。`configs/extension-moe-v7.json` 为机器可读清单。先跑三 calibration，再冻结模型参数与预测，最后采 holdout；不以 holdout 调参后仍称其独立。

## 分块机制特征与局部耗时对照的解释

三点 calibration（其中新增一例沿GMM2归约维F变化）仍不足以同时辨识多个流水参数，所以建议先使用带显式范围的单特征校准模型，保留固定耗时基线；不要硬拟合五个未知数再声称具备架构泛化能力。

先按每专家分块计算工作量：

\[
L_{e,1}=\lceil m_e/b_M\rceil\lceil F/b_N\rceil,\quad
L_{e,2}=\lceil m_e/b'_M\rceil\lceil K/b'_N\rceil,\quad
R_1=\lceil K/b_K\rceil,\quad R_2=\lceil F/b'_K\rceil.
\]

`b_M,b_N,b_K` 是 GMM1 实测或库 tiling 导出的行、半宽、归约块尺寸；带撇参数属于 GMM2。`L_e` 是该专家要分发的输出 tile 数，`R` 是每 tile 的归约循环数。空专家 m_e=0 贡献零 tile，不能把平均 token 数代入所有专家。直观上，两个各 1 行的专家并不等于一个 2 行专家：即使 FLOPs 相同，它们仍有独立 tile、group-list 处理和尾块。

可同时保存诊断特征：`E_active=Σ1[m_e>0]`（有效专家数）、`L=Σ_e L_e`（总任务数）、`m_max=max(m_e)`（最大专家负载）、`imbalance=E_active*m_max/T`（偏斜程度）。真实浮点操作数是 `FLOPs1=4TKF` 与 `FLOPs2=2TFK`，SwiGLU 处理元素数为 `TF`；它们可解释形状，却不能单独代表小 shape 时长。

以下二参数式只作同分支小shape的局部统计对照，不作为通用主模型、架构效率或A7/A8外推依据。其前提是观测确认当前 tile 调度/核数稳定：

\[
\widehat t_i=h_i+\alpha_i z_i,\quad
z_1=\max_c\sum_{j\in\mathcal J_{1,c}} R_1,\quad
z_2=\max_c\sum_{j\in\mathcal J_{2,c}} R_2.
\]

`t_i` 是组件 i 的 net kernel 时间代理，`h_i≥0` 是该分支启动/收尾固定成本，`α_i≥0` 是一个临界核 tile-归约轮次的有效时间，`J_i,c` 是按库调度规则分给核 c 的任务集合，`z_i` 是最忙核的工作轮次数。少量 calibration 只估 h/α 两个参数；若 z 相同或时长随 z 反向，使用非负受限拟合/固定基线，不输出虚假负斜率。通俗地说，总工作由许多核分担，但算子必须等最忙的核完成。

若 tiling/调度证据不够，`z` 只能用受限 FLOPs/有效专家 proxy，必须在总模型明确标为统计 proxy，不能标成已确认分核模型。扩到多 tile 后才进一步拆 `latency`（首次数据到达）、`II`（可再次发起间隔）和 Cube/Vector overlap。GMM1 是融合 CV，不能简单把 Cube 与 Vector 指令活跃时间相加，也不能把融合耗时当作单独 SwiGLU 实测。

整链代理 `t_chain=t_1+t_2` 仅表示两个顺序组件的净执行时间和；若真实部署需要 launch、通信、全局路由或系统延迟，应显式另建模型。本轮小 shape/MX ND 路线外的外推（其他 dtype、更多专家/空专家、跨 UB/L1/tile 门限、更大真实模型 shape、A7/A8）均未经校准。预测特征按 E 计算，O(E)，无需每次读取 trace；trace 只在离线校准时分析。

## 首次链验证与采集修复（实测更新）

`runs/model-v7/moe_cal_e1/run.log` 的真实 `(2,64,64,1)` 链于约51.4秒完成计算并打印 `SMOKE_PASS`：GMM1全部128值 `bad=0,max_abs_error=0`；4个E8M0 scale `bad=0,max_abs_error=0`；GMM2全部128值 `bad=0,max_abs_error=0.00269413`。这证实了当前shape下直接传递scale的ABI与完整数值正确性。

随后helper在shutdown超过180秒预算，第二DB只有空`KernelInfo`，整次采集判失败，不能纳入链耗时误差。第一DB完整但只作故障诊断原始数据；不把部分链trace拼成完整实测。helper日志显示两次runtime handle不同但entry都为0，分别创建`0000_entry_kernel_0.db`和`0001_entry_kernel_0.db`；raw日志两次task_id都为0、stream_id分别0/1，因此“task0 done”不能唯一标识launch。这是证实的身份重复/导出不完整现象；“同名entry导致锁等待”仍是待验证根因，不能写成已定位源码bug。

当时为排除多kernel导出问题，曾计划保持相同真实库caller，分别运行`grouped_matmul_swiglu_quant_v2 M K F E counts`（2组正确性）和`gmm2 M K F E counts`（1组正确性），每个进程只导出一个kernel。此排障方案现已撤销，后续r2单组件实验和r3成功链已确认真实根因为metadata生命周期。manifest保留所有失败/未运行状态，不延长180秒超时。GMM2独立输入是非退化FP8/E8M0测试数据，不声称它等于链中每个量化值；链数值正确性由上述真实链另行证明。若将来使用独立primitive备选方案，组合时间也只能标注为两个独立调用的指令span之和，不能称为端到端链实测。

## 生命周期根因与r3修复

独立GMM1的r2仍在约40.4秒完成所有正确性检查后卡在shutdown，因此“只因同名多kernel”假设已被反证。对仍活着的进程4092446做只读gdb栈采集并detach：主线程为`free → op::internal::BlockCache::~BlockCache() → __call_tls_dtors → exit`；该次DB的`KernelInfo`与`ExecutedInstructions`均为空，不接受其耗时。

安装包`libnnopbase.so`反汇编明确显示：`aclTensorList::aclTensorList`通过`memcpy_s`复制成员指针而不克隆tensor；`aclDestroyTensorList`在0xff320循环取成员，0xff330调用`aclDestroyTensor`。原caller的TensorList析构与Tensor RAII析构重复释放同一metadata，导致线程本地BlockCache损坏，进程退出时挂起。最小修复是TensorList构造成功后接管`aclTensor*&`并清空原指针；device Buffer仍由Tensor独立管理。没有绕过析构、强制退出或改动精度容差。

修复后先做CPU-only元数据自测：1000次真实libnnopbase建/销list，0次device allocation，打印`METADATA_SELFTEST_PASS`并正常exit=0（未启动CA）。r3增加对象释放、stream释放、device reset、ACL finalize与main return日志，便于确认整个清理链。旧r2失败/已手动中止/未运行case保存在manifest历史条目；r3先用完整链验证两个组件及退出，再推进剩余calibration。诊断结构记录为`runs/model-v7/moe-lifecycle-diagnosis.json`。

## r3完整链通过（无需加长超时）

`moe_cal_e1_chain_r3` 完整采集成功，npusim计时约67.2秒、外层墙钟68.205秒，原180秒预算不变；所有三组golden通过，新增cleanup日志走到`main_return`，进程正常exit=0。两份DB均完整并通过主采集器检查：GMM1 `7.004848485 µs`，GMM2 `3.247878788 µs`，串行kernel净指令时间和`10.252727273 µs`。该和仍不包含host/inter-launch idle，不能称为host端到端延迟。

raw日志在各同步stage区间内分别出现唯一`(stream_id,task_id)=(0,0)`、`(1,0)`，与helper chronological launch_id 0/1对应；虽然两个entry均名`entry_kernel_0`，阶段映射不再依赖CLI去重后的“task0 done”。两个不同完整DB及明确API阶段证明实际调用的两个组件。

成功证明生命周期修复消除了退出挂起；无需以独立primitive启动增加采集成本。manifest恢复3cal+1holdout的完整链，8个尚未运行的primitive r3候补标记superseded，不重复采同shape。其他cal/hold完成情况以最终results/v0.7报告为准。
