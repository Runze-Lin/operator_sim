# KDA v0.7：真实 ACLNN 调用、数值检查与受限耗时模型

本文件记录 KDA 两条路径的实现合同。**调用器已经链接到本机 CANN 9.2.1；CA 执行、选中 kernel 和误差统计以主流程生成的 v0.7 采集清单为准，编译成功本身不是实测通过。** 不改写此前 `kda-moe.md` 的历史结论。目标为 `Ascend950DT_9582`。

## 真实算子与验证矩阵

| 路径 | 实际接口 | 固定 shape 参数 | 预设校准 S | 预设留出 S | 每例检查组 |
|---|---|---|---|---|---|
| 递推 | `aclnnRecurrentKda` | B=Hq=Hv=1,K=V=128 | 1、2 | 4 | 3：O、最终 state、初始 state 不变 |
| 分块r2 | `aclnnChunkKdaFwd` | B=Hq=Hv=1,K=V=128,C=64 | 16、32 | 24 | 6：Aqk、Akk、累计 gate、O、最终 state、初始 state 不变 |

`B` 是本 rank 序列数，`S` 是本次调用 token 数，`Hq/Hv` 分别是 Q/K 和 value 的 head 数，`K/V` 是 key/value 特征宽度，`C` 是 chunk 容量。shape 顺序固定 `(B,S,Hq,Hv,K,V)`。调用参数都为 dense `BSND`、BF16 Q/K/V、FP32 gate/beta/state、`stateVFirst=false`、预激活 gate、无 QK 归一化、scale=1/sqrt(K)。最终 state 为 `[B,Hv,K,V]`，并不在 S 维保存所有中间状态。初始 state 采用非零输入且不允许原地修改。

分块路径分配所有必选输出，包括 `[B,Hv,S,C]` 的 Aqk/Akk；还导出累计 gate 和 final state。未导出的 W/U/QG/KG/VNew/H 传 nullptr，仍由库在私有工作区中计算，不能宣称这些私有量已逐元素验证。Aqk/Akk 包括上三角和尾部补零列，均有检查。每个输出先填 NaN，避免未写位置碰巧通过。

每例 CA 外部预算 180 秒，由主采集器串行执行；先校准两例，冻结模型与 holdout 预测，再执行第三例。原K=V16/S32泛化分支在180秒预算内未完成：只有辅助kernel `entry_kernel_2710` 的DB完整，主kernel `entry_kernel_1` 的DB无KernelInfo，并出现重复set_flag告警，不能计入完整耗时/精度。K16/S64与S48计划撤销并保存在manifest的retired_cases；改为最多先试K=V128/S16的arch35 key2尾块候选分支，预算仍180秒，首例失败则不继续该分支。

原六个预设 shape 不是广覆盖验证（chunk分支已在第二次预算失败后终止，仅recurrent保持活动），只建立这两条路径的最小闭环。模型不能据此声称覆盖更多 chunk key2执行组合/长度、多 batch/head、GVA、TND、state slot 或 raw-gate 分支。投机 8 对应 Q=9，**超过当前 RecurrentKda 的 S≤8 限制**；caller 明确拒绝，不能静默拆成 8+1。

## 数值公式及每个参数的含义

对一个序列及 value head，采用公开的 K×V state 布局：

\[
\bar H_t[d,e]=H_{t-1}[d,e]\exp(g_t[d]),
\quad r_t[e]=\beta_t\left(v_t[e]-\sum_d k_t[d]\bar H_t[d,e]\right),
\]
\[
H_t[d,e]=\bar H_t[d,e]+k_t[d]r_t[e],\qquad
O_t[e]=\alpha\sum_d q_t[d]H_t[d,e],\quad \alpha=K^{-1/2}.
\]

`t` 为本次序列 token，`d` 是 K 维坐标，`e` 是 V 维坐标；`H` 是历史状态，`g` 是已经激活的自然对数 gate，`exp(g)` 是每步保留历史的比例；`q/k/v` 为输入向量，`beta` 控制新信息的写入强度，`r` 是 value 与现有 state 预测之间的修正量，`alpha` 为注意力缩放系数，`O` 是输出。通俗地说，每来一个 token，先衰减旧记忆，再用预测误差修正记忆，最后用 query 读取更新后的记忆。历史长度压缩在 H 中，不能套用 FA 的上下文线性增长公式。

Chunk 使用同一个数学递推的分块形式。对块内 token i、j 和特征 d：

\[
G_{i,d}=\frac{\sum_{t=0}^{i}g_{t,d}}{\ln2},\quad
A^{qk}_{i,j}=\mathbf1_{j\le i}\alpha\sum_d q_{i,d}k_{j,d}2^{G_{i,d}-G_{j,d}},
\]
\[
L_{i,j}=\mathbf1_{j<i}\beta_i\sum_d k_{i,d}k_{j,d}2^{G_{i,d}-G_{j,d}},\quad A^{kk}=(I+L)^{-1}.
\]

`G` 是公开 `gkOut`，以 2 为底的块内累计 gate；`Aqk` 表示一个 query 能从当前及过去 token 读取多少信息；`L` 是严格下三角的 state 写入耦合矩阵；`I` 是单位矩阵；`Akk` 通过三角求逆消除同一块内 token 相互更新造成的重复计算。指示函数为条件成立时 1、否则 0。CPU 参考用 double 实现原始递推得到 O/state，并独立通过上述公式得到 G/Aqk/Akk；Akk 用前向代入求逆，无第三方库依赖。

输入具有正负交替、非恒定 beta/gate、非零 state，并在 CPU 上先还原 BF16 实际值。BF16 验收采用 `abs_err ≤ 2e-6 + 2^-6*abs(golden)`，比通用 BF16 标准更严格；FP32 为 `2^-16 + 2^-10*abs(golden)`。每个元素都必须通过并为有限数；另设 BF16/FP32 最大绝对误差 1/0.01 上界。小幅 value/state 控制了由块内 BF16 中间量带来的累计数值误差，未依据测量结果放宽容差。NaN、任何 bad>0 均失败，不允许只保留计时成功。

## 机制工作量与局部回归对照的适用范围

用户要求的默认方法应分离算子结构与硬件参数。下面的逻辑工作量用于机制模型；`a+b·work` 仅为本轮小样本经验回归对照，**不作为通用机制预测、不用于A7/A8默认外推**。通用方法以总建模策略中最终冻结的机制引擎为准。

递推逻辑工作量：

\[
W_{rec}=BSH_v(7KV+2V),\quad
w_{rec}=S(7KV+2V)\left\lceil\frac{BH_v}{P_v}\right\rceil,
\quad \widehat T_{rec}=a_{rec}+b_{rec}w_{rec}.
\]

`Wrec` 为整次调用逻辑标量操作数，不是 ISA 指令数；7KV 分别为 state 衰减 KV、预测矩阵向量乘 2KV、rank-one 更新 2KV、输出矩阵向量乘 2KV，2V 是 residual/beta 操作。`Pv` 为该 kernel 真实分配的有效 vector 核数，必须来自 launch/trace；括号上取整为最忙核任务轮数。`wrec` 为最忙核承担的工作代理，`arec` 为本条路由固定开销（同一 CA 时间口径，非 Python/ACL host 时延），`brec` 为每单位工作对应周期或纳秒，从两条校准样本估计并约束非负。单序列单 head 时，同一个 state 的 S 个 token 串行，不能除以整芯片 64 核。

分块阶段包括 Gate、Prepare、Post-WU、FwdH、Finalize。设有效 chunk 数 `M=ceil(S/C)`，逻辑任务数 `N=BH_vM`；对第 m 块，有效 token 数 `c_m=min(C,S-mC)`。可复用的工作特征为：

\[
F_{gate}=BSH_vK,\quad F_{pair}=BH_v\sum_m c_m^2K,\quad
F_{state}=BSH_vKV,\quad F_{inv}=BH_v\sum_m c_m^3.
\]

`Fgate` 表示 gate 累积元素数；`Fpair` 是 qk/kk 配对计算代理；`Fstate` 是状态读写及矩阵乘代理；`Finv` 是块内三角求逆工作代理。这些特征区分机制，不能拿六个参数拟合两个点。当前固定 C/K/V/B/H 的小样本仅允许受限两参数回归作为对照：

\[
\widehat T_{chunk}=a_{chunk}+b_{chunk}S.
\]

`achunk` 为本分支一次调用的固定阶段与同步开销，`bchunk` 为该固定 chunk 配置下每个有效 token 的经验增量；参数均由校准集非负拟合。通俗地说，先付一次启动和固定 tile 计算成本，再加处理更多有效行的代价。如果 kernel 按 C=64 固定 tile 执行，16/32 的耗时可能非常接近，拟合斜率会趋近零，固定耗时基线可能同样好。此时不能根据逻辑 FLOPs 强迫耗时线性翻倍；需要从 trace 判断哪些阶段仍执行完整 tile。若跨 64 token 到新 chunk、切换不同 K/V generic key 或增加 head，此模型立即超出校准域，不能用本轮低误差推断覆盖这些分支。

以上模型是受限路由代理，不是已辨识完整流水 DAG。两例校准无法分离每条流水的发起间隔与依赖延迟；读延迟不可直接当成每轮服务时间。部署预测只做常数规模算术，不重新运行 CA、不重新解析 trace。CA trace 可用于离线辨识路由和流水参数；新样本只验证，不重新拟合。

## 复现和来源

```sh
bash scripts/build-extension-kda.sh
# 仅编译。CA 执行由主采集器按 configs/extension-kda-v7.json 进行。
# binary 参数示例：build/smoke/extension_kda recurrent 1
#                    build/smoke/extension_kda chunk 16 128
```

实际调用 `toolchains/cann/9.2.1/cann-9.2.1` 的 ACLNN/ops-transformer 安装包。头文件、源文件及配置 SHA256 沿用 `docs/extensions/kda-moe-evidence.json` 的已核对证据；参考文档位于固定 `third_party/ops-transformer` 提交 `14d022b1a4c3df263bf697534e3c25c7c992ff5d`。参考知识卡 `ops/catlass-dsl/guides/operator_playbooks/linear_attention/kda/chunk_kda_fwd.md` 为 950/stable，但对应相关上游；不把其说明当成实际选中的 `.o` 身份。实际身份、tiling key、blockDim、CA 秒数和数据角色必须由本轮采集结果补齐。

## key2 与完整快流水的区分

固定参考设计 `attention/chunk_kda_fwd/docs/ChunkKdaFwd_design.md` §5.2–5.4 说明：tiling key2只检查C=64、K=V=128，不限制tail；dense FwdH另需BF16/dense/S%64=0，内联Gate另需raw FP32+A_log+useGateInKernel+safeGate，Prepare/Post-WU融合另需safeGate、dense-aligned、Hv偶数。本caller的S16/Hv1/预激活gate并不满足这些执行组合。r2实际出现`0001_entry_kernel_2.db`，和原K16的`entry_kernel_1`不同，但未获得完整主kernel DB/数值输出前不能宣称快路径验证成功。安装源码`arch35/chunk_kda_fwd_impl.h`也明确在useDenseFwdH=false时转RunGenericBackEnd；不能只按K/V命名fast。同步告警的具体根因未证实，不据此擅自修改库内核。

最终r2 S16候选同样在180.55秒达到墙钟预算，主kernel未完成、无六组完整数值检查，时延校准/验证均N/A。按预算停止本轮chunk分支；S32校准和S24留出没有执行，记录于manifest的retired_cases，不能算独立验证。Recurrent两条校准已完整通过，独立S4等待冻结后由主流程执行。
