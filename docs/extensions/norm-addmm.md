# RMSNorm 与 Addmm：首版结构模型及验证准备

## Summary

本轮新增两个 **未校准结构模型**：可以计算数学工作量、逻辑张量字节数和有限范围内的源码 tiling 候选；不能返回可信耗时。所有 `predicted_us`、利用率均为 `null`，时延校准样本数为0。已完成4项离线单元测试、真实ACLNN调用器编译，以及主线程追加的两个极小CA调用：RMSNorm `[1,128]`、Addmm `[M=2,N=16,K=16,self=bias]` 均通过全部输出正确性检查。它们没有用于拟合通用时延，因此没有预测误差百分比，也不把旧Matmul的校准搬到Addmm。完整采集身份和成本见 [smoke-950dt.json](smoke-950dt.json)。

| 极小验证 | 实际DB entry | CA指令窗 µs | 采集墙钟 s | 范围 |
|---|---|---:|---:|---|
| RMSNorm `[1,128]` | entry_kernel_1388 | 3.453939 | 40.74 | y的128元素和rstd的1元素均通过；0x1388=5000，与NORMAL源码候选key一致，只支持本点判断 |
| Addmm `[2,16,16]`，self `[16]` | entry_kernel_2 | 2.992121 | 38.80 | 完整32元素通过，本点只观察到一个kernel；entry 2不是全域唯一身份，未导出ELF |

调用入口：`operator_model.extensions.norm_addmm.plan(op, shape, **attrs)`，输出为可 JSON 序列化字典。目标为 **CANN 9.2.1 / Ascend950DT_9582 / DAV_3510**。来源身份详见 [证据清单](norm-addmm-evidence.json)。

## 库接口和实现身份

| 算子 | 实际调用的已安装接口 | 已确认库/源码信息 | 尚未确认 |
|---|---|---|---|
| RMSNorm | `aclnnRmsNormGetWorkspaceSize` / `aclnnRmsNorm` | `include/aclnnop/aclnn_rms_norm.h`；`opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_nn/rms_norm/RmsNorm_normal_all_bf16_high_performance.o` | 除已测[1,128]之外的dispatch、指令次数与时延泛化 |
| Addmm | `aclnnAddmmGetWorkspaceSize` / `aclnnAddmm` | `include/aclnnop/aclnn_addmm.h`；ops-nn `matmul/mat_mul_v3/op_host/op_api/aclnn_addmm.cpp` 有多种计算图分支 | CANN 包是否与此开源 revision 完全一致；实际 kernel 序列及其二进制身份 |

表中 SDK 相对路径根为 `/home/l00925865/operator_sim/toolchains/cann/9.2.1/cann-9.2.1/`。开源实现固定在 `third_party/ops-nn` commit `602e4ad985da00aad28c16432841e9c5069e10de`，不是以当前网站最新版冒充已安装二进制。后续每例应记录实际下发的所有 kernels、object hash、tiling 和 launch 数。

此次遵照 `knowledge-query` 搜索并阅读全文阅读 `rms_norm_950.md`（CANN 9.1，旧源码 revision）。发现固定源码新增 transpose/SplitAR 路径，所以实际模型以固定源码为候选依据，不照搬旧卡的完整性结论。Addmm 搜索仅返回无关模型卡；不声称知识库已覆盖 Addmm，使用本机 SDK 头文件与固定源码补证。

## RMSNorm 方法与公式

初始范围是连续 ND 输入 `x[R,C]`、`gamma[C]`，沿最后一维归一化；输出 `y[R,C]` 和 FP32 `rstd[R,1]`。输入支持 BF16/FP16/FP32，gamma 可同 dtype 或 FP32。初版要求 epsilon 有限且大于 0（保守子集），默认 $10^{-6}$。

\[
r_r=\left(C^{-1}\sum_{j=0}^{C-1}x_{rj}^2+\epsilon\right)^{-1/2},\quad
 y_{rj}=x_{rj}r_r\gamma_j.
\]

数学乘法数为 $R(3C+1)$：平方 $RC$、乘均值因子 $R$、输出两次乘法 $2RC$。数学加法数为 $RC$：归约 $R(C-1)$ 加 epsilon $R$；另列 $R$ 次 rsqrt，不把它按一次 FMA 计。因此不含 rsqrt 的标量 FLOPs 为 $R(4C+1)$。这些量不是 VF/ISA 指令数；向量掩码、归约树、精度转换和寄存器操作必须另外校准。

若 x、gamma 的单元素字节数分别为 $b_x,b_g$，逻辑张量字节数为

\[
B_{logical}=2RCb_x+Cb_g+4R.
\]

其中 x 读、y 写、一次逻辑 gamma 读和 rstd 写分别占上述项。每核重复 gamma 加载、cache、对齐和中间缓冲不在此量里，不能直接当成 HBM 实测流量。

当前仅移植 `REGBASE_NORMAL`（源码 key 5000）的 tiling 候选。SDK 配置提供 $P=64$ AIV、UB $U=253952$ B、block 32 B、FP32 向量宽度 $V=64$。设

\[
f=\lceil R/P\rceil,\quad P_{used}=\lceil R/f\rceil,\quad
 C_a=\operatorname{align}(C,32/b_x),\quad F=2^{\lfloor\log_2(C_a-1)\rfloor},
\]
\[
L=\operatorname{align}(\lceil F/V\rceil,32),\qquad
u=\left\lfloor\frac{U-1024-2048-C_ab_g}{4C_ab_x+12+L}\right\rfloor.
\]

$1024$ B 是 host 提前扣除的 UB 空间；$2048$ B 是 arch35 实现保留量（常量名 `RETAINED_SIZE_1K`，实际值为 2048，不能按名字误读）。分母的 $4C_ab_x$ 表示 x/y 两组 double buffer；$12+L$ 对应归约辅助空间，严格保持源码的字节算式。gamma 为每核常驻一行。`C_a≤16384` 且 $u≥1$ 时，每次行 tile 为 $t=\min(u,f)$，最长核迭代数 $\lceil f/t\rceil$，尾核行数 $R-(P_{used}-1)f$。

极短列、每核行数大于 512 的新 transpose 路径，以及 full-row 不适用时的 SplitD/SplitAR 路径，当前返回 `unresolved_dispatch`，不会继续套 full-row 分核数。模板开关和 deterministic 配置也会影响真实调度。示例 `[65,5120]` 源码候选为 33 个 AIV、前 32 核各 2 行、尾核 1 行；该数字仍不是 CA 观测。

以后校准时需要把读响应延迟 $L_r$ 与发起间隔 $I_r$ 分开，按实际 double-buffer 依赖得到稳态间隔。一个待校准表达形式是

\[
T=T_{launch}+\max_c\{T_{fill,c}+(n_c-1)I_c+T_{drain,c}\}.
\]

只有看到指令事件、依赖和独立 shape 验证后，才确定 $I_c$ 的资源约束与参数。目前此式为校准设计，没有用于输出数值耗时。

## Addmm 方法与源码分支

`shape=(M,N,K)` 表示 `mat1[M,K]`、`mat2[K,N]`；`self_shape` 可为标量、`[N]`、`[1,N]`、`[M,1]`、`[M,N]` 等合法右对齐广播形状。当前是连续 ND、同 dtype 输入输出、独立 output，非 WeightNz/非 inplace。

\[
Y=\beta\operatorname{broadcast}(S)+\alpha AB,\qquad F_{mm}=2MNK.
\]

按算法习惯用 $2MNK$ 计算 GEMM FLOPs（包含累积的常见近似），另外列 $MN$ 合并加法，以及 alpha/beta 非 1 时的 $MN$ 级数学缩放。beta 缩放广播前 self 可能更便宜，因此这些是稠密代数工作上界，不是实际 kernel 计数。逻辑张量字节数为

\[
B_{logical}=b(MK+KN+|S|+MN).
\]

省略了中间张量读写、dtype cast 和多 kernel 启动；alpha/beta 为零时可能跳过整个分支，当前工作量仍明确标为稠密上界。

固定源码中的优先次序包括：空 tensor/零系数、`NeedToConvertBias`、满足条件的 inplace FP32 GemmV3、DAV_3510 FusedMatMul(add)、一般 Matmul + Muls/Add/Axpy 组合。FusedMatMul(add) 的主要门控是：AIV:AIC=2:1、alpha/beta 均接近 1（FP32 epsilon）、self 必须是完整 `[M,N]`、同 dtype ND；FP32 只接受 cubeMathType=3。**[N] bias 不满足这个融合分支。** 上游 bias 特化有更高优先级，且已安装包可能不同，因此 `fused_add_source_predicates_match=true` 仍不等于运行中一定 dispatch 到该分支。

接口返回 `matmul_only_candidate` 时仅提供已有 Matmul 工作划分作为后续排查线索；`candidate_is_actual_addmm_mapping=false`，不代表 Addmm 的实际分核和融合，不使用其已校准耗时。

## 独立验证计划与误差分析边界

| 算子/shape | 目的 | 目前状态 |
|---|---|---|
| RMSNorm `[1,5120]`, `[9,5120]` | decode 低行数、Q=9 验证 | 结构模型，未采集 |
| RMSNorm `[65,5120]` | 分核尾数变化 | 单元测试通过，未采集 |
| RMSNorm `[128,7168]` | 常见较宽隐藏维 | 结构模型，未采集 |
| Addmm `[9,128,64]`, self `[128]` | 小 decode/broadcast 路径 | 编译就绪，未采集 |
| Addmm `[9,128,64]`, self `[9,128]` | 同 shape 对照完整矩阵融合候选 | 编译就绪，未采集 |

这些是代表性接口验证形状，不宣称是某 checkpoint 的真实命中。与 128k/512k KV 长度不同，RMSNorm 的 $R$ 应由实际 chunk 或本次 query token 数确定，不能把总 KV 长度直接当作当前归一化行数。

初始误差风险按证据等级区分：已证实的是源码分支差异、UB 保留量和不同 self 广播语义；尚未证实的是其他shape的二进制分支、launch 头开销、多核尾波、实际流水重叠与 cache 重用。这些缺口需要完整的库调用链 trace；Addmm 一次 API 若有多个 kernels，不能仅取第一个 kernel 对齐总调用耗时。

## 验证资产与仿真预算

- `python3 -m unittest discover -s tests -p test_extension_norm_addmm.py -v`：4 项通过，覆盖尾核、full-row 不适用、Addmm 广播字节与融合候选、非法输入。
- `bash scripts/build-extension-norm-addmm.sh`：成功，C++17，链接本机 `libopapi/libnnopbase/libascendcl`。编译通过不表示算子已执行。
- `smoke/extension_norm_addmm.cpp`：真实 `aclnnRmsNorm`/`aclnnAddmm`，无自定义替代 kernel。BF16 输入先截断后用于 FP64 CPU golden；RMSNorm 同时比较 y/rstd。按 `ops-precision-standard` 使用 BF16 atol/rtol=$2^{-6}$、硬绝对误差上限 1，FP32 rstd atol=$2^{-16}$、rtol=$2^{-10}$、上限 0.01；smoke 要求所有元素通过，比 99% 门限更严格。
- 后续可用的参数：`rmsnorm 9 5120`、`addmm 9 128 64 bias`、`addmm 9 128 64 matrix`。仅编译，没有执行这些命令。所有 CA 由主线程统一调度；本子任务 CA 用时和 trace 新增量均为 0。
