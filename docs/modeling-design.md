# 昇腾算子建模方案草案

本项目服务于模型系统仿真：以有限的离线 CA Model 采样，构建可快速查询的算子耗时模型，并支持 A5 到 A7、A8 的条件性外推。建议采用 tile 级资源与依赖模型，结合真实 tiling、分核策略和少量残差校准。耗时是主要预测目标，利用率是派生指标。

方案版本为 v0.2，更新于 2026 年 11 月 17 日。本文中的采样预算、精度和速度数值均为建议目标，不是已测结果。A7/A8 的硬件参数为场景输入，不代表已核实产品规格。本轮只更新计划，并按追加授权安装 CANNBOT；CANN 安装、算子编译、CA 采样与模型实跑均留待执行阶段。

## 对原始设想的判断

| 原始设想 | 判断及调整 |
| --- | --- |
| 从三个模型实跑抽取算子 | 正确。以设备端 kernel 及其实际调用参数为单位；模型名称本身无法唯一确定算子集合。 |
| 使用 A5 的现有算子和 CA Model | 正确。必须同时固定 SoC、编译器、算子源码/二进制、tiling 和 CA 版本。 |
| 单核利用率加分核轮次得到总耗时 | 可作起点。需补充尾块、异构任务、核间依赖和共享带宽；不要把平均核时间代替最慢核完成时间。 |
| Vector 指令数推导理论时间，再除以 CA 时间 | 可以定义执行效率，但要区分指令类别、repeat/mask、吞吐与延迟、依赖、流水和有效工作量。 |
| Cube 使用同样方式，并忽略内部访存 | Cube 可用相同框架。内部访存可以简化为搬运阶段，但不能全部删掉，否则会把供数不足误归因于 Cube 效率。 |
| C/V 分别计时后按硬件比例外推 | 在同实现、工作量和充分并行等条件下可作基线。正式模型要重排依赖和资源，重新判断瓶颈。 |
| 每个算子与 CA 校准并迭代 | 正确。必须预留不参与迭代的测试集，区分拟合 CA 的误差、CA 对真机的偏差、跨代外推误差。 |

## 平台与工具的证据

用户已指定 A5 使用 950DT。vLLM Ascend 的官方 [Dockerfile.a5](https://github.com/vllm-project/vllm-ascend/blob/main/Dockerfile.a5) 在 11 月 16 日读取的内容使用 `ascend950dt_9582` 和 `quay.io/ascend/cann:9.2.1-950-ubuntu22.04-py3.12`。这只是已观察到的配套参考，不代表已确认的最新安装包。正式实验应锁定具体 SKU、源码 commit、软件包版本和校验值，不能持续跟随 main。

Ascend 官方 [msprobe 构建脚本](https://github.com/Ascend/msprobe/blob/master/build.sh) 将 950 映射到 `dav-3510`。`dv120` 未找到可靠映射。产品名、算子库编译目标、底层 ISA 架构和仿真器目标属于不同层次，不能互换，也不能仅凭这个架构映射确定具体 SKU 的核数和频率。

本机为 aarch64、openEuler 22.03，已安装 CANN 8.5.0。已枚举的 simulator 目录没有显式 950DT/950PR 名称；部分目标使用内部名称，尚不能认定它们对应 A5。`npu-smi info` 在本次执行环境初始化失败，不能据此判断物理机器无 NPU。现有环境尚未通过 A5 仿真 smoke test。

官方 [msOpProf 仿真指南](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_simulator_user_guide.md) 说明：950 支持仿真流水分析；可观察指令执行次数、执行周期及同步关系，A5 需显式选择 `--soc-version`。但 950 的部分直接统计指标不支持，包括热点视图的 Vector Unit Utilization、UB Bank Conflict、Process Bytes，以及 PMSampling、Overhead 选项。应以实际安装版本的功能探测为准。

官方 [仿真数据格式](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_simulator_performance_data.md) 给出 `core*_instr_exe.csv`、`core*_code_exe.csv` 和 `trace.json`。前者包含 `instr/pipe/call_count/cycles/detail` 等字段；时间线用于分析重叠。指令周期求和不是 kernel 墙钟时间，汇总文件与逐核文件也不能重复计数。对 950 缺失的字节数等字段，优先从 tiling、指令参数和布局推导；无法推导的量应标为未知。

安装策略按用户要求改为用户目录安装：计划前缀 `/home/l00925865/operator_sim/toolchains/cann/<version>/`，选择执行时官方可获取、支持 aarch64 和 950DT CA 的最新兼容版本。精确 9.x 版本从官方包清单核实，若最新版不能兼容目标则记录具体原因，不静默降级。安装前核对 openEuler、glibc、Python、编译器、profiler 和 ops 配套；采用独立环境脚本，不修改系统 CANN、驱动和全局 shell。先跑 dynamicMXquant、matmul、FA 各一个小案例并验证输出正确性与 trace，再扩展。CA 名称不等于所有内存、启动和通信行为都达到真机精度，需记录功能/时序仿真模式、建模范围和时钟。

## 算子范围与真实工作负载

三个模型分别维护工作负载清单。固定 checkpoint revision、模型配置、vLLM/vLLM Ascend/TorchNPU commit、量化与 KV dtype、TP/EP/DP/PP/CP、prefill/decode、图模式、推测解码、融合开关、batch/context 分布和缓存状态。Kimi K3 与 DeepSeek V4.1 Flash 的视觉分支作为独立可选范围；首批建议文本推理，不能把这个子集宣称为完整多模态覆盖。

### 已确定的工作负载范围

- MTP 草拟长度 d 覆盖所有整数 0 至 16，主模型每请求校验长度 q=d+1，覆盖 1 至 17。记录草拟次数、实际提交校验长度、接受长度及拒绝后的状态处理；主模型与 draft/MTP 路径分别计时。
- decode 和 prefill 都覆盖。Decode 包括普通 q=1、MTP 校验 q=2 至 17，以及同一批内 q 不同的 ragged 情形；图模式若执行 padding，另存有效 q 与实际执行 q。Prefill 的 q 是实际 prompt/chunk 长度，不受 17 限制；chunked prefill、带已有 KV 的 prefill 和混合批次分别记录。
- EP 候选规模固定为 `4, 8, 16, 32, 64, 72, 128, 144, 256, 288, 384, 512, 576`，不将非 2 次幂规模取整。
- 并行组合支持 attention DP + MoE EP（用户所称 DEP）、attention 按 head TP + MoE EP（用户所称 TEP），并允许 MoE TP+EP 或纯 TP 与 attention 策略组合。为避免命名歧义，实际配置记录 attention TP/DP、expert TP/EP/DP 及各 rank group 的映射，不从 DEP/TEP 缩写推导 world size。

本地 shape 由 rank 映射器生成。attention TP 的 query heads 按合法切分规则计算，KV heads 可能切分或复制，不能直接把全部 KV 流量除以 TP。MoE EP 决定 expert 放置与 token dispatch；MoE TP 决定各 expert 矩阵的具体切分维度及归约。每层存本地 expert 集合、每 expert token 数、路由 top-k、共享专家及 padding。平均分布仅作一种场景，还要覆盖空 expert、倾斜和尾 rank。

例如同一 attention replica 中 B 个请求都校验 q 个 token，token-wise 算子的有效 M 通常为 B×q；ragged 时是 sum(q_i)，随后再考虑真实 token 分片及 padding。head TP 通常不把 token 数除以 TP，EP dispatch 后各 expert 的 M 也不再等于这个总数。并行规模不是 kernel shape 的直接除数。

在校验集生成层面覆盖全部 EP 和 q，昂贵的 CA 测量按最终本地 kernel 签名去重。相同本地 shape、布局、分支和资源条件可以复用算子样本；通信拓扑、远端等待和并发背景不同则由系统层保留，通信融合 kernel 不适用简单去重。合法性检查覆盖 head/维度可分性、expert placement 和运行时限制；不合法组合标为不支持，不能伪造实跑样本。

已取得三个官方模型 config 并固定 revision，见 `model-configs.md` 和 `configs/models/sources.json`。完整 checkpoint 不作为 shape 推导或 CA smoke case 的前置条件；hidden size、head 数、expert 数及矩阵维度从这些配置读取，再结合推理实现与请求场景生成本地 shape。TP 候选首批可用 1/2/4/8/16，再按配置过滤；不将它们宣称为已获三个模型支持的部署组合。

官方教程提供了这些候选路径：

- [DeepSeek V4.1 Flash](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/DeepSeek-V4.1-Flash.md)：CSA2、低比特 KV、mHC、Engram 和 DSpark；950DT 有官方 MXFP checkpoint 路径。
- [Kimi K3](https://docs.vllm.ai/projects/ascend/en/v0.30.0/tutorials/models/Kimi-K3.html)：KDA 与周期性 Gated MLA、LatentMoE，视觉分支另计。
- [DeepSeek V4 Pro](https://docs.vllm.ai/projects/ascend/en/v0.30.0/tutorials/models/DeepSeek-V4-Pro.html)：混合压缩 attention、mHC、MoE；950DT 的原始 checkpoint 涉及混合 MXFP 精度。

这些是候选算子族的依据，并非本项目的实跑清单。首批应覆盖 GEMM/量化 GEMM/grouped GEMM、norm/activation/quant、稀疏或压缩 attention/indexer、KDA、MoE 路由与置换、mHC、KV 管理。Engram、通信融合和视觉路径根据实际配置纳入。

抽取流程：静态源码梳理候选调用 → 固定配置运行代表请求 → 通过框架与设备 trace 对齐实际 kernel → 记录 tilingKey、blockDim、布局及参数 → 独立 replay kernel。不能为了获得清单而在 CA 中仿真整个巨型模型。若没有实跑条件，静态清单标为候选，等待后续 trace 确认。

算子实现不限于 ops-transformer 和 ops-nn，还可能来自 ops-math、vLLM 自定义 C++/Ascend C、TorchNPU op-plugin、Triton Ascend、FLA 等。官方 [ops-transformer](https://gitcode.com/cann/ops-transformer) 列出 A5 的 attention、delta-rule、mHC、MoE 和通信融合实现，并要求 CANN 与源码版本匹配；[ops-math](https://gitcode.com/cann/ops-math) 提供基础数值算子。最终以调用链为准。

一条样本的身份至少包含：语义算子、实际 kernel、源码/二进制 hash、SoC、dtype、layout/stride、shape、属性、tilingKey、tile 参数、blockDim、缓存条件和软件版本。动态 MoE 要记录每个 expert 的 token 数及倾斜，attention 要记录有效长度、mask、稀疏索引/页布局，不能只存张量外形。

## 预测模型

推荐分四层：kernel 与 tiling 选择、tile 资源需求、资源受限的依赖调度、多核映射。CA 用来离线标定这些层，在线系统仿真不调用 CA。

### 利用率定义

对资源 r、核 c，用同一个 kernel 设备时间窗口 T 定义：

- 忙碌占比：`U_busy(r,c) = busy_interval_union(r,c) / T`。用区间并集，不把重叠指令延迟相加。
- 有效工作效率：`U_useful(r,c) = useful_work(r,c) / (peak_rate(r,c) × T)`。纯 Cube 可用有效 FLOPs，异构 Vector 指令应分别报告类别效率或定义经过说明的理想调度时间比。
- 执行工作效率：将上述 useful_work 换成 executed_work，计入 padding、无效 lane 和额外计算；二者差异反映算法/映射浪费。
- 核覆盖率：被分配任务的物理 AIC/AIV 数除以可用数；负载均衡另报告最慢核、平均核及各核任务数分布。
- 芯片有效效率：`sum(useful_work) / (total_peak_rate × T)`，分别对 C、V 等资源定义。不能把 AIC 与 AIV 简单加成一种“核”。

用户提出的 `ideal_time / CA_time` 可保留为一个指标，前提是 ideal_time 的资源、指令模型、工作量及时间窗口严格定义一致。它不自动等于硬件流水忙碌率，也不能作为跨代不变的常数。

### Vector 成本

源代码操作次数不等于动态机器指令次数。优先从 CA 指令统计与反汇编建立动态指令类别计数，再由源码和 tiling 推导同类 shape 的计数。需要处理 dtype、repeat、有效 mask、向量长度、访存步长、寄存器 load/store、reduction、exp/reciprocal、cast、标量控制和同步。

对每个实际硬件流水 p，定义归一化服务工作量 `W_p` 和吞吐 `q_p`，资源下界为 `D_p = W_p / q_p`；依赖链给出另一条下界。tile 时间至少满足 `T_tile >= max(max_p D_p, T_dependency_lower_bound)`。实际预测使用流水和依赖调度；相同硬件流水上的工作累加，可并行流水通过依赖约束重叠。

指令 latency 决定结果何时可被依赖者使用，issue interval/throughput 决定连续独立指令的服务速率，二者不能混用。微基准分别测独立指令流、依赖链、repeat/mask、混合流水和搬运，不能用一个每指令平均成本覆盖全部 Vector 指令。

### Cube 成本

同时保存有效 FLOPs（普通 GEMM 为 2MNK）与实际执行的分块矩阵工作量。按 dtype、矩阵块尺寸和指令吞吐计算 Cube 服务时间；计入 M/N/K 尾块、K 分段、split-K、归约、量化/反量化和 epilogue。理论峰值必须与真实精度、dense/sparse 口径匹配。

访存采用分层简化：HBM/L2 的有效流量与共享带宽、L1/L0/UB 的搬运阶段及启动延迟、缓冲容量与双缓冲约束。第一版可暂不模拟每条 cache line 或 bank，但保留 MTE、FIXPIPE 及必要寄存器搬运阶段。单独的 compute-only 结果可输出为乐观界，不能当作完整 kernel 时间。

### CV 融合成本

为一个代表 tile 构建阶段 DAG，如加载、QK Cube、softmax Vector、PV Cube、输出与同步。按实际实现加入前后 tile 依赖、缓冲区复用、跨 AIC/AIV 同步和资源容量。

节点调度的基本形式为 `start(v) = max(predecessor_finish, resource_ready, buffer_ready)`，`finish(v) = start(v) + duration(v)`。流水吞吐资源应允许按 issue interval 接受下一任务，依赖结果仍按 latency 就绪；模型粒度须与这一语义一致。

规则循环可压缩为 `T_core ≈ T_fill + (n_tiles - 1) × II + T_drain`。`II` 由资源竞争和循环依赖共同决定，不是永远等于 `max(t_cube,t_vector)`。不规则 attention/MoE 使用少量 tile 类型的事件调度或分桶调度。在线不回放每一条机器指令。

CA 上 C/V 的 busy 区间、首尾跨度、等待和关键路径贡献是不同统计量。时间线中两个流水的时间和可能超过 kernel 总时长；重新计算目标架构时，不能把源平台观测到的等待作为固定工作量再次加进去。

### 多核成本

从真实 host tiling 与调度公式得到核任务，不强行设置 `blockDim=1` 来代表真实运行。那样可能改变 tile、kernel 分支、缓冲和 CV 协作关系。优先分析正常多核运行的逐核 trace；若隔离单核，必须保持相同 tile 和分支，并记录实验差异。

对等长独立任务，`ceil(N_tiles / P)` 的轮次公式可作为基线。正式模型按核队列求最后完成时间，保留尾块、空核、负载倾斜和同步。计算资源允许本地聚合，共享 HBM/L2/MTE 路径需通过全芯片带宽约束或简化争用模型耦合；只用单核测量无法识别全芯片争用。

设备 kernel 时间与 host enqueue、host tiling、图启动和通信分开。CA 进程在 CPU 上运行了多少秒与模拟设备执行时间无关。系统仿真若已建模 host/通信，只计一次；通信计算融合则暴露依赖和资源占用，不能把整个融合 kernel 当纯计算再叠加一遍通信。

## 跨代外推

将硬件参数显式化：AIC/AIV 数量和配对关系、各时钟域、按 dtype/指令类别的每周期吞吐及延迟、MTE/FIXPIPE 能力、HBM/L2 带宽和容量、L1/L0/UB 容量、同步与启动、芯粒拓扑。A7/A8 未知参数应保留为场景输入并报告敏感度，不能从 A5 单平台观测中唯一推断。

源数据可移植的是语义工作量、依赖结构和在明确条件下不变的指令/分块需求；源平台的争用等待、利用率、最优 tiling 和机器二进制不能默认可移植。

提供两种外推模式：

1. 固定算法和 tile 模板，按目标资源重新映射，估计同实现策略的收益；这里不意味着 A5 二进制可直接在目标上执行。
2. 在目标容量和指令约束下重选 tile、分核和融合策略，估计可实现的调优潜力。没有目标编译器/算子实现时，该结果是条件性估计。

按用户给定假设，若 32→120 指 AIC 数，C:V 数量比从 1:2 到 1:1，且 C/V 都按 1.65→1.8 GHz 变化、每核每周期吞吐不变，则：

`S_C = (120/32) × (1.8/1.65) = 4.0909`

`S_V = (120/64) × (1.8/1.65) = 2.0455`

对工作量不变且充分并行的资源服务需求，`T_C7 = T_C5 / S_C`，`T_V7 = T_V5 / S_V`，所以 `T_V7/T_C7 = 2 × T_V5/T_C5`。若 A5 原来是 V 限制，在这个两资源近似下 A7 仍是 V 限制；条件是 T_C5/T_V5 表示服务需求而非包含等待的首尾跨度。

算子加速并不因此必为 2.0455 倍。例如仅有 16 个独立 tile、原来已全部并行时，新增核基本不带来收益，主要收益来自频率或单核能力。带宽瓶颈、CV 串行依赖、分核变化及每核能力变化也会改变结果。

没有 A7/A8 仿真器或硬件时，输出场景区间及其假设，称为敏感性分析区间，不能当作经过目标平台验证的统计置信区间。若有其他代际数据，用留出整代的平台验证迁移流程；它也不能证明更远代际已准确。

## 采样与校准开销

先以 tilingKey、算法分支、dtype/layout 和资源瓶颈划分区间，再采 shape。重点采 tile 边界前后、核轮次变化、buffer 容量临界点、K 分段切换、短/长序列、尾块和 MoE 倾斜。使用实际流量加权的常见 shape，再补少量异常边界，避免所有维度的笛卡尔积。

首轮只选择 dynamicMXquant、matmul、FA 三个算子族。预设最多 96 个不同 CA 案例：dynamicMXquant 24、matmul 24、FA 36，加共享微基准 12。三项 smoke test 和首批计时案例包含在预算内，失败尝试和重试另外消耗同一运行时间预算，不因失败无限扩张。该数量是启动上限而非覆盖承诺；不足时明确缺失的分支，再决定下一轮扩展。

先从预算中取 12 个代表案例估计仿真时间和 trace 大小。建议第一轮 CA 总预算为墙钟 2 小时、累计 8 CPU 小时、原始 trace 10 GiB，任一先达到就保存结果并停止扩展，报告未覆盖区域。快采样每例预设 120 秒超时；超时记为截尾失败，不能把超时值当设备延迟。必要的慢例进入独立诊断队列，单例最多 10 分钟且仍计入总预算。安装和编译时间单独核算，预算数值需以首批实测调整，不能承诺所有 shape 都在该时间内完成。

分组留出测试集后再主动补样；优先补影响总时间最大的分支，而不是平均扩张全部 shape。只在工具明确支持隔离并发时并行 CA 作业，每个作业独立输出与端口，并受 CPU/内存/IO 上限约束。不能把线程数提高当作仿真速度优化的主要手段。

大批案例优先收集可用的轻量统计，对有误差和新分支的案例才保存完整 trace。如工具不能独立关闭昂贵 trace，不能假定轻量模式必然省时。等价 tile 只做代表性细测，但保留少量完整多核运行验证争用。

首选有物理意义的系数与分段插值。只有解析模型在区间内留下稳定残差时，再引入小型回归/MLP；模型输入是资源需求、tile 数、尾块、工作集、分支等特征，不是仅靠算子名加全局 shape。未知区域输出外推标记，不静默套用最近样本。

主动补样优先级按出现频率、时间贡献、预测分歧和参数敏感度确定。缓存键覆盖源码/二进制、软件版本、SoC、shape/属性、tiling、输入分布和缓存状态，任一关键项变化即失效。

### 三个首轮算子的具体计划

| 算子 | 首先识别的实际执行结构 | 优先采样边界 | 快速模型 |
| --- | --- | --- | --- |
| dynamicMXquant | 分组读取、归约/amax、scale 计算、转换/打包、数据与 scale 写回；按实际 API 确认语义 | 小 M、量化组长度、组尾、向量宽度、tile 容量、核轮次；axis、输入/输出 dtype、round/scale 模式 | 各指令类别的分段服务成本，加首尾阶段、搬运及核任务聚合 |
| matmul | 真实 M/N/K tile、K 循环、装载、Cube、FIXPIPE/epilogue | decode 的瘦 M、prefill 的大 M、N/K 对齐及尾块、K 分段、split-K | 每种 tile 的流水包络，加完整 wave 和尾 wave 的聚合 |
| FA | QK、mask/softmax、PV、搬运、buffer 复用与同步 | q=1 至 17、prefill q/chunk、KV 长度、head dimension、GQA、页边界、split-KV/归约 | 有限阶段 DAG 折叠为周期模板；对 request/head/tile 类型做数量聚合 |

dynamicMXquant 将用户名称映射到固定版本中的实际接口，不能预先认定 V1/V2/V3 成本相同。[官方接口文档入口](https://gitcode.com/cann/ops-nn/blob/master/quant/dynamic_mx_quant/docs/aclnnDynamicMxQuantV3.md) 列出了 axis、量化目标、blocksize、scale 模式等参数；精确合法取值待所选包确认。FP4/FP8 与 scale 布局分开建模；零值、极端值等输入先作正确性检查，仅在执行路径随数值改变时增加性能分组。

matmul 的基线先选支持的普通 dtype 和明确 epilogue，以隔离 Cube 成本；随后用实际 MX 量化矩阵路径验证复用范围。grouped matmul 是 MoE 的必要扩展，不能因为单 matmul 达标就认为 MoE 已覆盖。

FA 首先验证标准 attention 的建模方法，但三个模型的实际 sparse/MLA/CSA/KDA 路径要各自识别，不能把标准 FA 校准误差直接作为这些路径的精度。每请求 q=1 至 17 均进入 workload 测试；初始昂贵采样可取 q={1,2,4,8,16,17} 加实现阈值两侧，未测 q 由同分支解析模型预测并标识覆盖级别。prefill chunk 与 context 初始按对数分布取点，再由模型配置和真实 workload 收敛。

### 在线仿真速度

以单个算子调用接口为核心：输入是本地 kernel 签名、硬件配置与必要状态，输出包括时间、C/V/搬运服务需求、核数与轮次、利用率、模型版本及覆盖标记。提供批量查询，避免每个 layer/rank 都跨进程或读取文件。

1. 精确签名缓存：同样的 kernel/shape/架构/状态直接返回；保留出现次数，模型层汇总重复 layer/rank。
2. 分支内解析快路径：对重复 tile 使用填充、稳态、排空闭式公式。多核用完整轮次与有限尾块类别聚合，复杂度随 tile 类型及核映射复杂度增长，不随全部指令数增长。
3. 有界诊断路径：复杂 CV 只对小规模代表依赖图求周期与瓶颈，再压缩；线上不展开所有 KV tiles。未见分支返回明确低可信/不支持标记，或显式标注的保守界，不能在线启动 CA 或悄悄退化成高可信 Roofline。

矩阵分块、规则因果 mask 的完整块/边界块数量采用公式计数。跨代重选 tiling 在离线或硬件配置首次加载时，仅搜索按容量约束筛选的小候选集合并缓存，不在每个调用中重新穷举。ragged 请求与 expert 负载优先使用精确直方图；如果压缩长度桶，单独校验压缩误差并保留长尾，避免隐藏最慢 rank。

验收同时记录冷启动、缓存命中/未命中、批量吞吐、P50/P95 查询耗时、缓存内存上限和每轮模型仿真的额外墙钟时间；不存在仅凭算法复杂度就认定速度达标的步骤。

## 验证与验收

每个案例先过数值正确性。按 shape 区间、tiling 分支、工作负载或平台分组留出验证/测试数据，避免相邻 shape 随机划分形成数据泄漏。可预设约 60/20/20 的组划分；最终测试集不参与调参，测试失败后修改模型应使用新留出集复验。

比较三个基础模型：原 Roofline、带单核效率及轮次的基线、tile 资源依赖模型。逐项消融头尾开销、尾块、多核调度、内部搬运、CV 依赖和残差项，只有确实改善留出结果的复杂度才保留。

建议第一版目标：CA 留出集的工作负载频次加权 WAPE 不高于 10%，P90 相对误差不高于 20%；它们需要按算子族和 prefill/decode 分别报告。小 kernel 同时报告绝对误差，不让相对误差失真掩盖总体结果。

加权指标采用 `sum(freq_i × abs(pred_i-ref_i)) / sum(freq_i × ref_i)`。端到端还需报告实际阶段 DAG 的 makespan 偏差，不能只对 kernel 时间求和，也不能让正负偏差互相抵消后宣称逐算子准确。

快路径目标为已缓存 shape 查询约 10 微秒以内、未缓存解析评估约 1 毫秒以内；正式接入前以系统仿真的总额外开销为最终约束，例如不超过原仿真预算的 5%。这些数字尚未实测，并且与语言、调用接口和机器有关。

误差诊断按：kernel/tiling 是否匹配 → 指令工作量是否正确 → 资源服务成本 → 依赖和重叠 → 尾块与核负载 → 共享带宽与缓存 → CA 覆盖范围。避免直接给每个 shape 增加一个自由拟合系数。

CA 作为第一阶段参考是合理的。若以后能用真机，对代表案例单独建立 CA 与真机偏差报告，不能把“对 CA 误差小”写成“对真机准确”。

## 相关研究的取舍

### CANNBOT 与 OpModel 的复用评估

本地获取的官方 CANNBOT 源码中，`ops-perf-optimize/skills/ops-opmodel-toolkit` 已包含 YAML 模型、分核、DAG、流水、硬件 profile 与搜索接口资料。实施前优先做能力核查，复用能覆盖本项目的结构和解析结果，避免先行重写整套事件模拟器。CANNBOT 知识库用于查询 950 的硬件机制、API、算子实现和来源，知识卡不代替 CA 校准数据。

根据当前读取的技能文档：普通完整 case 采用 SimPy，物理模型另有后端和约束；部分入口不支持任意硬件覆盖，示例中的共享服务参数存在合成假设。因此，OpModel 暂定位为离线建模与诊断候选后端，不能直接认定可满足在线查询速度或 A5 到 A7/A8 的外推要求。

首轮能力检查按四项决定复用范围：是否能输入真实 tiling/分核及分类指令成本；是否表达 C/V 与搬运依赖及共享资源；是否能在同一工作量口径下覆盖目标硬件；是否可导出轻量流水模板供批量快速查询。缺失的部分通过本项目适配层补充，若接口或性能不满足，则保留自建轻量解析路径。所有检查和示例仿真在执行阶段进行，本轮仅安装与阅读。

安装核查补充：OpModel 技能文档已安装，但 Python 3.12/aarch64 环境在 PyPI 及官方安装器默认华为云镜像均未找到可安装的 `opmodel` 发行版。尚不能判断是发行渠道还是平台支持问题，不能将文档能力表记为本地运行能力；它不作为本项目开始建模的硬依赖。

下列研究提供设计参考，论文中的 GPU 结果不能直接作为本项目的昇腾准确率承诺。

| 研究 | 可吸收的思路 | 使用边界 |
| --- | --- | --- |
| [LLMCompass，ISCA 2024](https://github.com/PrincetonUniversity/LLMCompass) | 硬件参数化、算子映射与分层性能建模 | 有公开实现；可参考模块划分，不直接复用 GPU 参数。 |
| [NeuSight，ASPLOS 2025](https://arxiv.org/abs/2407.13853) 与 [代码](https://github.com/scai-tech/NeuSight) | 以 tile 为预测单元，用物理约束改善跨硬件泛化 | 学习部分需要代表性训练数据；只有 A5 数据不足以验证跨代学习。 |
| [PipeWeave，ISCA 2026](https://github.com/zksainx/pipeweave) | 流水解析特征结合小型学习模型 | 已有公开 artifact；适合作残差层参考，其 NVIDIA 模型不能直接套昇腾。 |
| [TileSight，2026 预印本](https://arxiv.org/html/2607.22432v1) | tile 资源向量、合法重叠、填充/稳态/排空 | 与本项目抽象接近；所读版本称发表后开源，当前不将其视为已核验可复用代码。 |
| [ForeSight/UnitSight，2026 年 11 月预印本](https://arxiv.org/html/2611.13762v1) | 区分固定依赖延迟与随资源压力变化的排队等待 | 主要讨论 NVIDIA 同微架构配置变化；跨 A5/A7 时固定延迟本身也必须重新核实。 |

建议组合为：白盒 tile 调度作为主干，A5 CA 校准资源成本与少量残差，用按分支主动补样控制成本。保留 Roofline 作为下界和回归基线。

## 分阶段交付

1. 环境与复用核查：在用户目录锁定 950DT SKU 与最新兼容 CANN 软件集合，检查 CANNBOT/OpModel 接口适用范围，完成三类 smoke test，明确 CA 输出字段及限制。
2. 工作负载闭环：选择三个模型各一套官方支持配置，区分文本/多模态与 prefill/decode，形成可追溯 kernel 清单及独立 replay。
3. 最小模型闭环：以 dynamicMXquant、matmul、FA 为三个例子；完成不超过首轮预算的采样、预测、留出评估、查询耗时测试和误差归因。
4. 扩展覆盖：补充 grouped GEMM/MoE、KDA、mHC、indexer/KV 等真实热点，以累计设备时间及系统关键路径覆盖率排序。
5. 跨代分析：实现固定策略与重新 tiling 两种模式，发布 A7/A8 参数敏感度、瓶颈变化和适用范围。
6. 系统接入：提供批量耗时查询、资源占用、计算/搬运重叠信息、模型版本及外推标记，验证仿真开销和端到端影响。

Git 维护设计、配置、runner、解析器、性能模型和小型 golden 数据；大型 trace/安装包/权重存仓库外，用 hash 和清单索引。方案确定后再创建 GitHub 仓库并固定公开范围。
