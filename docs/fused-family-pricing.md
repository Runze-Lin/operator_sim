# 原生混合算子：指令类别法如何替换一个 VF

同一实现的两种报价使用相同的 dtype、分核、tile 和 DMA/Cube 参数。差别在 **AIV 收到数据以后、发出下一份结果以前，占用多久**。没有访存、没有 VF 或没有实现，这三个命题并不等价；例如 A4W4 的 scale 要读取，但原生 Cube 消费 scale，不需要另外插入一次 AIV 反量化。

**把一段程序换成周期。** 设寄存器宽度为 `R` 字节（950 默认 256），FP32 每向量 `R/4` 个值。先按行、精度和尾部掩码算出每类寄存器指令数 `n[j]`；`i[j]` 是同类指令可再次发起的间隔，`l[j]` 是结果可被消费者读取的延迟，均以每个忙 AIV 的周期为单位。`d[j]` 只计一条生产者依赖链上的指令数，不能用全 tile 的 `n[j]` 代替。总发射槽数 `I`、算术端口数 `A`、UB 端口数 `U` 默认均为 2；这是接口场景假设。单 VF 的周期预算为：

```text
issue = Σ n[j] / I                      dep = Σ d[j] l[j]
alu   = Σ(算术、转换、规约等 n[j] i[j]) / A
ub    = (n[load] i[load] + n[store] i[store]) / U
Cvf   = max(issue, dep, alu, ub, n[store] i[store], max_j n[j] i[j])
```

这里取最大是给**能重叠发起**的资源定价；必须等待的先后阶段再按下表组合。一次读寄存器的延迟不等于每条读指令的发起间隔。默认 II 对 exp/divide 为 4、其他类为 1；默认依赖延迟继承 FA 既有 950 trace 的类别代理（load/store 9、规约 16、exp 18、divide 17、算术/转换 7 周期；predicate 1 为假设）。`chip.details.vf_<class>_ii_cycles`、`vf_<class>_latency_cycles`、`vector_register_bytes` 均可替换。真实二进制指令数、寄存器分配和相关性没有从这些假设自动得到。呈现的 `instruction_programs` 是每种局部程序的样例，不是全芯片的动态指令直方图。

| 实现合同 | AIV 工作与具体算例 | 如何放回旧的时序 |
|---|---|---|
| BF16 Matmul、A4W4 原生 Matmul、列 bias Addmm、GMM2、原生 MXFP4 routed GMM V5 | 无单独 VF。示例：[A4W4 与 A8W4](examples/mx-handoff.md)。 | 直接保留原设备窗、状态和活动核，`vector_model=not_applicable`。 |
| A8W4，AIV 展开 W4 | 一次 `[w,k]` 的 packed 权重页有 `g=w·ceil(k/(2R))` 个输入寄存器；候选展开程序含 `g` load、`2g` 谓词、`4g` 位算术、`2g` 转换和 `2g` 写。例如 `w=4,k=1024,R=256`，共 8 个输入寄存器。 | 每一页的 `W ready=max(W DMA retire,AIV free)+setup+Cvf`；再等待 UB→L1、空闲 L1 slot 和 L0→MMAD。额外 K 页或波次依照原 SWAT 环调度；一部分差异会被 Cube 或搬运隐藏。 |
| LightningIndexer、BF16 候选过滤投影 | 一行 128 个 key、8 个头，FP32 按头有 `x=8·ceil(128/64)=16` 组；模型假设 `3x−2=46` 组 ReLU、乘权和跨头相加。排序另按 256 key、8 次 radix 分解，超过 8192 key 的下一段携带 512 个优胜项；候选输出的有序映射需要独立的比较交换层。 | Cube 仍做点积；每页取 Cube、读、打分间隔的最大值，末页打分后再排序并排空写。dense LI 若用旧的大 Q 模板，改忙核 SIMD 预算后重算每个并发队列组合，其他服务照旧。 |
| A5 QSLI MXFP4 | 每 512 个候选 key 分到两个 AIV；单 AIV 给 256 个 score 做跨头组合，子核并行。跨核合并先读分片列表，再按合并轮次选择并重排索引，最后写结果；不会在每核串行收取全芯片合并费。 | 保留已观察的首发、布局复用、单/多团队 MMAD 间隔及跨行控制。替换本地 score、radix 和 merge-select 成本，然后重新比较关键团队、HBM 和 L2；因此两个方法共用部分 CA 信息，不能将这项比较称作完全脱离 CA 的外推。 |
| Recurrent KDA | 一条 `[S,K,V]` 链以 `K·ceil(V/(R/4))` 个状态向量计数；衰减、`K·state`、值修正、秩一更新和 `Q·state` 均计入。`K=V=128` 时每步为 256 个状态寄存器组。 | 以 `S` 个串行更新替换互斥 SIMD 成本；多头和 batch 沿原独立状态链划分，按忙核轮数重复单链体，和共享带宽下界取最大。此代理不复原条带内每个广播、规约依赖或 spill，因而对已知实现仍推荐旧法。 |
| FLA Chunk KDA V2 | Prepare 的 gate/prefix、三角带打包、两块 32 阶求逆、右端构造；FwdH 的初态转换、修正值和状态更新各有一段独立程序；Finalize 保留 Cube 和 DMA。`[1,128,1,1,128,128]` 有两个 64-token 的 chunk。 | 替换到具体 VF descriptor 上；FwdH 保留首块/末块、H 状态边、W/U 预取、L1 信用等待，三个实 kernel 的窗串行相加。V1 无可比整窗，继续 N/A。 |
| MX GroupedMatmulSwigluQuant V2 与串行 MoE FFN | `[T,K,F,E]=[8,256,128,2]`、expert counts `[1,7]`，有 `ceil(128/256)+ceil(896/256)=5` 个向量任务；每个忙 AIV 以最多 256 值进行 sigmoid（exp/div）、门控乘法、group32 规约、转换、写 FP8/E8M0。 | 旧模板记录每个实际关键核各组队列的并发区间。将类别预算按保存的 SIMD 活跃窗份额分配，各区间以同时活跃队列服务的最大值计费，取最晚核。双算子合同最后加独立 GMM2；若全被隐藏，两法整窗可以相同。 |

列说明：**实现合同**区分 dtype 和具体路径；**AIV 工作**描述指令推断，不能当作 CA 计数；**放回旧时序**是控制、搬运和计算并行的边界。分核、硬件资源、寄存器程序可独立替换；一块新芯片若改变了实际算法、寄存器类型或 AIC:AIV 配对，必须重选模板，不能仅按频率缩放。

**候选 Indexer 的比较证据。** [逐用例 CSV](../results/v2.3/candidate-methods.csv) 和[生成脚本](../calibration/candidate_indexer/compare_vf_methods.py) 使用已归档的相同 950 窗口，0 次新增 CA。上表的类别数量是根据源算法提出的程序假设，没有用这些整窗的时长反推系数。旧模板和 QSLI 的时序控制仍然包含历史 CA 证据。下表没有新冻结后的独立样本。

| 范围 | 记录数 | 旧法 WAPE | 类别法 WAPE |
|---|---:|---:|---:|
| 950DT MXFP4 原生 QSLI 静态 metadata，整窗回放 | 8 | 7.48% | 7.38% |
| 950PR 同口径 | 4 | 7.90% | 8.52% |
| DT/PR BF16，dense 或 compact 的显式组件链 | 6 | N/A | N/A |

列说明：**记录数**是物理执行次数，同 shape 不同页表宽度和布局各占一条；**WAPE**=`Σ|预测−CA|/ΣCA`。CSV 的 `shape` 按 `[B,Q,H,D,L,C,K]`，`attrs` 为调用属性，`chip/dtype` 指目标，`legacy_us/classes_us` 为两套预测（µs），`ca_us` 是同实现整窗，`*_error_pct` 是有符号相对误差，`source` 是本地原始 JSON 的文件名（不含扩展名）。BF16 的 `component_sum_us` 是 5 或 13 个显式库 kernel 的 CA 串行和，不是单个假想融合 kernel 的实测；保留作参考但不计算误差。强制单团队的诊断未混入正常分核计分。对于已测路径，当前数据不支持统一换默认方法。
