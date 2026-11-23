# 未发布芯片上的 MXFP4 Matmul 怎么估

取单卡稠密 Matmul，M=128、N=256、K=512，A 和 B 均已是沿 K 分组的 MXFP4 E2M1，假设 FP32 累加、BF16 输出、无 bias。推导从打包完毕的张量开始。此处有用运算、逻辑字节和条件服务下界可以复核；目标芯片整窗、动态指令次数和 CA 误差均为 N/A。`estimate("matmul", ..., dtype="mxfp4")` 当前返回 `unsupported`。

### 与芯片无关的工作量

- 有用运算为 $2MNK$ = 33,554,432 ops。FMA 计两次。
- 两个 4-bit 元素共用一个字节，A、B 的 payload 合计 $(M+N)K/2$ = 98,304 B。
- 每 32 个元素共享一个 8-bit E8M0 scale。本例 K 整除 32 且 64 对齐，所以 scale 占 $(M+N)K/32$ = 6,144 B。A、B 总共读 **104,448 B**，BF16 输出写 **65,536 B**。K 有尾部时按每行、每列分别向上取整，并再考虑实现的成对 scale 排布、格式转换和对齐。[OCP MX 规格 §5.2](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)

### 从工作量推到可能的指令

950PR 的 CANNBot 资料记录了原生 MX 路线：读入打包数据和 scale 到 L1，以带 scale 的搬运装入 L0A/L0B，Cube 执行矩阵乘累加，最后转换并写出。由此可为未来芯片提出一个原生候选。这个证据支持阶段依赖和数据类型，970/980 上具体指令编码、每条覆盖的矩阵、发起间隔和流水深度仍是待验证量。[CANNBot 的 950PR MX 记录](https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-145-microscaling-mx-format-mxfp8-mxfp4-cube-path-with.md)

假设输出 tile 取 64×128、K 页取 128，无 Split-K。一共 4 个输出块，每块依次处理 4 个 K 页，合计 **16 个逻辑页任务**。这个例子的页内 A/B payload 和 scale 共 (64+128)×(128/2+128/32)=13,056 B，L1 双缓冲的最低数据预算为 26,112 B，另需满足实际分形对齐；每块 FP32 累加需 32,768 B。每页可能需多次搬运和多条 MMAD。若目标证据给出一条指令的矩阵范围 $(m_i,n_i,k_i)$，在没有合并、掩码或重算的假设下，可对每个 tile 累加 $\lceil m/m_i\rceil\lceil n/n_i\rceil\lceil k/k_i\rceil$。真实发起数仍需目标二进制或 trace 核对。

若 970DT 沿用这个 tile，最多 4 个 AIC 同时工作。给定 120 个 AIC 的 4-bit 稠密峰值 14,160 TOP/s，且每核能力相同，候选的理想 Cube 服务下界为 $33{,}554{,}432/(14{,}160\times10^6\times4/120)$ ≈ **0.0711 µs**。若 A/B 首读来自冷 HBM，按 14.4 TB/s 得到读入下界 **0.0073 µs**。L2 命中会改变外存字节；输出是否在整窗内落 HBM 也要单列。端到端估时还需要首读响应、页间发起间隔、MMAD 填充/尾块代价和末次写回。对同一资源需求取上限和重叠，依赖导致的首、尾延迟要另外保留。[970DT 情景配置](../../src/operator_model/data/chips/Ascend970DT.json)

### 为什么必须区分 A4W4 和 A8W4

950 的一个已知混合精度实现对 E4M3FN 激活、FP4 权重走另一条路：AIV 从打包权重解出 FP8 送到 L1，AIC 消费 FP8 及 scale，末端还有 Vector 处理。它使用两个 AIV 配一个 AIC 的交接协议。若 970 情景为 1:1，必须为解包吞吐、缓冲槽和交接另设排程，再比较 Cube 与 Vector 谁占据关键路径。对于一种全新的融合算子同理，先确定数据交接和输出语义，再决定可以沿用哪些资源价格。[950 参考实现与固定源码](https://gitcode.com/cann/ops-transformer/blob/be33d95bb755cedec3108f70805c1874d1792540/gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_vcv_basic_block.h)
