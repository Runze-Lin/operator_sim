# 附录 A　算子、精度与可复算入口

下列公式用来确定提交给排程器的工作。T、D 通常表示本卡行数和通道数，M、N、K 为矩阵维度，B 为 batch，H 为本地头数。$\lceil x\rceil$ 表示向上取整。读写量给出张量边界字节；当有重复加载、片上中间结果或压缩格式时，另建实际执行账。下面每个算例都有输入、阶段参数来源和可用的 CA 对照，附录 B 为每项合同列出相应计分集。

## A.1　逐元素、量化、路由及行归约

| 实现与格式 | 工作怎样得到，哪一步限制完成时刻 |
|---|---|
| [DynamicMXQuant](../examples/quant.md)，BF16→MXFP8 | R×C 输入有 $R\lceil C/32\rceil$ 个组。向量工作 $R(2C-\lceil C/32\rceil)$，值与 scale 分别写回；双槽中的下一次读要等相应槽可复用。 |
| [动态 MXFP4](../examples/quant-mxfp4.md)，BF16→E2M1/E8M0 | 同样每 32 值共享 scale，输出值按两值一字节打包。取实际行布局、组尾和 scale 填充，另计编码与写回。 |
| [逐 token FP8](../examples/quant-token-fp8.md)，BF16→FP8/FP32 scale | 一行只有一个 FP32 scale。每行先规约绝对最大值，再缩放、转换并落盘；输入 2RC 字节，输出值与 scale 共 RC+4R 字节。 |
| [CacheLoad](../examples/cache_load.md)，packed E2M1→BF16 | 每一条访问按索引取得 packed 值与 scale 后展开。显式 AntiMxQuant 可对 CA；融合 gather 的索引搬运与命中条件另建。 |
| [RMSNorm](../examples/rmsnorm.md)，BF16→BF16/FP32 | $s=1/\sqrt{(\sum_j x_j^2)/D+\varepsilon}$，输出 $x\cdot s\cdot\gamma$；统计量另写 4T 字节。按忙核及 UB 容量计算行块，末块 SIMD 和写回可部分重叠。 |
| [Cast](../examples/cast_bf16_fp32.md)，BF16→FP32 | X 个值读 2X、写 4X 字节；对最忙核求首块、双槽复用间隔和尾写，再检查共享接口。 |
| [Sigmoid](../examples/sigmoid_fp32.md)，FP32 | X 个值各读写 4X 字节。每个 UB 块 n 个值含 $\lceil n/64\rceil$ 个向量组；指数服务和短尾单算。 |
| [MoE gating Top-K](../examples/moe_gating_top_k.md)，FP32→FP32/INT32 | T 个 token、E 个专家，选择 k 个，写 8Tk 字节。每行扫描、排序服务依赖 $\lceil E/64\rceil$，关键核承担至多 $\lceil T/\mathrm{AIV}\rceil$ 行。 |
| [AttnResFwd](../examples/attn_res_fwd.md)，BF16 | [T,D,S,N] 指 token、通道、槽容量与有效残差数；按 N 个残差的评分和混合计数，每段向量组数随 $\lceil D/64\rceil$ 增长。物化写回与 UB 驻留各有独立证据。 |

## A.2　矩阵与融合注意力

| 实现与格式 | 工作怎样得到，哪一步限制完成时刻 |
|---|---|
| [MatMulV3](../examples/matmul.md)，BF16×BF16 | 有用工作 2MNK；输出母块为 $\lceil M/b_m\rceil\lceil N/b_n\rceil$。拆分尾波、L1 页、L0 片，最后一次 FIXP 退休后输出才完整。 |
| [Addmm](../examples/addmm.md)，BF16 带列 bias | 2MNK 个矩阵操作，另读 2N 字节 bias，有 MN 个逻辑加法。校准合同取 $\alpha=\beta=1$ 和可融合列偏置。 |
| [原生 A4W4](../examples/mx-native-tail.md)，MXFP4→BF16 | 两侧 packed FP4 和 E8M0 尺度送入 Cube；尾波余 r 个母块可按核数拆长轴，核间用局部完成条件交接。 |
| [展开 A8W4](../examples/mx-handoff.md)，MXFP8×MXFP4→BF16 | 有用工作仍为 2MNK，另计 W 的解包与展开；每页要等自己的 AIV 交付 W，后续页展开可与当前页 MMAD 并行。 |
| [FlashAttention](../examples/fa.md)，BF16/FP32 统计 | 稠密 QK 和 PV 有用操作 $4BHQ\cdot\mathrm{KV}\cdot D$。关键为每核内 Q 与 KV 的实际页；首轮按依赖，后续按最慢资源或发起间隔，末端计输出。 |
| [Lightning Indexer](../examples/lightning_indexer.md)，BF16→INT32 | 按本地 query、头及有效 key 计点积工作，再计权重、整流、头间规约和选择。有效 key 范围由该合同的掩码与索引决定。 |
| [候选索引 BF16](../examples/candidate-filtered-indexer.md) 与 [原生 MXFP4 QSLI](../examples/flash-a5-qsli.md) | BF16 参考的全量评分、候选筛选，与按各行候选记录打分的原生路径分别计数。稀疏页次序和跨核归并决定关键团队；固定 metadata 的实测只覆盖 kernel。 |

## A.3　mHC、KDA 与专家组合

| 实现与格式 | 工作怎样得到，哪一步限制完成时刻 |
|---|---|
| [mHC Pre](../examples/mhc_pre.md) | 对 N 条残差流生成 $N^2+2N$ 个系数，投影宽度 ND。按 token 块和通道分区给投影、搬运、控制阶段分配服务。 |
| [Sinkhorn](../examples/mhc_sinkhorn.md)，FP32 | 每个 token 的 N×N 系数经交替行列归一；迭代 I 轮，有 I 次前后依赖链。短序列的启动与前序驻留可占主要时间。 |
| [融合 PreSinkhorn](../examples/mhc_pre_sinkhorn.md) | 同一设备窗串接投影、交接和归一。按实际 MMAD 和末段向量组估计，已校准分支使用 HF32 投影。 |
| [mHC Post](../examples/mhc_post.md) 与 [vLLM HcPost](../examples/hc_post_vllm.md) | 每个 token 的 N 条残差流按系数混合为 N 条输出，混合主项随 $N^2D$ 增长。库实现与扩展实现分别排程，有序流可识别热态或扩核。 |
| [拆分的 mHC 调用](../examples/mhc.md) | 总设备服务为 Pre、Sinkhorn、Post 三个串行窗之和；中间量在接口上物化。前两段融合的实现改用融合窗。 |
| [KDA recurrent](../examples/kda_recurrent.md) 与 [chunk V2](../examples/kda_chunk.md) | recurrent 按 token 递推同一状态，后步等前步；chunk 分支按分块及其内部子阶段排程。单状态链和 V2 有计分窗，其他分支沿用各自资格。 |
| [GMM1–SwiGLU–MXQuant](../examples/grouped_matmul_swiglu_quant_v2.md) | 各专家 token 数 mₑ 单独取整，$\sum_e m_e=T$；上投影 K→2F 有 4TKF 个 Cube 操作，随后激活和 MX 量化。 |
| [GMM2](../examples/gmm2.md) 与 [显式 A4W4 GroupedMatmulV5](../examples/routed-grouped-matmul.md) | 下投影 F→K 有 2TFK 个操作，每专家独算尾块。原生 MXFP4 与 MXFP8 按实际操作数和 scale 格式计价。 |
| [MoE FFN](../examples/moe_ffn.md) | 将两次分组投影按 producer/consumer 连接，逻辑字节包括中间值和 scale；分发与回收由系统通信节点另计。 |

长 prefill 图还包含重排、掩码、地址准备、通信、混合精度投影和不能取得完整 CA 的融合路线。调用方应在逐条合同中检查实现假设、返回精度与数据层级。[长 prefill 路由](../examples/prefill-routes.md) · [系统 Vector](../examples/system-vector-routes.md) · [系统 Cube](../examples/system-cube-routes.md) · [系统 Attention](../examples/system-attention-routes.md) · [通信](../examples/system-communication.md)
