# 精度覆盖与 950 实现证据

系统接入时先确定每个张量的编码、scale 的数据类型与粒度，以及谁负责量化、反量化和写回。「4 位」本身无法决定 Cube 的指令、读写字节数或 kernel 数。以下盘点以 CANN 9.2.1 的 `ascend950` 安装包、950DT/PR 已有 CA 收据和固定版本的公开接口说明为据。

| 系统需要的计算 | 当前可通过 `estimate` 得到有 CA 依据的时间 | 同一套 CANN 里的扩展线索 | 新组合的 DT / PR 耗时证据 |
|---|---|---|---|
| 稠密 Matmul | BF16；MX A4W4 原生和 MX A8W4 融合展开，均输出 BF16 | 两条 MX 路线均已逐元素验算，并核对实际派发 ELF 与 MMAD 输入 | 两端均有冻结独立窗，最新样本数及 WAPE 见[总误差表](error-analysis.md) |
| 注意力 | `fa` 的 BF16 Q/K/V 路线 | A8C8 全量化推理；A4C4 有独立的块稀疏算子；分页稠密 A4C4、A8C4 需另找实现或建组合 | 低精度 FA 均 N/A |
| 独立量化 / 反量化 | BF16 → block32 MXFP8/MXFP4、逐 token E4M3FN + FP32；显式 AntiMxQuant 的连续 FP4 → BF16 | 其他目的格式与布局需匹配 kernel；FP8 cache 反量化已有同形双端 CA | [FP4 量化](errors/quant_mxfp4.md) · [逐 token FP8](errors/quant_token_fp8.md) · [两种 CacheLoad](examples/cache_load.md) |
| MoE | MXFP8 GMM1/SwiGLU/GMM2、FP32 路由；显式 V5 的双 MXFP4 专家分组单投影 | 输出头分组 A8W4、混合激活 scale、与 SwiGLU/量化融合的 4 位路线须重新核验 | [A4W4 专家分组误差](errors/routed_grouped_matmul_mxfp4_v5.md) · [MoE FFN](errors/moe_ffn.md) |

这里的「候选」表示能在本机找到接口或目标芯片的注册项，须再验证选定 shape 的 tiling、输出和完整指令窗。970DT 均为架构外推；表中 N/A 的路线尚无可声明的跨代误差。[调用契约](integration.md) · [双端实测误差](error-analysis.md) · [稠密 MX 收据](../calibration/reference/mx-matmul.json)

稠密 A4W4 和 A8W4 均以 BF16 为最终输出。已支持的两次串行调用与 MoE 内融合的输出量化，分别在[精度、分块与输出算例](modeling-strategy.md#精度如何改变分块计算和输出)中说明。

**Flash 的候选 Indexer。** 捕获参考有 BF16 Q/K/权重；A5 部署的 Q/K 是 MXFP4 E2M1 和 E8M0 scale。Cube 的点积累加为 FP32，FIX 后的 ReLU、权重归约在 BF16 Vector 上，输出为 INT32 位置。层 20 的全表索引与其四个稀疏消费层独立定价。[实现、运行证据与形状](errors/flash-a5-qsli.md) · [完整算例](examples/flash-a5-qsli.md)

新增稠密 FP16 Q/K/V、FP32 softmax、FP16 输出合同，默认采用指令类别推演。FP16 两个字节的存储和 FP32 内部计算分别计价。转换吞吐可显式覆盖。独立小形状及适用范围见[FA 试验报告](paper/13-fa-vector-pilot.md)。

## FA 的 A 和 C 分别代表什么

本文的 **A** 是传入注意力的 query（Q）激活，**C** 是实际存储的 K/V cache。prefill 没有分页 cache 时，C 指此次传入的 K/V。网络线性层的输入激活、KV 投影权重、注意力读到的 Q/K/V 可以各用不同精度。P 指 softmax 输出送入第二次矩阵乘的数据，中间累加、online softmax、跨块 rescale、最终 Y 也可各自选择格式。FP8 E4M3FN 配逐 head 的 FP32 scale 与采用 E8M0 分组 scale 的 MXFP8 要作为两份契约处理。

| 组合 | 950 安装包与公开接口支持到哪一步 | prefill / decode 应怎样验证 |
|---|---|---|
| **MX A4C4，块稀疏** | `aclnnBlockSparseAttentionV3` 有 Q/K/V 全为 FP4 E2M1 的注册项，输出 BF16 或 FP16，LSE 为 FP32。Q、K 使用 rowwise E8M0，V 使用 columnwise E8M0；quantMode 2/3 对应两种 scale 取整。950 使用 innerPrecise=4，MXFP4 的 online softmax 是 FP16，rescale 是 FP32。 | 可先做非分页 prefill：提供实际 block mask，两个块边长取 64 的倍数。例如单头、S=128、D=128，两个 64-token 块，分别检查一块/两块可见的数值与访问量。该接口要求 blockTable 为空；将 mask 全置一仍执行块稀疏实现。decode Q=1/9 及 paged cache 需单独验证。 |
| **A4C4，通用稠密 / 分页** | 已核对的 `FusedInferAttentionScore` 和 `QuantFlashAttn` 950 注册中未见 FP4 Q 或 INT4 Q。上述块稀疏正例只涵盖其自身语义。 | 两阶段均缺目标实现的 CA。若后代使用原生 FP4 QK 和另一精度 PV，要分别配置这两段的 Cube 峰值、P 的量化及 scale 读写。 |
| **A8C4，MX 或整数** | 在上述通用 950 推理注册中未见 Q=FP8 且 K/V=FP4，亦未见 Q=INT8 且 K/V=INT4。确有 BF16/FP16 Q + FP4 或 INT4 K/V 的 V5 注册，这表示 **A16C4**。FP4 的 per-token-group 使用 E8M0；INT4 的 per-channel/per-token 分支和 offset 有不同约束。 | 先决定谁把 C4 展开到什么格式。如果设计为显式展开再做 A8C8，须单列展开、必要的重新量化、临时 K/V 的写回与重读；如果设计为核内展开，则需按 K/V tile 计 AIV 工作和交接。V5 的整数反量化分支在 Q 长度大于 1 时限制 INT4/INT32，普通 decode 的可行性不能延用到 MTP Q=9。 |
| **A8C8，浮点** | `aclnnFusedInferAttentionScoreV5` 与专用 `QuantFlashAttn` 均有 950 浮点候选。后者区分 HiF8 per-tensor、MXFP8、E4M3 的 Q/K per-token-head + V per-head。其相应 P 为 FP8 E4M3、softmax FP32。V5 也有 MX 全量化 TND 及受约束的 GQA/Decode MLA 分支。 | 优先选择与模型相同的布局、Q/KV head 比、D、scale 和 mask。V5 的 GQA 全量化例要求 D=128、分页、NTD_TND；文档列 Q 长度至 64K、KV 至 256K。Decode MLA 全量化另有 KV 头数 1、Q 长度 1–16、Rope BF16 等约束。MTP Q=9 与 prefill 长 Q 分开取数。 |
| **A8C8，整数** | V5 有 Q/K/V=INT8 → BF16 的 950 注册。Decode MLA 的 INT8 全量化要求分页 NZ cache。专用 `QuantFlashAttn` 文档枚举 12 是 INT8 prefill、P INT8、softmax FP16，产品表将该场景列在 A2/A3；950 专用目录的三个 ELF 登记的是浮点元组。 | 可优先核对合法的 V5 INT8 decode；950 整数 prefill 要另验选定 API 与实现。不同产品、FP8 与 INT8 的误差分别统计。 |

950 的 `MixedQuantSparseFlashMla` 也应另列。该注册项的 Q 是 BF16，ori/cmp KV 是存储记录，记录可混排量化字段、RoPE 和 scale。模型的稀疏选择、记录长度、索引访问和有效块数都会改变字节量。仅按 KV 张量的外层 dtype 乘逻辑 D 会漏算或错算。[推理 V5](https://gitcode.com/cann/ops-transformer/blob/14d022b1a4c3df263bf697534e3c25c7c992ff5d/attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV5.md) · [专用 QuantFlashAttn](https://gitcode.com/cann/ops-transformer/blob/14d022b1a4c3df263bf697534e3c25c7c992ff5d/attention/quant_flash_attn/docs/torchapi_quant_flash_attn.md) · [块稀疏 V3](https://gitcode.com/cann/ops-transformer/blob/14d022b1a4c3df263bf697534e3c25c7c992ff5d/attention/block_sparse_attention/docs/aclnnBlockSparseAttentionV3.md) · [混合稀疏 MLA](https://gitcode.com/cann/ops-transformer/blob/14d022b1a4c3df263bf697534e3c25c7c992ff5d/attention/mixed_quant_sparse_flash_mla/docs/aclnnMixedQuantSparseFlashMla.md)

## 量化与反量化

| 方向 | 本机可核查的实现 | 建模和单测的主要区别 |
|---|---|---|
| BF16 / FP16 / FP32 → MXFP4 或 MXFP8 | `aclnnDynamicMxQuant` 的 12 个 OPP 对象覆盖三种源精度与 E2M1、E1M2、E4M3FN、E5M2 四种目的格式。发布模型计时 BF16 → E4M3FN 和 BF16 → E2M1 两条末轴 block32 路线。E2M1 的尾组清零、发起及两路写回已用 FP4 核级 trace 校准。 | 动态 MX 要做每组归约、生成 E8M0、缩放与转换。FP4 每个字节打包两个值，Y 写回、scale 写回和尾块填充分别计数。相同 R/C 的 BF16→E2M1 与既有 E4M3FN case 可作低成本对照。 |
| MXFP4 / MXFP8 + E8M0 → BF16 / FP16 / FP32 | `aclnnAntiMxQuant` 有上述四种源、三种目的的 12 个对象。FP4 最后一维要求偶数。 | 输入为载荷和已有 scale；操作包含解包、广播、缩放与高精度写回。输出更宽，需用自己的 MTE3 成本。用正负值、不同组的 scale 做逐元素独立 golden。 |
| BF16 / FP16 → 整数或普通 8-bit 浮点 | `aclnnDynamicQuantV2` 文档涵盖 INT8、INT4、打包 INT32、E4M3FN、E5M2、HiF8；950 `dynamic_quant/` 中可见 BF16/FP16 的 INT4、INT8 及 8-bit 浮点对象。 | per-token 的 scale/可选 offset 与 MX 的 block scale 形状不同。INT32 容器可以存 8 个 INT4，应按存储格式计流量；分开验证对称/非对称、饱和及尾部。 |
| 普通低精度 → 浮点 | 安装包 `AscendAntiQuant` 可见 INT8、HiF8、E4M3FN、E5M2 分别到 FP16/FP32 的对象，另有 AscendAntiQuantV2 将 E4M3FN 与 FP32 行 scale 解到 BF16，已取到双端整窗。更广的跨产品文档列出 INT4/INT32，但本次未确认 950 的直接整数 4-bit 反量化派发。 | 确认 scale/offset、广播轴及输出类型；独立算子计入设备启动和 GM 写回，融合反量化按实际落点计 UB/L1 交接。权重常驻预处理另计。 |

调度端按两个输出分别记账。例如每行 C=256 的 BF16 输入为 512 B，block32 的 E8M0 scale 为 8 B，MXFP4 输出为 128 B；反向生成 BF16 时则读 128+8 B、写 512 B。这是单行逻辑载荷，分核重复读取、尾块和事务对齐由目标实现定价。[DynamicMxQuant](https://gitcode.com/cann/ops-nn/blob/602e4ad985da00aad28c16432841e9c5069e10de/quant/dynamic_mx_quant/docs/aclnnDynamicMxQuant.md) · [AntiMxQuant](https://gitcode.com/cann/ops-nn/blob/602e4ad985da00aad28c16432841e9c5069e10de/quant/anti_mx_quant/docs/aclnnAntiMxQuant.md) · [DynamicQuantV2](https://gitcode.com/cann/ops-nn/blob/602e4ad985da00aad28c16432841e9c5069e10de/quant/dynamic_quant_v2/docs/aclnnDynamicQuantV2.md)

## MoE 的 W4A4 和 W4A8

W4A8 表示 **4-bit 专家权重 × 8-bit 输入激活**。路由 gate 的 logits、top-k 索引、专家收到的行数、GMM1 后的激活、GMM2 的输出都要另给。给定本卡专家行数向量 `expert_counts`，第 e 个专家处理 mₑ 行，GMM1 为 `[mₑ,H] × [H,2F]`；SwiGLU 将宽度变为 F，随后量化到 GMM2 的输入格式。GMM2 为 `[mₑ,F] × [F,H]`，再按路由权重合并。专家数、总行数相同的均匀路由和热点专家可能需要不同波数。

| 权重 / 激活及用途 | GMM1、SwiGLU 与中间精度 | GMM2 与发布状态 |
|---|---|---|
| MX W4A4 | `aclnnGroupedMatmulSwigluQuantV2` 有 E2M1 × E2M1、ND 权重的 950 条目，输出可为 E2M1 或 E4M3FN，另带 E8M0。GMM1→GMM2 全程 W4A4 要选择 E2M1 中间输出。 | `aclnnGroupedMatmulV5` 有 E2M1 × E2M1 → BF16、groupType=0 的 MoE 候选。双侧 E8M0、转置 ND 权重的独立专家 V5 投影已双端校准，W4 融合整链待测。 |
| MX W4A8 | `aclnnGroupedMatmulSwigluQuantWeightNzV2` 的 MXA8W4 用 E4M3FN 激活、E2M1 权重，输出 E4M3FN；包内相应权重为 FRACTAL_NZ 类。按专门示例构造转置 NZ 权重和 scale；要求 K 为 32 倍数且 N≥128。V3 的同名 WeightNz 接口文档限定 MXFP8 单 Tensor，应核对具体版本。 | `aclnnGroupedMatmulWeightNz` 的 MX_A8W4 示例先用 `aclnnWeightQuantPreprocess` 处理 W 和 scale，产物供分组乘使用。模型加载时可复用的预处理单计一次，若每次动态重排则计入每次请求。两个子算子以及展开时是否使用 AIV 都须用实际 trace 核对。 |
| INT W4A4 / W4A8 | `aclnnGroupedMatmulV5` 的整数表和 950 `grouped_matmul_apt` 有 INT4×INT4、INT8×INT4 到 BF16 的条目。950 的 GMM1-SwiGLU V2 注册中未核对到对应的 INT4 权重融合项；老代的 MSD INT8×INT4 示例有不同的平台脚注。 | 可先对独立整数 GMM 做 CA，再构造 SwiGLU 和独立量化的多 kernel 路线。只有能衔接 GMM1 输出、SwiGLU、GMM2 输入及 scale 的实际派发序列才可计整链时间。 |

`GroupedMatmulFinalizeRouting` 的注册里也有 FP4×FP4、FP8×FP4 → FP32 的组合。它将专家等贡献合并到 token 上，输出语义、地址分布和 dtype 与普通 GMM2→BF16 有别，须建立单独的融合算例。当前 `moe_ffn` 的计时口径见现有算例，EP 通信、全网路由和模型加载费用由上层显式安排。[当前 MoE 算例](examples/moe_ffn.md) · [GMM V5](https://gitcode.com/cann/ops-transformer/blob/14d022b1a4c3df263bf697534e3c25c7c992ff5d/gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md) · [GMM WeightNz](https://gitcode.com/cann/ops-transformer/blob/14d022b1a4c3df263bf697534e3c25c7c992ff5d/gmm/grouped_matmul/docs/aclnnGroupedMatmulWeightNz.md) · [GMM1 WeightNzV2](https://gitcode.com/cann/ops-transformer/blob/14d022b1a4c3df263bf697534e3c25c7c992ff5d/gmm/grouped_matmul_swiglu_quant_v2/docs/aclnnGroupedMatmulSwigluQuantWeightNzV2.md)

## 用模型配置选路线

| 已固定的 config | 能从权重元数据读出什么 | 运行时仍需给出的数据 |
|---|---|---|
| DeepSeek-V4.1-Flash | `quant_method=fp8`、动态 activation、权重块 [32,32]、scale_fmt=ue8m0、expert_dtype=fp4 | 本次 dispatch 的专家输入是 4/8/16 位，具体的 cache 记录、稀疏模式与本卡路由 |
| DeepSeek-V4-Pro | expert_dtype=fp4；全局 FP8 E4M3、动态 activation、权重块 [128,128]、scale_fmt=ue8m0 | 区分普通层的二维块量化与 E8M0/block32 的算子契约；注意力和 routed expert 各核对实际实现 |
| Kimi-K3 | text config 声明 MXFP4 packed 权重、group_size=32，input_activations / output_activations / kv_cache_scheme 为 null。ignore 排除了 self_attn 和 shared_experts 等。 | 分开看 routed/shared experts、KDA 与常规注意力。运行时激活及 KV 的编码不能从权重位宽推出 |

单卡 shape 可先从 config、TP/EP、prefill chunk、batch 和投机长度推出；实际 cache 策略、分块、专家负载和精度路线用框架 dispatch 或运行配置确认。一个有效的核对样例是 decode batch=1、投机 8 时主模型 Q=9；不能假设所有只支持单 Q 的 kernel 都能接这个 shape。[三个固定 revision 与 SHA256](../examples/workloads/models/sources.json) · [本地 shape 推导](../examples/workloads/README.md)

## 如何复核与选择少量 CA

静态检查定位到 `$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend950/ops_transformer/`。对同一 `binList` 条目核对每个命名输入和输出的 dtype、format、可选 scale 及 `binInfo.jsonFilePath`，再到 `kernel/ascend950/ops_transformer/<目录>/` 核对对应 ELF 存在。接口约束以固定开源快照与本机头文件互证；不同版本的签名、产品脚注和可选参数仍须靠实际调用判定。

| 安装包注册文件（.json） | SHA256 前 12 位 | 可用于定位的 ELF 文件名片段 |
|---|---|---|
| fused_infer_attention_score_apt | 9376988837d9 | 614961c4… 为 E4M3 Q/K/V→BF16；442588f1… 为 BF16 Q 与 FP4 K/V→BF16 |
| quant_flash_attn / block_sparse_attention | fb5b8735c061 / ad76861ec3ed | 10e20636… 为 E4M3；567a5cfd… 为块稀疏 E2M1 Q/K/V→BF16。后者通用注册的可选 scale 字段写为 float32，V3 MX 专门契约要求 E8M0，准确派发需以 GetWorkspaceSize 和运行结果核对 |
| grouped_matmul_swiglu_quant_v2_apt | 6c83408c21d2 | 2b9df07c… 为 MX A4W4→E2M1，41068c7d… 为 MX A8W4→E4M3FN |
| grouped_matmul_apt / grouped_matmul_finalize_routing_apt | 06e971b39d57 / 2ce6a39b5f76 | 40ca02ea… 为 GMM A4W4→BF16，0023c686… 为 GMM A8W4→BF16；d150bedd… 为 A4W4 + finalize→FP32 |

最省采集的顺序是先测独立正向、反向 MXFP4，再验证 MX W4A4 的 GMM1 和 GMM2，中间用相同 E8M0 与已知的 expert_counts。混合 W4A8 首先验证预处理结果能被 NZ 接口接受。FA 先取 A8C8 的短 prefill 与 Q=1/9 decode 做合法性和数值检查，再单测 MX A4C4 的 block-mask 路线。完整计时应匹配模型实际的密集/稀疏、分页与精度契约。每个新入口都需冻结预测、检查全部输出、核对实际 kernel ELF 并提取全部核心首指令至末退休；只有通过这些检查后才将对应 shape 和误差加入总表。[CA 验证协议](../calibration/README.md)
