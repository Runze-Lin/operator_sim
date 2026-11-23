# 950DT 长上下文 decode 的系统算子覆盖

这份清单回答两个接入问题：给定三个模型的 128Ki 历史、一次 9-token 校验，rank 0 实际捕获了哪些逻辑算子？按系统现用 Roofline 账本，哪些算子值得优先替换？27 次运行共 **171,232 个节点、80 个不同原始调用名**。最大的四项依次为候选过滤索引、分组矩阵乘、FP4 动态量化和 cache 读取解包，合计 **80.4%** 的已定价时间。账本没有设备指令、派发 ELF 或 CA 实测微秒数。子算子在 operator-sim 的计时证据见[逐算子 CA 误差表](error-analysis.md)。

## 九组实验和静态显存

以下每格按**本卡 B=1 / 16 / 64** 排列。显存列来自每个 run 的 `results/l1/inference.json → statistics[0].memory.peak_gb`，采用十进制 GB；时间列来自同文件的单轮 `duration_us`，换算为 ms。每轮同时计入 target 一次和 draft 一次。三种布局的 (TP, DP, EP) 分别为 (8,1,8)、(8,4,32)、(1,64,64)。全局 batch 为本卡 B×DP。

| 模型 preset | 布局 | rank 0 静态峰值 GB | rank 0 部分定价的单轮时间 ms |
|---|---|---:|---:|
| deepseek-ai/DeepSeek-V4-Pro | TP8-EP8 | 114.35 / 120.63 / 140.73 | 36.94 / 67.23 / 167.17 |
| deepseek-ai/DeepSeek-V4-Pro | TP8-EP32 | 33.49 / 39.77 / 59.88 | 11.62 / 41.90 / 145.38 |
| deepseek-ai/DeepSeek-V4-Pro | DP64-EP64 | 37.22 / 43.52 / 63.67 | 12.19 / 75.45 / 301.44 |
| deepseek-ai/DeepSeek-V4.1-Flash | TP8-EP8 | 70.17 / 75.62 / 93.07 | 21.33 / 150.07 / 563.58 |
| deepseek-ai/DeepSeek-V4.1-Flash | TP8-EP32 | 41.07 / 46.52 / 63.97 | 12.20 / 140.95 / 555.70 |
| deepseek-ai/DeepSeek-V4.1-Flash | DP64-EP64 | 237.45 / 242.91 / 260.39 | 12.47 / 152.08 / 608.15 |
| moonshotai/Kimi-K3.1-Pro | TP8-EP8 | 192.38 / 215.32 / 288.73 | 63.16 / 119.42 / 307.73 |
| moonshotai/Kimi-K3.1-Pro | TP8-EP32 | 56.77 / 79.71 / 153.13 | 20.64 / 77.00 / 274.18 |
| moonshotai/Kimi-K3.1-Pro | DP64-EP64 | 71.52 / 115.50 / 256.24 | 26.58 / 170.30 / 680.01 |

这 27 个静态峰值按两位小数逐项与用户给出的九组一致。以 TP8/EP32 为例，三个模型各自的 b1、b16、b64 子目录均有独立的 target/draft 压缩图和 L1 账本；`native/000_round_metadata.json` 的 target_runtime 也分别记录 TP=8、DP=4、EP=32。没有从其他布局补数。表描述已生成的分析任务；系统配置的 144 GB 与 operator-sim 的具体 `Ascend950DT_9582` 容量 103.079215104 GB 不同，例如 Kimi 的 TP8/EP8 三个配置均超过两种容量，仍能导出分析图。任务成功和真机显存可运行性分别判断。

额外的 `v4-pro-c128-boundary` 是 TP8/EP8、B=1、历史 131198 的压缩边界样例，峰值 114.350689212 GB，单轮 36.943817 ms。它未进入本页 27 次统计。

所有常规任务的 `kv_seq_len=131072`、`kv_capacity=132114`，请求生成 1024 个 token。导出一轮 DSpark，8 个候选加一个 anchor，target 的查询位置为 [131072,131081)，draft 为 [131081,131090)。时间统计按 6 个接受的 draft token 加目标 token，共 7 个；它不代表跑完 1024 个输出。TP 分头、EP 分专家，并非所有子层的维度都除以 TP。Pro 的 H=7168、61 层、128 个注意力头、indexer 64×128、384 专家取 6；Flash 对应 H=5120、40 层、64 头、indexer 32×128、384 取 6。二者的注意力 head_dim 均为 512。Kimi 此处确切 preset 为 **Kimi-K3.1-Pro**，H=7168、93 层、96 头、896 专家取 16，KDA 的头宽为 128，MLA 的 qk_nope/qk_rope/v 维度为 128/64/128。每个算子用其捕获的本卡 tensors 判定尺寸。

## 如何读账本

- 每个 case 有两张 native rank-0 图和两份 verify L1 CSV。按 `(case,stage,rank,node_id)` 一一连接 `native/000_rank0_{target,draft}.json.gz` 与 `results/l1/{000_verify_target,001_verify_draft}.csv`，并核对 raw_name、节点数、状态和汇总。此处按原始名区分 `aten.masked_fill.Scalar` 和带下划线的原位版本。相同层在 draft 中再次执行时保留其独立时间。
- L1 的 98,920 个 success 已由系统后端返回结果，64,365 个 skipped_zero 在基线记零，6,078 个 placeholder 和 1,869 个 miss 没有价格。没有失败记录。27 组的 rank-0 部分定价合计 **4,814,851.03 µs**。本导出 serial 与 exposed 合计一致。各组是不同的运行设定，求和只为给这批样本排优先级；不能当作生产部署的平均延迟，也不能把零价格节点视为免费。
- 同批 Chrome trace 只导出 **101,665** 个可与 L1 对齐的 operator 事件，另含展示元数据。以 Pro/TP8/EP8/B1 的 target 为例，native 和总账均为 7,760 个节点，Chrome 仅有 3,939 个；被省略的节点里有 210 个 success 的正时间事件。按 Chrome 可见行统计会少算。汇总 `modeling-inputs.csv.gz` 跳过 skipped 节点，`expert-routing.csv` 保存 dispatch 的路由信息，两者用于补语义，均不再累加时间。
- `native` 是系统捕获图的目录名。一张图可同时含模板实现的探索节点、ATen 原语、通信占位和融合节点。展开已单列的 `cache_load` 后要检查消费者是否真正读写新 BF16 张量；改成直接读压缩 KV 的融合实现时，必须重新划定中间物化边界。账本未给出真实 ACLNN 派发、权重预打包是否常驻、cache 命中、候选 block ID 内容、每个核的执行记录。本页只对有证据的输入契约建立联系。

## 优先缺口与可接入条件

| 次序 | 捕获合同与基线预算 | 建模与校准边界 |
|---|---|---|
| 1 | Flash 候选过滤 54 次，1,467,808.02 µs。每个 Flash run 的 target 有第 24/28/32/36 层四次，draft 的第 36 层出现两次。B1 节点有 BF16 Q[9,32,128]、K[1,131081,128]、头权重[9,32]、计数[9]、INT32 block IDs[9,2048] → INT32[9,512]。 | `candidate_filtered_indexer` 提供工作量。系统参考的打分量为 2×T×H×K×D，另加 T×C 比较量；其中 K 是每请求 key 行数，C=2048 是 block 个数。上游 `candidate_blocks` 的 block_size=8，若先筛候选，应按去重后的有效 block 覆盖行数算工作。实际值依赖 ID、有效长度和因果截断；2048 个 block 不能直接当作 2048 行。两种顺序及全窗时间均需实际 kernel 证据。 |
| 2 | GroupedMatmul 共 4,545 次，848,425.21 µs。3,582 次为路由专家 A4W4（812,778 µs），963 次为按输出头分组的投影 A8W4（35,647 µs）。 | `routed_grouped_matmul` 提供专家支路单投影的工作量，逐条关联本地专家行数与分发容量。3,582 个节点均有上游直方图，全部 963 个 head 分组仍需独立合同，A 用逐 token FP32 scale。显式选择转置 ND 权重和 E8M0 的 `aclnnGroupedMatmulV5` 已在两端独立验证，包括 Flash 的真实投影。图导出未记录原生派发。[专家投影算例](examples/routed-grouped-matmul.md) |
| 3 | FP4 动态 MX 量化共 17,748 次，791,696.08 µs。如 Pro 的 BF16[9,7168] → FP4[9,7168] 与 E8M0[9,112,2]，每 32 元素一 scale。 | 对应 `quant_mxfp4`，输出载荷、打包和单独 scale 写回按 FP4 模型核算。另有 **8,253 次**名称含 dynamic_mx_quant 的 FP8 参考节点，总计 23,599.76 µs，实际输出 E4M3 与逐 token **FP32** scale；不能归到原 BF16→MXFP8 E8M0 的验证集。27 个 run 的 FP4 量化共有 135 个形状，14,400 个输入连续，3,348 个是非连续视图。连续与视图分别统计，双端已匹配的调用次数随新增实测在下页更新。逐 token FP8 可明确选择单独的 `quant_token_fp8` 计时。[全部 135 个 shape 与双端结果](system-quant-shapes.md) · [设备误差表](error-analysis.md) |
| 4 | CacheLoad 共 3,213 次，763,472.87 µs。其中 FP4→BF16 2,295 次／758,698 µs，FP8→BF16 918 次／4,775 µs。Kimi B1 中 FP4[1,131081,576] 配 E8M0[1,131081,9,2]；Pro 的小窗 FP8[1,128,512] 配 FP32[1,128]。 | 独立 `cache_load` 工作量入口分别描述 MXFP4/block32 与 FP8/per-row 的公开物化合同。连续 FP4 可显式选 AntiMxQuant，并有双端独立校准；FP8 显式 AscendAntiQuant 已跑通一个实图几何。27 个常规 run 的 FP4/FP8 调用中分别有 636/186 次涉及非连续输入或输出。[两种格式与实测](examples/cache_load.md) |
| 5–8 | V4-Pro 量化索引参考 top1024、V4 的稀疏共享 KV 参考、稠密双 MX4、Kimi draft 的 BF16 推理 FA，合计 487,535 µs。 | 前两项的导出明确 `native_schema_compatible=false`；稠密双 MX4 可沿本仓原生 Matmul 建模，接入时核对 W 物理布局、scale 方向和预打包。推理 FA 的 12:2 Q/KV 头以及“历史缓存+当前 9 行”合同需单独适配；密集单头 FA 的 CA 不能直接复用。 |

例如 Pro/TP8/EP8/B1 中的共享专家量化乘是 FP4[9,7168]×FP4[7168,768]，导出 W scale 为 [112,768,2]、A scale 为 [9,112,2]。同一图的注意力投影 A8W4 为 FP8[9,7168]×FP4[7168,1536]，权重仍用 E8M0，**激活 scale 是 FP32[9]**。本仓 `matmul` 的 A8W4 路线 CA 校准使用两侧 MX/E8M0；这两个系统节点须分开处理。[精度合同与 950 路线](precision-coverage.md)

## 哪些输入能交给 950 的 CA

- **已有双端完整窗和预测。** 连续 FP4 动态量化可直接调用。匹配独立实现时，可选择逐 token FP8 的 DynamicQuantV2、FP4 CacheLoad 的 AntiMxQuant、路由 A4W4 分组投影的 GroupedMatmulV5。既有 BF16 Matmul、RMSNorm、稠密 FA、普通 LightningIndexer 及 MoE 组件也有完整窗，应逐项匹配本卡形状、精度、输出和历史。Flash 的 963 个输出头分组 A8W4 节点与路由专家的双 MXFP4 合同不同。[当前模型和误差](error-analysis.md)
- **已跑通，尚缺跨形状价格。** FP8 CacheLoad `[1,128,512]`、非连续 FP4 量化 `[9,128]`、BF16 同形残差加 `[9,7168]` 和 FP32 行广播乘 `[9,4]×[9,1]` 都能取得各自实现的整窗。后两项按导出输入选择 CANN ops-math 的 aclnnAdd 和 aclnnMul；每个 SKU 输出逐元素核对，并以实际 ELF 和全体核的 END 核验。应用框架如融合了基础算子，须按融合后的边界重新计费。[实测值与误差口径](error-analysis.md#已实测但尚无通用价格)
- **参考张量与现有库签名存在差别。** 账单中排第 5 的量化索引给 BF16 Q/K/权重和位置，top1024；950 的 V2 模式接收量化 Q/K 及其 scale。第 6 的稀疏共享 KV 用 FP8 Q 和四组单独的 scale，已登记的 MixedQuantSparseFlashMla 要求 FP16/BF16 Q 以及规定格式的打包 KV。第 8 的 draft FA 合并历史与临时 KV，其 BF16 GQA 样本 Q 为 [9,12,64]、新 K/V 为 [9,2,64]，另有 [1,131081,2,2,64] 历史存储。给单体库调用前需确定有效 KV 的接续和物化成本。普通 top512 索引和稠密 FA 的已有 CA 不能直接计入这三项误差。[量化索引 V2 的输入](https://gitcode.com/cann/ops-transformer/blob/14d022b1a4c3df263bf697534e3c25c7c992ff5d/attention/quant_lightning_indexer_v2/docs/aclnnQuantLightningIndexerV2.md) · [稀疏 MLA 的输入](https://gitcode.com/cann/ops-transformer/blob/14d022b1a4c3df263bf697534e3c25c7c992ff5d/attention/mixed_quant_sparse_flash_mla/docs/aclnnMixedQuantSparseFlashMla.md)
- **图中没有对应的独立 AI Core 时窗。** Flash 候选过滤在固定版库中缺少可消费候选列表的 950 前向。表末 13 个视图、别名或分配类名字的设备成本取决于是否物化；4 个集合通信与 EP 分发/合并名字的端到端时间依赖拓扑和其他 rank。其余有 950 注册项而未实跑的名字在末列逐项标明。[候选过滤的实现边界](examples/candidate-filtered-indexer.md)

## 全部原始调用名

按已定价时间由高到低排序。单位为这 27 次 rank-0 运行中所有同名节点的 µs 总和。样例列取一个真实节点的首个输入 → 首个输出，维度采用捕获逻辑 shape，FP4 的两个值物理打包为一字节。没有输出的原位节点记为「原位」。“950 CA 证据”一列按调用签名区分已运行、只有库注册、参考节点、视图和通信。注册名来自本机 CANN 9.2.1 的 `opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend950/`，表示有候选 AI Core 实现；还须实际派发、逐输出 golden、ELF 和全核退休才能产生误差数据。标为“参数核对”的项目还需逐节点检查其余张量、配置和实现路线。完整 schema、参考实现标签和系统基线时间均可经文末命令重建。

| # | raw_name | 节点数 | 基线 µs | 一例首入 → 首出 | operator-sim 接入条件 | 950 CA 证据 |
|---:|---|---:|---:|---|---|---|
| 1 | `inference::candidate_filtered_indexer` | 54 | 1,467,808.02 | `BF16[9,32,128] → I32[9,512]` | 两种筛选顺序的工作量；完整 CA 待测 | 本版 950 无候选消费前向；普通 LI 的 blockTable 不接受候选集合 |
| 2 | `npu::npu_grouped_matmul` | 4,545 | 848,425.21 | `E4M3[18,4096] → BF16[18,1024]` | A4W4 专家投影可选 V5 计时；head 分组 A8W4 待建 | A4W4 专家 V5 已实跑；A8W4 输出头待匹配 |
| 3 | `npu::npu_dynamic_mx_quant` | 17,748 | 791,696.08 | `BF16[9,7168] → E2M1[9,7168]` | BF16→MXFP4 + E8M0，独立精度入口 | 连续 FP4 已实跑；strided 另有两次派发 |
| 4 | `inference::cache_load` | 3,213 | 763,472.87 | `E4M3[1,128,512] → BF16[1,128,512]` | 按存储精度、scale 和布局选择，FP4 显式可定价 | 两种格式均已实跑；FP8 逐行只核验 [1,128,512] |
| 5 | `vllm_ascend::quant_lightning_indexer_reference` | 270 | 162,661.25 | `BF16[9,64,128] → I32[9,1024]` | top1024、连续参考 K；待量化索引路线 | 950 QuantLightningIndexerV2 要求量化 Q/K 和 scale，参考输入为 BF16 连续 K |
| 6 | `vllm_ascend::sparse_attn_sharedkv_reference` | 963 | 126,309.66 | `E4M3[9,16,512] → BF16[9,16,512]` | 需共享稀疏 KV、窗口、压缩和量化合同 | 950 MixedQuantSparseFlashMla 要求 Q=BF16/FP16，捕获 Q=FP8 且 scale 单列 |
| 7 | `npu::npu_quant_matmul` | 8,181 | 113,582.54 | `E2M1[9,7168] → BF16[9,768]` | 双侧 E8M0 的 A4W4；核对权重布局 | 稠密 A4W4 已测；激活 scale、W 转置与布局须相符 |
| 8 | `npu::npu_fused_infer_attention_score` | 45 | 84,981.86 | `BF16[9,12,64] → BF16[9,12,64]` | BF16 GQA，历史/临时 KV 和 LSE 待适配 | 950 注册 `fused_infer_attention_score_apt`；该图签名尚未核验 |
| 9 | `aten.mm.default` | 2,241 | 79,430.26 | `F32[9,7168] → F32[9,384]` | BF16 子集可核对 Matmul，FP32 另建 | BF16 Matmul 已测；FP32 另核对内部精度与输出 |
| 10 | `inference::kimi_dsa_indexer_score` | 216 | 76,685.33 | `U8[9,32,64] → I32[9,2048]` | FP4 packed 查询及 cache，top2048 待建 | 参考节点；缺可消费该 packed 编码的 top2048 实现 |
| 11 | `inference::mixed_precision_matmul` | 4,302 | 72,714.65 | `E4M3[9,7168] → BF16[9,1536]` | A/token-FP32 scale；双 MX 的 A8W4 需适配 | 参考节点；需逐行 FP32 激活 scale、MX 权重的混合实现 |
| 12 | `inference::lightning_indexer` | 54 | 49,337.64 | `BF16[9,32,128] → I32[9,512]` | BF16 top512；实图头组等与已校准 G=8 核对 | 普通 LightningIndexer 已测 G=8；实际 G=32 待单独测 |
| 13 | `inference::dynamic_mx_quant_fp8_reference` | 8,253 | 23,599.76 | `BF16[9,7168] → E4M3[9,7168]` | E4M3+逐 token FP32 scale，可选择独立量化合同 | 显式 DynamicQuantV2 已实跑；图给定 E4M3FN 值、每行 FP32 scale |
| 14 | `inference::attention_residual` | 1,683 | 20,169.07 | `BF16[9,7168] → BF16[9,7168]` | 核对 prefix 写回、有效 bank 与物化合同 | AttnResFwd 物化路线已测；需确认 bank、输出边界 |
| 15 | `_C_ascend::npu_hc_post` | 1,926 | 16,715.48 | `BF16[1,9,7168] → BF16[1,9,4,7168]` | 参数核对 vLLM hc_post，保留 token 及流顺序 | vLLM HcPost 原生组件已测；核对 N=4、流历史 |
| 16 | `npu::npu_rms_norm` | 8,244 | 15,805.37 | `BF16[9,7168] → BF16[9,7168]` | BF16 RMSNorm，核对 epsilon、rstd 输出和核状态 | BF16 RMSNorm 原生已测；检查 rstd 和布局 |
| 17 | `aten.add.Tensor` | 3,618 | 14,093.57 | `BF16[9,7168] → BF16[9,7168]` | ATen 独立原语待建／按实际融合边界去重 | BF16 [9,7168] 同形加法已双端实测；当前只有 CA |
| 18 | `inference::kimi_gated_dsa_attention` | 216 | 13,354.54 | `E2M1[9,12,512] → BF16[9,12,512]` | FP4 配门控、稀疏选取和 MLA，待建 | 参考节点；缺完整 FP4 稀疏、门控及 MLA 实现 |
| 19 | `_C_ascend::compressor` | 819 | 11,445.95 | `BF16[9,7168] → BF16[0,512]` | KV 压缩与量化，核对完成分组再建 | 950 注册 `compressor`；该图签名尚未核验 |
| 20 | `_C_ascend::npu_hc_pre_v2` | 1,926 | 10,758.09 | `BF16[9,4,7168] → BF16[9,7168]` | 融合 20 轮 Sinkhorn，需匹配输出与 kernel | 已有 PreSinkhorn 组件；V2 扩展 ABI 和输出需对齐 |
| 21 | `inference::matmul_rhs_transposed` | 828 | 9,630.36 | `BF16[9,7168] → F32[9,896]` | BF16×BF16→FP32，与现有 BF16 输出不同 | 950 注册 `mat_mul_v3`；该图签名尚未核验 |
| 22 | `inference::situ` | 1,665 | 8,909.79 | `BF16[9,8448] → BF16[9,4224]` | 独立门控激活，核对输入拆分 | 图给出 SITU 的 beta=4、linear_beta=25，需相同语义的实现 |
| 23 | `aten.clone.default` | 3,564 | 7,364.07 | `BF16[9,4,7168] → BF16[9,4,7168]` | 复制，需物理流量和实际物化 | 950 注册 `view_copy_apt`；该图签名尚未核验 |
| 24 | `aten._to_copy.default` | 1,818 | 6,580.04 | `I64[9] → I32[9]` | BF16→FP32 子集可核对 Cast，其余按类型建 | BF16→FP32 的 Cast 已测；其他源和目的精度另验 |
| 25 | `vllm_ascend::causal_conv1d_update_reference` | 621 | 6,462.37 | `BF16[9,4608] → BF16[9,4608]` | 卷积和历史更新，待建 | 950 注册 `causal_conv1d_apt`；该图签名尚未核验 |
| 26 | `npu::npu_swiglu` | 1,971 | 3,704.59 | `BF16[9,768] → BF16[9,384]` | 独立激活；融合 MX GMM 不可直接定价 | 950 有独立 SwiGLU；核对图上 silu 的 limit=10 是否实际生效 |
| 27 | `_C_ascend::inplace_partial_rotary_mul` | 3,321 | 2,026.97 | `BF16[9,1,16,512] → 原位` | 部分通道原位 RoPE 待建 | 950 注册 `inplace_partial_rotary_mul_apt`；该图签名尚未核验 |
| 28 | `inference::grouped_gated_rms_norm_reference` | 621 | 1,448.72 | `BF16[9,12,128] → BF16[9,12,128]` | 含 sigmoid 门控及分组，普通 RMSNorm 仅组件 | 参考节点；需带 gate 的核，已有 RMSNorm 只能计组件 |
| 29 | `inference::quantized_batched_matmul` | 432 | 1,250.89 | `E2M1[12,9,128] → BF16[12,9,512]` | 逐头批次 A4W4，需组间调度与 scale | 参考节点；需批次和 scale 布局匹配的量化实现 |
| 30 | `aten.cat.default` | 162 | 627.06 | `BF16[9,7168] → BF16[9,21504]` | ATen 独立原语待建／按实际融合边界去重 | 950 注册 `concat_v2_apt`；该图签名尚未核验 |
| 31 | `aten.mul.Tensor` | 108 | 537.71 | `F32[9,4] → F32[9,4]` | ATen 独立原语待建／按实际融合边界去重 | FP32 [9,4]×[9,1] 广播乘已双端实测；当前只有 CA |
| 32 | `aten.pow.Tensor_Scalar` | 36 | 537.48 | `F32[9,28672] → F32[9,28672]` | ATen 独立原语待建／按实际融合边界去重 | 950 注册 `pow_apt`；该图签名尚未核验 |
| 33 | `aten.mean.dim` | 90 | 520.69 | `BF16[9,4,7168] → BF16[9,7168]` | ATen 独立原语待建／按实际融合边界去重 | 950 注册 `reduce_mean_apt`；该图签名尚未核验 |
| 34 | `inference::mla_output_gate` | 216 | 503.88 | `BF16[9,1536] → BF16[9,1536]` | MLA 输出门控待建 | 参考节点；需确定门控融合边界 |
| 35 | `aten.sum.dim_IntList` | 36 | 335.92 | `F32[9,4,7168] → F32[9,7168]` | ATen 独立原语待建／按实际融合边界去重 | 950 注册 `reduce_sum_apt`；该图签名尚未核验 |
| 36 | `_C_ascend::moe_gating_top_k` | 828 | 192.44 | `F32[2,896] → F32[2,16]` | 先核对打分、group 与 bias；已测为 sigmoid | 原生 sigmoid Top-K 组件已测；核实打分、bias、组数 |
| 37 | `inference::engram_gate` | 18 | 182.42 | `BF16[9,4,5120] → BF16[9,4,5120]` | Engram 门控待建 | 参考节点；缺 Engram 门控合成实现 |
| 38 | `inference::kimi_dsa_indexer_query_quant` | 216 | 173.21 | `BF16[9,32,128] → U8[9,32,64]` | packed 查询、权重、scale 合同待建 | 参考节点；需查询量化和权重的完整 ABI |
| 39 | `inference::cache_store` | 1,926 | 150.51 | `BF16[9,512] → E4M3[1,128,512]` | 涉及索引散写与量化，读取模型仅覆盖反向 | 参考节点；需写入索引、量化及写回完整实现 |
| 40 | `aten.embedding.default` | 81 | 109.53 | `BF16[16160,7168] → BF16[9,7168]` | ATen 独立原语待建／按实际融合边界去重 | 950 注册 `embedding`；该图签名尚未核验 |
| 41 | `npu::npu_rotary_mul` | 243 | 108.42 | `BF16[1,9,1,64] → BF16[1,9,1,64]` | RoPE 待建 | 950 注册 `rotary_position_embedding_apt`；该图签名尚未核验 |
| 42 | `aten.masked_fill_.Scalar` | 51 | 99.85 | `BF16[9,7168] → BF16[9,7168]` | ATen 独立原语待建／按实际融合边界去重 | 950 注册 `masked_fill_apt`；该图签名尚未核验 |
| 43 | `_C_ascend::moe_gating_top_k_hash` | 963 | 95.72 | `F32[2,384] → F32[2,6]` | 含 token hash 表，需独立路由合同 | 含 token hash；普通 Top-K 的 CA 不覆盖 |
| 44 | `aten.index.Tensor` | 3,078 | 60.50 | `BF16[1048576,64] → BF16[9,64]` | ATen 独立原语待建／按实际融合边界去重 | 950 注册 `index_apt`；该图签名尚未核验 |
| 45 | `inference::c2_ring_compressor` | 27 | 42.22 | `F32[9,512] → BF16[4,512]` | 环形与完成行写回待建 | 参考节点；需值依赖的环形池实现 |
| 46 | `aten.constant_pad_nd.default` | 802 | 39.24 | `BF16[9,7168] → BF16[16,7168]` | ATen 独立原语待建／按实际融合边界去重 | 950 注册 `pad_v3_apt`；该图签名尚未核验 |
| 47 | `inference::candidate_blocks` | 27 | 21.00 | `I32[9,512] → I32[9,2048]` | 候选块生成独立计费；上游 block_size=8 | 候选块生产者；本版 950 未找到配套前向 |
| 48 | `aten.addmm.default` | 9 | 18.75 | `F32[1] → F32[8,1]` | 源为 FP32，现有 Addmm CA 为 BF16 | 已测 BF16 aclnnAddmm；此图 self、alpha、beta 及输出 dtype 需对齐 |
| 49 | `aten.neg.default` | 963 | 18.72 | `BF16[9,1,1,64] → BF16[9,1,1,64]` | ATen 独立原语待建／按实际融合边界去重 | 950 注册 `neg_apt`；该图签名尚未核验 |
| 50 | `aten.sigmoid.default` | 684 | 15.19 | `F32[9,4] → F32[9,4]` | FP32 beta sigmoid 组件可核对 strides/dtype | 连续 FP32 Sigmoid 已测；逐节点核对布局 |
| 51 | `inference::engram_dequantize` | 18 | 10.93 | `U8[9,864] → BF16[9,768]` | Engram packed 格式解码待建 | 参考节点；缺编码和解码的对应实现 |
| 52 | `inference::engram_lookup` | 18 | 9.19 | `I64[9,24] → U8[9,864]` | 索引散读、量化 scale 待建 | 参考节点；需实际哈希表和分组 scale 访问 |
| 53 | `inference::state_update` | 216 | 8.67 | `BF16[9,128] → U8[1,132114,68]` | Kimi 状态更新待建 | 参考节点；需确认增量状态布局和物化 |
| 54 | `_C_ascend::compressor_metadata` | 540 | 4.74 | `BF16[1048576,64] → BF16[2,1,1,64]` | 压缩位置与元信息待建 | 需显式 metadata 派发；AI CPU 元数据可能由系统层计 |
| 55 | `inference::engram_hash` | 9 | 1.73 | `I64[1,9] → I64[9,2,24]` | 哈希与索引元数据待建 | 参考节点；需实际哈希实现 |
| 56 | `aten.masked_fill.Scalar` | 51 | 0.07 | `I64[9] → I64[9]` | ATen 独立原语待建／按实际融合边界去重 | 950 注册 `masked_fill_apt`；该图签名尚未核验 |
| 57 | `aten.sub.Tensor` | 51 | 0.06 | `I64[9] → I64[9]` | ATen 独立原语待建／按实际融合边界去重 | 950 注册 `sub_apt`；该图签名尚未核验 |
| 58 | `inference::compressor_source_positions` | 54 | 0.06 | `I64[9] → I64[4]` | 压缩源位置元数据待建 | 参考节点；需确定位置映射与实际执行层 |
| 59 | `aten.lt.Scalar` | 51 | 0.03 | `I64[9] → bool[9]` | ATen 独立原语待建／按实际融合边界去重 | 950 注册 `less_apt`；该图签名尚未核验 |
| 60 | `aten.ge.Scalar` | 51 | 0.03 | `I64[9] → bool[9]` | ATen 独立原语待建／按实际融合边界去重 | 950 注册 `greater_equal_apt`；该图签名尚未核验 |
| 61 | `aten.rsqrt.default` | 36 | 0.02 | `F32[9,1] → F32[9,1]` | ATen 独立原语待建／按实际融合边界去重 | 950 注册 `rsqrt_apt`；该图签名尚未核验 |
| 62 | `aten.bitwise_or.Tensor` | 51 | 0.01 | `bool[9] → bool[9]` | ATen 独立原语待建／按实际融合边界去重 | 950 注册 `bitwise_or_apt`；该图签名尚未核验 |
| 63 | `aten.view.default` | 20,097 | 0.00 | `I64[1,9] → I64[9]` | 导出按视图/分配记零 | 图元数据或别名；若消费者要求物化须单测复制/转置 |
| 64 | `aten.unsqueeze.default` | 12,516 | 0.00 | `bool[9] → bool[9,1]` | 导出按视图/分配记零 | 图元数据或别名；若消费者要求物化须单测复制/转置 |
| 65 | `collective::all_reduce` | 2,496 | 0.00 | `BF16[9,7168] → BF16[9,7168]` | 通信占位，系统互连定价 | 多卡通信或通算融合；单卡 AI Core CA 不能核验端到端 |
| 66 | `aten.expand.default` | 36 | 0.00 | `BF16[9,1,7168] → BF16[9,4,7168]` | 导出按视图/分配记零 | 图元数据或别名；若消费者要求物化须单测复制/转置 |
| 67 | `aten.select.int` | 1,170 | 0.00 | `BF16[9,1,512] → BF16[9,512]` | 导出按视图/分配记零 | 图元数据或别名；若消费者要求物化须单测复制/转置 |
| 68 | `aten.alias.default` | 4,437 | 0.00 | `E4M3[1,128,512] → E4M3[1,128,512]` | 导出按视图/分配记零 | 图元数据或别名；若消费者要求物化须单测复制/转置 |
| 69 | `aten.empty.memory_format` | 4,815 | 0.00 | `∅ → I32[2]` | 导出按视图/分配记零 | 图元数据或别名；若消费者要求物化须单测复制/转置 |
| 70 | `aten.transpose.int` | 3,465 | 0.00 | `E2M1[7168,512] → E2M1[512,7168]` | 导出按视图/分配记零 | 图元数据或别名；若消费者要求物化须单测复制/转置 |
| 71 | `aten.slice.Tensor` | 9,171 | 0.00 | `E4M3[1,1033,512] → E4M3[1,1024,512]` | 导出按视图/分配记零 | 图元数据或别名；若消费者要求物化须单测复制/转置 |
| 72 | `aten.permute.default` | 3,852 | 0.00 | `E4M3[9,2,4096] → E4M3[2,9,4096]` | ATen 独立原语待建／按实际融合边界去重 | 图元数据或别名；若消费者要求物化须单测复制/转置 |
| 73 | `aten._unsafe_view.default` | 2,349 | 0.00 | `E4M3[2,9,4096] → E4M3[18,4096]` | 导出按视图/分配记零 | 图元数据或别名；若消费者要求物化须单测复制/转置 |
| 74 | `aten.squeeze.dim` | 2,169 | 0.00 | `BF16[1,9,4,7168] → BF16[9,4,7168]` | 导出按视图/分配记零 | 图元数据或别名；若消费者要求物化须单测复制/转置 |
| 75 | `aten.t.default` | 2,250 | 0.00 | `F32[384,7168] → F32[7168,384]` | 导出按视图/分配记零 | 图元数据或别名；若消费者要求物化须单测复制/转置 |
| 76 | `npu::npu_moe_distribute_dispatch_v2` | 1,791 | 0.00 | `E2M1[2,7168] → E2M1[48,7168]` | 通信占位，EP 路由、交换和本地搬运 | 多卡通信或通算融合；单卡 AI Core CA 不能核验端到端 |
| 77 | `npu::npu_moe_distribute_combine_v2` | 1,791 | 0.00 | `E4M3[48,7168] → BF16[2,7168]` | 通信占位，EP 反交换和合并 | 多卡通信或通算融合；单卡 AI Core CA 不能核验端到端 |
| 78 | `collective::all_gather` | 1,248 | 0.00 | `BF16[2,7168] → BF16[16,7168]` | 系统互连缺价 | 多卡通信或通算融合；单卡 AI Core CA 不能核验端到端 |
| 79 | `aten.split_with_sizes.default` | 1,890 | 0.00 | `BF16[9,6284] → BF16[9,1536]` | 导出按视图/分配记零 | 图元数据或别名；若消费者要求物化须单测复制/转置 |
| 80 | `vllm_ascend::recurrent_kda_reference` | 621 | 0.00 | `E2M1[1,9,12,128] → BF16[1,9,12,128]` | 基线 miss，FP4 状态链缺 CA，BF16/FP16 不代替 | 单链 BF16 KDA 已测；本图 packed FP4 状态合同待实现 |

## 复核与接入

准备系统导出的 `decode-950dt-128k` 目录。本次数据位于 `system-sim-workspace/preview-0.1.0-preview.1/experiments/decode-950dt-128k`，`summary.csv` 的 SHA-256 为 `ffbcb8f0cdcf788becf1a8edb2cce46b552c5fba371a5eaf86dbfdd3bbb32c89`，`operators.csv` 为 `4db19363e8925a6b031531a36aefa9ec42999ed4d024197b55f7d7b658cbd691`。下面的只读脚本逐个验证 native 图、L1 账单与系统聚合表，对每个原始调用名再按输出和精度属性分组。不需要 CANN 或真机。[审计代码](../calibration/system_workload.py)

```bash
python -m calibration.system_workload "$EXPORT" --verify-trace > audit.json
python -m calibration.system_workload "$EXPORT" --case v41-flash --op 'inference::candidate_filtered_indexer' > flash-indexer.jsonl
python -m calibration.system_workload "$EXPORT" --case v4-pro-c128-boundary --include-boundary
```

第二条按发生顺序返回每个节点的 rank、stage、ID、scope、依赖、输入输出的 dtype/shape/stride、底层 storage 容量、原始属性和 L1 时间。storage 容量是底层分配大小，切片的逻辑读写量应按有效 tensor 范围计。同一 backing storage 的别名、原地更新、融合算子的临时张量和 DP 其他 rank 的时间均不可直接相加。专家分组还要读取同目录 `expert-routing.csv` 的本卡行数；对冻结的 CA 样本是否覆盖该精度、布局与 shape，可再查[支持合同与 API](integration.md)。
