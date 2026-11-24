# 附录 B　逐合同的双端误差与精度

DT、PR 栏分别给出不同形状数/窗口数，以及 WAPE/最大绝对误差。开发和独立窗数跨两端合计。冻结独立优先，未有独立数据的合同按开发回放展示。N/A 表示该问题缺可比证据。点击合同名称可查每个 shape 的 CA、预测、970DT 外推和冷读写 U 范围。

| 执行合同 | 主精度 → 输出 | 数据角色 | 950DT | 950PR |
|---|---|---|---|---|
| [DynamicMXQuant](../errors/quant.md) | BF16 → MXFP8 | 回放 · 112/0 | 58/59 · 4.74% / 13.61% | 53/53 · 7.71% / 16.83% |
| [DynamicMXQuant，MXFP4](../errors/quant_mxfp4.md) | BF16 → E2M1、E8M0 scale | 独立 · 8/20 | 10/10 · 7.85% / 15.00% | 10/10 · 9.70% / 18.56% |
| [逐 token FP8 量化](../errors/quant_token_fp8.md) | BF16 → E4M3FN、每行 FP32 scale | 独立 · 6/20 | 10/10 · 7.43% / 22.77% | 10/10 · 8.54% / 21.29% |
| [CacheLoad，显式 AntiMxQuant](../errors/cache_load_mxfp4_antimx.md) | E2M1、E8M0 → BF16 | 独立 · 4/20 | 10/10 · 9.93% / 21.86% | 10/10 · 11.87% / 26.53% |
| [路由专家，显式 GroupedMatmulV5](../errors/routed_grouped_matmul_mxfp4_v5.md) | E2M1 × E2M1、双侧 E8M0 → BF16 | 独立 · 6/20 | 10/10 · 9.82% / 21.07% | 10/10 · 6.30% / 10.59% |
| [Matmul](../errors/matmul.md) | BF16 × BF16 → BF16 | 回放 · 131/0 | 59/61 · 3.66% / 29.01% | 70/70 · 5.20% / 15.06% |
| [Matmul，原生 A4W4](../errors/matmul_mx_native.md) | MXFP4 × MXFP4 → BF16 | 独立 · 28/20 | 10/10 · 3.89% / 13.59% | 10/10 · 6.87% / 14.50% |
| [Matmul，展开 A8W4](../errors/matmul_mx_mixed.md) | MXFP8 × MXFP4 → BF16 | 独立 · 28/20 | 10/10 · 8.80% / 35.73% | 10/10 · 11.32% / 56.67% |
| [FA](../errors/fa.md) | BF16 Q/K/V → BF16，FP32 统计 | 回放 · 88/0 | 38/43 · 1.84% / 12.43% | 45/45 · 5.05% / 17.72% |
| [Lightning Indexer](../errors/lightning_indexer.md) | BF16 Q/K/权重 → INT32 索引 | 回放 · 24/0 | 13/14 · 5.87% / 15.67% | 10/10 · 4.14% / 10.88% |
| [候选块 Indexer，BF16 参考](../errors/candidate_filtered_indexer.md) | BF16 Q/K/权重、INT64 位置、INT32 候选块 → INT32 | 原生融合 · 0/0；显式组合另表 | 0/0 · N/A | 0/0 · N/A |
| [Flash A5 QSLI 含 metadata、前后的完整调用](../errors/flash-a5-qsli.md) | MXFP4 Q/K、E8M0 scale、FP32 权重、BF16 头归约 → INT32 位置 | 部署整窗 · 0/0；普通 QLI 对照另表 | 0/0 · N/A | 0/0 · N/A |
| [Flash A5 QSLI，原生 kernel 固定分核表](../errors/flash-a5-qsli.md) | MXFP4、E8M0 → INT32 | 原生拆行开发回放 6/0（每端 2 个几何、3 个窗） | 2/3 · 10.21% / 16.29% | 2/3 · 8.21% / 9.93% |
| [Flash A5 QSLI，32-query 并发 prefill](../errors/flash-a5-qsli.md) | MXFP4、E8M0 → INT32 | 已知页序，开发 2/0 | 1/1 · 3.79% / 3.79% | 1/1 · 7.68% / 7.68% |
| [Flash A5 QSLI，64-query 行间候选页变化](../errors/flash-a5-qsli.md) | MXFP4、E8M0 → INT32 | 已知页序，开发 1/0 | 1/1 · 7.40% / 7.40% | 0/0 · N/A |
| [Flash A5 QSLI，只传 shape](../errors/flash-a5-qsli.md) | MXFP4 Q/K、E8M0 → INT32 | 同形布局 2/1；只对揭盲窗计分 | 1/1 · 9.44% / 9.44% | 0/0 · N/A |
| [Flash A5 QSLI，输入实际候选页序](../errors/flash-a5-qsli.md) | MXFP4 Q/K、E8M0 → INT32 | 同形布局 2/1；只对揭盲窗计分 | 1/1 · 9.44% / 9.44% | 0/0 · N/A |
| [Addmm](../errors/addmm.md) | BF16 矩阵和 bias → BF16 | 回放 · 54/0 | 33/34 · 5.67% / 12.82% | 20/20 · 5.71% / 13.61% |
| [RMSNorm，compact CA](../errors/rmsnorm-compact-ca.md) | BF16 → BF16 Y、FP32 rstd | 回放 · 57/0 | 53/57 · 19.03% / 46.64% | 0/0 · N/A |
| [RMSNorm，msopprof](../errors/rmsnorm-msopprof.md) | BF16 → BF16 Y、FP32 rstd | 回放 · 262/0 | 90/126 · 17.94% / 49.86% | 98/136 · 18.55% / 49.61% |
| [RMSNorm，可见前序](../errors/rmsnorm-visible-stream.md) | BF16 → BF16 Y、FP32 rstd | 回放 · 120/0 | 30/60 · 5.35% / 20.16% | 30/60 · 5.98% / 24.37% |
| [mHC Pre](../errors/mhc_pre.md) | BF16 激活、FP32 投影参数 → BF16、FP32 | 回放 · 34/0 | 23/24 · 3.21% / 9.62% | 10/10 · 6.10% / 10.75% |
| [mHC Sinkhorn](../errors/mhc_sinkhorn.md) | FP32 → FP32 | 回放 · 120/0 | 50/74 · 10.83% / 37.01% | 36/46 · 10.79% / 35.64% |
| [mHC Sinkhorn，已知前序](../errors/mhc_sinkhorn-known-history.md) | FP32 → FP32 | 回放 · 86/0 | 48/50 · 4.07% / 31.41% | 35/36 · 3.73% / 12.92% |
| [mHC PreSinkhorn](../errors/mhc_pre_sinkhorn.md) | BF16 激活、FP32 参数 → BF16、FP32 | 回放 · 44/0 | 23/24 · 5.97% / 12.89% | 20/20 · 5.39% / 14.00% |
| [mHC Post，仅 shape](../errors/mhc_post-shape.md) | BF16 激活、FP32 系数 → BF16 | 回放 · 239/0 | 113/151 · 27.94% / 63.56% | 56/88 · 28.02% / 67.40% |
| [mHC Post，有序流](../errors/mhc_post-stream.md) | BF16 激活、FP32 系数 → BF16 | 回放 · 239/0 | 113/151 · 8.38% / 45.13% | 56/88 · 6.83% / 31.78% |
| [mHC 拆分链](../errors/mhc.md) | BF16、FP32 → BF16 | 回放 · 34/0 | 23/24 · 2.61% / 7.20% | 10/10 · 4.19% / 9.37% |
| [vLLM HcPost](../errors/vllm_hc_post.md) | BF16 激活、FP32 系数 → BF16 | 回放 · 30/0 | 15/15 · 2.32% / 5.33% | 15/15 · 8.49% / 17.79% |
| [KDA recurrent（单状态链）](../errors/kda_recurrent.md) | BF16 Q/K/V、FP32 状态 → BF16、FP32 | 回放 · 24/0 | 13/14 · 6.53% / 17.55% | 10/10 · 4.35% / 10.41% |
| [KDA chunk FLA V2](../errors/fla_kda_v2.md) | BF16 Q/K/V 与 gate、FP32 beta/状态 → BF16、FP32 | 回放 · 35/0 | 18/18 · 2.55% / 6.76% | 17/17 · 1.32% / 3.15% |
| [MoE GMM2](../errors/gmm2.md) | MXFP8 × MXFP8 → BF16 | 回放 · 25/0 | 14/15 · 8.22% / 26.86% | 10/10 · 7.91% / 20.59% |
| [MoE 路由专家 A4W4 分组乘法](../errors/routed_grouped_matmul_mxfp4.md) | MXFP4 × MXFP4、双侧 E8M0 → BF16 | 待测 · 0/0 | 0/0 · N/A | 0/0 · N/A |
| [GMM1–SwiGLU–MXQuant](../errors/grouped_matmul_swiglu_quant_v2.md) | MXFP8 × MXFP8 → MXFP8 | 回放 · 25/0 | 14/15 · 8.44% / 32.74% | 10/10 · 10.77% / 34.79% |
| [MoE FFN](../errors/moe_ffn.md) | MXFP8 × MXFP8 → BF16 | 回放 · 25/0 | 14/15 · 6.00% / 23.01% | 10/10 · 7.83% / 23.61% |
| [MoE sigmoid Top-K](../errors/moe_gating_top_k.md) | FP32 → FP32、INT32 | 回放 · 22/0 | 12/12 · 7.72% / 16.18% | 10/10 · 5.72% / 11.02% |
| [Cast BF16→FP32](../errors/cast_bf16_fp32.md) | BF16 → FP32 | 回放 · 65/0 | 33/33 · 5.03% / 17.10% | 32/32 · 3.31% / 19.83% |
| [KDA beta Sigmoid](../errors/sigmoid_fp32.md) | FP32 → FP32 | 回放 · 43/0 | 23/23 · 5.98% / 12.85% | 20/20 · 16.39% / 26.96% |
| [Kimi AttnResFwd，物化输出](../errors/attn_res_fwd.md) | BF16 激活 → BF16 | 回放 · 42/0 | 21/21 · 4.29% / 9.76% | 21/21 · 5.04% / 11.42% |
| [Kimi AttnResFwd，UB 驻留推理 prefill](../errors/attn_res_fwd-resident-inference-prefill.md) | BF16 激活 → BF16 | 待测 · 0/0 | 0/0 · N/A | 0/0 · N/A |
| [CacheLoad，MXFP4](../errors/cache_load_mxfp4.md) | packed E2M1、E8M0 scale → BF16 | 待测 · 0/0 | 0/0 · N/A | 0/0 · N/A |
| [CacheLoad，FP8](../errors/cache_load_fp8.md) | E4M3FN、逐行 FP32 scale → BF16 | 待测 · 0/0 | 0/0 · N/A | 0/0 · N/A |
| [CacheLoad，显式 AscendAntiQuant](../errors/measured-only.md) | E4M3FN、FP32 → BF16 | 仅实测 · 各 1 窗 | 1/1 · N/A | 1/1 · N/A |
| [动态 MXFP4 量化，行跨步视图](../errors/measured-only.md) | BF16 → E2M1、E8M0 | 仅实测 · 各 1 窗 | 1/1 · N/A | 1/1 · N/A |
| [残差加](../errors/measured-only.md) | BF16 → BF16 | 仅实测 · 各 1 窗 | 1/1 · N/A | 1/1 · N/A |
| [按行广播乘](../errors/measured-only.md) | FP32 → FP32 | 仅实测 · 各 1 窗 | 1/1 · N/A | 1/1 · N/A |

[完整证据、各合同 U 范围与计分定义](../error-analysis.md)
