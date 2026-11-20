# 以当前公开调用重新核对两端误差

下表汇总各来源中已归档且按该口径可计时的样本，包含开发回放。每格是“不同几何数 / 窗数，WAPE / 最大绝对误差”。同一时序的不同实现、同形的前序、不同计时提供者按表内协议处理。单独的新形验收、分组尾差及原生 wrapper 在总误差分析列出。

| 算子及输入合同 | 950DT | 950PR | 双端是否通过此表两项限值 | 数据 |
|---|---:|---:|---|---|
| addmm | 33 / 34 · 5.67% / 12.82% | 20 / 20 · 5.71% / 13.61% | 通过 | [replay.json](replay.json) |
| attn_res_fwd | 21 / 21 · 4.29% / 9.76% | 21 / 21 · 5.04% / 11.42% | 通过 | [replay.json](replay.json) |
| cast_bf16_fp32 | 33 / 33 · 5.03% / 17.10% | 32 / 32 · 3.31% / 19.83% | 通过 | [cast/score.json](../v0.28/cast/score.json) · [replay.json](replay.json) |
| fa | 38 / 43 · 1.84% / 12.43% | 44 / 44 · 5.70% / 17.72% | 通过 | [replay.json](replay.json) |
| fla_kda_v2 | 18 / 18 · 2.55% / 6.76% | 17 / 17 · 1.32% / 3.15% | 通过 | [replay.json](replay.json) |
| gmm2 | 14 / 15 · 8.22% / 26.86% | 10 / 10 · 7.91% / 20.59% | 通过 | [replay.json](replay.json) |
| grouped_matmul_swiglu_quant_v2 | 14 / 15 · 8.44% / 32.74% | 10 / 10 · 10.77% / 34.79% | 通过 | [replay.json](replay.json) |
| kda_chunk · fused V1 | 0 · N/A | 0 · N/A | 缺合格时延 | [kda/report.md](kda/report.md) |
| kda_recurrent | 13 / 14 · 6.53% / 17.55% | 10 / 10 · 4.35% / 10.41% | 通过 | [replay.json](replay.json) |
| lightning_indexer | 13 / 14 · 12.40% / 31.71% | 10 / 10 · 11.88% / 26.51% | 通过 | [replay.json](replay.json) |
| matmul | 47 / 49 · 15.56% / 107.49% | 59 / 59 · 12.26% / 108.81% | 有超限 | [matmul/score.json](matmul/score.json) |
| mhc | 23 / 24 · 2.61% / 7.20% | 10 / 10 · 4.19% / 9.37% | 通过 | [replay.json](replay.json) |
| mhc_post · shape | 113 / 151 · 30.29% / 79.81% | 56 / 88 · 31.07% / 78.23% | 有超限 | [post/score.json](post/score.json) |
| mhc_post · visible stream | 113 / 151 · 8.20% / 58.52% | 56 / 88 · 7.13% / 57.22% | 有超限 | [post/score.json](post/score.json) |
| mhc_pre | 23 / 24 · 3.21% / 9.62% | 10 / 10 · 6.10% / 10.75% | 通过 | [replay.json](replay.json) |
| mhc_pre_sinkhorn | 23 / 24 · 5.97% / 12.89% | 20 / 20 · 5.39% / 14.00% | 通过 | [replay.json](replay.json) |
| mhc_sinkhorn | 44 / 57 · 13.88% / 39.20% | 31 / 52 · 11.90% / 80.84% | 有超限 | [fusion/independent.json](../v0.28/fusion/independent.json) · [fusion/replay.json](../v0.28/fusion/replay.json) · [replay.json](replay.json) |
| moe_ffn | 14 / 15 · 6.00% / 23.01% | 10 / 10 · 7.83% / 23.61% | 通过 | [replay.json](replay.json) |
| moe_gating_top_k | 12 / 12 · 7.72% / 16.18% | 10 / 10 · 5.72% / 11.02% | 通过 | [replay.json](replay.json) |
| quant | 58 / 59 · 4.74% / 13.61% | 53 / 53 · 7.71% / 16.83% | 通过 | [replay.json](replay.json) |
| rmsnorm · compact CA | 53 / 57 · 19.03% / 46.64% | 0 · N/A | 缺合格时延 | [rms/stream-replay.json](../v0.28/rms/stream-replay.json) |
| rmsnorm · msopprof | 90 / 126 · 17.94% / 49.86% | 98 / 136 · 18.55% / 49.61% | 通过 | [rms/stream-replay.json](../v0.28/rms/stream-replay.json) |
| rmsnorm · visible stream | 30 / 60 · 5.35% / 20.16% | 30 / 60 · 5.98% / 24.37% | 通过 | [rms/stream-replay.json](../v0.28/rms/stream-replay.json) |
| sigmoid_fp32 | 23 / 23 · 5.98% / 12.85% | 20 / 20 · 16.39% / 26.96% | 通过 | [replay.json](replay.json) |
| vllm_hc_post | 15 / 15 · 2.32% / 5.33% | 14 / 14 · 2.43% / 5.59% | 通过 | [replay.json](replay.json) |

Matmul 此处用公开路由。RMSNorm 和内置 Post 的有序行使用从真实顺序可重建的历史，compact 和 msopprof 指令口径分列。KDA V2 三段和扩展 HcPost 各有单独的实现合同。[全部独立验收和未满足的分组](../../docs/error-analysis.md) · [各协议来源和未舍入数值](evaluation.json)
