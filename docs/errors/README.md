# 按合同查阅 CA 完整窗

共 36 个输入与计时合同、2,180 条记录。选合同可查看每次调用的本卡 shape、输入精度和前序、模型与 CA 设备耗时、查询耗时及 970DT 外推。同一 shape 的不同运行各占一行。

可比窗指能够同时取得等效 Roofline 理想时间和同一输入完整模型时间的窗口。合同汇总按 `Σ理想时间 / Σ模型时间` 计算，分子和分母取完全相同的可比窗。单次占比可超过 100%，例如来自 L2 的数据也可能计入逻辑搬运量。这里的分母是设备预测，CA 用于校验预测误差。970DT 列只有外推。

[总误差与计时口径](../error-analysis.md) · [建模方法](../modeling-strategy.md) · [逐窗原始数据](../../calibration/reference/windows.jsonl)

## 开发回放

| 合同 | DT 可比/完整窗 | DT 等效利用率 | PR 可比/完整窗 | PR 等效利用率 |
| --- | --- | --- | --- | --- |
| [quant](quant.md) | 59/59 | 16.341% | 53/53 | 35.715% |
| [matmul](matmul.md) | 61/61 | 59.767% | 70/70 | 94.568% |
| [fa](fa.md) | 43/43 | 1.763% | 45/45 | 5.466% |
| [lightning_indexer](lightning_indexer.md) | 14/14 | 1.518% | 10/10 | 3.362% |
| [addmm](addmm.md) | 34/34 | 1.644% | 20/20 | 5.941% |
| [rmsnorm · compact CA](rmsnorm-compact-ca.md) | 57/57 | 7.340% | 0/0 | N/A |
| [rmsnorm · msopprof](rmsnorm-msopprof.md) | 126/126 | 28.625% | 136/136 | 65.821% |
| [rmsnorm · visible stream](rmsnorm-visible-stream.md) | 60/60 | 7.239% | 60/60 | 20.695% |
| [mhc_pre](mhc_pre.md) | 0/24 | N/A | 0/10 | N/A |
| [mhc_sinkhorn](mhc_sinkhorn.md) | 74/74 | 0.075% | 46/46 | 0.288% |
| [mhc_sinkhorn · known history](mhc_sinkhorn-known-history.md) | 50/50 | 0.106% | 36/36 | 0.349% |
| [mhc_pre_sinkhorn](mhc_pre_sinkhorn.md) | 0/24 | N/A | 0/20 | N/A |
| [mhc_post · shape](mhc_post-shape.md) | 151/151 | 8.732% | 88/88 | 31.538% |
| [mhc_post · stream](mhc_post-stream.md) | 151/151 | 7.855% | 88/88 | 28.059% |
| [mhc](mhc.md) | 0/24 | N/A | 0/10 | N/A |
| [vllm_hc_post](vllm_hc_post.md) | 15/15 | 2.820% | 15/15 | 59.923% |
| [kda_recurrent](kda_recurrent.md) | 14/14 | 0.160% | 10/10 | 0.438% |
| [fla_kda_v2](fla_kda_v2.md) | 18/18 | 0.384% | 17/17 | 1.120% |
| [gmm2](gmm2.md) | 15/15 | 0.293% | 10/10 | 0.456% |
| [grouped_matmul_swiglu_quant_v2](grouped_matmul_swiglu_quant_v2.md) | 15/15 | 0.241% | 10/10 | 0.387% |
| [moe_ffn](moe_ffn.md) | 15/15 | 0.257% | 10/10 | 0.409% |
| [moe_gating_top_k](moe_gating_top_k.md) | 12/12 | 0.152% | 10/10 | 0.502% |
| [cast_bf16_fp32](cast_bf16_fp32.md) | 33/33 | 109.093% | 32/32 | 163.149% |
| [sigmoid_fp32](sigmoid_fp32.md) | 23/23 | 79.042% | 20/20 | 131.887% |
| [attn_res_fwd](attn_res_fwd.md) | 21/21 | 0.272% | 21/21 | 0.774% |
| [attn_res_fwd · resident inference prefill](attn_res_fwd-resident-inference-prefill.md) | 0/0 | N/A | 0/0 | N/A |
| [matmul_mx_native](matmul_mx_native.md) | 14/14 | 9.519% | 14/14 | 24.765% |
| [matmul_mx_mixed](matmul_mx_mixed.md) | 14/14 | 9.433% | 14/14 | 25.145% |
| [quant_mxfp4](quant_mxfp4.md) | 4/4 | 0.523% | 4/4 | 1.597% |
| [candidate_filtered_indexer](candidate_filtered_indexer.md) | 0/0 | N/A | 0/0 | N/A |
| [cache_load_mxfp4](cache_load_mxfp4.md) | 0/0 | N/A | 0/0 | N/A |
| [cache_load_fp8](cache_load_fp8.md) | 0/0 | N/A | 0/0 | N/A |
| [routed_grouped_matmul_mxfp4](routed_grouped_matmul_mxfp4.md) | 0/0 | N/A | 0/0 | N/A |
| [quant_token_fp8](quant_token_fp8.md) | 3/3 | 1.310% | 3/3 | 3.756% |
| [cache_load_mxfp4_antimx](cache_load_mxfp4_antimx.md) | 2/2 | 0.235% | 2/2 | 0.673% |
| [routed_grouped_matmul_mxfp4_v5](routed_grouped_matmul_mxfp4_v5.md) | 3/3 | 1.344% | 3/3 | 3.839% |
| **全部合同窗** | 1101/1173 | 22.119% | 847/887 | 57.999% |

## 冻结独立

| 合同 | DT 可比/完整窗 | DT 等效利用率 | PR 可比/完整窗 | PR 等效利用率 |
| --- | --- | --- | --- | --- |
| [quant](quant.md) | 0/0 | N/A | 0/0 | N/A |
| [matmul](matmul.md) | 0/0 | N/A | 0/0 | N/A |
| [fa](fa.md) | 0/0 | N/A | 0/0 | N/A |
| [lightning_indexer](lightning_indexer.md) | 0/0 | N/A | 0/0 | N/A |
| [addmm](addmm.md) | 0/0 | N/A | 0/0 | N/A |
| [rmsnorm · compact CA](rmsnorm-compact-ca.md) | 0/0 | N/A | 0/0 | N/A |
| [rmsnorm · msopprof](rmsnorm-msopprof.md) | 0/0 | N/A | 0/0 | N/A |
| [rmsnorm · visible stream](rmsnorm-visible-stream.md) | 0/0 | N/A | 0/0 | N/A |
| [mhc_pre](mhc_pre.md) | 0/0 | N/A | 0/0 | N/A |
| [mhc_sinkhorn](mhc_sinkhorn.md) | 0/0 | N/A | 0/0 | N/A |
| [mhc_sinkhorn · known history](mhc_sinkhorn-known-history.md) | 0/0 | N/A | 0/0 | N/A |
| [mhc_pre_sinkhorn](mhc_pre_sinkhorn.md) | 0/0 | N/A | 0/0 | N/A |
| [mhc_post · shape](mhc_post-shape.md) | 0/0 | N/A | 0/0 | N/A |
| [mhc_post · stream](mhc_post-stream.md) | 0/0 | N/A | 0/0 | N/A |
| [mhc](mhc.md) | 0/0 | N/A | 0/0 | N/A |
| [vllm_hc_post](vllm_hc_post.md) | 0/0 | N/A | 0/0 | N/A |
| [kda_recurrent](kda_recurrent.md) | 0/0 | N/A | 0/0 | N/A |
| [fla_kda_v2](fla_kda_v2.md) | 0/0 | N/A | 0/0 | N/A |
| [gmm2](gmm2.md) | 0/0 | N/A | 0/0 | N/A |
| [grouped_matmul_swiglu_quant_v2](grouped_matmul_swiglu_quant_v2.md) | 0/0 | N/A | 0/0 | N/A |
| [moe_ffn](moe_ffn.md) | 0/0 | N/A | 0/0 | N/A |
| [moe_gating_top_k](moe_gating_top_k.md) | 0/0 | N/A | 0/0 | N/A |
| [cast_bf16_fp32](cast_bf16_fp32.md) | 0/0 | N/A | 0/0 | N/A |
| [sigmoid_fp32](sigmoid_fp32.md) | 0/0 | N/A | 0/0 | N/A |
| [attn_res_fwd](attn_res_fwd.md) | 0/0 | N/A | 0/0 | N/A |
| [attn_res_fwd · resident inference prefill](attn_res_fwd-resident-inference-prefill.md) | 0/0 | N/A | 0/0 | N/A |
| [matmul_mx_native](matmul_mx_native.md) | 10/10 | 13.002% | 10/10 | 32.742% |
| [matmul_mx_mixed](matmul_mx_mixed.md) | 10/10 | 12.113% | 10/10 | 30.025% |
| [quant_mxfp4](quant_mxfp4.md) | 10/10 | 15.073% | 10/10 | 42.096% |
| [candidate_filtered_indexer](candidate_filtered_indexer.md) | 0/0 | N/A | 0/0 | N/A |
| [cache_load_mxfp4](cache_load_mxfp4.md) | 0/0 | N/A | 0/0 | N/A |
| [cache_load_fp8](cache_load_fp8.md) | 0/0 | N/A | 0/0 | N/A |
| [routed_grouped_matmul_mxfp4](routed_grouped_matmul_mxfp4.md) | 0/0 | N/A | 0/0 | N/A |
| [quant_token_fp8](quant_token_fp8.md) | 10/10 | 1.584% | 10/10 | 4.532% |
| [cache_load_mxfp4_antimx](cache_load_mxfp4_antimx.md) | 10/10 | 0.252% | 10/10 | 0.727% |
| [routed_grouped_matmul_mxfp4_v5](routed_grouped_matmul_mxfp4_v5.md) | 10/10 | 15.676% | 10/10 | 35.287% |
| **全部合同窗** | 60/60 | 10.701% | 60/60 | 27.372% |

## 只有 CA 实测的输入

[FP8 cache 解包、非连续 FP4 量化、残差加和广播乘的双端整窗](measured-only.md)。这些输入还没有可独立评价的价格。

## 其它精度与实现

表内合同以参考数据的当前登记为准。表外精度与实现（如 A8W4 专家 GMM 或低精度 FA）应按其各自的完整窗确定 CA 误差与等效利用率，缺少匹配证据时为 N/A。[精度与实现覆盖](../precision-coverage.md)

## 复算与更新

在安装了本仓库包的环境，从仓库根目录执行 `python -m calibration.report_pages --check` 可重算全部页面并检查归档耗时。更新参考记录后，执行 `python -m calibration.report_pages` 重新生成。[生成工具](../../calibration/report_pages.py)
