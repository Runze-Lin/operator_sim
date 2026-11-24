# 按合同查阅 CA 完整窗

共 36 个输入与计时合同、2,180 条记录。选合同可查看每次调用的本卡 shape、输入精度和前序、模型与 CA 设备耗时、查询耗时及 970DT 外推。同一 shape 的不同运行各占一行。

可比窗指能够同时取得逻辑 Roofline 基准时间和同一输入完整模型时间的窗口。下表分芯片与角色取全部可比窗的最小值和最大值。两个 U 使用同一个逻辑基准；原窗的分母是默认排程时间，冷读写的分母是所有逻辑读写都经过 HBM 时的场景预测。CA 误差只对原窗计分；970DT 只有外推。

[总误差与计时口径](../error-analysis.md) · [建模方法](../modeling-strategy.md) · [逐窗原始数据](../../calibration/reference/windows.jsonl)

已知物理下界覆盖 281/2,180 个窗口，其中模型低于下界 0 窗、CA 低于下界 0 窗。当前可核实的项为稠密 Cube 乘加。完整物理利用率还需要同窗 HBM 计数及适用的 Vector 指令吞吐。

## 开发回放

| 合同 | DT 可比/完整窗 | DT 原窗 U 范围 | DT 冷读写 U 范围 | PR 可比/完整窗 | PR 原窗 U 范围 | PR 冷读写 U 范围 |
| --- | --- | --- | --- | --- | --- | --- |
| [quant](quant.md) | 59/59 | 0.00115%-68.911% | 0.00115%-68.911% | 53/53 | 0.00386%-127.306% | 0.00386%-100.000% |
| [matmul](matmul.md) | 61/61 | 0.00906%-98.443% | 0.00906%-98.443% | 70/70 | 0.026%-136.136% | 0.026%-100.000% |
| [fa](fa.md) | 43/43 | 0.048%-3.051% | 0.048%-3.051% | 45/45 | 0.138%-8.717% | 0.138%-8.717% |
| [lightning_indexer](lightning_indexer.md) | 14/14 | 0.658%-2.402% | 0.658%-2.402% | 10/10 | 1.875%-5.284% | 1.875%-5.284% |
| [addmm](addmm.md) | 34/34 | 0.00545%-6.509% | 0.00545%-6.509% | 20/20 | 0.016%-15.438% | 0.016%-15.438% |
| [rmsnorm · compact CA](rmsnorm-compact-ca.md) | 57/57 | 0.010%-46.796% | 0.010%-46.796% | 0/0 | N/A | N/A |
| [rmsnorm · msopprof](rmsnorm-msopprof.md) | 126/126 | 0.051%-106.557% | 0.051%-100.000% | 136/136 | 0.029%-194.644% | 0.029%-100.000% |
| [rmsnorm · visible stream](rmsnorm-visible-stream.md) | 60/60 | 0.032%-51.210% | 0.032%-51.210% | 60/60 | 0.097%-128.472% | 0.097%-100.000% |
| [mhc_pre](mhc_pre.md) | 0/24 | N/A | N/A | 0/10 | N/A | N/A |
| [mhc_sinkhorn](mhc_sinkhorn.md) | 74/74 | 0.000905%-1.618% | 0.000905%-1.618% | 46/46 | 0.00164%-4.250% | 0.00164%-4.250% |
| [mhc_sinkhorn · known history](mhc_sinkhorn-known-history.md) | 50/50 | 0.000802%-1.569% | 0.000802%-1.569% | 36/36 | 0.00146%-4.118% | 0.00146%-4.118% |
| [mhc_pre_sinkhorn](mhc_pre_sinkhorn.md) | 0/24 | N/A | N/A | 0/20 | N/A | N/A |
| [mhc_post · shape](mhc_post-shape.md) | 151/151 | 0.012%-41.582% | 0.012%-41.582% | 88/88 | 0.034%-101.512% | 0.034%-100.000% |
| [mhc_post · stream](mhc_post-stream.md) | 151/151 | 0.00657%-37.884% | 0.00657%-37.884% | 88/88 | 0.019%-106.745% | 0.019%-100.000% |
| [mhc](mhc.md) | 0/24 | N/A | N/A | 0/10 | N/A | N/A |
| [vllm_hc_post](vllm_hc_post.md) | 15/15 | 0.073%-17.000% | 0.073%-17.000% | 15/15 | 0.208%-127.658% | 0.208%-100.000% |
| [kda_recurrent](kda_recurrent.md) | 14/14 | 0.091%-0.356% | 0.091%-0.356% | 10/10 | 0.255%-1.002% | 0.255%-1.002% |
| [fla_kda_v2](fla_kda_v2.md) | 18/18 | 0.155%-0.787% | 0.155%-0.787% | 17/17 | 0.442%-2.240% | 0.442%-2.240% |
| [gmm2](gmm2.md) | 15/15 | 0.032%-0.933% | 0.032%-0.933% | 10/10 | 0.093%-1.404% | 0.093%-1.404% |
| [grouped_matmul_swiglu_quant_v2](grouped_matmul_swiglu_quant_v2.md) | 15/15 | 0.030%-0.941% | 0.030%-0.941% | 10/10 | 0.084%-0.992% | 0.084%-0.992% |
| [moe_ffn](moe_ffn.md) | 15/15 | 0.030%-0.938% | 0.030%-0.938% | 10/10 | 0.087%-1.103% | 0.087%-1.103% |
| [moe_gating_top_k](moe_gating_top_k.md) | 12/12 | 0.00711%-0.434% | 0.00711%-0.434% | 10/10 | 0.051%-1.228% | 0.051%-1.228% |
| [cast_bf16_fp32](cast_bf16_fp32.md) | 33/33 | 0.0049%-157.680% | 0.0049%-100.000% | 32/32 | 0.014%-287.266% | 0.014%-100.000% |
| [sigmoid_fp32](sigmoid_fp32.md) | 23/23 | 0.000892%-127.615% | 0.000892%-100.000% | 20/20 | 0.060%-181.433% | 0.060%-100.000% |
| [attn_res_fwd](attn_res_fwd.md) | 21/21 | 0.019%-0.847% | 0.019%-0.847% | 21/21 | 0.055%-2.405% | 0.055%-2.405% |
| [attn_res_fwd · resident inference prefill](attn_res_fwd-resident-inference-prefill.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [matmul_mx_native](matmul_mx_native.md) | 14/14 | 0.113%-28.991% | 0.113%-28.991% | 14/14 | 0.336%-55.055% | 0.336%-55.055% |
| [matmul_mx_mixed](matmul_mx_mixed.md) | 14/14 | 0.114%-28.297% | 0.114%-28.297% | 14/14 | 0.355%-71.790% | 0.355%-71.790% |
| [quant_mxfp4](quant_mxfp4.md) | 4/4 | 0.018%-1.079% | 0.018%-1.079% | 4/4 | 0.056%-3.283% | 0.056%-3.283% |
| [candidate_filtered_indexer](candidate_filtered_indexer.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [cache_load_mxfp4](cache_load_mxfp4.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [cache_load_fp8](cache_load_fp8.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [routed_grouped_matmul_mxfp4](routed_grouped_matmul_mxfp4.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [quant_token_fp8](quant_token_fp8.md) | 3/3 | 0.094%-2.376% | 0.094%-2.376% | 3/3 | 0.268%-6.867% | 0.268%-6.867% |
| [cache_load_mxfp4_antimx](cache_load_mxfp4_antimx.md) | 2/2 | 0.00266%-0.386% | 0.00266%-0.386% | 2/2 | 0.00761%-1.104% | 0.00761%-1.104% |
| [routed_grouped_matmul_mxfp4_v5](routed_grouped_matmul_mxfp4_v5.md) | 3/3 | 0.168%-2.444% | 0.168%-2.444% | 3/3 | 0.480%-6.983% | 0.480%-6.983% |
| **全部合同窗** | 1101/1173 | 0.000802%-157.680% | 0.000802%-100.000% | 847/887 | 0.00146%-287.266% | 0.00146%-100.000% |

## 冻结独立

| 合同 | DT 可比/完整窗 | DT 原窗 U 范围 | DT 冷读写 U 范围 | PR 可比/完整窗 | PR 原窗 U 范围 | PR 冷读写 U 范围 |
| --- | --- | --- | --- | --- | --- | --- |
| [quant](quant.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [matmul](matmul.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [fa](fa.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [lightning_indexer](lightning_indexer.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [addmm](addmm.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [rmsnorm · compact CA](rmsnorm-compact-ca.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [rmsnorm · msopprof](rmsnorm-msopprof.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [rmsnorm · visible stream](rmsnorm-visible-stream.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [mhc_pre](mhc_pre.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [mhc_sinkhorn](mhc_sinkhorn.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [mhc_sinkhorn · known history](mhc_sinkhorn-known-history.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [mhc_pre_sinkhorn](mhc_pre_sinkhorn.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [mhc_post · shape](mhc_post-shape.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [mhc_post · stream](mhc_post-stream.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [mhc](mhc.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [vllm_hc_post](vllm_hc_post.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [kda_recurrent](kda_recurrent.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [fla_kda_v2](fla_kda_v2.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [gmm2](gmm2.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [grouped_matmul_swiglu_quant_v2](grouped_matmul_swiglu_quant_v2.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [moe_ffn](moe_ffn.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [moe_gating_top_k](moe_gating_top_k.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [cast_bf16_fp32](cast_bf16_fp32.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [sigmoid_fp32](sigmoid_fp32.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [attn_res_fwd](attn_res_fwd.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [attn_res_fwd · resident inference prefill](attn_res_fwd-resident-inference-prefill.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [matmul_mx_native](matmul_mx_native.md) | 10/10 | 5.010%-28.554% | 5.010%-28.554% | 10/10 | 13.751%-53.927% | 13.751%-53.927% |
| [matmul_mx_mixed](matmul_mx_mixed.md) | 10/10 | 3.516%-20.527% | 3.516%-20.527% | 10/10 | 9.544%-55.846% | 9.544%-55.846% |
| [quant_mxfp4](quant_mxfp4.md) | 10/10 | 0.072%-49.597% | 0.072%-49.597% | 10/10 | 0.220%-109.246% | 0.220%-100.000% |
| [candidate_filtered_indexer](candidate_filtered_indexer.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [cache_load_mxfp4](cache_load_mxfp4.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [cache_load_fp8](cache_load_fp8.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [routed_grouped_matmul_mxfp4](routed_grouped_matmul_mxfp4.md) | 0/0 | N/A | N/A | 0/0 | N/A | N/A |
| [quant_token_fp8](quant_token_fp8.md) | 10/10 | 0.186%-4.677% | 0.186%-4.677% | 10/10 | 0.531%-13.349% | 0.531%-13.349% |
| [cache_load_mxfp4_antimx](cache_load_mxfp4_antimx.md) | 10/10 | 0.00133%-0.864% | 0.00133%-0.864% | 10/10 | 0.00381%-2.528% | 0.00381%-2.528% |
| [routed_grouped_matmul_mxfp4_v5](routed_grouped_matmul_mxfp4_v5.md) | 10/10 | 0.064%-64.608% | 0.064%-64.608% | 10/10 | 0.182%-83.912% | 0.182%-83.912% |
| **全部合同窗** | 60/60 | 0.00133%-64.608% | 0.00133%-64.608% | 60/60 | 0.00381%-109.246% | 0.00381%-100.000% |

## 只有 CA 实测的输入

[FP8 cache 解包、非连续 FP4 量化、残差加和广播乘的双端整窗](measured-only.md)。这些输入还没有可独立评价的价格。

## 其它精度与实现

表内合同以参考数据的当前登记为准。表外精度与实现（如 A8W4 专家 GMM 或低精度 FA）应按其各自的完整窗确定 CA 误差与逻辑时间系数，缺少匹配证据时为 N/A。[精度与实现覆盖](../precision-coverage.md)

## 复算与更新

在安装了本仓库包的环境，从仓库根目录执行 `python -m calibration.report_pages --check` 可重算全部页面并检查归档耗时。更新参考记录后，执行 `python -m calibration.report_pages` 重新生成。[生成工具](../../calibration/report_pages.py)
