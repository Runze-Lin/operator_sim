# 系统调用的 Vector 方法对照

按 **rank-local 原始调用名**列出三模型 prefill/decode 捕获。每个方法使用相同的 shape、dtype、布局、并行元数据和芯片；更换的是 Vector 阶段，而非 Cube 吞吐、HBM 假设或系统 DES。第一张表是系统调用维度的索引；校准数据只属于具体子合同，不能外推为整行或整图实测。

调用 `FullOperatorSimBackend(vector_model="auto")`；可显式传 `"calibrated"`（原排程，包括未校准的旧设计）或 `"instruction_classes"`。只接原生 lowerings 的 `OperatorSimBackend` 也支持相同选项。选择器进入系统缓存标识。严格未知签名继续返回 unsupported。

两法保留相同首发、分核、UB 分页与资源依赖。指令法将有意义的工作（如注意力每个 score 的 exp、规约和量化；A8W4 每个展开权重的解包、转换与写 UB）折成寄存器发起次数。对选中忙核，按 256 B 寄存器、计算位宽、分给该核的元素数向上取整；总发起、单类间隔、UB 端口、ALU 端口取最大周期数。默认 2 发起槽、2 端口、exp/divide 4 周期是待检验的架构先验；可在芯片配置中更换。新实现默认类别法，沿用已验证实现的路径默认原法。通信本身不对应 VF；有本地 SUM/反量化融合的通信只替换本地计算。视图及纯 DMA 无此阶段。

本次 86 个原始名中，有 54 个至少在一个系统样本上改变整窗；其余可能是无 VF、同价、被 Cube/搬运/通信遮住，或低于排程分辨率。35 个原始名有可配对的计时合同，共 684 个 DT/PR 窗口；普通 H64 LightningIndexer 的组件回放不计入 quant_lightning_indexer_reference 的整窗误差。相同调用的 Python 热态墙钟中位数，原法 60.9 µs、类别法 76.8 µs。无证据支持将全部已知原生程序改默认。

| 原始调用名 | 系统签名数 | auto（样本路径） | DT 同签名整算子差值范围 | DT CA：原 / 类别 | PR CA：原 / 类别 | CA 窗数 DT / PR |
|---|---:|---|---:|---:|---:|---:|
| `_C_ascend::compressor` | 2 | 类别 | +45.77 … +103.30% | N/A | N/A | 0 / 0 |
| `_C_ascend::compressor_metadata` | 2 | 类别 | +0.00 … +0.00% | N/A | N/A | 0 / 0 |
| `_C_ascend::inplace_partial_rotary_mul` | 2 | 原 | +0.00 … +0.04% | 1.13% / 1.09% | 2.66% / 2.66% | 1 / 1 |
| `_C_ascend::moe_gating_top_k` | 2 | 原 | -71.66 … -29.82% | 7.72% / 27.42% | 5.72% / 26.21% | 12 / 10 |
| `_C_ascend::moe_gating_top_k_hash` | 4 | 原, 类别 | -62.90 … +0.92% | 2.77% / 2.77% | 3.02% / 3.02% | 2 / 2 |
| `_C_ascend::npu_hc_post` | 2 | 原 | -12.61 … -7.65% | 2.32% / 6.87% | 8.49% / 4.97% | 15 / 15 |
| `_C_ascend::npu_hc_pre_v2` | 2 | 原 | -59.60 … -17.16% | 3.53% / 18.83% | 3.10% / 19.12% | 1 / 1 |
| `aten._to_copy.default` | 6 | 原, 类别 | +0.00 … +1.94% | 5.03% / 4.85% | 3.33% / 3.30% | 35 / 34 |
| `aten._unsafe_view.default` | 6 | 无 VF | — | N/A | N/A | 0 / 0 |
| `aten.add.Tensor` | 7 | 原, 类别 | +0.00 … +1.17% | 0.38% / 0.26% | 36.35% / 36.35% | 2 / 2 |
| `aten.addmm.default` | 1 | 原 | +0.00 … +0.00% | 26.46% / 26.46% | 10.71% / 10.71% | 2 / 2 |
| `aten.alias.default` | 12 | 无 VF | — | N/A | N/A | 0 / 0 |
| `aten.bitwise_or.Tensor` | 2 | 原 | +0.12 … +0.20% | 0.77% / 0.78% | 1.39% / 1.50% | 2 / 2 |
| `aten.cat.default` | 4 | 无 VF | +0.00 … +0.00% | 15.31% / 15.31% | 12.89% / 12.89% | 2 / 2 |
| `aten.clone.default` | 8 | 无 VF | +0.00 … +0.00% | N/A | N/A | 0 / 0 |
| `aten.constant_pad_nd.default` | 3 | 无 VF | +0.00 … +0.00% | 9.71% / 9.71% | 9.20% / 9.20% | 2 / 2 |
| `aten.copy_.default` | 1 | 无 VF | +0.00 … +0.00% | N/A | N/A | 0 / 0 |
| `aten.embedding.default` | 2 | 原 | +0.00 … +0.72% | 7.31% / 7.31% | 7.40% / 7.40% | 2 / 2 |
| `aten.empty.memory_format` | 2 | 无 VF | — | N/A | N/A | 0 / 0 |
| `aten.expand.default` | 2 | 无 VF | — | N/A | N/A | 0 / 0 |
| `aten.ge.Scalar` | 2 | 原 | +0.08 … +3.67% | 0.29% / 0.53% | 1.44% / 1.28% | 2 / 2 |
| `aten.index.Tensor` | 2 | 原 | +0.49 … +3.23% | 7.25% / 7.25% | 8.99% / 8.99% | 2 / 2 |
| `aten.lt.Scalar` | 2 | 原 | +0.04 … +1.89% | 4.57% / 4.51% | 0.00% / 0.12% | 2 / 2 |
| `aten.masked_fill.Scalar` | 2 | 原 | +0.09 … +4.20% | N/A | N/A | 0 / 0 |
| `aten.masked_fill_.Scalar` | 2 | 类别 | +0.00 … +1.17% | 0.45% / 0.25% | 3.40% / 3.19% | 2 / 2 |
| `aten.mean.dim` | 3 | 原, 类别 | +0.11 … +2.66% | 8.81% / 12.16% | 9.15% / 12.52% | 2 / 2 |
| `aten.mm.default` | 4 | 原, 无 VF | +0.00 … +0.00% | 12.53% / 12.53% | 18.37% / 18.37% | 6 / 6 |
| `aten.mul.Tensor` | 6 | 原 | +0.00 … +2.16% | 1.67% / 1.51% | 8.02% / 8.20% | 2 / 2 |
| `aten.neg.default` | 3 | 原 | +0.23 … +1.59% | 5.06% / 4.88% | 1.93% / 1.73% | 2 / 2 |
| `aten.permute.default` | 8 | 无 VF | — | N/A | N/A | 0 / 0 |
| `aten.pow.Tensor_Scalar` | 2 | 原 | +0.00 … +2.61% | 10.11% / 9.63% | 14.97% / 14.47% | 2 / 2 |
| `aten.rsqrt.default` | 2 | 原 | +0.21 … +4.26% | 7.95% / 6.54% | 12.52% / 11.10% | 2 / 2 |
| `aten.select.int` | 4 | 无 VF | — | N/A | N/A | 0 / 0 |
| `aten.sigmoid.default` | 2 | 原 | +0.00 … +0.28% | 5.98% / 6.48% | 16.39% / 16.72% | 23 / 20 |
| `aten.slice.Tensor` | 15 | 无 VF | — | N/A | N/A | 0 / 0 |
| `aten.split_with_sizes.default` | 4 | 无 VF | — | N/A | N/A | 0 / 0 |
| `aten.squeeze.dim` | 2 | 无 VF | — | N/A | N/A | 0 / 0 |
| `aten.sub.Tensor` | 2 | 原 | +0.10 … +3.67% | 42.99% / 42.89% | 47.50% / 47.40% | 2 / 2 |
| `aten.sum.dim_IntList` | 2 | 原 | +0.00 … +1.30% | 2.98% / 5.29% | 3.82% / 6.36% | 2 / 2 |
| `aten.t.default` | 4 | 无 VF | — | N/A | N/A | 0 / 0 |
| `aten.transpose.int` | 6 | 无 VF | — | N/A | N/A | 0 / 0 |
| `aten.unsqueeze.default` | 10 | 无 VF | — | N/A | N/A | 0 / 0 |
| `aten.view.default` | 18 | 无 VF | — | N/A | N/A | 0 / 0 |
| `aten.zeros_like.default` | 1 | 无 VF | +0.00 … +0.00% | N/A | N/A | 0 / 0 |
| `collective::all_gather` | 2 | 无 VF | +0.00 … +0.00% | N/A | N/A | 0 / 0 |
| `collective::all_reduce` | 2 | 类别 | +0.00 … +0.00% | N/A | N/A | 0 / 0 |
| `inference::attention_residual` | 2 | 无 VF | +0.00 … +0.00% | N/A | N/A | 0 / 0 |
| `inference::c2_ring_compressor` | 2 | 类别 | +0.00 … +0.00% | N/A | N/A | 0 / 0 |
| `inference::cache_load` | 5 | 原 | -40.69 … +0.43% | 8.28% / 8.09% | 12.35% / 11.68% | 14 / 14 |
| `inference::cache_store` | 10 | 类别 | +0.03 … +12.63% | N/A | N/A | 0 / 0 |
| `inference::candidate_blocks` | 2 | 类别 | +0.00 … +0.00% | N/A | N/A | 0 / 0 |
| `inference::candidate_filtered_indexer` | 8 | 原 | -17.56 … +2.60% | N/A | N/A | 0 / 0 |
| `inference::compressor_source_positions` | 2 | 类别 | +0.10 … +1.27% | N/A | N/A | 0 / 0 |
| `inference::dynamic_mx_quant_fp8_reference` | 4 | 原 | +0.10 … +0.71% | 6.23% / 6.07% | 6.83% / 6.96% | 13 / 13 |
| `inference::engram_dequantize` | 2 | 类别 | +0.05 … +2.34% | N/A | N/A | 0 / 0 |
| `inference::engram_gate` | 2 | 类别 | +1.07 … +7.73% | N/A | N/A | 0 / 0 |
| `inference::engram_hash` | 2 | 类别 | +0.05 … +4.31% | N/A | N/A | 0 / 0 |
| `inference::engram_lookup` | 2 | 类别 | +0.00 … +0.63% | N/A | N/A | 0 / 0 |
| `inference::grouped_gated_rms_norm_reference` | 2 | 类别 | +0.45 … +4.46% | N/A | N/A | 0 / 0 |
| `inference::kimi_dsa_indexer_k_norm_quant_insert` | 1 | 类别 | +11.19 … +11.19% | N/A | N/A | 0 / 0 |
| `inference::kimi_dsa_indexer_query_quant` | 2 | 类别 | +0.34 … +6.99% | N/A | N/A | 0 / 0 |
| `inference::kimi_dsa_indexer_score` | 2 | 类别 | +37.23 … +64.60% | N/A | N/A | 0 / 0 |
| `inference::kimi_gated_dsa_attention` | 2 | 类别 | +0.00 … +58.50% | N/A | N/A | 0 / 0 |
| `inference::lightning_indexer` | 8 | 原, 分阶段：原及类别 | -12.32 … -6.86% | N/A | N/A | 0 / 0 |
| `inference::matmul_rhs_transposed` | 2 | 原, 无 VF | +0.00 … +0.00% | 8.69% / 8.69% | 2.90% / 2.90% | 2 / 2 |
| `inference::mixed_precision_matmul` | 2 | 类别 | +0.08 … +1.91% | N/A | N/A | 0 / 0 |
| `inference::mla_output_gate` | 2 | 类别 | +0.00 … +1.04% | N/A | N/A | 0 / 0 |
| `inference::quantized_batched_matmul` | 2 | 类别 | +0.53 … +4.32% | N/A | N/A | 0 / 0 |
| `inference::situ` | 2 | 类别 | +2.66 … +2.69% | N/A | N/A | 0 / 0 |
| `inference::state_update` | 1 | 类别 | +0.12 … +0.12% | N/A | N/A | 0 / 0 |
| `npu::npu_dynamic_mx_quant` | 6 | 原 | -18.44 … +0.26% | 6.99% / 8.71% | 8.69% / 9.32% | 14 / 14 |
| `npu::npu_fused_infer_attention_score` | 1 | 原 | +0.00 … +0.00% | 6.81% / 4.64% | 5.09% / 4.58% | 2 / 2 |
| `npu::npu_grouped_matmul` | 4 | 类别, 无 VF | +0.00 … +2.60% | 8.48% / 8.48% | 5.53% / 5.53% | 13 / 13 |
| `npu::npu_moe_distribute_combine_v2` | 2 | 类别 | +0.00 … +0.00% | N/A | N/A | 0 / 0 |
| `npu::npu_moe_distribute_dispatch_v2` | 2 | 无 VF | +0.00 … +0.00% | N/A | N/A | 0 / 0 |
| `npu::npu_quant_matmul` | 2 | 无 VF | +0.00 … +0.00% | 4.43% / 4.43% | 6.15% / 6.15% | 24 / 24 |
| `npu::npu_rms_norm` | 4 | 原 | +0.91 … +29.09% | 17.94% / 16.62% | 18.55% / 17.32% | 126 / 136 |
| `npu::npu_rotary_mul` | 1 | 类别 | +0.04 … +0.04% | N/A | N/A | 0 / 0 |
| `npu::npu_swiglu` | 2 | 类别 | +2.02 … +9.70% | 2.26% / 0.27% | 2.75% / 4.79% | 1 / 1 |
| `vllm_ascend::causal_conv1d_fn_reference` | 1 | 类别 | +33.55 … +33.55% | N/A | N/A | 0 / 0 |
| `vllm_ascend::causal_conv1d_update_reference` | 1 | 原 | +0.81 … +0.81% | 15.50% / 15.50% | 11.02% / 11.75% | 2 / 2 |
| `vllm_ascend::chunk_kda_fwd_reference` | 1 | 类别 | +43.53 … +43.53% | N/A | N/A | 0 / 0 |
| `vllm_ascend::l2norm_fwd_reference` | 1 | 类别 | +0.00 … +0.00% | N/A | N/A | 0 / 0 |
| `vllm_ascend::quant_lightning_indexer_reference` | 2 | 原 | -8.07 … -6.68% | N/A | N/A | 0 / 0 |
| `vllm_ascend::recurrent_kda_reference` | 1 | 类别 | +40.50 … +40.50% | N/A | N/A | 0 / 0 |
| `vllm_ascend::sparse_attn_sharedkv_reference` | 7 | 类别 | +0.00 … +23.46% | N/A | N/A | 0 / 0 |

表头：**原始调用名**是系统图符号；**系统签名数**为本页保留的代表节点数（同一算子可有不同精度、角色和执行阶段，不是所有 shape 数）；**auto** 为自动模式对这些样本实际选择，纯搬运/元数据不存在 VF。**DT 同签名差值**是 `(类别设备时间 / 原设备时间 − 1) × 100%` 的范围，正数为类别法较慢；原时间全零时不作除法。**DT/PR CA** 为对应芯片可重放合同的原法 WAPE / 类别法 WAPE，`WAPE=Σ|预测−CA|/ΣCA`；**CA 窗数**为同样两组 WAPE 的分母样本数。N/A 表示不能给两法配对的合格整窗；范围是方法差异，不是 CA 误差。

共 86 个调用名、293 份系统输入，来自三模型各一份 decode 和 prefill 图中的 target/draft。按阶段、输入 dtype/秩和主要实现标签选取首个实际节点（因此不是对所有序列长度或所有系统配置重新跑 DES）。芯片采用原生 DT/PR 配置、CA window 定义；不覆盖系统硬件配置覆盖值或可选 cold-IO 下界。所有 CA 是已存档窗口的回放，包含旧方法开发样本，不是本次新冻结独立验证。类指令在系统设计路线上是源算法推演，不是实测的指令序列。

[图输入与来源](../results/v2.3/system-inputs.json.gz) · [逐节点双芯片报价、分支、耗时](../results/v2.3/system-quotes.json.gz) · [逐窗口双方法预测、真值、数据角色和 kernel](../results/v2.3/system-windows.json) · [原始完整误差及采集身份](../calibration/reference/system-full-ca.csv)。报价中 `wall_us` 是四次运行去掉首次后取中位数的本机墙钟微秒，不含图转换或 CA；`device_us` 才是预计的芯片运行时间。再现：`PYTHONPATH=src:. python -m calibration.examples.system_vector_methods`。
