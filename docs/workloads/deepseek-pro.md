# DeepSeek-V4-Pro · 整批 prefill 的逐节点单卡张量

indexer.weights_proj 的每卡 N=64、K=7168。单 token 及 MTP8、B8×32Ki 的 M 均从本步调度 token 数得到。这些投影使用 BF16 ND 独立库合同做双端 CA 对照。[含普通 decode、MTP 与大 prefill 的 33 个 Matmul 几何](../../results/v0.29/matmul/workloads.md)

**单卡 8×256Ki 的实测。** 全部 2,097,152 行在一个 `aclnnCast` 中完成，950DT 预测 227.018 µs、CA 247.788 µs。950PR 为 506.517 / 529.287 µs。下表使用模型的 64 个 indexer 系数；内部 hidden 宽度仍为 7168。[第 62–66 个 UB 块的同轴图](../../results/v0.27/memory/pro-256ki-dt-blocks-62-66.png) · [100 轮算例](../examples/cast_bf16_fp32.md)

**单卡 8×128k 的实测。** 64 个 indexer 系数升精度得到 BF16→FP32 [1048576,64]，在 950DT 上完整运行 50 轮/核。模型为 99.918 µs，CA 为 105.374 µs。逐节点表中的别的通道数取决于各自的模型位置。[同一批次逐 UB 块的模型与 CA](../../results/v0.23/pro-prefill/pro-b8-128ki-dt-plot.png) · [完整计时与校验](../../results/v0.23/pro-prefill/pro-b8-128ki-dt.json)

每个表对应 8 条等长请求一次提交，attention TP8、DP1，MoE EP8。层号从 0 开始。权重写为源码的逻辑形状，KV 表同时包含持久化或条件性缓冲。专家行数采用本情景的均衡路由。表内给出完整轴长度，可以按具体节点直接查询。

三个时延列的单位是 µs。950 的单元格是“模型 / 同形 CA”，970DT 是配置外推。只有算子、shape 和属性一致的独立库调用才显示正式指令窗。原生 wrapper 和单独检查过的 BF16 ND 投影另有证据。此情景的残差激活为 BF16，beta Sigmoid 是 FP32→FP32；动态 MX 量化是 BF16→FP8 E4M3FN 和 E8M0 scale。“仅形状”表示只导出了结构，没有独立后端计时合同。视部署而定的算子可能被融合。[调度与参数来源](../prefill-shapes.md) · [独立库采集的角色及误差](../error-analysis.md) · [完整配置、条件和其他并行拓扑](../../results/v0.23/workloads/contracts.jsonl.gz)

## 每请求 32,768 token · 单卡 262,144 token

| 节点与层号 | 输入、输出及参数形状 | 单算子合同 | DT 模型 / CA | PR 模型 / CA | 970DT |
|---|---|---|---:|---:|---:|
| `embedding.lookup`<br>input | ids <code>[262144]</code><br>out <code>[262144,7168]</code> | 仅形状 | — | — | — |
| `lm_head.last_position`<br>output | x <code>[8,7168]</code><br>weight_logical <code>[16160,7168]</code><br>y <code>[8,16160]</code> | 仅形状 | — | — | — |
| `mhc.attn.pre`<br>0–60 | residual <code>[262144,4,7168]</code><br>fn <code>[24,28672]</code><br>post <code>[262144,4]</code><br>mix <code>[262144,4,4]</code><br>out <code>[262144,7168]</code> | 仅形状 | — | — | — |
| `mhc.attn.post`<br>0–60 | x <code>[1,262144,7168]</code><br>residual <code>[1,262144,4,7168]</code><br>post <code>[1,262144,4]</code><br>mix <code>[1,262144,4,4]</code><br>out <code>[1,262144,4,7168]</code> | `mhc_post`<br>implementation="vllm_hc_post" | 10661.002 / — | 24090.285 / — | 5211.613 |
| `mhc.ffn.pre`<br>0–60 | residual <code>[262144,4,7168]</code><br>fn <code>[24,28672]</code><br>post <code>[262144,4]</code><br>mix <code>[262144,4,4]</code><br>out <code>[262144,7168]</code> | 仅形状 | — | — | — |
| `mhc.ffn.post`<br>0–60 | x <code>[1,262144,7168]</code><br>residual <code>[1,262144,4,7168]</code><br>post <code>[1,262144,4]</code><br>mix <code>[1,262144,4,4]</code><br>out <code>[1,262144,4,7168]</code> | `mhc_post`<br>implementation="vllm_hc_post" | 10661.002 / — | 24090.285 / — | 5211.613 |
| `input_layernorm`<br>0–60 | x <code>[262144,7168]</code><br>weight <code>[7168]</code><br>out <code>[262144,7168]</code> | `rmsnorm` | 1849.354 / — | 5291.876 / — | 770.501 |
| `post_attention_layernorm`<br>0–60 | x <code>[262144,7168]</code><br>weight <code>[7168]</code><br>out <code>[262144,7168]</code> | `rmsnorm` | 1849.354 / — | 5291.876 / — | 770.501 |
| `moe.router_input.float`<br>0–60 | bf16 <code>[262144,7168]</code><br>fp32 <code>[262144,7168]</code> | `cast_bf16_fp32` | 3531.621 / — | 7984.362 / — | 1720.257 |
| `dsa.wq_a`<br>0–60 | x <code>[262144,7168]</code><br>weight_logical <code>[1536,7168]</code><br>y <code>[262144,1536]</code> | 仅形状 | — | — | — |
| `dsa.q_norm`<br>0–60 | x <code>[262144,1536]</code><br>weight <code>[1536]</code><br>y <code>[262144,1536]</code> | 仅形状 | — | — | — |
| `dsa.wq_b`<br>0–60 | x <code>[262144,1536]</code><br>weight_logical <code>[8192,1536]</code><br>y <code>[262144,8192]</code> | 仅形状 | — | — | — |
| `dsa.q_norm_without_weight_and_rope`<br>0–60 | q <code>[262144,16,512]</code><br>rope_suffix <code>[262144,16,64]</code> | 仅形状 | — | — | — |
| `dsa.wkv`<br>0–60 | x <code>[262144,7168]</code><br>weight_logical <code>[512,7168]</code><br>y <code>[262144,512]</code> | 仅形状 | — | — | — |
| `dsa.kv_norm_rope_swa_cache`<br>0–60 | current_kv <code>[262144,1,512]</code><br>weight <code>[512]</code><br>resident_window_logical <code>[8,128,1,512]</code> | 仅形状 | — | — | — |
| `dsa.wo_a`<br>0–60 | x <code>[262144,2,4096]</code><br>weight_logical <code>[2,1024,4096]</code><br>y <code>[262144,2,1024]</code> | 仅形状 | — | — | — |
| `dsa.wo_b`<br>0–60 | x <code>[262144,2048]</code><br>weight_logical <code>[7168,2048]</code><br>y <code>[262144,7168]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>0–1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59 | q <code>[262144,16,512]</code><br>out <code>[262144,16,512]</code><br>sink <code>[16]</code><br>compressed_kv_logical <code>[8,256,1,512]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | q <code>[262144,16,512]</code><br>out <code>[262144,16,512]</code><br>sink <code>[16]</code><br>compressed_kv_logical <code>[8,8192,1,512]</code><br>indices_capacity_per_query <code>[262144,1024]</code> | 仅形状 | — | — | — |
| `compressor.kv`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | x <code>[262144,7168]</code><br>weight_logical <code>[1024,7168]</code><br>y <code>[262144,1024]</code> | 仅形状 | — | — | — |
| `compressor.gate`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | x <code>[262144,7168]</code><br>weight_logical <code>[1024,7168]</code><br>y <code>[262144,1024]</code> | 仅形状 | — | — | — |
| `compressor.pool_store`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | new_rows <code>[65536,1,512]</code><br>history_logical <code>[8,8192,1,512]</code> | 仅形状 | — | — | — |
| `indexer.compressor.kv`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | x <code>[262144,7168]</code><br>weight_logical <code>[256,7168]</code><br>y <code>[262144,256]</code> | 仅形状 | — | — | — |
| `indexer.compressor.gate`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | x <code>[262144,7168]</code><br>weight_logical <code>[256,7168]</code><br>y <code>[262144,256]</code> | 仅形状 | — | — | — |
| `indexer.compressor.store`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | new_rows <code>[65536,1,128]</code><br>history_logical <code>[8,8192,1,128]</code> | 仅形状 | — | — | — |
| `compressor.kv`<br>0–1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59 | x <code>[262144,7168]</code><br>weight_logical <code>[512,7168]</code><br>y <code>[262144,512]</code> | 仅形状 | — | — | — |
| `compressor.gate`<br>0–1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59 | x <code>[262144,7168]</code><br>weight_logical <code>[512,7168]</code><br>y <code>[262144,512]</code> | 仅形状 | — | — | — |
| `compressor.pool_store`<br>0–1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59 | new_rows <code>[2048,1,512]</code><br>history_logical <code>[8,256,1,512]</code> | 仅形状 | — | — | — |
| `indexer.wq_b`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | x <code>[262144,1536]</code><br>weight_logical <code>[8192,1536]</code><br>y <code>[262144,8192]</code> | 仅形状 | — | — | — |
| `indexer.weights_proj`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | x <code>[262144,7168]</code><br>weight_logical <code>[64,7168]</code><br>y <code>[262144,64]</code> | [独立 BF16 ND Matmul](../../results/v0.28/matmul/README.md) | 1514.284 / 待回收 | 2688.910 / 待回收 | 382.559 |
| `indexer.weights_scale`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | x <code>[262144,64]</code><br>scalar <code>[]</code><br>y <code>[262144,64]</code> | 仅形状 | — | — | — |
| `indexer.prepare_dsa_weights.float`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | scaled_bf16 <code>[262144,64]</code><br>fp32 <code>[262144,64]</code> | `cast_bf16_fp32` | [16.592 / 17.041](../examples/figures/prefill-pro-32ki-dt.png) | [25.030 / 23.762](../examples/figures/prefill-pro-32ki-pr.png) | 9.951 |
| `indexer.select`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | q <code>[262144,64,128]</code><br>weights <code>[262144,64]</code><br>indices <code>[262144,1024]</code> | 仅形状 | — | — | — |
| `output.hc_head_and_norm`<br>output | x <code>[262144,4,7168]</code><br>out <code>[262144,7168]</code> | 仅形状 | — | — | — |
| `parallel.attention_row_reduce`<br>0–60 | partial <code>[262144,7168]</code><br>result <code>[262144,7168]</code> | 仅形状 | — | — | — |
| `moe.gate_logits`<br>0–60 | x <code>[262144,7168]</code><br>weight_logical <code>[384,7168]</code><br>y <code>[262144,384]</code> | 仅形状 | — | — | — |
| `moe.hash_router`<br>0–2 | logits <code>[262144,384]</code><br>weights <code>[262144,6]</code><br>ids <code>[262144,6]</code><br>input_ids <code>[262144]</code> | 仅形状 | — | — | — |
| `moe.sqrtsoftplus_topk`<br>3–60 | logits <code>[262144,384]</code><br>weights <code>[262144,6]</code><br>ids <code>[262144,6]</code> | 仅形状 | — | — | — |
| `moe.dispatch_combine`<br>0–60 | rank_source_tokens <code>[262144,7168]</code><br>logical_received_rows <code>[196608,7168]</code><br>combined <code>[262144,7168]</code> | 仅形状 | — | — | — |
| `moe.routed_grouped_ffn`<br>0–60 | x <code>[196608,7168]</code><br>w_gate_up_logical <code>[48,6144,7168]</code><br>gate_up <code>[196608,6144]</code><br>activation <code>[196608,3072]</code><br>w_down_logical <code>[48,7168,3072]</code><br>out <code>[196608,7168]</code> | 仅形状 | — | — | — |
| `moe.shared_gate_up`<br>0–60 | x <code>[262144,7168]</code><br>weight_logical <code>[768,7168]</code><br>y <code>[262144,768]</code> | 仅形状 | — | — | — |
| `moe.shared_activation`<br>0–60 | gate_up <code>[262144,768]</code><br>out <code>[262144,384]</code> | 仅形状 | — | — | — |
| `moe.shared_down`<br>0–60 | x <code>[262144,384]</code><br>weight_logical <code>[7168,384]</code><br>y <code>[262144,7168]</code> | 仅形状 | — | — | — |

## 每请求 131,072 token · 单卡 1,048,576 token

| 节点与层号 | 输入、输出及参数形状 | 单算子合同 | DT 模型 / CA | PR 模型 / CA | 970DT |
|---|---|---|---:|---:|---:|
| `embedding.lookup`<br>input | ids <code>[1048576]</code><br>out <code>[1048576,7168]</code> | 仅形状 | — | — | — |
| `lm_head.last_position`<br>output | x <code>[8,7168]</code><br>weight_logical <code>[16160,7168]</code><br>y <code>[8,16160]</code> | 仅形状 | — | — | — |
| `mhc.attn.pre`<br>0–60 | residual <code>[1048576,4,7168]</code><br>fn <code>[24,28672]</code><br>post <code>[1048576,4]</code><br>mix <code>[1048576,4,4]</code><br>out <code>[1048576,7168]</code> | 仅形状 | — | — | — |
| `mhc.attn.post`<br>0–60 | x <code>[1,1048576,7168]</code><br>residual <code>[1,1048576,4,7168]</code><br>post <code>[1,1048576,4]</code><br>mix <code>[1,1048576,4,4]</code><br>out <code>[1,1048576,4,7168]</code> | `mhc_post`<br>implementation="vllm_hc_post" | 42631.403 / — | 96612.797 / — | 20831.753 |
| `mhc.ffn.pre`<br>0–60 | residual <code>[1048576,4,7168]</code><br>fn <code>[24,28672]</code><br>post <code>[1048576,4]</code><br>mix <code>[1048576,4,4]</code><br>out <code>[1048576,7168]</code> | 仅形状 | — | — | — |
| `mhc.ffn.post`<br>0–60 | x <code>[1,1048576,7168]</code><br>residual <code>[1,1048576,4,7168]</code><br>post <code>[1,1048576,4]</code><br>mix <code>[1,1048576,4,4]</code><br>out <code>[1,1048576,4,7168]</code> | `mhc_post`<br>implementation="vllm_hc_post" | 42631.403 / — | 96612.797 / — | 20831.753 |
| `input_layernorm`<br>0–60 | x <code>[1048576,7168]</code><br>weight <code>[7168]</code><br>out <code>[1048576,7168]</code> | `rmsnorm` | 7487.285 / — | 21400.250 / — | 3069.475 |
| `post_attention_layernorm`<br>0–60 | x <code>[1048576,7168]</code><br>weight <code>[7168]</code><br>out <code>[1048576,7168]</code> | `rmsnorm` | 7487.285 / — | 21400.250 / — | 3069.475 |
| `moe.router_input.float`<br>0–60 | bf16 <code>[1048576,7168]</code><br>fp32 <code>[1048576,7168]</code> | `cast_bf16_fp32` | 14208.032 / — | 32143.553 / — | 6939.836 |
| `dsa.wq_a`<br>0–60 | x <code>[1048576,7168]</code><br>weight_logical <code>[1536,7168]</code><br>y <code>[1048576,1536]</code> | 仅形状 | — | — | — |
| `dsa.q_norm`<br>0–60 | x <code>[1048576,1536]</code><br>weight <code>[1536]</code><br>y <code>[1048576,1536]</code> | 仅形状 | — | — | — |
| `dsa.wq_b`<br>0–60 | x <code>[1048576,1536]</code><br>weight_logical <code>[8192,1536]</code><br>y <code>[1048576,8192]</code> | 仅形状 | — | — | — |
| `dsa.q_norm_without_weight_and_rope`<br>0–60 | q <code>[1048576,16,512]</code><br>rope_suffix <code>[1048576,16,64]</code> | 仅形状 | — | — | — |
| `dsa.wkv`<br>0–60 | x <code>[1048576,7168]</code><br>weight_logical <code>[512,7168]</code><br>y <code>[1048576,512]</code> | 仅形状 | — | — | — |
| `dsa.kv_norm_rope_swa_cache`<br>0–60 | current_kv <code>[1048576,1,512]</code><br>weight <code>[512]</code><br>resident_window_logical <code>[8,128,1,512]</code> | 仅形状 | — | — | — |
| `dsa.wo_a`<br>0–60 | x <code>[1048576,2,4096]</code><br>weight_logical <code>[2,1024,4096]</code><br>y <code>[1048576,2,1024]</code> | 仅形状 | — | — | — |
| `dsa.wo_b`<br>0–60 | x <code>[1048576,2048]</code><br>weight_logical <code>[7168,2048]</code><br>y <code>[1048576,7168]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>0–1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59 | q <code>[1048576,16,512]</code><br>out <code>[1048576,16,512]</code><br>sink <code>[16]</code><br>compressed_kv_logical <code>[8,1024,1,512]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | q <code>[1048576,16,512]</code><br>out <code>[1048576,16,512]</code><br>sink <code>[16]</code><br>compressed_kv_logical <code>[8,32768,1,512]</code><br>indices_capacity_per_query <code>[1048576,1024]</code> | 仅形状 | — | — | — |
| `compressor.kv`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | x <code>[1048576,7168]</code><br>weight_logical <code>[1024,7168]</code><br>y <code>[1048576,1024]</code> | 仅形状 | — | — | — |
| `compressor.gate`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | x <code>[1048576,7168]</code><br>weight_logical <code>[1024,7168]</code><br>y <code>[1048576,1024]</code> | 仅形状 | — | — | — |
| `compressor.pool_store`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | new_rows <code>[262144,1,512]</code><br>history_logical <code>[8,32768,1,512]</code> | 仅形状 | — | — | — |
| `indexer.compressor.kv`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | x <code>[1048576,7168]</code><br>weight_logical <code>[256,7168]</code><br>y <code>[1048576,256]</code> | 仅形状 | — | — | — |
| `indexer.compressor.gate`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | x <code>[1048576,7168]</code><br>weight_logical <code>[256,7168]</code><br>y <code>[1048576,256]</code> | 仅形状 | — | — | — |
| `indexer.compressor.store`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | new_rows <code>[262144,1,128]</code><br>history_logical <code>[8,32768,1,128]</code> | 仅形状 | — | — | — |
| `compressor.kv`<br>0–1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59 | x <code>[1048576,7168]</code><br>weight_logical <code>[512,7168]</code><br>y <code>[1048576,512]</code> | 仅形状 | — | — | — |
| `compressor.gate`<br>0–1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59 | x <code>[1048576,7168]</code><br>weight_logical <code>[512,7168]</code><br>y <code>[1048576,512]</code> | 仅形状 | — | — | — |
| `compressor.pool_store`<br>0–1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59 | new_rows <code>[8192,1,512]</code><br>history_logical <code>[8,1024,1,512]</code> | 仅形状 | — | — | — |
| `indexer.wq_b`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | x <code>[1048576,1536]</code><br>weight_logical <code>[8192,1536]</code><br>y <code>[1048576,8192]</code> | 仅形状 | — | — | — |
| `indexer.weights_proj`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | x <code>[1048576,7168]</code><br>weight_logical <code>[64,7168]</code><br>y <code>[1048576,64]</code> | 仅形状 | — | — | — |
| `indexer.weights_scale`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | x <code>[1048576,64]</code><br>scalar <code>[]</code><br>y <code>[1048576,64]</code> | 仅形状 | — | — | — |
| `indexer.prepare_dsa_weights.float`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | scaled_bf16 <code>[1048576,64]</code><br>fp32 <code>[1048576,64]</code> | `cast_bf16_fp32` | [99.918 / 105.374](../../results/v0.23/pro-prefill/pro-b8-128ki-dt-plot.png) | [218.908 / 224.719](../../results/v0.23/pro-prefill/pro-b8-128ki-pr-plot.png) | 42.536 |
| `indexer.select`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | q <code>[1048576,64,128]</code><br>weights <code>[1048576,64]</code><br>indices <code>[1048576,1024]</code> | 仅形状 | — | — | — |
| `output.hc_head_and_norm`<br>output | x <code>[1048576,4,7168]</code><br>out <code>[1048576,7168]</code> | 仅形状 | — | — | — |
| `parallel.attention_row_reduce`<br>0–60 | partial <code>[1048576,7168]</code><br>result <code>[1048576,7168]</code> | 仅形状 | — | — | — |
| `moe.gate_logits`<br>0–60 | x <code>[1048576,7168]</code><br>weight_logical <code>[384,7168]</code><br>y <code>[1048576,384]</code> | 仅形状 | — | — | — |
| `moe.hash_router`<br>0–2 | logits <code>[1048576,384]</code><br>weights <code>[1048576,6]</code><br>ids <code>[1048576,6]</code><br>input_ids <code>[1048576]</code> | 仅形状 | — | — | — |
| `moe.sqrtsoftplus_topk`<br>3–60 | logits <code>[1048576,384]</code><br>weights <code>[1048576,6]</code><br>ids <code>[1048576,6]</code> | 仅形状 | — | — | — |
| `moe.dispatch_combine`<br>0–60 | rank_source_tokens <code>[1048576,7168]</code><br>logical_received_rows <code>[786432,7168]</code><br>combined <code>[1048576,7168]</code> | 仅形状 | — | — | — |
| `moe.routed_grouped_ffn`<br>0–60 | x <code>[786432,7168]</code><br>w_gate_up_logical <code>[48,6144,7168]</code><br>gate_up <code>[786432,6144]</code><br>activation <code>[786432,3072]</code><br>w_down_logical <code>[48,7168,3072]</code><br>out <code>[786432,7168]</code> | 仅形状 | — | — | — |
| `moe.shared_gate_up`<br>0–60 | x <code>[1048576,7168]</code><br>weight_logical <code>[768,7168]</code><br>y <code>[1048576,768]</code> | 仅形状 | — | — | — |
| `moe.shared_activation`<br>0–60 | gate_up <code>[1048576,768]</code><br>out <code>[1048576,384]</code> | 仅形状 | — | — | — |
| `moe.shared_down`<br>0–60 | x <code>[1048576,384]</code><br>weight_logical <code>[7168,384]</code><br>y <code>[1048576,7168]</code> | 仅形状 | — | — | — |

## 每请求 262,144 token · 单卡 2,097,152 token

| 节点与层号 | 输入、输出及参数形状 | 单算子合同 | DT 模型 / CA | PR 模型 / CA | 970DT |
|---|---|---|---:|---:|---:|
| `embedding.lookup`<br>input | ids <code>[2097152]</code><br>out <code>[2097152,7168]</code> | 仅形状 | — | — | — |
| `lm_head.last_position`<br>output | x <code>[8,7168]</code><br>weight_logical <code>[16160,7168]</code><br>y <code>[8,16160]</code> | 仅形状 | — | — | — |
| `mhc.attn.pre`<br>0–60 | residual <code>[2097152,4,7168]</code><br>fn <code>[24,28672]</code><br>post <code>[2097152,4]</code><br>mix <code>[2097152,4,4]</code><br>out <code>[2097152,7168]</code> | 仅形状 | — | — | — |
| `mhc.attn.post`<br>0–60 | x <code>[1,2097152,7168]</code><br>residual <code>[1,2097152,4,7168]</code><br>post <code>[1,2097152,4]</code><br>mix <code>[1,2097152,4,4]</code><br>out <code>[1,2097152,4,7168]</code> | `mhc_post`<br>implementation="vllm_hc_post" | 85258.605 / — | 193309.479 / — | 41658.290 |
| `mhc.ffn.pre`<br>0–60 | residual <code>[2097152,4,7168]</code><br>fn <code>[24,28672]</code><br>post <code>[2097152,4]</code><br>mix <code>[2097152,4,4]</code><br>out <code>[2097152,7168]</code> | 仅形状 | — | — | — |
| `mhc.ffn.post`<br>0–60 | x <code>[1,2097152,7168]</code><br>residual <code>[1,2097152,4,7168]</code><br>post <code>[1,2097152,4]</code><br>mix <code>[1,2097152,4,4]</code><br>out <code>[1,2097152,4,7168]</code> | `mhc_post`<br>implementation="vllm_hc_post" | 85258.605 / — | 193309.479 / — | 41658.290 |
| `input_layernorm`<br>0–60 | x <code>[2097152,7168]</code><br>weight <code>[7168]</code><br>out <code>[2097152,7168]</code> | `rmsnorm` | 15004.526 / — | 42878.083 / — | 6133.959 |
| `post_attention_layernorm`<br>0–60 | x <code>[2097152,7168]</code><br>weight <code>[7168]</code><br>out <code>[2097152,7168]</code> | `rmsnorm` | 15004.526 / — | 42878.083 / — | 6133.959 |
| `moe.router_input.float`<br>0–60 | bf16 <code>[2097152,7168]</code><br>fp32 <code>[2097152,7168]</code> | `cast_bf16_fp32` | 28443.245 / — | 64355.808 / — | 13899.273 |
| `dsa.wq_a`<br>0–60 | x <code>[2097152,7168]</code><br>weight_logical <code>[1536,7168]</code><br>y <code>[2097152,1536]</code> | 仅形状 | — | — | — |
| `dsa.q_norm`<br>0–60 | x <code>[2097152,1536]</code><br>weight <code>[1536]</code><br>y <code>[2097152,1536]</code> | 仅形状 | — | — | — |
| `dsa.wq_b`<br>0–60 | x <code>[2097152,1536]</code><br>weight_logical <code>[8192,1536]</code><br>y <code>[2097152,8192]</code> | 仅形状 | — | — | — |
| `dsa.q_norm_without_weight_and_rope`<br>0–60 | q <code>[2097152,16,512]</code><br>rope_suffix <code>[2097152,16,64]</code> | 仅形状 | — | — | — |
| `dsa.wkv`<br>0–60 | x <code>[2097152,7168]</code><br>weight_logical <code>[512,7168]</code><br>y <code>[2097152,512]</code> | 仅形状 | — | — | — |
| `dsa.kv_norm_rope_swa_cache`<br>0–60 | current_kv <code>[2097152,1,512]</code><br>weight <code>[512]</code><br>resident_window_logical <code>[8,128,1,512]</code> | 仅形状 | — | — | — |
| `dsa.wo_a`<br>0–60 | x <code>[2097152,2,4096]</code><br>weight_logical <code>[2,1024,4096]</code><br>y <code>[2097152,2,1024]</code> | 仅形状 | — | — | — |
| `dsa.wo_b`<br>0–60 | x <code>[2097152,2048]</code><br>weight_logical <code>[7168,2048]</code><br>y <code>[2097152,7168]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>0–1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59 | q <code>[2097152,16,512]</code><br>out <code>[2097152,16,512]</code><br>sink <code>[16]</code><br>compressed_kv_logical <code>[8,2048,1,512]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | q <code>[2097152,16,512]</code><br>out <code>[2097152,16,512]</code><br>sink <code>[16]</code><br>compressed_kv_logical <code>[8,65536,1,512]</code><br>indices_capacity_per_query <code>[2097152,1024]</code> | 仅形状 | — | — | — |
| `compressor.kv`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | x <code>[2097152,7168]</code><br>weight_logical <code>[1024,7168]</code><br>y <code>[2097152,1024]</code> | 仅形状 | — | — | — |
| `compressor.gate`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | x <code>[2097152,7168]</code><br>weight_logical <code>[1024,7168]</code><br>y <code>[2097152,1024]</code> | 仅形状 | — | — | — |
| `compressor.pool_store`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | new_rows <code>[524288,1,512]</code><br>history_logical <code>[8,65536,1,512]</code> | 仅形状 | — | — | — |
| `indexer.compressor.kv`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | x <code>[2097152,7168]</code><br>weight_logical <code>[256,7168]</code><br>y <code>[2097152,256]</code> | 仅形状 | — | — | — |
| `indexer.compressor.gate`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | x <code>[2097152,7168]</code><br>weight_logical <code>[256,7168]</code><br>y <code>[2097152,256]</code> | 仅形状 | — | — | — |
| `indexer.compressor.store`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | new_rows <code>[524288,1,128]</code><br>history_logical <code>[8,65536,1,128]</code> | 仅形状 | — | — | — |
| `compressor.kv`<br>0–1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59 | x <code>[2097152,7168]</code><br>weight_logical <code>[512,7168]</code><br>y <code>[2097152,512]</code> | 仅形状 | — | — | — |
| `compressor.gate`<br>0–1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59 | x <code>[2097152,7168]</code><br>weight_logical <code>[512,7168]</code><br>y <code>[2097152,512]</code> | 仅形状 | — | — | — |
| `compressor.pool_store`<br>0–1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59 | new_rows <code>[16384,1,512]</code><br>history_logical <code>[8,2048,1,512]</code> | 仅形状 | — | — | — |
| `indexer.wq_b`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | x <code>[2097152,1536]</code><br>weight_logical <code>[8192,1536]</code><br>y <code>[2097152,8192]</code> | 仅形状 | — | — | — |
| `indexer.weights_proj`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | x <code>[2097152,7168]</code><br>weight_logical <code>[64,7168]</code><br>y <code>[2097152,64]</code> | 仅形状 | — | — | — |
| `indexer.weights_scale`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | x <code>[2097152,64]</code><br>scalar <code>[]</code><br>y <code>[2097152,64]</code> | 仅形状 | — | — | — |
| `indexer.prepare_dsa_weights.float`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | scaled_bf16 <code>[2097152,64]</code><br>fp32 <code>[2097152,64]</code> | `cast_bf16_fp32` | [227.018 / 247.788](../../results/v0.27/memory/pro-256ki-dt-blocks-62-66.png) | [506.517 / 529.287](../../results/v0.23/pro-prefill/pro-b8-256ki-pr.json) | 104.673 |
| `indexer.select`<br>2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60 | q <code>[2097152,64,128]</code><br>weights <code>[2097152,64]</code><br>indices <code>[2097152,1024]</code> | 仅形状 | — | — | — |
| `output.hc_head_and_norm`<br>output | x <code>[2097152,4,7168]</code><br>out <code>[2097152,7168]</code> | 仅形状 | — | — | — |
| `parallel.attention_row_reduce`<br>0–60 | partial <code>[2097152,7168]</code><br>result <code>[2097152,7168]</code> | 仅形状 | — | — | — |
| `moe.gate_logits`<br>0–60 | x <code>[2097152,7168]</code><br>weight_logical <code>[384,7168]</code><br>y <code>[2097152,384]</code> | 仅形状 | — | — | — |
| `moe.hash_router`<br>0–2 | logits <code>[2097152,384]</code><br>weights <code>[2097152,6]</code><br>ids <code>[2097152,6]</code><br>input_ids <code>[2097152]</code> | 仅形状 | — | — | — |
| `moe.sqrtsoftplus_topk`<br>3–60 | logits <code>[2097152,384]</code><br>weights <code>[2097152,6]</code><br>ids <code>[2097152,6]</code> | 仅形状 | — | — | — |
| `moe.dispatch_combine`<br>0–60 | rank_source_tokens <code>[2097152,7168]</code><br>logical_received_rows <code>[1572864,7168]</code><br>combined <code>[2097152,7168]</code> | 仅形状 | — | — | — |
| `moe.routed_grouped_ffn`<br>0–60 | x <code>[1572864,7168]</code><br>w_gate_up_logical <code>[48,6144,7168]</code><br>gate_up <code>[1572864,6144]</code><br>activation <code>[1572864,3072]</code><br>w_down_logical <code>[48,7168,3072]</code><br>out <code>[1572864,7168]</code> | 仅形状 | — | — | — |
| `moe.shared_gate_up`<br>0–60 | x <code>[2097152,7168]</code><br>weight_logical <code>[768,7168]</code><br>y <code>[2097152,768]</code> | 仅形状 | — | — | — |
| `moe.shared_activation`<br>0–60 | gate_up <code>[2097152,768]</code><br>out <code>[2097152,384]</code> | 仅形状 | — | — | — |
| `moe.shared_down`<br>0–60 | x <code>[2097152,384]</code><br>weight_logical <code>[7168,384]</code><br>y <code>[2097152,7168]</code> | 仅形状 | — | — | — |

[返回三个模型的对照表](../prefill-shapes.md) · [模型与 CA 流水图集](../pipeline-comparisons.md)
