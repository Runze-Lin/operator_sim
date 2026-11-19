# DeepSeek-V4.1-Flash · 整批 prefill 的逐节点单卡张量

每个表对应 8 条等长请求一次提交，attention TP8、DP1，MoE EP8。层号从 0 开始。权重写为源码的逻辑形状，KV 表同时包含持久化或条件性缓冲。专家行数采用本情景的均衡路由。表内给出完整轴长度，可以按具体节点直接查询。

三个时延列的单位是 µs。950 的单元格是“模型 / 同形 CA”，970DT 是配置外推。只有算子、shape 和属性一致的独立库调用才显示正式指令窗。原生 wrapper 和单独检查过的 BF16 ND 投影另有证据。此情景的残差激活为 BF16，beta Sigmoid 是 FP32→FP32；动态 MX 量化是 BF16→FP8 E4M3FN 和 E8M0 scale。“仅形状”表示只导出了结构，没有独立后端计时合同。视部署而定的算子可能被融合。[调度与参数来源](../prefill-shapes.md) · [独立库采集的角色及误差](../error-analysis.md) · [完整配置、条件和其他并行拓扑](../../results/v0.23/workloads/contracts.jsonl.gz)

## 每请求 32,768 token · 单卡 262,144 token

| 节点与层号 | 输入、输出及参数形状 | 单算子合同 | DT 模型 / CA | PR 模型 / CA | 970DT |
|---|---|---|---:|---:|---:|
| `embedding.lookup`<br>input | ids <code>[262144]</code><br>out <code>[262144,5120]</code> | 仅形状 | — | — | — |
| `lm_head.last_position`<br>output | x <code>[8,5120]</code><br>weight_logical <code>[16160,5120]</code><br>y <code>[8,16160]</code> | 仅形状 | — | — | — |
| `mhc.attn.pre`<br>0–39 | residual <code>[262144,4,5120]</code><br>fn <code>[24,20480]</code><br>post <code>[262144,4]</code><br>mix <code>[262144,4,4]</code><br>out <code>[262144,5120]</code> | 仅形状 | — | — | — |
| `mhc.attn.post`<br>0–39 | x <code>[1,262144,5120]</code><br>residual <code>[1,262144,4,5120]</code><br>post <code>[1,262144,4]</code><br>mix <code>[1,262144,4,4]</code><br>out <code>[1,262144,4,5120]</code> | `mhc_post`<br>implementation="vllm_hc_post" | — / — | — / — | — |
| `mhc.ffn.pre`<br>0–39 | residual <code>[262144,4,5120]</code><br>fn <code>[24,20480]</code><br>post <code>[262144,4]</code><br>mix <code>[262144,4,4]</code><br>out <code>[262144,5120]</code> | 仅形状 | — | — | — |
| `mhc.ffn.post`<br>0–39 | x <code>[1,262144,5120]</code><br>residual <code>[1,262144,4,5120]</code><br>post <code>[1,262144,4]</code><br>mix <code>[1,262144,4,4]</code><br>out <code>[1,262144,4,5120]</code> | `mhc_post`<br>implementation="vllm_hc_post" | — / — | — / — | — |
| `input_layernorm`<br>0–39 | x <code>[262144,5120]</code><br>weight <code>[5120]</code><br>out <code>[262144,5120]</code> | `rmsnorm` | 1312.284 / — | 3757.462 / — | 551.800 |
| `post_attention_layernorm`<br>0–39 | x <code>[262144,5120]</code><br>weight <code>[5120]</code><br>out <code>[262144,5120]</code> | `rmsnorm` | 1312.284 / — | 3757.462 / — | 551.800 |
| `moe.router_input.float`<br>0–39 | bf16 <code>[262144,5120]</code><br>fp32 <code>[262144,5120]</code> | `cast_bf16_fp32` | 2514.820 / — | 5697.327 / — | 1223.154 |
| `dsa.wq_a`<br>0–39 | x <code>[262144,5120]</code><br>weight_logical <code>[1280,5120]</code><br>y <code>[262144,1280]</code> | 仅形状 | — | — | — |
| `dsa.q_norm`<br>0–39 | x <code>[262144,1280]</code><br>weight <code>[1280]</code><br>y <code>[262144,1280]</code> | 仅形状 | — | — | — |
| `dsa.wq_b`<br>0–39 | x <code>[262144,1280]</code><br>weight_logical <code>[4096,1280]</code><br>y <code>[262144,4096]</code> | 仅形状 | — | — | — |
| `dsa.q_norm_without_weight_and_rope`<br>0–39 | q <code>[262144,8,512]</code><br>rope_suffix <code>[262144,8,64]</code> | 仅形状 | — | — | — |
| `dsa.wkv`<br>0–39 | x <code>[262144,5120]</code><br>weight_logical <code>[512,5120]</code><br>y <code>[262144,512]</code> | 仅形状 | — | — | — |
| `dsa.kv_norm_rope_swa_cache`<br>0–39 | current_kv <code>[262144,1,512]</code><br>weight <code>[512]</code><br>resident_window_logical <code>[8,128,1,512]</code> | 仅形状 | — | — | — |
| `dsa.wo_a`<br>0–39 | x <code>[262144,1,4096]</code><br>weight_logical <code>[1,1024,4096]</code><br>y <code>[262144,1,1024]</code> | 仅形状 | — | — | — |
| `dsa.wo_b`<br>0–39 | x <code>[262144,1024]</code><br>weight_logical <code>[5120,1024]</code><br>y <code>[262144,5120]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>0–1 | q <code>[262144,8,512]</code><br>out <code>[262144,8,512]</code><br>sink <code>[8]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>2–7 | q <code>[262144,8,512]</code><br>out <code>[262144,8,512]</code><br>sink <code>[8]</code><br>compressed_kv_logical <code>[8,16384,1,512]</code><br>indices_capacity_per_query <code>[262144,512]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>8–13 | q <code>[262144,8,512]</code><br>out <code>[262144,8,512]</code><br>sink <code>[8]</code><br>compressed_kv_logical <code>[8,16384,1,512]</code><br>indices_capacity_per_query <code>[262144,512]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>14–19 | q <code>[262144,8,512]</code><br>out <code>[262144,8,512]</code><br>sink <code>[8]</code><br>compressed_kv_logical <code>[8,16384,1,512]</code><br>indices_capacity_per_query <code>[262144,512]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>20–23 | q <code>[262144,8,512]</code><br>out <code>[262144,8,512]</code><br>sink <code>[8]</code><br>compressed_kv_logical <code>[8,32768,1,512]</code><br>indices_capacity_per_query <code>[262144,512]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>24–27 | q <code>[262144,8,512]</code><br>out <code>[262144,8,512]</code><br>sink <code>[8]</code><br>compressed_kv_logical <code>[8,32768,1,512]</code><br>indices_capacity_per_query <code>[262144,512]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>28–31 | q <code>[262144,8,512]</code><br>out <code>[262144,8,512]</code><br>sink <code>[8]</code><br>compressed_kv_logical <code>[8,32768,1,512]</code><br>indices_capacity_per_query <code>[262144,512]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>32–35 | q <code>[262144,8,512]</code><br>out <code>[262144,8,512]</code><br>sink <code>[8]</code><br>compressed_kv_logical <code>[8,32768,1,512]</code><br>indices_capacity_per_query <code>[262144,512]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>36–39 | q <code>[262144,8,512]</code><br>out <code>[262144,8,512]</code><br>sink <code>[8]</code><br>compressed_kv_logical <code>[8,32768,1,512]</code><br>indices_capacity_per_query <code>[262144,512]</code> | 仅形状 | — | — | — |
| `compressor.kv`<br>20 | x <code>[262144,5120]</code><br>weight_logical <code>[512,5120]</code><br>y <code>[262144,512]</code> | 仅形状 | — | — | — |
| `compressor.pool_store`<br>20 | new_rows <code>[262144,1,512]</code><br>history_logical <code>[8,32768,1,512]</code> | 仅形状 | — | — | — |
| `indexer.wk`<br>20 | x <code>[262144,512]</code><br>weight_logical <code>[128,512]</code><br>y <code>[262144,128]</code> | 仅形状 | — | — | — |
| `indexer.norm_cache`<br>20 | key_projection_rows <code>[262144,1,128]</code><br>new_unique_rows <code>[262144,1,128]</code><br>cache_logical <code>[8,32768,1,128]</code> | 仅形状 | — | — | — |
| `compressor.input_cast`<br>2, 8, 14 | bf16 <code>[262144,5120]</code><br>fp32 <code>[262144,5120]</code> | `cast_bf16_fp32` | 2514.820 / — | 5697.327 / — | 1223.154 |
| `compressor.kv`<br>2, 8, 14 | x <code>[262144,5120]</code><br>weight_logical <code>[512,5120]</code><br>y <code>[262144,512]</code> | 仅形状 | — | — | — |
| `compressor.gate`<br>2, 8, 14 | x <code>[262144,5120]</code><br>weight_logical <code>[512,5120]</code><br>y <code>[262144,512]</code> | 仅形状 | — | — | — |
| `compressor.pool_store`<br>2, 8, 14 | new_rows <code>[131072,1,512]</code><br>history_logical <code>[8,16384,1,512]</code> | 仅形状 | — | — | — |
| `indexer.wk`<br>2, 8, 14 | x <code>[262144,512]</code><br>weight_logical <code>[128,512]</code><br>y <code>[262144,128]</code> | 仅形状 | — | — | — |
| `indexer.norm_cache`<br>2, 8, 14 | key_projection_rows <code>[262144,1,128]</code><br>new_unique_rows <code>[131072,1,128]</code><br>cache_logical <code>[8,16384,1,128]</code> | 仅形状 | — | — | — |
| `indexer.wq_b`<br>2, 8, 14, 20, 24, 28, 32, 36 | x <code>[262144,1280]</code><br>weight_logical <code>[4096,1280]</code><br>y <code>[262144,4096]</code> | 仅形状 | — | — | — |
| `indexer.weights_proj`<br>2, 8, 14, 20, 24, 28, 32, 36 | x <code>[262144,5120]</code><br>weight_logical <code>[32,5120]</code><br>y <code>[262144,32]</code> | 仅形状 | — | — | — |
| `indexer.project_weights.float`<br>2, 8, 14, 20, 24, 28, 32, 36 | bf16 <code>[262144,32]</code><br>fp32 <code>[262144,32]</code> | `cast_bf16_fp32` | [9.850 / 9.970](../examples/figures/prefill-flash-32ki-dt.png) | [13.046 / 14.080](../examples/figures/prefill-flash-32ki-pr.png) | 6.656 |
| `indexer.weights_scale`<br>2, 8, 14, 20, 24, 28, 32, 36 | x <code>[262144,32]</code><br>scalar <code>[]</code><br>y <code>[262144,32]</code> | 仅形状 | — | — | — |
| `indexer.select`<br>2, 8, 14, 20, 24, 28, 32, 36 | q <code>[262144,32,128]</code><br>weights <code>[262144,32]</code><br>indices <code>[262144,512]</code> | 仅形状 | — | — | — |
| `indexer.candidate_blocks`<br>20 | block_ids <code>[262144,1,2048]</code> | 仅形状 | — | — | — |
| `indexer.filtered_topk`<br>24, 28, 32, 36 | block_ids <code>[262144,1,2048]</code><br>indices <code>[262144,512]</code> | 仅形状 | — | — | — |
| `engram.lookup`<br>1, 14 | token_mask <code>[262144]</code><br>rows <code>[262144,6144]</code> | 仅形状<br>engram_config enabled | — | — | — |
| `engram.wkv`<br>1, 14 | x <code>[262144,6144]</code><br>weight_logical <code>[25600,6144]</code><br>y <code>[262144,25600]</code> | 仅形状<br>engram_config enabled | — | — | — |
| `engram.gate`<br>1, 14 | residual <code>[262144,4,5120]</code><br>key <code>[262144,4,5120]</code><br>value <code>[262144,5120]</code><br>out <code>[262144,4,5120]</code> | 仅形状<br>engram_config enabled | — | — | — |
| `output.hc_collapse_and_norm`<br>output | x <code>[262144,4,5120]</code><br>out <code>[262144,5120]</code> | 仅形状 | — | — | — |
| `parallel.attention_row_reduce`<br>0–39 | partial <code>[262144,5120]</code><br>result <code>[262144,5120]</code> | 仅形状 | — | — | — |
| `moe.gate_logits`<br>0–39 | x <code>[262144,5120]</code><br>weight_logical <code>[384,5120]</code><br>y <code>[262144,384]</code> | 仅形状 | — | — | — |
| `moe.sqrtsoftplus_topk`<br>0–39 | logits <code>[262144,384]</code><br>weights <code>[262144,6]</code><br>ids <code>[262144,6]</code> | 仅形状 | — | — | — |
| `moe.dispatch_combine`<br>0–39 | rank_source_tokens <code>[262144,5120]</code><br>logical_received_rows <code>[196608,5120]</code><br>combined <code>[262144,5120]</code> | 仅形状 | — | — | — |
| `moe.routed_grouped_ffn`<br>0–39 | x <code>[196608,5120]</code><br>w_gate_up_logical <code>[48,4608,5120]</code><br>gate_up <code>[196608,4608]</code><br>activation <code>[196608,2304]</code><br>w_down_logical <code>[48,5120,2304]</code><br>out <code>[196608,5120]</code> | 仅形状 | — | — | — |
| `moe.shared_gate_up`<br>0–39 | x <code>[262144,5120]</code><br>weight_logical <code>[576,5120]</code><br>y <code>[262144,576]</code> | 仅形状 | — | — | — |
| `moe.shared_activation`<br>0–39 | gate_up <code>[262144,576]</code><br>out <code>[262144,288]</code> | 仅形状 | — | — | — |
| `moe.shared_down`<br>0–39 | x <code>[262144,288]</code><br>weight_logical <code>[5120,288]</code><br>y <code>[262144,5120]</code> | 仅形状 | — | — | — |

## 每请求 131,072 token · 单卡 1,048,576 token

| 节点与层号 | 输入、输出及参数形状 | 单算子合同 | DT 模型 / CA | PR 模型 / CA | 970DT |
|---|---|---|---:|---:|---:|
| `embedding.lookup`<br>input | ids <code>[1048576]</code><br>out <code>[1048576,5120]</code> | 仅形状 | — | — | — |
| `lm_head.last_position`<br>output | x <code>[8,5120]</code><br>weight_logical <code>[16160,5120]</code><br>y <code>[8,16160]</code> | 仅形状 | — | — | — |
| `mhc.attn.pre`<br>0–39 | residual <code>[1048576,4,5120]</code><br>fn <code>[24,20480]</code><br>post <code>[1048576,4]</code><br>mix <code>[1048576,4,4]</code><br>out <code>[1048576,5120]</code> | 仅形状 | — | — | — |
| `mhc.attn.post`<br>0–39 | x <code>[1,1048576,5120]</code><br>residual <code>[1,1048576,4,5120]</code><br>post <code>[1,1048576,4]</code><br>mix <code>[1,1048576,4,4]</code><br>out <code>[1,1048576,4,5120]</code> | `mhc_post`<br>implementation="vllm_hc_post" | — / — | — / — | — |
| `mhc.ffn.pre`<br>0–39 | residual <code>[1048576,4,5120]</code><br>fn <code>[24,20480]</code><br>post <code>[1048576,4]</code><br>mix <code>[1048576,4,4]</code><br>out <code>[1048576,5120]</code> | 仅形状 | — | — | — |
| `mhc.ffn.post`<br>0–39 | x <code>[1,1048576,5120]</code><br>residual <code>[1,1048576,4,5120]</code><br>post <code>[1,1048576,4]</code><br>mix <code>[1,1048576,4,4]</code><br>out <code>[1,1048576,4,5120]</code> | `mhc_post`<br>implementation="vllm_hc_post" | — / — | — / — | — |
| `input_layernorm`<br>0–39 | x <code>[1048576,5120]</code><br>weight <code>[5120]</code><br>out <code>[1048576,5120]</code> | `rmsnorm` | 5339.602 / — | 15264.085 / — | 2193.658 |
| `post_attention_layernorm`<br>0–39 | x <code>[1048576,5120]</code><br>weight <code>[5120]</code><br>out <code>[1048576,5120]</code> | `rmsnorm` | 5339.602 / — | 15264.085 / — | 2193.658 |
| `moe.router_input.float`<br>0–39 | bf16 <code>[1048576,5120]</code><br>fp32 <code>[1048576,5120]</code> | `cast_bf16_fp32` | 10140.828 / — | 22953.892 / — | 4951.425 |
| `dsa.wq_a`<br>0–39 | x <code>[1048576,5120]</code><br>weight_logical <code>[1280,5120]</code><br>y <code>[1048576,1280]</code> | 仅形状 | — | — | — |
| `dsa.q_norm`<br>0–39 | x <code>[1048576,1280]</code><br>weight <code>[1280]</code><br>y <code>[1048576,1280]</code> | 仅形状 | — | — | — |
| `dsa.wq_b`<br>0–39 | x <code>[1048576,1280]</code><br>weight_logical <code>[4096,1280]</code><br>y <code>[1048576,4096]</code> | 仅形状 | — | — | — |
| `dsa.q_norm_without_weight_and_rope`<br>0–39 | q <code>[1048576,8,512]</code><br>rope_suffix <code>[1048576,8,64]</code> | 仅形状 | — | — | — |
| `dsa.wkv`<br>0–39 | x <code>[1048576,5120]</code><br>weight_logical <code>[512,5120]</code><br>y <code>[1048576,512]</code> | 仅形状 | — | — | — |
| `dsa.kv_norm_rope_swa_cache`<br>0–39 | current_kv <code>[1048576,1,512]</code><br>weight <code>[512]</code><br>resident_window_logical <code>[8,128,1,512]</code> | 仅形状 | — | — | — |
| `dsa.wo_a`<br>0–39 | x <code>[1048576,1,4096]</code><br>weight_logical <code>[1,1024,4096]</code><br>y <code>[1048576,1,1024]</code> | 仅形状 | — | — | — |
| `dsa.wo_b`<br>0–39 | x <code>[1048576,1024]</code><br>weight_logical <code>[5120,1024]</code><br>y <code>[1048576,5120]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>0–1 | q <code>[1048576,8,512]</code><br>out <code>[1048576,8,512]</code><br>sink <code>[8]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>2–7 | q <code>[1048576,8,512]</code><br>out <code>[1048576,8,512]</code><br>sink <code>[8]</code><br>compressed_kv_logical <code>[8,65536,1,512]</code><br>indices_capacity_per_query <code>[1048576,512]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>8–13 | q <code>[1048576,8,512]</code><br>out <code>[1048576,8,512]</code><br>sink <code>[8]</code><br>compressed_kv_logical <code>[8,65536,1,512]</code><br>indices_capacity_per_query <code>[1048576,512]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>14–19 | q <code>[1048576,8,512]</code><br>out <code>[1048576,8,512]</code><br>sink <code>[8]</code><br>compressed_kv_logical <code>[8,65536,1,512]</code><br>indices_capacity_per_query <code>[1048576,512]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>20–23 | q <code>[1048576,8,512]</code><br>out <code>[1048576,8,512]</code><br>sink <code>[8]</code><br>compressed_kv_logical <code>[8,131072,1,512]</code><br>indices_capacity_per_query <code>[1048576,512]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>24–27 | q <code>[1048576,8,512]</code><br>out <code>[1048576,8,512]</code><br>sink <code>[8]</code><br>compressed_kv_logical <code>[8,131072,1,512]</code><br>indices_capacity_per_query <code>[1048576,512]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>28–31 | q <code>[1048576,8,512]</code><br>out <code>[1048576,8,512]</code><br>sink <code>[8]</code><br>compressed_kv_logical <code>[8,131072,1,512]</code><br>indices_capacity_per_query <code>[1048576,512]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>32–35 | q <code>[1048576,8,512]</code><br>out <code>[1048576,8,512]</code><br>sink <code>[8]</code><br>compressed_kv_logical <code>[8,131072,1,512]</code><br>indices_capacity_per_query <code>[1048576,512]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>36–39 | q <code>[1048576,8,512]</code><br>out <code>[1048576,8,512]</code><br>sink <code>[8]</code><br>compressed_kv_logical <code>[8,131072,1,512]</code><br>indices_capacity_per_query <code>[1048576,512]</code> | 仅形状 | — | — | — |
| `compressor.kv`<br>20 | x <code>[1048576,5120]</code><br>weight_logical <code>[512,5120]</code><br>y <code>[1048576,512]</code> | 仅形状 | — | — | — |
| `compressor.pool_store`<br>20 | new_rows <code>[1048576,1,512]</code><br>history_logical <code>[8,131072,1,512]</code> | 仅形状 | — | — | — |
| `indexer.wk`<br>20 | x <code>[1048576,512]</code><br>weight_logical <code>[128,512]</code><br>y <code>[1048576,128]</code> | 仅形状 | — | — | — |
| `indexer.norm_cache`<br>20 | key_projection_rows <code>[1048576,1,128]</code><br>new_unique_rows <code>[1048576,1,128]</code><br>cache_logical <code>[8,131072,1,128]</code> | 仅形状 | — | — | — |
| `compressor.input_cast`<br>2, 8, 14 | bf16 <code>[1048576,5120]</code><br>fp32 <code>[1048576,5120]</code> | `cast_bf16_fp32` | 10140.828 / — | 22953.892 / — | 4951.425 |
| `compressor.kv`<br>2, 8, 14 | x <code>[1048576,5120]</code><br>weight_logical <code>[512,5120]</code><br>y <code>[1048576,512]</code> | 仅形状 | — | — | — |
| `compressor.gate`<br>2, 8, 14 | x <code>[1048576,5120]</code><br>weight_logical <code>[512,5120]</code><br>y <code>[1048576,512]</code> | 仅形状 | — | — | — |
| `compressor.pool_store`<br>2, 8, 14 | new_rows <code>[524288,1,512]</code><br>history_logical <code>[8,65536,1,512]</code> | 仅形状 | — | — | — |
| `indexer.wk`<br>2, 8, 14 | x <code>[1048576,512]</code><br>weight_logical <code>[128,512]</code><br>y <code>[1048576,128]</code> | 仅形状 | — | — | — |
| `indexer.norm_cache`<br>2, 8, 14 | key_projection_rows <code>[1048576,1,128]</code><br>new_unique_rows <code>[524288,1,128]</code><br>cache_logical <code>[8,65536,1,128]</code> | 仅形状 | — | — | — |
| `indexer.wq_b`<br>2, 8, 14, 20, 24, 28, 32, 36 | x <code>[1048576,1280]</code><br>weight_logical <code>[4096,1280]</code><br>y <code>[1048576,4096]</code> | 仅形状 | — | — | — |
| `indexer.weights_proj`<br>2, 8, 14, 20, 24, 28, 32, 36 | x <code>[1048576,5120]</code><br>weight_logical <code>[32,5120]</code><br>y <code>[1048576,32]</code> | 仅形状 | — | — | — |
| `indexer.project_weights.float`<br>2, 8, 14, 20, 24, 28, 32, 36 | bf16 <code>[1048576,32]</code><br>fp32 <code>[1048576,32]</code> | `cast_bf16_fp32` | [36.368 / 35.535](../examples/figures/prefill-flash-128ki-dt.png) | [88.943 / 70.711](../examples/figures/prefill-flash-128ki-pr.png) | 16.542 |
| `indexer.weights_scale`<br>2, 8, 14, 20, 24, 28, 32, 36 | x <code>[1048576,32]</code><br>scalar <code>[]</code><br>y <code>[1048576,32]</code> | 仅形状 | — | — | — |
| `indexer.select`<br>2, 8, 14, 20, 24, 28, 32, 36 | q <code>[1048576,32,128]</code><br>weights <code>[1048576,32]</code><br>indices <code>[1048576,512]</code> | 仅形状 | — | — | — |
| `indexer.candidate_blocks`<br>20 | block_ids <code>[1048576,1,2048]</code> | 仅形状 | — | — | — |
| `indexer.filtered_topk`<br>24, 28, 32, 36 | block_ids <code>[1048576,1,2048]</code><br>indices <code>[1048576,512]</code> | 仅形状 | — | — | — |
| `engram.lookup`<br>1, 14 | token_mask <code>[1048576]</code><br>rows <code>[1048576,6144]</code> | 仅形状<br>engram_config enabled | — | — | — |
| `engram.wkv`<br>1, 14 | x <code>[1048576,6144]</code><br>weight_logical <code>[25600,6144]</code><br>y <code>[1048576,25600]</code> | 仅形状<br>engram_config enabled | — | — | — |
| `engram.gate`<br>1, 14 | residual <code>[1048576,4,5120]</code><br>key <code>[1048576,4,5120]</code><br>value <code>[1048576,5120]</code><br>out <code>[1048576,4,5120]</code> | 仅形状<br>engram_config enabled | — | — | — |
| `output.hc_collapse_and_norm`<br>output | x <code>[1048576,4,5120]</code><br>out <code>[1048576,5120]</code> | 仅形状 | — | — | — |
| `parallel.attention_row_reduce`<br>0–39 | partial <code>[1048576,5120]</code><br>result <code>[1048576,5120]</code> | 仅形状 | — | — | — |
| `moe.gate_logits`<br>0–39 | x <code>[1048576,5120]</code><br>weight_logical <code>[384,5120]</code><br>y <code>[1048576,384]</code> | 仅形状 | — | — | — |
| `moe.sqrtsoftplus_topk`<br>0–39 | logits <code>[1048576,384]</code><br>weights <code>[1048576,6]</code><br>ids <code>[1048576,6]</code> | 仅形状 | — | — | — |
| `moe.dispatch_combine`<br>0–39 | rank_source_tokens <code>[1048576,5120]</code><br>logical_received_rows <code>[786432,5120]</code><br>combined <code>[1048576,5120]</code> | 仅形状 | — | — | — |
| `moe.routed_grouped_ffn`<br>0–39 | x <code>[786432,5120]</code><br>w_gate_up_logical <code>[48,4608,5120]</code><br>gate_up <code>[786432,4608]</code><br>activation <code>[786432,2304]</code><br>w_down_logical <code>[48,5120,2304]</code><br>out <code>[786432,5120]</code> | 仅形状 | — | — | — |
| `moe.shared_gate_up`<br>0–39 | x <code>[1048576,5120]</code><br>weight_logical <code>[576,5120]</code><br>y <code>[1048576,576]</code> | 仅形状 | — | — | — |
| `moe.shared_activation`<br>0–39 | gate_up <code>[1048576,576]</code><br>out <code>[1048576,288]</code> | 仅形状 | — | — | — |
| `moe.shared_down`<br>0–39 | x <code>[1048576,288]</code><br>weight_logical <code>[5120,288]</code><br>y <code>[1048576,5120]</code> | 仅形状 | — | — | — |

## 每请求 262,144 token · 单卡 2,097,152 token

| 节点与层号 | 输入、输出及参数形状 | 单算子合同 | DT 模型 / CA | PR 模型 / CA | 970DT |
|---|---|---|---:|---:|---:|
| `embedding.lookup`<br>input | ids <code>[2097152]</code><br>out <code>[2097152,5120]</code> | 仅形状 | — | — | — |
| `lm_head.last_position`<br>output | x <code>[8,5120]</code><br>weight_logical <code>[16160,5120]</code><br>y <code>[8,16160]</code> | 仅形状 | — | — | — |
| `mhc.attn.pre`<br>0–39 | residual <code>[2097152,4,5120]</code><br>fn <code>[24,20480]</code><br>post <code>[2097152,4]</code><br>mix <code>[2097152,4,4]</code><br>out <code>[2097152,5120]</code> | 仅形状 | — | — | — |
| `mhc.attn.post`<br>0–39 | x <code>[1,2097152,5120]</code><br>residual <code>[1,2097152,4,5120]</code><br>post <code>[1,2097152,4]</code><br>mix <code>[1,2097152,4,4]</code><br>out <code>[1,2097152,4,5120]</code> | `mhc_post`<br>implementation="vllm_hc_post" | — / — | — / — | — |
| `mhc.ffn.pre`<br>0–39 | residual <code>[2097152,4,5120]</code><br>fn <code>[24,20480]</code><br>post <code>[2097152,4]</code><br>mix <code>[2097152,4,4]</code><br>out <code>[2097152,5120]</code> | 仅形状 | — | — | — |
| `mhc.ffn.post`<br>0–39 | x <code>[1,2097152,5120]</code><br>residual <code>[1,2097152,4,5120]</code><br>post <code>[1,2097152,4]</code><br>mix <code>[1,2097152,4,4]</code><br>out <code>[1,2097152,4,5120]</code> | `mhc_post`<br>implementation="vllm_hc_post" | — / — | — / — | — |
| `input_layernorm`<br>0–39 | x <code>[2097152,5120]</code><br>weight <code>[5120]</code><br>out <code>[2097152,5120]</code> | `rmsnorm` | 10709.360 / — | 30606.250 / — | 4382.575 |
| `post_attention_layernorm`<br>0–39 | x <code>[2097152,5120]</code><br>weight <code>[5120]</code><br>out <code>[2097152,5120]</code> | `rmsnorm` | 10709.360 / — | 30606.250 / — | 4382.575 |
| `moe.router_input.float`<br>0–39 | bf16 <code>[2097152,5120]</code><br>fp32 <code>[2097152,5120]</code> | `cast_bf16_fp32` | 20308.837 / — | 45962.645 / — | 9922.452 |
| `dsa.wq_a`<br>0–39 | x <code>[2097152,5120]</code><br>weight_logical <code>[1280,5120]</code><br>y <code>[2097152,1280]</code> | 仅形状 | — | — | — |
| `dsa.q_norm`<br>0–39 | x <code>[2097152,1280]</code><br>weight <code>[1280]</code><br>y <code>[2097152,1280]</code> | 仅形状 | — | — | — |
| `dsa.wq_b`<br>0–39 | x <code>[2097152,1280]</code><br>weight_logical <code>[4096,1280]</code><br>y <code>[2097152,4096]</code> | 仅形状 | — | — | — |
| `dsa.q_norm_without_weight_and_rope`<br>0–39 | q <code>[2097152,8,512]</code><br>rope_suffix <code>[2097152,8,64]</code> | 仅形状 | — | — | — |
| `dsa.wkv`<br>0–39 | x <code>[2097152,5120]</code><br>weight_logical <code>[512,5120]</code><br>y <code>[2097152,512]</code> | 仅形状 | — | — | — |
| `dsa.kv_norm_rope_swa_cache`<br>0–39 | current_kv <code>[2097152,1,512]</code><br>weight <code>[512]</code><br>resident_window_logical <code>[8,128,1,512]</code> | 仅形状 | — | — | — |
| `dsa.wo_a`<br>0–39 | x <code>[2097152,1,4096]</code><br>weight_logical <code>[1,1024,4096]</code><br>y <code>[2097152,1,1024]</code> | 仅形状 | — | — | — |
| `dsa.wo_b`<br>0–39 | x <code>[2097152,1024]</code><br>weight_logical <code>[5120,1024]</code><br>y <code>[2097152,5120]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>0–1 | q <code>[2097152,8,512]</code><br>out <code>[2097152,8,512]</code><br>sink <code>[8]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>2–7 | q <code>[2097152,8,512]</code><br>out <code>[2097152,8,512]</code><br>sink <code>[8]</code><br>compressed_kv_logical <code>[8,131072,1,512]</code><br>indices_capacity_per_query <code>[2097152,512]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>8–13 | q <code>[2097152,8,512]</code><br>out <code>[2097152,8,512]</code><br>sink <code>[8]</code><br>compressed_kv_logical <code>[8,131072,1,512]</code><br>indices_capacity_per_query <code>[2097152,512]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>14–19 | q <code>[2097152,8,512]</code><br>out <code>[2097152,8,512]</code><br>sink <code>[8]</code><br>compressed_kv_logical <code>[8,131072,1,512]</code><br>indices_capacity_per_query <code>[2097152,512]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>20–23 | q <code>[2097152,8,512]</code><br>out <code>[2097152,8,512]</code><br>sink <code>[8]</code><br>compressed_kv_logical <code>[8,262144,1,512]</code><br>indices_capacity_per_query <code>[2097152,512]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>24–27 | q <code>[2097152,8,512]</code><br>out <code>[2097152,8,512]</code><br>sink <code>[8]</code><br>compressed_kv_logical <code>[8,262144,1,512]</code><br>indices_capacity_per_query <code>[2097152,512]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>28–31 | q <code>[2097152,8,512]</code><br>out <code>[2097152,8,512]</code><br>sink <code>[8]</code><br>compressed_kv_logical <code>[8,262144,1,512]</code><br>indices_capacity_per_query <code>[2097152,512]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>32–35 | q <code>[2097152,8,512]</code><br>out <code>[2097152,8,512]</code><br>sink <code>[8]</code><br>compressed_kv_logical <code>[8,262144,1,512]</code><br>indices_capacity_per_query <code>[2097152,512]</code> | 仅形状 | — | — | — |
| `dsa.sparse_attention`<br>36–39 | q <code>[2097152,8,512]</code><br>out <code>[2097152,8,512]</code><br>sink <code>[8]</code><br>compressed_kv_logical <code>[8,262144,1,512]</code><br>indices_capacity_per_query <code>[2097152,512]</code> | 仅形状 | — | — | — |
| `compressor.kv`<br>20 | x <code>[2097152,5120]</code><br>weight_logical <code>[512,5120]</code><br>y <code>[2097152,512]</code> | 仅形状 | — | — | — |
| `compressor.pool_store`<br>20 | new_rows <code>[2097152,1,512]</code><br>history_logical <code>[8,262144,1,512]</code> | 仅形状 | — | — | — |
| `indexer.wk`<br>20 | x <code>[2097152,512]</code><br>weight_logical <code>[128,512]</code><br>y <code>[2097152,128]</code> | 仅形状 | — | — | — |
| `indexer.norm_cache`<br>20 | key_projection_rows <code>[2097152,1,128]</code><br>new_unique_rows <code>[2097152,1,128]</code><br>cache_logical <code>[8,262144,1,128]</code> | 仅形状 | — | — | — |
| `compressor.input_cast`<br>2, 8, 14 | bf16 <code>[2097152,5120]</code><br>fp32 <code>[2097152,5120]</code> | `cast_bf16_fp32` | 20308.837 / — | 45962.645 / — | 9922.452 |
| `compressor.kv`<br>2, 8, 14 | x <code>[2097152,5120]</code><br>weight_logical <code>[512,5120]</code><br>y <code>[2097152,512]</code> | 仅形状 | — | — | — |
| `compressor.gate`<br>2, 8, 14 | x <code>[2097152,5120]</code><br>weight_logical <code>[512,5120]</code><br>y <code>[2097152,512]</code> | 仅形状 | — | — | — |
| `compressor.pool_store`<br>2, 8, 14 | new_rows <code>[1048576,1,512]</code><br>history_logical <code>[8,131072,1,512]</code> | 仅形状 | — | — | — |
| `indexer.wk`<br>2, 8, 14 | x <code>[2097152,512]</code><br>weight_logical <code>[128,512]</code><br>y <code>[2097152,128]</code> | 仅形状 | — | — | — |
| `indexer.norm_cache`<br>2, 8, 14 | key_projection_rows <code>[2097152,1,128]</code><br>new_unique_rows <code>[1048576,1,128]</code><br>cache_logical <code>[8,131072,1,128]</code> | 仅形状 | — | — | — |
| `indexer.wq_b`<br>2, 8, 14, 20, 24, 28, 32, 36 | x <code>[2097152,1280]</code><br>weight_logical <code>[4096,1280]</code><br>y <code>[2097152,4096]</code> | 仅形状 | — | — | — |
| `indexer.weights_proj`<br>2, 8, 14, 20, 24, 28, 32, 36 | x <code>[2097152,5120]</code><br>weight_logical <code>[32,5120]</code><br>y <code>[2097152,32]</code> | 仅形状 | — | — | — |
| `indexer.project_weights.float`<br>2, 8, 14, 20, 24, 28, 32, 36 | bf16 <code>[2097152,32]</code><br>fp32 <code>[2097152,32]</code> | `cast_bf16_fp32` | [99.918 / 106.757](../examples/figures/prefill-flash-256ki-dt.png) | [232.748 / 223.493](../examples/figures/prefill-flash-256ki-pr.png) | 42.536 |
| `indexer.weights_scale`<br>2, 8, 14, 20, 24, 28, 32, 36 | x <code>[2097152,32]</code><br>scalar <code>[]</code><br>y <code>[2097152,32]</code> | 仅形状 | — | — | — |
| `indexer.select`<br>2, 8, 14, 20, 24, 28, 32, 36 | q <code>[2097152,32,128]</code><br>weights <code>[2097152,32]</code><br>indices <code>[2097152,512]</code> | 仅形状 | — | — | — |
| `indexer.candidate_blocks`<br>20 | block_ids <code>[2097152,1,2048]</code> | 仅形状 | — | — | — |
| `indexer.filtered_topk`<br>24, 28, 32, 36 | block_ids <code>[2097152,1,2048]</code><br>indices <code>[2097152,512]</code> | 仅形状 | — | — | — |
| `engram.lookup`<br>1, 14 | token_mask <code>[2097152]</code><br>rows <code>[2097152,6144]</code> | 仅形状<br>engram_config enabled | — | — | — |
| `engram.wkv`<br>1, 14 | x <code>[2097152,6144]</code><br>weight_logical <code>[25600,6144]</code><br>y <code>[2097152,25600]</code> | 仅形状<br>engram_config enabled | — | — | — |
| `engram.gate`<br>1, 14 | residual <code>[2097152,4,5120]</code><br>key <code>[2097152,4,5120]</code><br>value <code>[2097152,5120]</code><br>out <code>[2097152,4,5120]</code> | 仅形状<br>engram_config enabled | — | — | — |
| `output.hc_collapse_and_norm`<br>output | x <code>[2097152,4,5120]</code><br>out <code>[2097152,5120]</code> | 仅形状 | — | — | — |
| `parallel.attention_row_reduce`<br>0–39 | partial <code>[2097152,5120]</code><br>result <code>[2097152,5120]</code> | 仅形状 | — | — | — |
| `moe.gate_logits`<br>0–39 | x <code>[2097152,5120]</code><br>weight_logical <code>[384,5120]</code><br>y <code>[2097152,384]</code> | 仅形状 | — | — | — |
| `moe.sqrtsoftplus_topk`<br>0–39 | logits <code>[2097152,384]</code><br>weights <code>[2097152,6]</code><br>ids <code>[2097152,6]</code> | 仅形状 | — | — | — |
| `moe.dispatch_combine`<br>0–39 | rank_source_tokens <code>[2097152,5120]</code><br>logical_received_rows <code>[1572864,5120]</code><br>combined <code>[2097152,5120]</code> | 仅形状 | — | — | — |
| `moe.routed_grouped_ffn`<br>0–39 | x <code>[1572864,5120]</code><br>w_gate_up_logical <code>[48,4608,5120]</code><br>gate_up <code>[1572864,4608]</code><br>activation <code>[1572864,2304]</code><br>w_down_logical <code>[48,5120,2304]</code><br>out <code>[1572864,5120]</code> | 仅形状 | — | — | — |
| `moe.shared_gate_up`<br>0–39 | x <code>[2097152,5120]</code><br>weight_logical <code>[576,5120]</code><br>y <code>[2097152,576]</code> | 仅形状 | — | — | — |
| `moe.shared_activation`<br>0–39 | gate_up <code>[2097152,576]</code><br>out <code>[2097152,288]</code> | 仅形状 | — | — | — |
| `moe.shared_down`<br>0–39 | x <code>[2097152,288]</code><br>weight_logical <code>[5120,288]</code><br>y <code>[2097152,5120]</code> | 仅形状 | — | — | — |

[返回三个模型的对照表](../prefill-shapes.md) · [模型与 CA 流水图集](../pipeline-comparisons.md)
