# 80 个捕获调用的精度、覆盖和设备证据

按完整后端的累计串行预算排序。每行覆盖该名称全部实例；“代表窗”只覆盖列明的实现与实验范围。双端指 950DT 和 950PR。全量 27 组用任务的 950DT-system 资源，970DT 只作为算子级外推。缺匹配整窗的设计没有 CA 误差。

[Vector 每个调用的例子与误差](examples/system-vector-routes.md) · [矩阵逐 shape 双端误差](examples/system-cube-routes.md) · [注意力与状态更新逐 shape 误差](examples/system-attention-routes.md) · [独立算子的逐窗误差](errors/README.md) · [通信算例](examples/system-communication.md)

| 全图调用组 | 原始名数 | 节点数 | 原串行 ms | 完整串行 ms |
|---|---:|---:|---:|---:|
| Vector、量化和状态 | 42 | 70,624 | 1690.68 | 2414.80 |
| 矩阵和分组乘 | 7 | 20,538 | 1125.05 | 793.18 |
| 注意力、压缩和时序 | 13 | 4,473 | 531.31 | 579.09 |
| Flash 候选过滤 | 1 | 54 | 1467.81 | 65.32 |
| TP/EP 通信 | 4 | 7,326 | 0.00 | 208.74 |
| 视图和空分配 | 13 | 68,217 | 0.00 | 0.00 |

| 原始调用 | 次数 | 输入→输出（第一个实图例） | 模型路径与 CA 边界 | 最大查询 ms |
|---|---:|---|---|---:|
| `npu::npu_dynamic_mx_quant` | 17,748 | BF16[2,7168] → FP4×2[2,3584], E8M0[2,112,2] | 原生 DynamicMxQuant 双端；Slice＋原生量化双端组件 | 8.740 |
| `inference::cache_load` | 3,213 | FP8[1,128,512], float32[1,128] → BF16[1,128,512] | 原生 AntiMxQuant 双端；AscendAntiQuantV2 双端代表窗；异布局 MX 解包设计 | 0.331 |
| `npu::npu_grouped_matmul` | 4,545 | FP8[18,4096], FP4×2[2,2048,1024], E8M0[2,64,1024,2], float32[18] → BF16[18,1024] | 分头 token scale＋MX 权重设计；原生 GMM V5 双端 | 0.941 |
| `inference::kimi_dsa_indexer_score` | 216 | uint8[9,32,64], uint8[9,32,4], uint8[1,132114,68], float32[9,32], int64[9], int32[9,2048] → int32[9,2048] | FP4 混排缓存，MX 原生接口边界见例页 | 0.163 |
| `vllm_ascend::quant_lightning_indexer_reference` | 270 | BF16[9,64,128], BF16[1,32770,128], BF16[9,64], int64[9] → int32[9,1024] | BF16 普通 LI 组件；量化参考调用按图精度 | 0.144 |
| `npu::npu_quant_matmul` | 8,181 | FP4×2[9,192], FP4×2[192,7168], E8M0[6,7168,2], E8M0[9,6,2] → BF16[9,7168] | 原生 A4W4 双端 | 0.485 |
| `aten.mm.default` | 2,241 | BF16[8,512], BF16[512,129280] → BF16[8,129280] | 原生转置权重 aclnnMm 双端；原生严格 FP32 aclnnMm 双端 | 0.273 |
| `collective::all_reduce` | 2,496 | BF16[9,7168] → BF16[9,7168] | 通信场景；互连整窗 N/A | 0.082 |
| `inference::dynamic_mx_quant_fp8_reference` | 8,253 | BF16[1,128,512] → FP8[1,128,512], float32[1,128] | 原生 DynamicQuantV2 双端 | 0.308 |
| `inference::mixed_precision_matmul` | 4,302 | FP8[9,1536], FP4×2[768,8192], E8M0[24,8192,2], float32[9] → BF16[9,8192] | token FP32 scale＋MX 权重设计 | 0.284 |
| `_C_ascend::npu_hc_pre_v2` | 1,926 | BF16[9,4,7168], float32[24,28672], float32[3], float32[24] → BF16[9,7168], float32[9,4], float32[9,4,4] | MhcPreSinkhorn 等价实现，宽通道验证 | 0.781 |
| `npu::npu_fused_infer_attention_score` | 45 | BF16[9,12,64], BF16[9,2,64], BF16[9,2,64], BF16[1,131081,2,2,64] → BF16[9,12,64], float32[12,9] | FIA V5 Q9 代表窗；历史+临时 KV 长形外推 | 0.160 |
| `inference::candidate_filtered_indexer` | 54 | BF16[9,32,128], BF16[1,131090,128], BF16[9,32], int64[9], int32[9,2048] → int32[9,512] | Gather/LI/Sort 组件双端；全窗 N/A | 0.509 |
| `vllm_ascend::recurrent_kda_reference` | 621 | FP4×2[1,9,12,64], FP4×2[1,9,12,64], FP4×2[1,9,12,64], E8M0[1,9,12,2,2], E8M0[1,9,12,2,2], E8M0[1,9,12,2,2], BF16[1,9,12,128], float32[1,9,12], float32[12], float32[1536], float32[1,12,128,128] → BF16[1,9,12,128] | 按实现设计；匹配整窗 N/A | 0.117 |
| `inference::lightning_indexer` | 54 | BF16[9,32,128], BF16[1,131090,128], BF16[9,32], int64[9] → int32[9,512] | 普通 LI 双端组件；候选生产侧输出为设计 | 0.187 |
| `npu::npu_rms_norm` | 8,244 | BF16[8,7168], BF16[7168] → BF16[8,7168], float32[8,1] | 原生 RmsNorm 双端；非标准权重/布局设计 | 12.040 |
| `npu::npu_moe_distribute_combine_v2` | 1,791 | FP8[48,7168], int32[2,6], int32[6912], int32[384], float32[2,6], int32[1], float32[48], float32[48] → BF16[2,7168] | 通信场景；互连整窗 N/A | 0.152 |
| `_C_ascend::npu_hc_post` | 1,926 | BF16[1,9,7168], BF16[1,9,4,7168], float32[1,9,4], float32[1,9,4,4] → BF16[1,9,4,7168] | vLLM 扩展双端代表窗 | 0.317 |
| `vllm_ascend::sparse_attn_sharedkv_reference` | 963 | FP8[9,16,512], FP8[1,128,512], FP8[9,512], float32[9,16], float32[1,128], float32[9], int64[9], float32[16] → BF16[9,16,512] | 按实现设计；匹配整窗 N/A | 0.273 |
| `collective::all_gather` | 1,248 | BF16[2,7168] → BF16[16,7168] | 通信场景；互连整窗 N/A | 0.086 |
| `npu::npu_moe_distribute_dispatch_v2` | 1,791 | FP4×2[2,3584], int32[2,6], float32[2,6], bool[2], E8M0[2,112,2] → FP4×2[48,3584], E8M0[48,112,2], int32[6912], int64[48], int32[384], int32[1], float32[48] | 通信场景；互连整窗 N/A | 0.163 |
| `inference::attention_residual` | 1,683 | BF16[9,7168], BF16[8,9,7168], BF16[7168], BF16[1,7168] → BF16[9,7168] | 按实现设计；匹配整窗 N/A | 0.231 |
| `aten.add.Tensor` | 3,618 | BF16[9,7168], BF16[9,7168] → BF16[9,7168] | 原生 Add BF16 代表窗 | 0.139 |
| `_C_ascend::compressor` | 819 | BF16[9,7168], FP4×2[1024,3584], FP4×2[1024,3584], float32[1,8,2048], float32[4,1024], float32[512], BF16[2,64], BF16[2,64], int32[1,1], int32[2], int32[1], E8M0[112,1024,2], E8M0[112,1024,2] → BF16[2,512] | 按实现设计；匹配整窗 N/A | 0.153 |
| `inference::matmul_rhs_transposed` | 828 | BF16[9,7168], BF16[896,7168] → float32[9,896] | BF16→FP32 Router，独立库探针见矩阵例页 | 0.176 |
| `_C_ascend::inplace_partial_rotary_mul` | 3,321 | BF16[9,1,1,512], BF16[9,1,1,64], BF16[9,1,1,64] → ∅ | 原生 partial RoPE 双端 | 0.147 |
| `aten.clone.default` | 3,564 | BF16[1,128,512] → BF16[1,128,512] | 按实现设计；匹配整窗 N/A | 0.156 |
| `inference::situ` | 1,665 | BF16[112,6144] → BF16[112,3072] | 按实现设计；匹配整窗 N/A | 0.111 |
| `inference::kimi_gated_dsa_attention` | 216 | FP4×2[9,12,256], FP4×2[9,12,32], FP4×2[1,131081,288], E8M0[9,12,8,2], E8M0[9,12,1,2], E8M0[1,131081,9,2], int32[9,2048] → BF16[9,12,512] | 按实现设计；匹配整窗 N/A | 0.149 |
| `npu::npu_swiglu` | 1,971 | BF16[48,6144] → BF16[48,3072] | SwiGlu 双端；带裁剪设计，SwiGlu 组件双端 | 0.143 |
| `vllm_ascend::causal_conv1d_update_reference` | 621 | BF16[9,4608], BF16[4,4608], BF16[1,11,4608] → BF16[9,4608] | 原生 CausalConv1dUpdate 双端 | 0.103 |
| `_C_ascend::moe_gating_top_k` | 828 | float32[2,896], float32[896] → float32[2,16], int32[2,16], float32[2,896] | 原生 MoeGatingTopK 双端代表窗 | 0.184 |
| `inference::cache_store` | 1,926 | BF16[9,512], int64[9], FP8[1,128,512], float32[1,128] → FP8[1,128,512], float32[1,128] | 按实现设计；匹配整窗 N/A | 0.144 |
| `aten._to_copy.default` | 1,818 | float32[9,7168] → BF16[9,7168] | aclnnCast 双端代表基元；异精度/布局外推；原生 Cast 双端 | 0.315 |
| `_C_ascend::moe_gating_top_k_hash` | 963 | float32[2,384], float32[384] → float32[2,6], int32[2,6], float32[2,384] | Hash TopKV2 原生代表窗；无 TID 的带偏置排名设计 | 0.402 |
| `aten.mean.dim` | 90 | float32[9,28672] → float32[9,1] | aclnnMean 双端代表基元；异精度/布局外推 | 0.260 |
| `aten.index.Tensor` | 3,078 | BF16[1048576,64], int64[9] → BF16[9,64] | aclnnGatherV2 双端代表基元；异精度/布局外推 | 0.173 |
| `inference::grouped_gated_rms_norm_reference` | 621 | BF16[9,12,128], BF16[9,12,128], BF16[128] → BF16[9,12,128] | 按实现设计；匹配整窗 N/A | 0.092 |
| `aten.constant_pad_nd.default` | 802 | BF16[9,7168] → BF16[16,7168] | aclnnConstantPadNd 双端代表基元；异精度/布局外推 | 0.111 |
| `inference::quantized_batched_matmul` | 432 | FP4×2[12,9,256], FP4×2[12,512,64], E8M0[12,9,8,2], E8M0[12,512,2,2] → BF16[12,9,128] | 异轴双 MX scale 展开设计 | 0.244 |
| `aten.sigmoid.default` | 684 | float32[8,1] → float32[8,1] | 原生 FP32 Sigmoid 双端 | 0.247 |
| `aten.neg.default` | 963 | BF16[9,64] → BF16[9,64] | aclnnNeg 双端代表基元；异精度/布局外推 | 0.165 |
| `npu::npu_rotary_mul` | 243 | BF16[1,9,1,64], BF16[1,9,1,64], BF16[1,9,1,64] → BF16[1,9,1,64] | RoPE 核心组件校准 | 0.146 |
| `inference::kimi_dsa_indexer_query_quant` | 216 | BF16[9,32,128], BF16[9,32] → uint8[9,32,64], uint8[9,32,4], float32[9,32] | 按实现设计；匹配整窗 N/A | 0.099 |
| `inference::mla_output_gate` | 216 | BF16[9,1536], BF16[9,1536] → BF16[9,1536] | 按实现设计；匹配整窗 N/A | 0.089 |
| `inference::state_update` | 216 | BF16[9,128], float32[128], float32[128], int64[9], uint8[1,132114,68] → uint8[1,132114,68] | 按实现设计；匹配整窗 N/A | 0.116 |
| `aten.cat.default` | 162 | BF16[8,7168], BF16[8,512] → BF16[8,7680] | aclnnCat 双端代表基元；异精度/布局外推 | 0.116 |
| `aten.mul.Tensor` | 108 | float32[9,4,1], float32[9,4,7168] → float32[9,4,7168] | 原生 Mul FP32 广播代表窗 | 0.124 |
| `aten.pow.Tensor_Scalar` | 36 | float32[9,28672] → float32[9,28672] | aclnnPowTensorScalar 双端代表基元；异精度/布局外推 | 0.133 |
| `_C_ascend::compressor_metadata` | 540 | BF16[1048576,64], BF16[1048576,64], int32[2], int32[1], int32[1,1] → BF16[2,1,1,64], BF16[2,1,1,64], int32[2] | 按实现设计；匹配整窗 N/A | 0.111 |
| `aten.sum.dim_IntList` | 36 | float32[9,4,7168] → float32[9,7168] | aclnnReduceSum 双端代表基元；异精度/布局外推 | 0.095 |
| `aten.embedding.default` | 81 | BF16[129280,512], int64[8] → BF16[8,512] | aclnnEmbedding 双端代表基元；异精度/布局外推 | 0.130 |
| `inference::engram_gate` | 18 | BF16[9,4,5120], BF16[9,25600], BF16[4,5120], BF16[4,5120] → BF16[9,4,5120] | 按实现设计；匹配整窗 N/A | 0.108 |
| `aten.lt.Scalar` | 51 | int64[9] → bool[9] | aclnnLtScalar 双端代表基元；异精度/布局外推 | 0.232 |
| `aten.masked_fill_.Scalar` | 51 | BF16[9,7168], bool[9,1] → BF16[9,7168] | aclnnInplaceMaskedFillScalar 双端代表基元；异精度/布局外推 | 0.137 |
| `inference::engram_dequantize` | 18 | uint8[9,864] → BF16[9,768] | 按实现设计；匹配整窗 N/A | 0.110 |
| `aten.ge.Scalar` | 51 | int64[9] → bool[9] | aclnnGeScalar 双端代表基元；异精度/布局外推 | 0.114 |
| `aten.masked_fill.Scalar` | 51 | int64[9], bool[9] → int64[9] | 按实现设计；匹配整窗 N/A | 0.109 |
| `aten.sub.Tensor` | 51 | int64[9] → int64[9] | aclnnSubs 双端代表基元；异精度/布局外推 | 0.111 |
| `inference::engram_lookup` | 18 | int64[9,24], int8[48000217,256], float32[48000217,8] → uint8[9,864] | 按实现设计；匹配整窗 N/A | 0.232 |
| `inference::c2_ring_compressor` | 27 | float32[9,512], float32[9,512], BF16[9,512], float32[1,32,1024], BF16[4,512] → BF16[4,512] | 按实现设计；匹配整窗 N/A | 0.105 |
| `aten.bitwise_or.Tensor` | 51 | bool[9], bool[9] → bool[9] | aclnnBitwiseOrTensor 双端代表基元；异精度/布局外推 | 1.166 |
| `inference::candidate_blocks` | 27 | int32[9,512] → int32[9,2048] | 按实现设计；匹配整窗 N/A | 0.100 |
| `inference::compressor_source_positions` | 54 | int64[9] → int64[9] | 按实现设计；匹配整窗 N/A | 0.120 |
| `aten.rsqrt.default` | 36 | float32[9,1] → float32[9,1] | aclnnRsqrt 双端代表基元；异精度/布局外推 | 0.113 |
| `aten.addmm.default` | 9 | float32[1], float32[8,7680], float32[7680,1] → float32[8,1] | 原生 FP32 aclnnAddmm 双端 | 0.162 |
| `inference::engram_hash` | 9 | int64[1,9], int32[129280], int64[2,24], int64[2,24], int64[2,4], int32[1,3] → int64[9,2,24] | 按实现设计；匹配整窗 N/A | 0.113 |
| `aten.view.default` | 20,097 | BF16[1,8,7168] → BF16[8,7168] | 张量别名，设备时间为零 | 0.360 |
| `aten.unsqueeze.default` | 12,516 | BF16[9,1,64] → BF16[1,9,1,64] | 张量别名，设备时间为零 | 0.069 |
| `aten.expand.default` | 36 | BF16[9,1,7168] → BF16[9,4,7168] | 张量别名，设备时间为零 | 0.041 |
| `aten.select.int` | 1,170 | BF16[9,1,512] → BF16[9,512] | 张量别名，设备时间为零 | 0.046 |
| `aten.alias.default` | 4,437 | BF16[8,129280] → BF16[8,129280] | 张量别名，设备时间为零 | 0.103 |
| `aten.empty.memory_format` | 4,815 | ∅ → bool[2] | 空分配，设备时间为零 | 0.059 |
| `aten.transpose.int` | 3,465 | float32[9,2] → float32[2,9] | 张量别名，设备时间为零 | 0.059 |
| `aten.slice.Tensor` | 9,171 | BF16[1,9,7168] → BF16[1,8,7168] | 张量别名，设备时间为零 | 0.079 |
| `aten.permute.default` | 3,852 | BF16[2,9,1024] → BF16[9,2,1024] | 张量别名，设备时间为零 | 0.055 |
| `aten._unsafe_view.default` | 2,349 | BF16[9,2,1024] → BF16[9,2048] | 张量别名，设备时间为零 | 0.073 |
| `aten.squeeze.dim` | 2,169 | BF16[1,9,1,64] → BF16[9,1,64] | 张量别名，设备时间为零 | 0.043 |
| `aten.t.default` | 2,250 | BF16[129280,512] → BF16[512,129280] | 张量别名，设备时间为零 | 0.142 |
| `aten.split_with_sizes.default` | 1,890 | BF16[9,12,192] → BF16[9,12,128], BF16[9,12,64] | 张量别名，设备时间为零 | 0.044 |

所有输入及输出示例为本卡物理存储形状，FP4×2 的一个字节存两个数；kernel 选择还读取逻辑宽度、stride、量化 scale 和属性。全量 CSV 保留每个任务和阶段、具体方法、相同几何不同属性对应的设备时间范围。它可以用 pandas.read_csv 直接解压读取。[逐几何明细](../calibration/reference/system-full-shapes.csv.gz) · [调用次数与路线分布](../calibration/reference/system-full-operators.csv)
