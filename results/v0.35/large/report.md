# 已登记大形的逐算子预测与 CA

当前公开后端 0.34.0。21 条路线中，20 条有登记的大形，另列单链 KDA 的合法序列上界。本表所选大形中有 6 条取得同 shape、同实现的 DT/PR 双端完整指令计时，合格指令端点共 13 个。表中耗时单位均为 µs。

| 路线 | 本地输入 shape | 950DT 预测 | DT 完整 CA | DT 实测下界 | DT wrapper | 950PR 预测 | PR 完整 CA | 970DT 外推 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| addmm | [65536, 128, 512] | 32.646 | N/A | N/A | N/A | 53.193 | N/A | 12.404 |
| attn_res_fwd | [512, 7168, 8, 1] | 45.078 | N/A | N/A | N/A | 55.623 | N/A | 27.478 |
| cast_bf16_fp32 | [2097152, 64] | 227.018 | 247.788 | N/A | N/A | 506.517 | 529.287 | 104.673 |
| fa | [9, 131072, 64, 1, 1] | 274.987 | 274.821 | N/A | N/A | 274.988 | 265.893 | 252.093 |
| fla_kda_v2 | [1, 1024, 12, 12, 128, 128] | 161.606 | N/A | N/A | N/A | 178.811 | N/A | 111.971 |
| gmm2 | [4096, 7168, 384, 1] | 2390.962 | N/A | N/A | N/A | 2581.778 | N/A | 1372.233 |
| grouped_matmul_swiglu_quant_v2 | [4096, 7168, 384, 1] | 5758.662 | N/A | N/A | N/A | 6570.864 | N/A | 1620.208 |
| kda_recurrent (single chain) | [1, 8, 1, 1, 128, 256] | 67.203 | N/A | N/A | N/A | 68.615 | N/A | 61.538 |
| lightning_indexer | [1, 9, 8192, 8, 128] | 24.653 | N/A | N/A | N/A | 24.768 | N/A | 22.594 |
| matmul | [1048576, 128, 512] | 409.270 | 405.617 | N/A | N/A | 847.926 | 804.867 | 87.608 |
| mhc | [4096, 4, 5120] | 25641.154 | N/A | N/A | N/A | 29774.539 | N/A | 9655.858 |
| mhc_post | [1024, 4, 5120] | 36.705 | N/A | N/A | N/A | 48.464 | N/A | 20.041 |
| mhc_pre | [4096, 4, 5120] | 20100.112 | N/A | N/A | N/A | 23219.433 | N/A | 6767.275 |
| mhc_pre_sinkhorn | [512, 4, 4096] | 43.892 | N/A | N/A | N/A | 54.095 | N/A | 34.296 |
| mhc_sinkhorn | [4097, 4] | 8.105 | 8.964 | N/A | N/A | 8.815 | 9.112 | 5.546 |
| moe_ffn | [4096, 7168, 384, 1] | 8149.624 | N/A | N/A | N/A | 9152.642 | N/A | 2992.441 |
| moe_gating_top_k | [4096, 896, 16] | 125.683 | N/A | N/A | N/A | 154.458 | N/A | 64.095 |
| quant | [262144, 7168] | 1200.415 | N/A | N/A | 1778.988 | 3992.864 | N/A | 527.779 |
| rmsnorm | [8192, 5120] | 39.379 | 44.868 | N/A | N/A | 61.622 | 70.485 | 21.582 |
| sigmoid_fp32 | [2097152, 12] | 45.897 | 42.686 | N/A | N/A | 104.717 | 88.593 | 19.353 |
| vllm_hc_post | [1, 1024, 4, 5120] | 45.169 | N/A | 38.186 | N/A | 52.850 | 44.868 | 24.696 |

完整 CA 取原合格端点的设备指令窗。FA 的 DT 来自 compact CA，PR 来自官方逐指令采集。Quant 的 DT 数值通过且 wrapper 已结束，指令整窗缺失，因此单列。vLLM HcPost 的 DT 缺三个核的结束标记，PR 已从同一采集的逐 launch 副本恢复完整窗。相同数值四舍五入后可能出现在不同算子，具体来源按行给出。970DT 全部为相同实现假设下的估计。

## 几何与证据

| 路线 | 工作负载与适用范围 | DT 指令误差 | PR 指令误差 | 原始登记 / 端点证据 |
|---|---|---:|---:|---|
| addmm | Flash B8×8192 调度 token，显式列 bias 的单算子变体 | N/A | N/A | [登记](../../v0.34/addmm/freeze.json) · [DT](../../v0.34/addmm/score.json) · [PR](../../v0.34/addmm/score.json) |
| attn_res_fwd | Kimi 512 token prefill，7168 通道，1 个有效 cached residual | N/A | N/A | [登记](../../v0.34/attn/freeze.json) · [DT](../../v0.34/attn/score.json) · [PR](../../v0.34/attn/score.json) |
| cast_bf16_fp32 | Pro B8×256Ki，64 列输入转换 | -8.38% | -4.30% | [登记](../../v0.32/large/coverage.json) · [DT](../../v0.23/pro-prefill/pro-b8-256ki-dt-blocks.json) · [PR](../../v0.23/pro-prefill/pro-b8-256ki-pr-blocks.json) |
| fa | 单头稠密 BF16，对 128Ki KV 做 Q9 校验，D=64 控制形 | +0.06% | +3.42% | [登记](../../v0.29/large/fa/freeze.json) · [DT](../../v0.29/large/fa/freeze.json) · [PR](../../v0.29/large/fa/fa-131072-Ascend950PR_957b.json) |
| fla_kda_v2 | Kimi TP8，单请求 prefill 分块 1024 token、12 本地头，强制 V2 | N/A | N/A | [登记](../../v0.29/large/cube/freeze.json) · [DT](../../v0.29/large/cube/matrix.json) · [PR](../../v0.29/large/cube/matrix.json) |
| gmm2 | Pro B8×32Ki，top6 均衡 EP384、专家 TP8，每卡单专家 4096 行 | N/A | N/A | [登记](../../v0.29/large/cube/freeze.json) · [DT](../../v0.29/large/cube/matrix.json) · [PR](../../v0.29/large/cube/matrix.json) |
| grouped_matmul_swiglu_quant_v2 | Pro B8×32Ki，top6 均衡 EP384、专家 TP8，每卡单专家 4096 行 | N/A | N/A | [登记](../../v0.29/large/cube/freeze.json) · [DT](../../v0.29/large/cube/matrix.json) · [PR](../../v0.29/large/cube/matrix.json) |
| kda_recurrent (single chain) | 单链最多 8 步，V=256 为库边界；Kimi V=128 单列在下方 | N/A | N/A | [登记](../../v0.34/recurrent/freeze.json) · [DT](../../v0.34/recurrent/freeze.json) · [PR](../../v0.34/recurrent/freeze.json) |
| lightning_indexer | Q9、8192 个可用 key，稠密 top512；容量对应 32Ki 的 4:1 压缩 | N/A | N/A | [登记](../../v0.34/indexer/freeze.json) · [DT](../../v0.34/indexer/score.json) · [PR](../../v0.34/indexer/score.json) |
| matmul | Flash B8×128Ki，indexer.wk 的 BF16 投影几何 | +0.90% | +5.35% | [登记](../../v0.32/large/coverage.json) · [DT](../../v0.28/matmul/long-prefill/131072-dt.json) · [PR](../../v0.28/matmul/long-prefill/131072-pr.json) |
| mhc | Flash B8×512，拆分库三段合计 | N/A | N/A | [登记](../../v0.29/large/fusion/freeze.json) · [DT](../../v0.29/large/fusion/score.json) · [PR](../../v0.29/large/fusion/score.json) |
| mhc_post | Flash 宽 5120，独立库 MhcPost，带 h_res | N/A | N/A | [登记](../../v0.31/post/freeze.json) · [DT](../../v0.31/post/score.json) · [PR](../../v0.31/post/score.json) |
| mhc_pre | Flash B8×512，拆分库前处理，导出中间量 | N/A | N/A | [登记](../../v0.29/large/fusion/freeze.json) · [DT](../../v0.29/large/fusion/stage-score.json) · [PR](../../v0.29/large/fusion/stage-score.json) |
| mhc_pre_sinkhorn | 库融合分支 512 token、宽 4096、20 次迭代控制形 | N/A | N/A | [登记](../../v0.29/large/fusion/freeze.json) · [DT](../../v0.32/large/coverage.json) · [PR](../../v0.32/large/coverage.json) |
| mhc_sinkhorn | N=4，I=4，超过整批 4096 的尾行控制；历史 resident_aiv 为 DT57/PR52 | -9.58% | -3.26% | [登记](../../v0.32/large/archive.json) · [DT](../../v0.32/large/archive.json) · [PR](../../v0.32/large/archive.json) |
| moe_ffn | Pro B8×32Ki，top6 均衡 EP384、专家 TP8，每卡单专家 4096 行 | N/A | N/A | [登记](../../v0.29/large/cube/freeze.json) · [DT](../../v0.29/large/cube/matrix.json) · [PR](../../v0.29/large/cube/matrix.json) |
| moe_gating_top_k | Kimi B8×512，896 专家中取 16 | N/A | N/A | [登记](../../v0.29/large/vector/freeze.json) · [DT](../gating/score.json) · [PR](../gating/score.json) |
| quant | Kimi B8×32Ki，hidden=7168 | N/A | N/A | [登记](../../v0.29/large/vector/freeze.json) · [DT](../../v0.20/full-quant/native.json) · [PR](../../v0.32/large/coverage.json) |
| rmsnorm | Flash B4×2048 调度 token，hidden=5120 | -12.23% | -12.57% | [登记](../../v0.29/large/vector/freeze.json) · [DT](../../v0.33/large/rms-8192-5120-dt.json) · [PR](../../v0.33/large/rms-8192-5120-pr.json) |
| sigmoid_fp32 | Kimi B8×256Ki，每 token 12 个本地头 | +7.52% | +18.20% | [登记](../../v0.32/large/coverage.json) · [DT](../../v0.23/prefill/kimi-256ki-dt.json) · [PR](../../v0.23/prefill/kimi-256ki-pr.json) |
| vllm_hc_post | Flash 宽 5120，vLLM HcPost 自定义实现 | N/A | +17.79% | [登记](../../v0.31/post/freeze.json) · [DT](../../v0.33/hcpost/partial.json) · [PR](../hcpost/pr-full.json) |

误差为 100 × (公开预测 / 完整 CA − 1)。价格按表中合同调用公开 API，不传实测的核数、tiling 或历史 resident_aiv。4097 行 Sinkhorn 的设备历史和 4 次迭代是该样本的条件，生产 20 次迭代须单独验证。N/A 表示本表未取得相应证据；每条采集的预算、完整性和 kernel 身份沿端点链接可查。

## 同一路线的其他实测

| 路线 | shape | 目标 | 当前预测 µs | 完整 CA µs | 收据 |
|---|---|---|---:|---:|---|
| kda_recurrent | [1, 8, 1, 1, 128, 128] | Ascend950DT_9582 | 39.399 | 35.396 | [来源](../../v0.32/replay.json) |
| kda_recurrent | [1, 8, 1, 1, 128, 128] | Ascend950PR_957b | 40.164 | 39.450 | [来源](../../v0.32/replay.json) |
| lightning_indexer | [1, 1, 8192, 8, 128] | Ascend950DT_9582 | 21.871 | 22.709 | [来源](../../v0.34/indexer/score.json) |
| lightning_indexer | [1, 9, 4096, 8, 128] | Ascend950DT_9582 | 14.801 | 15.334 | [来源](../../v0.34/indexer/score.json) |

## Matmul 整批 prefill 的数据量阶梯

Flash 的 indexer.wk 在 B=8 时，本地形状为 [8 × 每请求输入长度,128,512]。以下按整批一次提交，均为连续 BF16。32Ki 和 128Ki 两档参与过模型开发。64Ki 的 DT 在预测冻结时已开跑且未读取结果，256Ki 为登记留出。使用 chunk 的部署按实际调度行数另行查询。

| 每请求 token | 本地 M | DT 预测 / 完整 CA µs | PR 预测 / 完整 CA µs | 970DT 外推 µs |
|---:|---:|---:|---:|---:|
| 32Ki | 262,144 | 85.246 / [85.993](../../v0.28/matmul/262144-128-512-dt-drain.json) | 176.124 / [174.953](../../v0.28/matmul/262144-128-512-pr-drain.json) | 25.617 |
| 64Ki | 524,288 | 167.563 / N/A | 368.529 / N/A | 44.067 |
| 128Ki | 1,048,576 | 409.270 / [405.617](../../v0.28/matmul/long-prefill/131072-dt.json) | 847.926 / [804.867](../../v0.28/matmul/long-prefill/131072-pr.json) | 87.608 |
| 256Ki | 2,097,152 | 892.683 / N/A | 1806.823 / N/A | 180.829 |

同一投影输入长度从 32Ki 增至 128Ki，行数增至 4 倍。双端实测时间均增长超过 4 倍，不能仅凭这两个开发点断定大形严格线性。64Ki 的原始采集预算为 6 小时，只有完整数值、设备资源和逐核退休均通过才计入。[64Ki 运行与终态](../../v0.33/matmul/64ki-state.json) · [三模型 B=1/9/11/16 的 decode 与 MTP 逐形表](../../v0.32/matmul/decode.md)

63 次公开查询仅计 estimate 调用，当前主机进程内中位 0.434 ms、最大 3.919 ms。离线 CA 沿用原始采集。本表可用同一生成器重新读取已更新的端点计分文件以及完整 HcPost 收据。所有预测、精确参数、端点资格、源文件哈希与查询时间见[机读表](coverage.json)。
