# 三个模型的单卡 Matmul 几何与实测

共 33 个不同的 [M,N,K]，其中 16 个已有 DT、PR 同形完整 CA。每个几何只计一次。输入 X[M,K]、权重 W[K,N] 均为连续 BF16 ND。表中预测统一用公开路由，各条件候选的同集比较见分表。两个 SKU 的 CA 只有在数值、库实现和全核结束均通过时才填入。

| 模型 · 投影 | 场景 B×Q | [M,N,K] | DT 公开 / CA (µs) | PR 公开 / CA (µs) | 970DT 公开外推 (µs) |
|---|---|---|---:|---:|---:|
| DeepSeek-V4-Pro · indexer.weights_proj | decode 1×1 | [1, 64, 7168] | 5.695 / 5.585 | 5.573 / 5.467 | 13.865 |
| DeepSeek-V4-Pro · indexer.weights_proj | decode 9×1 | [9, 64, 7168] | 5.731 / 6.079 | 5.658 / 5.367 | 13.869 |
| DeepSeek-V4-Pro · indexer.weights_proj | decode 11×1 | [11, 64, 7168] | 5.741 / 5.688 | 5.679 / 5.833 | 13.870 |
| DeepSeek-V4-Pro · indexer.weights_proj | decode 16×1 | [16, 64, 7168] | 5.763 / 6.108 | 5.732 / 5.619 | 13.872 |
| Kimi-K3 · mla.q_b_proj | decode 1×1 | [1, 2304, 1536] | 8.618 / 6.680 | 8.621 / 7.494 | 5.633 |
| DeepSeek-V4.1-Flash · dsa.wq_a | decode 1×9 | [9, 1280, 5120] | 11.999 / 10.215 | 13.004 / 13.642 | 10.999 |
| Kimi-K3 · mla.q_b_proj | decode 11×1 | [11, 2304, 1536] | 8.628 / 6.939 | 8.653 / 7.608 | 5.638 |
| DeepSeek-V4.1-Flash · dsa.wkv | decode 16×1 | [16, 512, 5120] | 11.930 / 8.023 | 11.930 / 8.254 | 10.935 |
| DeepSeek-V4.1-Flash · indexer.weights_proj | decode 9×9 | [81, 32, 5120] | 7.691 / 6.333 | 7.691 / 6.184 | 7.050 |
| Kimi-K3 · mla.q_b_proj | decode 11×9 | [99, 2304, 1536] | 9.168 / 9.407 | 9.388 / 9.847 | 5.866 |
| DeepSeek-V4-Pro · indexer.weights_proj | decode 16×9 | [144, 64, 7168] | 15.309 / 7.378 | 15.309 / 7.332 | 14.033 |
| DeepSeek-V4.1-Flash · indexer.wk | prefill 1×32768 | [32768, 128, 512] | 19.153 / 15.562 | 28.820 / 26.690 | 7.777 |
| Kimi-K3 · mla.q_b_proj | prefill 8×512 | [4096, 2304, 1536] | 80.760 / 73.615 | 91.820 / 84.198 | 26.297 |
| DeepSeek-V4.1-Flash · indexer.weights_proj | prefill 8×512 | [4096, 32, 5120] | 20.572 / 19.116 | 34.024 / 29.570 | 9.509 |
| DeepSeek-V4.1-Flash · indexer.wk | prefill 8×32768 | [262144, 128, 512] | 113.505 / 85.993 | 206.156 / 174.953 | 31.871 |
| DeepSeek-V4.1-Flash · indexer.weights_proj | prefill 8×32768 | [262144, 32, 5120] | 1082.452 / 待返回 | 1921.347 / 待返回 | 273.952 |
| DeepSeek-V4-Pro · indexer.weights_proj | prefill 8×32768 | [262144, 64, 7168] | 1514.284 / 待返回 | 2688.910 / 待返回 | 382.559 |
| Kimi-K3 · mla.kv_b_proj | prefill 8×32768 | [262144, 3072, 512] | 2098.195 / 待返回 | 5754.504 / 待返回 | 522.738 |
| DeepSeek-V4.1-Flash · indexer.wk | prefill 8×131072 | [1048576, 128, 512] | 437.251 / 405.617 | 1155.000 / 804.867 | 114.805 |
| DeepSeek-V4.1-Flash · indexer.wk | prefill 8×262144 | [2097152, 128, 512] | 868.573 / 待返回 | 2304.688 / 待返回 | 226.799 |
| DeepSeek-V4.1-Flash · indexer.weights_proj | decode 1×1 | [1, 32, 5120] | 5.337 / 待返回 | 5.211 / 待返回 | 6.915 |
| DeepSeek-V4.1-Flash · indexer.weights_proj | decode 4×1 | [4, 32, 5120] | 5.343 / 待返回 | 5.227 / 待返回 | 6.916 |
| DeepSeek-V4.1-Flash · indexer.weights_proj | decode 1×9 | [9, 32, 5120] | 5.355 / 待返回 | 5.254 / 待返回 | 6.917 |
| DeepSeek-V4.1-Flash · indexer.weights_proj | decode 16×1 | [16, 32, 5120] | 5.371 / 待返回 | 5.291 / 待返回 | 6.919 |
| DeepSeek-V4-Pro · indexer.weights_proj | decode 4×1 | [4, 64, 7168] | 5.709 / 待返回 | 5.604 / 待返回 | 13.867 |
| DeepSeek-V4-Pro · indexer.weights_proj | decode 8×1 | [8, 64, 7168] | 5.727 / 待返回 | 5.647 / 待返回 | 13.869 |
| DeepSeek-V4-Pro · indexer.weights_proj | decode 8×9 | [72, 64, 7168] | 15.262 / 待返回 | 15.262 / 待返回 | 13.990 |
| DeepSeek-V4.1-Flash · dsa.wq_a | decode 4×9 | [36, 1280, 5120] | 12.156 / 待返回 | 13.360 / 待返回 | 11.083 |
| Kimi-K3 · mla.q_b_proj | decode 9×1、decode 1×9 | [9, 2304, 1536] | 8.626 / 未获合格窗 | 8.647 / 未获合格窗 | 5.637 |
| Kimi-K3 · mla.q_b_proj | decode 16×1 | [16, 2304, 1536] | 8.632 / 6.840 | 8.669 / 未获合格窗 | 5.640 |
| Kimi-K3 · mla.q_b_proj | decode 9×9 | [81, 2304, 1536] | 9.073 / 未获合格窗 | 9.253 / 9.008 | 5.788 |
| Kimi-K3 · mla.q_b_proj | decode 16×9 | [144, 2304, 1536] | 11.877 / 10.964 | 11.895 / 待返回 | 5.866 |
| Flash · indexer.weights_proj | 切 K 边界 | [3585, 32, 5120] | 20.558 / 待返回 | 30.250 / 待返回 | 7.329 |

Q 是本次派发中每个请求提交的行数。普通 decode 取 1，MTP 推 8 个 token 时主模型校验取 9。整批 B8×128Ki 与 B8×256Ki 的 M 分别为 1,048,576 和 2,097,152；使用 chunk 时从该次实际提交行数定价。表中时间只覆盖独立库调用。

[基础 28 形与原始回执](../../v0.28/matmul/README.md) · [Kimi 同宽 decode 与补测](decode/host-placement.md) · [Kimi 逐批误差](decode/report.md) · [切 K 边界](report.md) · [大形权重常驻模型与 CA](full-load/report.md)
