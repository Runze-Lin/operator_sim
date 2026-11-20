# 三模型 Matmul 的真实宽度、batch 和长轴覆盖

按单卡一次连续 BF16 ND 投影计时。输入矩阵为 A[M,K] 和 B[K,N]，输出为 C[M,N]。下列预测均来自 v0.31 公开 estimate()。CA 为完整计时窗，同几何有多个合格窗时取中位数。独立算子调用与服务侧的融合、补齐、量化或 chunk 配置可能不同。

| 数据集 | 不同 [M,N,K] | DT 已测 | PR 已测 | 同形两端均已测 |
|---|---:|---:|---:|---:|
| decode | 38 | 22 | 21 | 20 |
| prefill | 15 | 5 | 5 | 5 |

decode 的 M 等于本卡这一步实际提交的行数。B1 校验 8 个草稿加 1 个主模型位置，与 B9 普通 decode 都给 M=9，共用同一条设备证据。B9、11、16 同时校验 Q9 时对应 81、99、144 行。prefill 按每请求完整交付计算；有 chunk 时把 Q 替换为各请求本步提交的 token 数再求和。970DT 全为架构外推。表中 — 表示归档中缺完整 CA 设备时间。

## decode

| 模型 · 投影 | B×Q | [M,N,K] | 950DT 预测 / CA µs | 950PR 预测 / CA µs | 970DT 外推 µs |
|---|---|---|---:|---:|---:|
| DeepSeek-V4-Pro · indexer.weights_proj | 1×1 | [1, 64, 7168] | 5.695 / [5.585](../../../results/v0.28/matmul/1-64-7168-dt-drain.json) | 5.573 / [5.467](../../../results/v0.28/matmul/1-64-7168-pr-drain.json) | 13.865 |
| DeepSeek-V4-Pro · indexer.weights_proj | 4×1 | [4, 64, 7168] | 5.709 / — | 5.604 / — | 13.867 |
| DeepSeek-V4-Pro · indexer.weights_proj | 8×1 | [8, 64, 7168] | 5.727 / — | 5.647 / — | 13.869 |
| DeepSeek-V4-Pro · indexer.weights_proj | 1×9、9×1 | [9, 64, 7168] | 5.731 / [6.079](../../../results/v0.28/matmul/9-64-7168-dt-drain.json) | 5.658 / [5.367](../../../results/v0.28/matmul/9-64-7168-pr-drain.json) | 13.869 |
| DeepSeek-V4-Pro · indexer.weights_proj | 11×1 | [11, 64, 7168] | 5.741 / [5.688](../../../results/v0.28/matmul/11-64-7168-dt-drain.json) | 5.679 / [5.833](../../../results/v0.28/matmul/11-64-7168-pr-drain.json) | 13.870 |
| DeepSeek-V4-Pro · indexer.weights_proj | 16×1 | [16, 64, 7168] | 5.763 / [6.108](../../../results/v0.28/matmul/16-64-7168-dt-drain.json) | 5.732 / [5.619](../../../results/v0.28/matmul/16-64-7168-pr-drain.json) | 13.872 |
| DeepSeek-V4-Pro · indexer.weights_proj | 8×9 | [72, 64, 7168] | 6.149 / — | 6.104 / — | 13.990 |
| DeepSeek-V4-Pro · indexer.weights_proj | 9×9 | [81, 64, 7168] | 6.249 / — | 6.215 / — | 14.003 |
| DeepSeek-V4-Pro · indexer.weights_proj | 11×9 | [99, 64, 7168] | 6.475 / [7.056](../../../results/v0.31/matmul/device/99-64-7168-dt.json) | 6.461 / [6.888](../../../results/v0.31/matmul/device/99-64-7168-pr.json) | 14.015 |
| DeepSeek-V4-Pro · indexer.weights_proj | 16×9 | [144, 64, 7168] | 7.080 / [7.378](../../../results/v0.28/matmul/144-64-7168-dt-drain.json) | 7.113 / [7.332](../../../results/v0.28/matmul/144-64-7168-pr-drain.json) | 14.033 |
| DeepSeek-V4.1-Flash · dsa.wkv | 16×1 | [16, 512, 5120] | 8.315 / [8.023](../../../results/v0.28/matmul/16-512-5120-dt-drain.json) | 8.345 / [8.254](../../../results/v0.28/matmul/16-512-5120-pr-drain.json) | 10.935 |
| DeepSeek-V4.1-Flash · dsa.wq_a | 1×1 | [1, 1280, 5120] | 11.168 / — | 12.488 / — | 10.996 |
| DeepSeek-V4.1-Flash · dsa.wq_a | 1×9、9×1 | [9, 1280, 5120] | 11.288 / [10.215](../../../results/v0.28/matmul/9-1280-5120-dt-drain.json) | 12.631 / [13.642](../../../results/v0.28/matmul/9-1280-5120-pr-drain.json) | 10.999 |
| DeepSeek-V4.1-Flash · dsa.wq_a | 11×1 | [11, 1280, 5120] | 11.318 / [10.153](../../../results/v0.31/matmul/device/11-1280-5120-dt.json) | 12.667 / [13.579](../../../results/v0.31/matmul/device/11-1280-5120-pr.json) | 11.000 |
| DeepSeek-V4.1-Flash · dsa.wq_a | 16×1 | [16, 1280, 5120] | 11.394 / — | 12.756 / — | 11.002 |
| DeepSeek-V4.1-Flash · dsa.wq_a | 4×9 | [36, 1280, 5120] | 11.800 / — | 13.113 / — | 11.083 |
| DeepSeek-V4.1-Flash · dsa.wq_a | 9×9 | [81, 1280, 5120] | 12.653 / — | 13.916 / — | 11.133 |
| DeepSeek-V4.1-Flash · dsa.wq_a | 11×9 | [99, 1280, 5120] | 12.982 / — | 14.237 / — | 11.171 |
| DeepSeek-V4.1-Flash · dsa.wq_a | 16×9 | [144, 1280, 5120] | 13.777 / — | 15.040 / — | 11.188 |
| DeepSeek-V4.1-Flash · indexer.weights_proj | 1×1 | [1, 32, 5120] | 5.337 / — | 5.211 / — | 6.915 |
| DeepSeek-V4.1-Flash · indexer.weights_proj | 4×1 | [4, 32, 5120] | 5.343 / — | 5.227 / — | 6.916 |
| DeepSeek-V4.1-Flash · indexer.weights_proj | 1×9 | [9, 32, 5120] | 5.355 / — | 5.254 / — | 6.917 |
| DeepSeek-V4.1-Flash · indexer.weights_proj | 16×1 | [16, 32, 5120] | 5.371 / — | 5.291 / — | 6.919 |
| DeepSeek-V4.1-Flash · indexer.weights_proj | 9×9 | [81, 32, 5120] | 5.820 / [6.333](../../../results/v0.28/matmul/81-32-5120-dt-drain.json) | 5.754 / [6.184](../../../results/v0.28/matmul/81-32-5120-pr-drain.json) | 7.050 |
| DeepSeek-V4.1-Flash · indexer.wk | 1×1 | [1, 128, 512] | 4.230 / [4.161](../../../results/v0.30/matmul-dispatch/device/1-dt-retired.json) | 4.230 / [4.030](../../../results/v0.30/matmul-dispatch/device/1-pr-retired.json) | 3.877 |
| DeepSeek-V4.1-Flash · indexer.wk | 1×9、9×1 | [9, 128, 512] | 4.234 / [4.325](../../../results/v0.30/matmul-dispatch/device/9-dt-retired.json) | 4.234 / [4.127](../../../results/v0.30/matmul-dispatch/device/9-pr-retired.json) | 3.881 |
| DeepSeek-V4.1-Flash · indexer.wk | 11×1 | [11, 128, 512] | 4.235 / [4.310](../../../results/v0.30/matmul-dispatch/device/11-dt-retired.json) | 4.235 / [4.137](../../../results/v0.30/matmul-dispatch/device/11-pr-retired.json) | 3.882 |
| DeepSeek-V4.1-Flash · indexer.wk | 16×1 | [16, 128, 512] | 4.237 / [4.371](../../../results/v0.30/matmul-dispatch/device/16-dt-retired.json) | 4.237 / [4.186](../../../results/v0.30/matmul-dispatch/device/16-pr-retired.json) | 3.884 |
| DeepSeek-V4.1-Flash · indexer.wk | 9×9 | [81, 128, 512] | 4.380 / [4.402](../../../results/v0.30/matmul-dispatch/device/81-dt-retired.json) | 4.380 / [4.290](../../../results/v0.30/matmul-dispatch/device/81-pr-retired.json) | 4.015 |
| DeepSeek-V4.1-Flash · indexer.wk | 11×9 | [99, 128, 512] | 4.392 / [4.568](../../../results/v0.30/matmul-dispatch/device/99-dt-retired.json) | 4.392 / [4.332](../../../results/v0.30/matmul-dispatch/device/99-pr-retired.json) | 4.026 |
| DeepSeek-V4.1-Flash · indexer.wk | 16×9 | [144, 128, 512] | 4.412 / [4.479](../../../results/v0.30/matmul-dispatch/device/144-dt-retired.json) | 4.413 / [4.530](../../../results/v0.30/matmul-dispatch/device/144-pr-retired.json) | 4.045 |
| Kimi-K3 · mla.q_b_proj | 1×1 | [1, 2304, 1536] | 8.618 / [6.680](../../../results/v0.28/matmul/1-2304-1536-dt-drain.json) | 8.621 / [7.494](../../../results/v0.28/matmul/1-2304-1536-pr-drain.json) | 5.633 |
| Kimi-K3 · mla.q_b_proj | 1×9、9×1 | [9, 2304, 1536] | 8.626 / — | 8.647 / — | 5.637 |
| Kimi-K3 · mla.q_b_proj | 11×1 | [11, 2304, 1536] | 8.628 / [6.939](../../../results/v0.28/matmul/11-2304-1536-dt-drain.json) | 8.653 / [7.608](../../../results/v0.28/matmul/11-2304-1536-pr-drain.json) | 5.638 |
| Kimi-K3 · mla.q_b_proj | 16×1 | [16, 2304, 1536] | 8.632 / [6.840](../../../results/v0.29/matmul/decode/16-2304-1536-dt.json) | 8.669 / — | 5.640 |
| Kimi-K3 · mla.q_b_proj | 9×9 | [81, 2304, 1536] | 9.073 / — | 9.253 / [9.008](../../../results/v0.29/matmul/decode/81-2304-1536-pr.json) | 5.788 |
| Kimi-K3 · mla.q_b_proj | 11×9 | [99, 2304, 1536] | 9.168 / [9.407](../../../results/v0.28/matmul/99-2304-1536-dt-drain.json) | 9.388 / [9.847](../../../results/v0.28/matmul/99-2304-1536-pr-drain.json) | 5.866 |
| Kimi-K3 · mla.q_b_proj | 16×9 | [144, 2304, 1536] | 11.877 / [10.964](../../../results/v0.29/matmul/decode/144-2304-1536-dt.json) | 11.895 / — | 5.866 |

## prefill

| 模型 · 投影 | B×Q | [M,N,K] | 950DT 预测 / CA µs | 950PR 预测 / CA µs | 970DT 外推 µs |
|---|---|---|---:|---:|---:|
| DeepSeek-V4-Pro · indexer.weights_proj | 8×32,768 | [262144, 64, 7168] | 1514.284 / — | 2688.910 / — | 382.559 |
| DeepSeek-V4-Pro · indexer.weights_proj | 8×131,072 | [1048576, 64, 7168] | 6043.293 / — | 12193.509 / — | 1491.808 |
| DeepSeek-V4-Pro · indexer.weights_proj | 8×262,144 | [2097152, 64, 7168] | 12081.804 / — | 24363.732 / — | 2970.445 |
| DeepSeek-V4.1-Flash · indexer.weights_proj | 8×512 | [4096, 32, 5120] | 21.948 / [19.116](../../../results/v0.28/matmul/4096-32-5120-dt-drain.json) | 34.024 / [29.570](../../../results/v0.28/matmul/4096-32-5120-pr-drain.json) | 9.509 |
| DeepSeek-V4.1-Flash · indexer.weights_proj | 8×32,768 | [262144, 32, 5120] | 1082.452 / — | 1921.347 / — | 273.952 |
| DeepSeek-V4.1-Flash · indexer.wk | 1×32,768 | [32768, 128, 512] | 19.153 / [15.562](../../../results/v0.28/matmul/32768-128-512-dt-drain.json) | 28.820 / [26.690](../../../results/v0.28/matmul/32768-128-512-pr-drain.json) | 7.777 |
| DeepSeek-V4.1-Flash · indexer.wk | 8×32,768 | [262144, 128, 512] | 113.505 / [85.993](../../../results/v0.28/matmul/262144-128-512-dt-drain.json) | 206.156 / [174.953](../../../results/v0.28/matmul/262144-128-512-pr-drain.json) | 31.871 |
| DeepSeek-V4.1-Flash · indexer.wk | 8×65,536 | [524288, 128, 512] | 221.590 / — | 577.534 / — | 58.551 |
| DeepSeek-V4.1-Flash · indexer.wk | 8×131,072 | [1048576, 128, 512] | 437.251 / [405.617](../../../results/v0.28/matmul/long-prefill/131072-dt.json) | 1155.000 / [804.867](../../../results/v0.28/matmul/long-prefill/131072-pr.json) | 114.805 |
| DeepSeek-V4.1-Flash · indexer.wk | 8×262,144 | [2097152, 128, 512] | 868.573 / — | 2304.688 / — | 226.799 |
| Kimi-K3 · mla.kv_b_proj | 8×4,096 | [32768, 3072, 512] | 267.335 / — | 305.800 / — | 70.037 |
| Kimi-K3 · mla.kv_b_proj | 8×32,768 | [262144, 3072, 512] | 2098.195 / — | 5754.504 / — | 522.738 |
| Kimi-K3 · mla.kv_b_proj | 8×131,072 | [1048576, 3072, 512] | 8373.145 / — | 23013.315 / — | 2239.191 |
| Kimi-K3 · mla.kv_b_proj | 8×262,144 | [2097152, 3072, 512] | 16739.743 / — | 46019.822 / — | 4477.027 |
| Kimi-K3 · mla.q_b_proj | 8×512 | [4096, 2304, 1536] | 80.760 / [73.615](../../../results/v0.28/matmul/4096-2304-1536-dt-drain.json) | 91.820 / [84.198](../../../results/v0.28/matmul/4096-2304-1536-pr-drain.json) | 26.297 |

## 大 M 的斜率

同一 Flash indexer.wk 权重的两个相邻已测 M 之间，用 Δ设备时间 / Δ行数计算新增一行的平均成本。下表的倍数来自端点实测。输出乘法量随 M 线性增长，启动、并行轮次和读写越过缓存容量会改变局部斜率。

| 芯片 | 相邻已测 M | 行数倍数 | 实测时间倍数 | 增量 ns/行 |
|---|---|---:|---:|---:|
| 950DT | 32,768 → 262,144 | 8.00 | 5.53 | 0.3071 |
| 950DT | 262,144 → 1,048,576 | 4.00 | 4.72 | 0.4064 |
| 950PR | 32,768 → 262,144 | 8.00 | 6.55 | 0.6464 |
| 950PR | 262,144 → 1,048,576 | 4.00 | 4.60 | 0.8010 |

[模型 config、TP 及投影推导](../../v0.29/matmul/workloads.md) · [同形对比与覆盖角色](report.md) · [B11 的流水和指令](device/example.md) · [长 prefill 的容量模型与正式回执](../../v0.30/matmul/comparison.md)
