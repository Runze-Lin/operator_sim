# 三个模型的 Matmul 实测总表

共 28 个不同的每卡 [M,N,K]，950DT/PR 共 56 项。已返回 31 项，其中 31 项通过全数值及全核指令窗核验。所有输入与权重均为连续 BF16 ND，N、K 取目标投影在 TP8 下的真实维度。本表使用 v0.28 已封存的公开路由。独立性按各项原登记判断，总表同时包含开发回放与新形。

| 场景 · 模型 | 投影 | B×token | [M,N,K] | DT 公开预测 / CA (误差) µs | PR 公开预测 / CA (误差) µs | 970DT 外推 µs |
|---|---|---:|---|---:|---:|---:|
| decode · DeepSeek-V4-Pro | [indexer.weights_proj](report.md) | 1×1 | [1, 64, 7168] | 5.695 / [5.585 (+2.0%)](1-64-7168-dt-drain.json) | 5.573 / [5.467 (+1.9%)](1-64-7168-pr-drain.json) | 13.865 |
| decode · DeepSeek-V4-Pro | [indexer.weights_proj](report.md) | 9×1 | [9, 64, 7168] | 5.731 / [6.079 (-5.7%)](9-64-7168-dt-drain.json) | 5.658 / [5.367 (+5.4%)](9-64-7168-pr-drain.json) | 13.869 |
| decode · DeepSeek-V4-Pro | [indexer.weights_proj](report.md) | 11×1 | [11, 64, 7168] | 5.741 / [5.688 (+0.9%)](11-64-7168-dt-drain.json) | 5.679 / [5.833 (-2.6%)](11-64-7168-pr-drain.json) | 13.870 |
| decode · DeepSeek-V4-Pro | [indexer.weights_proj](report.md) | 16×1 | [16, 64, 7168] | 5.763 / [6.108 (-5.6%)](16-64-7168-dt-drain.json) | 5.732 / [5.619 (+2.0%)](16-64-7168-pr-drain.json) | 13.872 |
| decode · Kimi-K3 | [mla.q_b_proj](report.md) | 1×1 | [1, 2304, 1536] | 8.618 / [6.680 (+29.0%)](1-2304-1536-dt-drain.json) | 8.621 / [7.494 (+15.0%)](1-2304-1536-pr-drain.json) | 5.633 |
| decode · DeepSeek-V4.1-Flash | [dsa.wq_a](report.md) | 1×9 | [9, 1280, 5120] | 11.999 / [10.215 (+17.5%)](9-1280-5120-dt-drain.json) | 13.004 / [13.642 (-4.7%)](9-1280-5120-pr-drain.json) | 10.999 |
| decode · Kimi-K3 | [mla.q_b_proj](report.md) | 11×1 | [11, 2304, 1536] | 8.628 / [6.939 (+24.3%)](11-2304-1536-dt-drain.json) | 8.653 / [7.608 (+13.7%)](11-2304-1536-pr-drain.json) | 5.638 |
| decode · DeepSeek-V4.1-Flash | [dsa.wkv](report.md) | 16×1 | [16, 512, 5120] | 11.930 / [8.023 (+48.7%)](16-512-5120-dt-drain.json) | 11.930 / [8.254 (+44.5%)](16-512-5120-pr-drain.json) | 10.935 |
| decode · DeepSeek-V4.1-Flash | [indexer.weights_proj](report.md) | 9×9 | [81, 32, 5120] | 7.691 / [6.333 (+21.4%)](81-32-5120-dt-drain.json) | 7.691 / [6.184 (+24.4%)](81-32-5120-pr-drain.json) | 7.050 |
| decode · Kimi-K3 | [mla.q_b_proj](report.md) | 11×9 | [99, 2304, 1536] | 9.168 / [9.407 (-2.5%)](99-2304-1536-dt-drain.json) | 9.388 / [9.847 (-4.7%)](99-2304-1536-pr-drain.json) | 5.866 |
| decode · DeepSeek-V4-Pro | [indexer.weights_proj](report.md) | 16×9 | [144, 64, 7168] | 15.309 / [7.378 (+107.5%)](144-64-7168-dt-drain.json) | 15.309 / [7.332 (+108.8%)](144-64-7168-pr-drain.json) | 14.033 |
| prefill · DeepSeek-V4.1-Flash | [indexer.wk](report.md) | 1×32768 | [32768, 128, 512] | 19.153 / [15.562 (+23.1%)](32768-128-512-dt-drain.json) | 28.820 / [26.690 (+8.0%)](32768-128-512-pr-drain.json) | 7.777 |
| prefill · Kimi-K3 | [mla.q_b_proj](report.md) | 8×512 | [4096, 2304, 1536] | 80.760 / [73.615 (+9.7%)](4096-2304-1536-dt-drain.json) | 91.820 / [84.198 (+9.1%)](4096-2304-1536-pr-drain.json) | 26.297 |
| prefill · DeepSeek-V4.1-Flash | [indexer.weights_proj](report.md) | 8×512 | [4096, 32, 5120] | 20.572 / [19.116 (+7.6%)](4096-32-5120-dt-drain.json) | 34.024 / [29.570 (+15.1%)](4096-32-5120-pr-drain.json) | 9.509 |
| prefill · DeepSeek-V4.1-Flash | [indexer.wk](report.md) | 8×32768 | [262144, 128, 512] | 113.505 / [85.993 (+32.0%)](262144-128-512-dt-drain.json) | 206.156 / [174.953 (+17.8%)](262144-128-512-pr-drain.json) | 31.871 |
| prefill · DeepSeek-V4.1-Flash | [indexer.weights_proj](report.md) | 8×32768 | [262144, 32, 5120] | 1082.452 / 待回收 | 1921.347 / 待回收 | 273.952 |
| prefill · DeepSeek-V4-Pro | [indexer.weights_proj](wide-prefill/report.md) | 8×32768 | [262144, 64, 7168] | 1514.284 / 待回收 | 2688.910 / 待回收 | 382.559 |
| prefill · Kimi-K3 | [mla.kv_b_proj](wide-prefill/report.md) | 8×32768 | [262144, 3072, 512] | 2098.195 / 待回收 | 5754.504 / 待回收 | 522.738 |
| prefill · DeepSeek-V4.1-Flash | [indexer.wk](long-prefill/report.md) | 8×131072 | [1048576, 128, 512] | 437.251 / [405.617 (+7.8%)](long-prefill/131072-dt.json) | 1155.000 / 待回收 | 114.805 |
| prefill · DeepSeek-V4.1-Flash | [indexer.wk](long-prefill/report.md) | 8×262144 | [2097152, 128, 512] | 868.573 / 待回收 | 2304.688 / 待回收 | 226.799 |
| decode · DeepSeek-V4.1-Flash | [indexer.weights_proj](narrow/report.md) | 1×1 | [1, 32, 5120] | 5.337 / 待回收 | 5.211 / 待回收 | 6.915 |
| decode · DeepSeek-V4.1-Flash | [indexer.weights_proj](narrow/report.md) | 4×1 | [4, 32, 5120] | 5.343 / 待回收 | 5.227 / 待回收 | 6.916 |
| decode · DeepSeek-V4.1-Flash | [indexer.weights_proj](narrow/report.md) | 1×9 | [9, 32, 5120] | 5.355 / 待回收 | 5.254 / 待回收 | 6.917 |
| decode · DeepSeek-V4.1-Flash | [indexer.weights_proj](narrow/report.md) | 16×1 | [16, 32, 5120] | 5.371 / 待回收 | 5.291 / 待回收 | 6.919 |
| decode · DeepSeek-V4-Pro | [indexer.weights_proj](narrow/report.md) | 4×1 | [4, 64, 7168] | 5.709 / 待回收 | 5.604 / 待回收 | 13.867 |
| decode · DeepSeek-V4-Pro | [indexer.weights_proj](narrow/report.md) | 8×1 | [8, 64, 7168] | 5.727 / 待回收 | 5.647 / 待回收 | 13.869 |
| decode · DeepSeek-V4-Pro | [indexer.weights_proj](multi/report.md) | 8×9 | [72, 64, 7168] | 15.262 / 待回收 | 15.262 / 待回收 | 13.990 |
| decode · DeepSeek-V4.1-Flash | [dsa.wq_a](multi/report.md) | 4×9 | [36, 1280, 5120] | 12.156 / 待回收 | 13.360 / 待回收 | 11.083 |

M 为此次提交的总行数。B=1 的 MTP8 校验和 B=9 的普通 decode 可能得到相同的 M=9；B8×128Ki 和 B8×256Ki 分别为 1,048,576 和 2,097,152 行。实际部署中的量化、权重转置或其他融合会影响 kernel 选择。[单核分段与真实 MMAD](streamk-analysis.md) · [当前 WAPE 与直接复用 DT](public-report.md) · [重选输出块的候选及独立验证](multi/report.md) · [逐项数据与资格](overview.json)
