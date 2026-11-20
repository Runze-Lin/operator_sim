# Matmul 的 decode、MTP 与大 prefill 实测

表内形状均为单 rank 的 [M,N,K]。输入 X[M,K]、权重 W[K,N] 均为连续 BF16 ND，调用安装包 aclnnMatmul，逐元素核验全部 M×N 个输出。N、K 来自三模型在 TP8 下该投影的权重。生产运行的量化权重、转置形式、padding 或融合派发可能改变 kernel。

普通 decode 每个请求贡献一行；MTP8 校验每请求最多贡献九行。B=1、Q=9 与 B=9、Q=1 都可产生 M=9。同一权重和输入几何只验证一次。prefill 的 B8×512 表示一轮调度 4096 个 token，B8×32768 表示 262144 行一次提交。

冻结 16 个新 shape、两端 32 个采集。当前 30 个合格完整窗、0 个不合格、2 个待回收。冻结时三种配置的最大单次预测开销为 2.424 ms。970DT 全部是外推。

| 场景 | 模型 · 投影 | B×每请求 token | [M,N,K] | DT 预测 / CA (误差) µs | PR 预测 / CA (误差) µs | 970DT 外推 µs |
|---|---|---:|---|---:|---:|---:|
| decode | DeepSeek-V4-Pro · indexer.weights_proj | 1×1 | [1, 64, 7168] | 15.126 / 5.585 (+170.8%) | 15.126 / 5.467 (+176.7%) | 13.865 |
| decode | DeepSeek-V4-Pro · indexer.weights_proj | 9×1 | [9, 64, 7168] | 15.130 / 6.079 (+148.9%) | 15.130 / 5.367 (+181.9%) | 13.869 |
| decode | DeepSeek-V4-Pro · indexer.weights_proj | 11×1 | [11, 64, 7168] | 15.131 / 5.688 (+166.0%) | 15.131 / 5.833 (+159.4%) | 13.870 |
| decode | DeepSeek-V4-Pro · indexer.weights_proj | 16×1 | [16, 64, 7168] | 15.133 / 6.108 (+147.7%) | 15.133 / 5.619 (+169.3%) | 13.872 |
| decode | Kimi-K3 · mla.q_b_proj | 1×1 | [1, 2304, 1536] | 8.618 / 6.680 (+29.0%) | 8.621 / 7.494 (+15.0%) | 5.633 |
| decode | DeepSeek-V4.1-Flash · dsa.wq_a | 1×9 | [9, 1280, 5120] | 11.999 / 10.215 (+17.5%) | 13.004 / 13.642 (-4.7%) | 10.999 |
| decode | Kimi-K3 · mla.q_b_proj | 11×1 | [11, 2304, 1536] | 8.628 / 6.939 (+24.3%) | 8.653 / 7.608 (+13.7%) | 5.638 |
| decode | DeepSeek-V4.1-Flash · dsa.wkv | 16×1 | [16, 512, 5120] | 11.930 / 8.023 (+48.7%) | 11.930 / 8.254 (+44.5%) | 10.935 |
| decode | DeepSeek-V4.1-Flash · indexer.weights_proj | 9×9 | [81, 32, 5120] | 7.691 / 6.333 (+21.4%) | 7.691 / 6.184 (+24.4%) | 7.050 |
| decode | Kimi-K3 · mla.q_b_proj | 11×9 | [99, 2304, 1536] | 9.168 / 9.407 (-2.5%) | 9.388 / 9.847 (-4.7%) | 5.866 |
| decode | DeepSeek-V4-Pro · indexer.weights_proj | 16×9 | [144, 64, 7168] | 15.309 / 7.378 (+107.5%) | 15.309 / 7.332 (+108.8%) | 14.033 |
| prefill | DeepSeek-V4.1-Flash · indexer.wk | 1×32768 | [32768, 128, 512] | 19.153 / 15.562 (+23.1%) | 28.820 / 26.690 (+8.0%) | 7.777 |
| prefill | Kimi-K3 · mla.q_b_proj | 8×512 | [4096, 2304, 1536] | 80.760 / 73.615 (+9.7%) | 91.820 / 84.198 (+9.1%) | 26.297 |
| prefill | DeepSeek-V4.1-Flash · indexer.weights_proj | 8×512 | [4096, 32, 5120] | 20.572 / 19.116 (+7.6%) | 34.024 / 29.570 (+15.1%) | 9.509 |
| prefill | DeepSeek-V4.1-Flash · indexer.wk | 8×32768 | [262144, 128, 512] | 113.505 / 85.993 (+32.0%) | 206.156 / 174.953 (+17.8%) | 31.871 |
| prefill | DeepSeek-V4.1-Flash · indexer.weights_proj | 8×32768 | [262144, 32, 5120] | 1082.452 / 待回收 | 1921.347 / 待回收 | 273.952 |

| 场景 | DT 形数 · WAPE / 最大 | PR 形数 · WAPE / 最大 | DT/PR 比值 WAPE / 最大 | 直接用 DT 预测 PR，同形 WAPE |
|---|---:|---:|---:|---:|
| decode | 11 · 71.3% / 170.8% | 11 · 66.1% / 181.9% | 11 · 6.7% / 23.2% | 67.6 |
| prefill | 4 · 20.4% / 32.0% | 4 · 14.4% / 17.8% | 4 · 7.2% / 14.0% | 25.8 |

## 已返回的库分支和成本

| [M,N,K] | 芯片 | 预测 AIC/AIV | 实测 kernel · AIC/AIV · MMAD 数 | 墙钟 (s) | 资格 |
|---|---|---:|---|---:|---|
| [1, 64, 7168] | DT | 1/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_4098_mix_aic · 32/64 · 32 | 72.1 | [qualified](1-64-7168-dt-drain.json) |
| [1, 64, 7168] | PR | 1/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_4098_mix_aic · 28/56 · 28 | 78.1 | [qualified](1-64-7168-pr-drain.json) |
| [9, 64, 7168] | DT | 1/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_4098_mix_aic · 32/64 · 32 | 62.1 | [qualified](9-64-7168-dt-drain.json) |
| [9, 64, 7168] | PR | 1/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_4098_mix_aic · 28/56 · 28 | 72.1 | [qualified](9-64-7168-pr-drain.json) |
| [11, 64, 7168] | DT | 1/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_4098_mix_aic · 32/64 · 32 | 84.1 | [qualified](11-64-7168-dt-drain.json) |
| [11, 64, 7168] | PR | 1/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_4098_mix_aic · 28/56 · 28 | 104.2 | [qualified](11-64-7168-pr-drain.json) |
| [16, 64, 7168] | DT | 1/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_4098_mix_aic · 32/64 · 32 | 76.1 | [qualified](16-64-7168-dt-drain.json) |
| [16, 64, 7168] | PR | 1/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_4098_mix_aic · 28/56 · 28 | 74.1 | [qualified](16-64-7168-pr-drain.json) |
| [1, 2304, 1536] | DT | 18/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 · 18/0 · 216 | 78.1 | [qualified](1-2304-1536-dt-drain.json) |
| [1, 2304, 1536] | PR | 18/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 · 18/0 · 216 | 82.0 | [qualified](1-2304-1536-pr-drain.json) |
| [9, 1280, 5120] | DT | 20/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_4098_mix_aic · 32/64 · 420 | 110.2 | [qualified](9-1280-5120-dt-drain.json) |
| [9, 1280, 5120] | PR | 20/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_4098_mix_aic · 28/56 · 400 | 126.1 | [qualified](9-1280-5120-pr-drain.json) |
| [11, 2304, 1536] | DT | 18/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 · 18/0 · 216 | 78.1 | [qualified](11-2304-1536-dt-drain.json) |
| [11, 2304, 1536] | PR | 18/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 · 18/0 · 216 | 78.0 | [qualified](11-2304-1536-pr-drain.json) |
| [16, 512, 5120] | DT | 8/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_4098_mix_aic · 32/64 · 160 | 92.2 | [qualified](16-512-5120-dt-drain.json) |
| [16, 512, 5120] | PR | 8/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_4098_mix_aic · 28/56 · 168 | 94.1 | [qualified](16-512-5120-pr-drain.json) |
| [81, 32, 5120] | DT | 6/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_4098_mix_aic · 32/64 · 64 | 76.1 | [qualified](81-32-5120-dt-drain.json) |
| [81, 32, 5120] | PR | 6/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_4098_mix_aic · 28/56 · 56 | 72.1 | [qualified](81-32-5120-pr-drain.json) |
| [99, 2304, 1536] | DT | 18/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 · 18/0 · 216 | 90.1 | [qualified](99-2304-1536-dt-drain.json) |
| [99, 2304, 1536] | PR | 18/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 · 18/0 · 216 | 104.0 | [qualified](99-2304-1536-pr-drain.json) |
| [144, 64, 7168] | DT | 9/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_4098_mix_aic · 32/64 · 128 | 92.1 | [qualified](144-64-7168-dt-drain.json) |
| [144, 64, 7168] | PR | 9/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_4098_mix_aic · 28/56 · 112 | 90.1 | [qualified](144-64-7168-pr-drain.json) |
| [32768, 128, 512] | DT | 32/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_131074 · 32/0 · 2048 | 236.3 | [qualified](32768-128-512-dt-drain.json) |
| [32768, 128, 512] | PR | 28/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_131074 · 28/0 · 2048 | 316.2 | [qualified](32768-128-512-pr-drain.json) |
| [4096, 2304, 1536] | DT | 32/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 · 32/0 · 3840 | 1055.2 | [qualified](4096-2304-1536-dt-drain.json) |
| [4096, 2304, 1536] | PR | 28/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 · 28/0 · 3936 | 1224.6 | [qualified](4096-2304-1536-pr-drain.json) |
| [4096, 32, 5120] | DT | 32/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_4098_mix_aic · 32/64 · 1280 | 192.3 | [qualified](4096-32-5120-dt-drain.json) |
| [4096, 32, 5120] | PR | 26/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 · 26/0 · 2080 | 234.1 | [qualified](4096-32-5120-pr-drain.json) |
| [262144, 128, 512] | DT | 32/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_131074 · 32/0 · 16384 | 1396.1 | [qualified](262144-128-512-dt-drain.json) |
| [262144, 128, 512] | PR | 28/0 | MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_131074 · 28/0 · 16384 | 1805.1 | [qualified](262144-128-512-pr-drain.json) |

CA 从所有活动核最早发起计至最后退出。多 kernel 派发保留每个完整窗口及成本，只有冻结合同对应的一次独立 Matmul 窗参加聚合。WAPE 以实测时间加权；分母为零个合格样本时记 N/A。表中预测始终是采集前冻结的普通排程。[当前默认分支、同形误差与加速比](public-report.md) · [单输出分 K 的预登记验证](streamk-analysis.md)。两个队列按完成情况更新此表。[冻结的模型输入、预算与三芯片预测](freeze.json) · [逐形机器可读表](evaluation.json) · [单卡模型规格和切分](../../../docs/prefill-shapes.md)
