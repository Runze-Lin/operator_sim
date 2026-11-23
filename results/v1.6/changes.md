# v1.6 · 128Ki decode 的 950 原生实现校准

四条路线各新增 10 个几何的双端冻结独立验证，共 80 个计分窗。按库实现显式选择专家 GMM 或 FP4 CacheLoad。逐 token FP8 独立建模，packed FP4 根据开发窗修正尾部流水。

| 路线 | DT 独立 WAPE / 最大 | PR 独立 WAPE / 最大 | DT/PR 比值 WAPE |
| --- | --- | --- | --- |
| DynamicMXQuant，MXFP4 | 7.85% / 15.00% | 9.70% / 18.56% | 7.85% |
| 逐 token FP8 量化 | 7.43% / 22.77% | 8.54% / 21.29% | 4.25% |
| CacheLoad，显式 AntiMxQuant | 9.93% / 21.86% | 11.87% / 26.53% | 4.47% |
| 路由专家，显式 GroupedMatmulV5 | 9.82% / 21.07% | 6.30% / 10.59% | 4.54% |

packed FP4 旧价格在同集的 WAPE 为 11.29% / 10.69%。其余三条为新价格。8 个只有 CA 的窗与 2 个重复几何的新专家直方图单列。970DT 为外推；Flash 候选列表消费、非连续布局及图中的融合边界仍有限制。

| 实现 | 新 shape / 每端 | DT/PR 加速比 WAPE | 每个 CA 窗用时范围，秒（含开发） | 预测误差明细 |
| --- | --- | --- | --- | --- |
| DynamicMXQuant，MXFP4 | 10 | 7.85% | 51.1–209.4 | [全部 shape](../../docs/errors/quant_mxfp4.md) |
| 逐 token FP8 量化 | 10 | 4.25% | 47.8–76.6 | [全部 shape](../../docs/errors/quant_token_fp8.md) |
| CacheLoad，显式 AntiMxQuant | 10 | 4.47% | 54.8–71.5 | [全部 shape](../../docs/errors/cache_load_mxfp4_antimx.md) |
| 路由专家，显式 GroupedMatmulV5 | 10 | 4.54% | 56.1–208.0 | [全部 shape](../../docs/errors/routed_grouped_matmul_mxfp4_v5.md) |

四个合同内归档在线查询最大 11.770 ms。离线费用为单个端点的墙钟，与设备微秒分开。[总方法](../../docs/modeling-strategy.md) · [误差和未定价实测](../../docs/error-analysis.md) · [系统覆盖](../../docs/system-workload-coverage.md)
