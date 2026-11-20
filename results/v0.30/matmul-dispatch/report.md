# 同一投影从 decode 到整批 prefill

本表涵盖 **38 个场景几何**。其中 17 个已有双端完整设备时间。新增 host 对照每端查询 31 个几何，含 3 个旧阳性/阴性控制和 28 个其他几何。各端命中分别为 31/31、31/31。同一 [M,N,K] 的 B1×9 与 B9×1 共用一次测量。

每个输入是单卡一次提交的连续 BF16 ND 矩阵。普通 decode 每请求贡献 1 行，MTP8 主模型校验贡献 9 行。八条未切 chunk 的 prefill 各贡献完整 S 行。表内展示独立库调用；服务侧的量化权重、融合和 chunk 调度会改变实际派发。

| 模型 · 投影 | 阶段 B×Q | [M,N,K] | DT 预测 / CA µs | PR 预测 / CA µs | 970DT 外推 µs | 库 host 选择 DT / PR |
|---|---|---|---:|---:|---:|---|
| DeepSeek-V4-Pro · indexer.weights_proj | decode 1×1 | [1, 64, 7168] | 5.695 / [5.585](../../../results/v0.28/matmul/1-64-7168-dt-drain.json) | 5.573 / [5.467](../../../results/v0.28/matmul/1-64-7168-pr-drain.json) | 13.865 | 切 K + 归并 / 切 K + 归并 |
| DeepSeek-V4-Pro · indexer.weights_proj | decode 1×9、9×1 | [9, 64, 7168] | 5.731 / [6.079](../../../results/v0.28/matmul/9-64-7168-dt-drain.json) | 5.658 / [5.367](../../../results/v0.28/matmul/9-64-7168-pr-drain.json) | 13.869 | 切 K + 归并 / 切 K + 归并 |
| DeepSeek-V4-Pro · indexer.weights_proj | decode 11×1 | [11, 64, 7168] | 5.741 / [5.688](../../../results/v0.28/matmul/11-64-7168-dt-drain.json) | 5.679 / [5.833](../../../results/v0.28/matmul/11-64-7168-pr-drain.json) | 13.870 | 切 K + 归并 / 切 K + 归并 |
| DeepSeek-V4-Pro · indexer.weights_proj | decode 16×1 | [16, 64, 7168] | 5.763 / [6.108](../../../results/v0.28/matmul/16-64-7168-dt-drain.json) | 5.732 / [5.619](../../../results/v0.28/matmul/16-64-7168-pr-drain.json) | 13.872 | 切 K + 归并 / 切 K + 归并 |
| DeepSeek-V4-Pro · indexer.weights_proj | decode 9×9 | [81, 64, 7168] | 15.276 / — | 15.276 / — | 14.003 | 切 K + 归并 / 切 K + 归并 |
| DeepSeek-V4-Pro · indexer.weights_proj | decode 11×9 | [99, 64, 7168] | 15.289 / — | 15.289 / — | 14.015 | 切 K + 归并 / 切 K + 归并 |
| DeepSeek-V4-Pro · indexer.weights_proj | decode 16×9 | [144, 64, 7168] | 15.309 / [7.378](../../../results/v0.28/matmul/144-64-7168-dt-drain.json) | 15.309 / [7.332](../../../results/v0.28/matmul/144-64-7168-pr-drain.json) | 14.033 | 切 K + 归并 / 切 K + 归并 |
| DeepSeek-V4.1-Flash · indexer.wk | decode 1×1 | [1, 128, 512] | 4.230 / [4.161](../../../results/v0.30/matmul-dispatch/device/1-dt-retired.json) | 4.230 / [4.030](../../../results/v0.30/matmul-dispatch/device/1-pr-retired.json) | 3.877 | — / — |
| DeepSeek-V4.1-Flash · dsa.wq_a | decode 1×1 | [1, 1280, 5120] | 11.995 / — | 12.941 / — | 10.996 | 切 K + 归并 / 切 K + 归并 |
| DeepSeek-V4.1-Flash · indexer.wk | decode 1×9、9×1 | [9, 128, 512] | 4.234 / [4.325](../../../results/v0.30/matmul-dispatch/device/9-dt-retired.json) | 4.234 / [4.127](../../../results/v0.30/matmul-dispatch/device/9-pr-retired.json) | 3.881 | — / — |
| DeepSeek-V4.1-Flash · dsa.wq_a | decode 1×9、9×1 | [9, 1280, 5120] | 11.999 / [10.215](../../../results/v0.28/matmul/9-1280-5120-dt-drain.json) | 13.004 / [13.642](../../../results/v0.28/matmul/9-1280-5120-pr-drain.json) | 10.999 | 切 K + 归并 / 切 K + 归并 |
| DeepSeek-V4.1-Flash · indexer.wk | decode 11×1 | [11, 128, 512] | 4.235 / [4.310](../../../results/v0.30/matmul-dispatch/device/11-dt-retired.json) | 4.235 / [4.137](../../../results/v0.30/matmul-dispatch/device/11-pr-retired.json) | 3.882 | — / — |
| DeepSeek-V4.1-Flash · dsa.wq_a | decode 11×1 | [11, 1280, 5120] | 12.000 / — | 13.019 / — | 11.000 | 切 K + 归并 / 切 K + 归并 |
| DeepSeek-V4.1-Flash · indexer.wk | decode 16×1 | [16, 128, 512] | 4.237 / [4.371](../../../results/v0.30/matmul-dispatch/device/16-dt-retired.json) | 4.237 / [4.186](../../../results/v0.30/matmul-dispatch/device/16-pr-retired.json) | 3.884 | — / — |
| DeepSeek-V4.1-Flash · dsa.wq_a | decode 16×1 | [16, 1280, 5120] | 12.003 / — | 13.059 / — | 11.002 | 切 K + 归并 / 切 K + 归并 |
| DeepSeek-V4.1-Flash · indexer.wk | decode 9×9 | [81, 128, 512] | 4.380 / [4.402](../../../results/v0.30/matmul-dispatch/device/81-dt-retired.json) | 4.380 / [4.290](../../../results/v0.30/matmul-dispatch/device/81-pr-retired.json) | 4.015 | — / — |
| DeepSeek-V4.1-Flash · dsa.wq_a | decode 9×9 | [81, 1280, 5120] | 20.557 / — | 16.072 / — | 11.133 | 切 K + 归并 / 切 K + 归并 |
| DeepSeek-V4.1-Flash · indexer.wk | decode 11×9 | [99, 128, 512] | 4.392 / [4.568](../../../results/v0.30/matmul-dispatch/device/99-dt-retired.json) | 4.392 / [4.332](../../../results/v0.30/matmul-dispatch/device/99-pr-retired.json) | 4.026 | — / — |
| DeepSeek-V4.1-Flash · dsa.wq_a | decode 11×9 | [99, 1280, 5120] | 20.643 / — | 17.381 / — | 11.171 | 切 K + 归并 / 切 K + 归并 |
| DeepSeek-V4.1-Flash · indexer.wk | decode 16×9 | [144, 128, 512] | 4.412 / [4.479](../../../results/v0.30/matmul-dispatch/device/144-dt-retired.json) | 4.413 / [4.530](../../../results/v0.30/matmul-dispatch/device/144-pr-retired.json) | 4.045 | — / — |
| DeepSeek-V4.1-Flash · dsa.wq_a | decode 16×9 | [144, 1280, 5120] | 20.665 / — | 22.936 / — | 11.188 | 切 K + 归并 / 切 K + 归并 |
| Kimi-K3 · mla.q_b_proj | decode 1×1 | [1, 2304, 1536] | 8.618 / [6.680](../../../results/v0.28/matmul/1-2304-1536-dt-drain.json) | 8.621 / [7.494](../../../results/v0.28/matmul/1-2304-1536-pr-drain.json) | 5.633 | 纯 Cube / 纯 Cube |
| Kimi-K3 · mla.q_b_proj | decode 1×9、9×1 | [9, 2304, 1536] | 8.626 / — | 8.647 / — | 5.637 | 纯 Cube / 纯 Cube |
| Kimi-K3 · mla.q_b_proj | decode 11×1 | [11, 2304, 1536] | 8.628 / [6.939](../../../results/v0.28/matmul/11-2304-1536-dt-drain.json) | 8.653 / [7.608](../../../results/v0.28/matmul/11-2304-1536-pr-drain.json) | 5.638 | 纯 Cube / 纯 Cube |
| Kimi-K3 · mla.q_b_proj | decode 16×1 | [16, 2304, 1536] | 8.632 / — | 8.669 / — | 5.640 | 纯 Cube / 纯 Cube |
| Kimi-K3 · mla.q_b_proj | decode 9×9 | [81, 2304, 1536] | 9.073 / — | 9.253 / — | 5.788 | 纯 Cube / 纯 Cube |
| Kimi-K3 · mla.q_b_proj | decode 11×9 | [99, 2304, 1536] | 9.168 / [9.407](../../../results/v0.28/matmul/99-2304-1536-dt-drain.json) | 9.388 / [9.847](../../../results/v0.28/matmul/99-2304-1536-pr-drain.json) | 5.866 | 纯 Cube / 纯 Cube |
| Kimi-K3 · mla.q_b_proj | decode 16×9 | [144, 2304, 1536] | 11.877 / — | 11.895 / — | 5.866 | 纯 Cube / 纯 Cube |
| DeepSeek-V4-Pro · indexer.weights_proj | prefill 8×32768 | [262144, 64, 7168] | 1514.284 / — | 2688.910 / — | 382.559 | 纯 Cube / 纯 Cube |
| DeepSeek-V4-Pro · indexer.weights_proj | prefill 8×131072 | [1048576, 64, 7168] | 6043.293 / — | 12193.509 / — | 1491.808 | 纯 Cube / 纯 Cube |
| DeepSeek-V4-Pro · indexer.weights_proj | prefill 8×262144 | [2097152, 64, 7168] | 12081.804 / — | 24363.732 / — | 2970.445 | 纯 Cube / 纯 Cube |
| DeepSeek-V4.1-Flash · indexer.wk | prefill 8×32768 | [262144, 128, 512] | 113.505 / [85.993](../../../results/v0.28/matmul/262144-128-512-dt-drain.json) | 206.156 / [174.953](../../../results/v0.28/matmul/262144-128-512-pr-drain.json) | 31.871 | 纯 Cube / 纯 Cube |
| DeepSeek-V4.1-Flash · indexer.wk | prefill 8×65536 | [524288, 128, 512] | 221.590 / — | 577.534 / — | 58.551 | — / — |
| DeepSeek-V4.1-Flash · indexer.wk | prefill 8×131072 | [1048576, 128, 512] | 437.251 / [405.617](../../../results/v0.28/matmul/long-prefill/131072-dt.json) | 1155.000 / — | 114.805 | 纯 Cube / 纯 Cube |
| DeepSeek-V4.1-Flash · indexer.wk | prefill 8×262144 | [2097152, 128, 512] | 868.573 / — | 2304.688 / — | 226.799 | 纯 Cube / 纯 Cube |
| Kimi-K3 · mla.kv_b_proj | prefill 8×32768 | [262144, 3072, 512] | 2098.195 / — | 5754.504 / — | 522.738 | 纯 Cube / 纯 Cube |
| Kimi-K3 · mla.kv_b_proj | prefill 8×131072 | [1048576, 3072, 512] | 8373.145 / — | 23013.315 / — | 2239.191 | 纯 Cube / 纯 Cube |
| Kimi-K3 · mla.kv_b_proj | prefill 8×262144 | [2097152, 3072, 512] | 16739.743 / — | 46019.822 / — | 4477.027 | 纯 Cube / 纯 Cube |

DT 和 PR 实际调用 CANN 9.2.1 的 aclnnMatmulGetWorkspaceSize，逐形使用真实维度和足额分配的张量。表中 host 选择根据已做过设备复核的两类 workspace 签名解释。512 B 指示普通模板，512+262144×AIC数 B 指示带部分和工作区的模板。签名只能检验这两类分支；确切 tile、MMAD 发起数及设备时间需要完整设备窗。两个 SKU 的 host 整批墙钟分别为 36.3、42.6 秒。

新增 Flash indexer.wk 设备合同固定 N=128、K=512。七个 decode 几何共享同一权重，另测 B8×64Ki，共 16 个端点，只开一个 CA 槽。64Ki 是 32Ki 和 128Ki 两个已有 DT 大形之间的检查点，用于判断 L2 脏输出开始溢出后斜率是否可迁移。未知时间在表内留空，每个正式回执保留全输出数值、发行 ELF、逐核 END 与退休证据。续收回执后运行 report.py 即可更新。

[两卡 host 预测与输入来源](freeze.json) · [DT 原始 host 回执](Ascend950DT_9582-workspace.json) · [PR 原始 host 回执](Ascend950PR_957b-workspace.json) · [设备采前冻结与预算](device/freeze.json) · [设备作业状态](device/state.json) · [全表 JSON](coverage.json) · [已有 33 形档案](../../v0.29/matmul/workloads.md)
