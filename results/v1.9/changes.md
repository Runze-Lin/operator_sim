# v1.9 · SIMPlatform 全量后端

`operator_sim_full` 在 27 个 rank-0 decode 任务的 target/draft 图上为全部 80 个原始调用、171,232 个节点返回价格。63 个计算调用、4 个 TP/EP 通信和 13 个结构调用都有显式合同。每个节点保留原有资源与依赖。未知合同返回缺失原因，复算器拒绝接纳缺价的图。三种后端可独立选择。[结果与复算](../../docs/system-full-comparison.md) · [80 个名字的实现边界](../../docs/system-full-evidence.md)

| 同一组 27 个任务 | 原 Roofline | v1.8 混合 | v1.9 完整 |
|---|---:|---:|---:|
| 单轮 DES 时长之和 ms | 4814.851 | 3737.268 | 3992.324 |
| 接管原始计费预算 | — | 67.63% | 100% |

新采 108 个通过数值、核名单及设备识别检查的单芯片 CA 窗口，包括 60 个基础原语、16 个 Vector/融合窗口、20 个矩阵窗口和 12 个注意力/状态窗口。组件实验只核对组件。60 个基础原语在两种 SKU 的成功运行进程累计 1223.79 秒，各组有并发。超时、模拟 DMA 信号 11 和 AI CPU metadata 任务不支持的记录单列。二进制与完整 trace 留在本机，仓内保留可核验的 SHA、整窗、分核和独立输出检查。 [原语](../../calibration/system_primitives_ca/README.md) · [矩阵](../../docs/examples/system-cube-routes.md) · [Vector](../../docs/examples/system-vector-routes.md) · [注意力](../../docs/examples/system-attention-routes.md)

对 FP32 输出 Router 用已识别的 Stream-K 分核和归并，冻结后的 [144,896,7168] DT/PR 误差为 +12.5%/+4.4%。Hash 路由按源码先打分 E 个专家再 gather K 项，冻结后的 [9,384,6] 双端为 −5.4%/−5.9%。卷积从 D4608 迁移到 D9216 和 INT64 减法的大尺寸仍有分核误差。为保持数据角色清晰，单个校准点的整窗拟合不能证明独立精度。[逐 shape 误差与资格](../../docs/error-analysis.md)

7,131 次跳过系统结果缓存的在线查询，P95 1.011 ms，最大 12.040 ms。27 组原任务有 8 组静态峰值超过 144 GB，按原 compare 模式保留预测。每 rank 200 GB/s 及每步 1 µs 属于外部互连场景假设。没有多卡网络与整机真值；970DT 的算子结果为资源外推。[发布文件摘要与图指纹](release-receipt.json)
