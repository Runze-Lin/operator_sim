# v0.20 · 整批 prefill、末核时间轴与写回边界

按三份模型配置及固定版本 vLLM Ascend 导出 B8 下的 32k、128k、256k，每个 kernel 的 head_dim、hidden、TP/DP/EP、KV 与调度截面随模型计算。54 份形状清单附层号和条件分支，27 次外层 decoder 符号执行核对调用次数。候选融合节点单独标记；线上预测无需权重或 CANN。[真实几何](../../docs/prefill-shapes.md) · [固定依赖和范围](workloads/source-lock.json)

RMSNorm 与 BF16→FP32 Cast 现分别计算每核 MTE、共享 L2 及在设备窗口内必须离芯片的字节。输入预留和目标容量限制可留输出。新增 Kimi KDA 的独立 FP32 beta Sigmoid 及其工作量、分核、UB 四缓冲、exp/div 和 DMA 定价。参考模型的单请求 shape 上限现在与整批 quant／BF16 matmul 的资源外推分开。[全部公式与算例](../../docs/modeling-strategy.md)

复用了 MatMulV3 两波各两页的真实 trace。各计时路线现在有同轴上下图，分别标出模型关键路径或可识别的阶段时长、全核窗口与真实末核指令。KDA chunk 缺少合格的 CA 窗口。[Matmul](../../docs/examples/matmul.md) · [Quant 五轮 UB](../../docs/examples/quant.md) · [Kimi 32k 整批](../../docs/examples/sigmoid_fp32.md)

| 新冻结独立集 | 已完成 DT / PR / 成对（登记 shape） | DT WAPE / 最大 | PR WAPE / 最大 | DT/PR 倍速 WAPE / 最大 |
|---|---|---|---|---|
| RMSNorm | 3 / 3 / 3（10） | 7.81% / 12.90% | 12.35% / 19.83% | 3.77% / 8.65% |
| Kimi KDA beta Sigmoid | 4 / 4 / 4（10） | 40.84% / 55.54% | 102.28% / 120.50% | 19.48% / 30.32% |
| Cast BF16→FP32 | 1 / 0 / 0（4） | 21.76% / 21.76% | N/A | N/A |

Flash indexer 的 BF16 窄投影 [2048,32,5120] 在 DT 预测 12.123 µs，CA 13.023 µs。模型和 CA 各有 640 次 MMAD，模型的纯 Cube 路线漏掉实际 64 个 AIV 的末段。[模型与 CA 的块几何、首个 MMAD 和归并尾段](../../docs/examples/matmul-prefill.md)

[同集 v0.19 对照、逐 shape、970DT 外推和全部路线摘要](../../docs/error-analysis.md) · [冻结模型与输入指纹](model.json) · [数值、时间和数据角色](evaluation.json)

独立安装的 wheel 用三种芯片执行 1,089 个不同的单次查询。中位 0.083 ms，P95 0.548 ms，最慢 2.933 ms。以上 20 个已完成且有时间的正式窗口，CA 单次宿主墙钟 1.4–25.1 分钟；各进程累计 3.43 小时，并发时间重叠。[预测墙钟](wheel-smoke.json) · [RMS 每端](rms/costs.json) · [Cast 原始成本及低记录量实验](large/report.md) · [Sigmoid](newop/measurements.md)

32k Flash Cast 的 DT 原生通道记录显示 16 MiB 输入产生约 16.04 MiB 片外读，写命令为零。对同形 PR 的 Cast，旧误差 +162.9%，本次开发重放 −7.3%。RMSNorm [2048,5120] 的 PR 开发重放从 +77.64% 到 −6.46%，旧 PR 留出十例 WAPE 从 5.16% 上升至 9.04%。对 Sigmoid 的写回敏感性只用于诊断，正式表保留采集前冻结的模型。Flash 的全宽 Cast [262144,5120] 和 Kimi 的全宽动态量化 [262144,7168] 共用一个原生采集位，逐元素核对，wrapper 计时单列。[全宽量化合同](full-quant/freeze.json) · [整批形状逐项状态](../../docs/error-analysis.md#长度增加后设备耗时如何增长)
