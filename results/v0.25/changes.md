# v0.25 · 同一分块上的预测与 CA

| 路线 | 方法和范围 | 同集结果 |
|---|---|---|
| Pro indexer Cast | 8 条 128Ki 请求同卡整批，BF16 [1048576,64]，真实 56 个 PR AIV。复用冻结的预测，按读写地址将 57 个 UB 块对回实际指令。 | DT 模型 / CA 99.918 / 105.374 µs，PR 232.748 / 224.719 µs；970DT 外推 42.536 µs。PR 的第 29 块在模型中读晚 25.689 µs、写完成早 1.574 µs。[放大三块](pro-128ki-pr-middle.png) |
| FA | 三种配置用同一套页级资源排程，显式 calibrated 历史基线保留。 | 三个新形，DT WAPE 16.36% → 7.56%，PR 7.42%。DT/PR 比值 WAPE 18.60% → 2.85%，方向 2/3 → 1/3。[八页逐一对照](coverage/fa-dt-1008-pages.png) |
| RMSNorm | 可选标注中间派发的纯 AIC 合同，保留该 rank、该流的 RMS AIV 前序；不引入形状系数。 | DT 10 形，WAPE 21.87% → 7.19%，最大 38.34% → 22.23%；PR 10 形，WAPE 18.12% → 7.00%，最大 40.12% → 21.49%。多行组原始热身掩码的独立核实、复采和完整目标窗单独报告。[冻结后验证](mixed-stream/README.md) |
| 内置 mHC Post | 候选拆开每行四路读、Vector 发起、两行之间的流水间隔与最后一笔写回。 | 950DT 5 形 / 8 窗，给前序 WAPE 29.17% → 10.61%，最大 27.59%；950PR 3 形 / 6 窗，给前序 WAPE 16.78% → 2.39%，最大 6.67%。精确重复和普通复用分别评分。[原有价格与候选](post/report.md) · [逐 token 流水](post/two-token-repeat.png) |
| KDA V2 / 融合 V1 | V2 补计已登记的三段完整窗。V1 对新正式版本做独立同步核查。 | V2 累计 DT / PR 各 9 / 8 形，三段 WAPE 1.75% / 1.12%。 V1 新正式 DT 四输出正确，167 次重复 SET，正常耗时 N/A。[V1 实际证据](kda-v1/README.md) |

19 条可计时路线的去重回放含 848 个历史窗和 6 个新 FA 窗；内置 Post、RMS 与 V1 有单独的资格与误差统计。PR Cast/Sigmoid 在 L2 到两倍 L2 容量的两形组仍有 27.60% / 21.53% WAPE。新采集登记了同为 156 MiB 逻辑体量、读写比例不同的 Cast 和 Sigmoid，各在两种 950 上验证。970DT 只给情景外推。[大输入的尾三块](cache/flash-128ki-pr-tail.png) · [新形及冻结参数](cache/freeze.json) · [历史逐形与来源](coverage/report.md)

Pro PR 完整 CA 墙钟 30.28 分钟，FA 六次共 10.42 分钟，RMS 已核验的独立原始及复采进程合计 19.25 分钟。已安装 wheel 的 1,661 个常规调用最大 5.329 ms，带流历史的额外调用最大 9.958 ms。 Pro B8×256Ki 与全 hidden 长作业继续按预算排队，等待期间不填入误差统计。[安装件基准](wheel-check.json) · [Pro 256k 队列状态](../v0.23/pro-prefill/state.json)

[建模方法](../../docs/modeling-strategy.md) · [全模型误差和超限子组](../../docs/error-analysis.md) · [逐形测量与 970 外推](shape-errors.md) · [逐算子流水图](../../docs/pipeline-comparisons.md)
