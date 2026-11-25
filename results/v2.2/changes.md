# v2.2　指令类别模型的机制修订与适用范围

FA 在续页按寄存器组计算跨行状态指数；行和与概率共用寄存器。发起间隔和依赖延迟分别定价，加入行最大值存储屏障。DT 32Ki 的忙核指数从 3195 修为实测的 2687 条；相对正时长 RV 并集的低估从 33.9% 缩小至 9.0%。Cast、Sigmoid、逐 token FP8 量化增加显式的指令类别选项。[13 个复用窗的机制证据](fa-mechanism.md) · [非 FA 的程序、证据和限制](../../docs/nonfa-vector-classes.md)

首批冻结后 3 种 BF16 新形在 DT/PR 各有 3 个合格完整窗。整窗 WAPE（DT/PR）原法 18.28% / 22.69%，v2.1 类别法 2.98% / 5.91%，本版类别法 0.83% / 3.65%。两端忙核相对 RV 并集的 WAPE 均由旧类别的 37.35% 降至 15.56%。看到较短窗的实际 RV 占比后，另登记 KV3072 双端验证；价格冻结相同，仍高估 VF 26.74%，DT/PR 整窗 +4.30% / +6.12%。8 次 CA 进程累计约 713 秒。非 FA 用 134 个历史整窗和 66 个已有阶段观测，三个算子的类别法均未改善阶段 WAPE。[首批、补充样本、970DT 外推和各次采集成本](fa-validation/README.md) · [非 FA 逐形](nonfa-vector-comparison.csv)

热进程类别法 P95 为 FA 237 µs、Cast 87 µs、Sigmoid 86 µs、量化 76 µs。已有库合同默认原法，FP16 FA 和按行设计默认使用修订类别法；970DT 结果为架构外推。当前没有未知芯片的目标 VF 证据或独立精度验证。保持原始 v2.1 试验数据；复算历史版本请使用对应提交。

[完整报告附录 F](../../docs/paper/14-vector-transfer.md) · [查询墙钟样本统计](vector-cost.json) · [同形 DT/PR 耗时比、逐窗时间与 970DT 场景](vector-transfer.json) · [970DT 单参数扰动](vector-sensitivity.json)

在已安装当前源码、带 Matplotlib 的 Python 环境下，可在仓库根目录重算分析和图表。

```sh
python calibration/examples/vector_method_study.py --study transfer
python calibration/examples/vector_method_study.py --study sensitivity
python calibration/examples/vector_method_study.py --study cost
python calibration/examples/vector_method_study.py --study plot
```
