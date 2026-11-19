# v0.26 的方法与误差变化

| 验证合同 | 950DT | 950PR |
|---|---:|---:|
| FA 新形，shape 数 · WAPE / 最大 | 8 · 5.84% / 9.84% | 7 · 7.33% / 13.68% |
| FLA 26.9.1 KDA 三段新形，shape 数 · WAPE / 最大 | 10 · 1.61% / 3.08% | 10 · 0.99% / 2.09% |
| 内置 Post 紧邻重复，合格独立窗数 · 可选 WAPE / 最大 | 5 · 2.55% / 4.68% | 2 · 3.40% / 3.85% |
| 单独 Sinkhorn，已知前序候选，独立窗数 · WAPE / 最大 | 1 · 1.68% / 1.68% | 0 · N/A |

Cast 的诊断现在给出前两个空闲输入槽的发起时刻。一个 950DT 窗测得间隔 91 周期，其他九个旧窗在双端均重现；整窗沿用共享带宽和换槽预算。新做的 Flash B8×256Ki、Pro B8×128Ki 上下图使用同尺度，重画不增加 CA 时间。Pro B8×256Ki 单次读 256 MiB、写 512 MiB，前两笔发起、稳态完成前沿和两种缓存预测均已登记，采集沿用原长作业。[逐块证据和长作业进度](memory/README.md)

Post 显式开启精确重复定价；其它热态中，宽通道的 MTE2 仍会高估超过 2.5 µs。两种 Sinkhorn 候选只以 DT 阶段校准。独立版按每 8 行寄存器循环和 VF 首发/间隔定价，融合版分别给 K 分块 MMAD 和两段 Vector 定价。融合 PR 开发回放 WAPE 9.39% → 5.39%，DT 7.91% → 5.97%，小 token PR 有回退。默认合同和候选各自列出。[Post](post/report.md) · [独立 Sinkhorn](sink/report.md) · [融合 PreSinkhorn](presink/README.md)

KDA 三段计时使用原参数。T257 的 Aqk 和 T248 的 Akk 各一个元素未通过原数值协议，两端失败位置相同。冻结双精度发布算法参考保持原判，按实际 kernel 的中间精度顺序能重现两个偏差。该独立 CPU 检查与额外完整 CA 成本分别记录。[三段和失败数值](kda/README.md) · [双参考实验](kda/numeric-audit.md)

外部安装在 1680 个不同单次查询中最大 3.532 ms。所有 970DT 结果均为外推。shape-only 的 RMSNorm 和内置 Post、PR 容量过渡区的 Cast/Sigmoid 等仍有超限，融合 KDA V1 仍缺合格计时；整套目标尚未通过。[总建模方法](../../docs/modeling-strategy.md) · [总误差分析](../../docs/error-analysis.md) · [冻结及当前源码](model.json) · [统计快照](evaluation.json)

T259/H2 的新 CA 已完成事前双参考验证。DT 和 PR 的两个辅助输出每端合计 66,304 个元素，均逐位匹配已封存的设备精度参考，原算法口径则各有且仅有一项超差。该组耗时独立于主表，验收为 N/A。[双端逐元素对照](kda/numeric-audit-prospective.json)

普通换形的 Post 也登记了按行重叠的独立检查，仅复用已在队列中的窗口；[65,4,5120] 上的改善属于开发回放。片外缓存假设的四个独立容量窗口沿用原冻结计划、二进制、单槽和预算，可在原长作业仍运行且持续出现足够空闲 CPU 时开始。[换形的前向登记](post/warm-schedule.md) · [容量测试的调度条件](cache-queue/README.md)
