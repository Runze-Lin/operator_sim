# v0.24 · 逐块比较模型与实测

| 算子与输入 | 变化 | 同集结果 |
|---|---|---|
| Pro indexer Cast，B8×128k | 950DT 实际运行 50 轮 UB 转换。图按相同 GM 地址与 UB 槽对齐，再放大最后两个块。 | 冻结模型 99.918 µs，实测 105.374 µs，低估 5.18%。末块读启动已经落后 4.661 µs。970DT 外推 42.536 µs。|
| KDA FLA V2 | 一条 head 链的工作组按 W/U 槽释放预取下一块，H 读保留跨块写后读依赖。两种芯片、三种配置以相同公式计价。 | 两个新 shape、每端完整三段。状态阶段 WAPE 从 DT 24.92% / PR 20.49% 降到 7.65% / 3.87%，整次调用降到 2.24% / 1.34%。另八种新形已冻结，由后台队列串行补齐。|
| RMSNorm | 系统仿真从中途接入时，按可见同族调用推导已驻留 AIV 数的下界，约束下一次扩核的可行时长。逐行写回图揭示统计量与向量输出的退休相关性。 | 部分前序的十种新形 WAPE 由 DT 15.78% / PR 15.09% 降至 6.16% / 7.11%，新最大误差为 29.15% / 25.20%。完全无历史的独立误差仍超限。[逐形和双端速度比](rms-prefix/README.md) |
| 内置 mHC Post | 冻结的对照模型将 token 首发和同形立即重复的 channel 路径按启动、首次读、算术及退休分别计价。 | 分别记录首发、扩核、相邻重复、隔形回访。诊断图按同一 token 的四条输出流对齐。默认公开入口维持原有估计。[逐形新旧比较及适用范围](post/report.md) |
| 默认融合 KDA V1 | 用真实 Kimi cu_seqlens 与 dense 分支验证实际入口、物理指令和四组输出。 | DT/PR T2/H1 四组数值相同且通过，42 个相同错误地址、167/155 次重复 SET 阻断时延校准。DT T64/H2 的另一条 dense 分支也有重复 SET，原登记金标有 Aqk/Akk 超限。[实际异常、操作数回放与指纹](kda-v1/README.md) |

Pro 128k 采集花 25.30 分钟。首轮两种新 KDA 形状双端合计 10.98 分钟。RMSNorm 与 Post 的采集时间在各自回执按进程记录。线上 Python wheel 在 1,716 个不同常规输入上实测 P95 0.688 ms、最大 6.012 ms，另 39 次只给出部分历史的调用最大 11.407 ms。两种本机 Python 的 243 个回归测试通过。[安装版调用开销](wheel-check.json)

[Pro 最后两块放大图](pro-128ki-tail.png) · [KDA 两块的独立对照](kda-prefetch/dt-80-independent.png) · [RMSNorm 两笔输出写回](rms-prefix/pr-rms-output-rows.png) · [内置 Post 首发和热重复](post/post-ordered-pipeline.png)

[全套方法](../../docs/modeling-strategy.md) · [全量误差摘要](../../docs/error-analysis.md) · [本版逐形数据与 970 外推](shape-errors.md) · [KDA 新形累计进度](kda-prefetch/supplement/report.md) · [Pro 长作业登记及实测](../v0.23/pro-prefill/README.md)
