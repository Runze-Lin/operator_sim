# 整批 prefill 的 Cast 复核

当前结果、角色和成本见[分 shape 报告](../../../results/v0.20/large/report.md)，该目录属于离线验证。系统仿真器直接调用 `operator_model.estimate`。

- `registration.json`、`native-freeze.json` 保留首次采集前的 v0.19 基线。第二次冻结 `cast-v20-freeze.json` 标明当时已有数据、并行采集和新的独立样例。`pro-freeze.json` 在 Pro 的单独采集开始前同时登记两版。
- 已有 CANN 9.2.1 工具链时，`bash development/validation/large20/build.sh` 编译流式验证 host。它只分块生成、搬入及比对数据，每个完整二维输入调用一次 `aclnnCast`。DT/PR 正式组用两个 worker，各例最多 3 小时和 24 GiB 日志。
- 正式组使用 `python3 development/validation/large20/campaign.py collect`，Pro 增补使用 `python3 development/validation/large20/pro.py collect`。同一目录已有结果会跳过。中断后先保全尚无成本记录的原始目录再发起重试。
- DT 的 4 小时全宽原生 CA 使用 `python3 development/validation/large20/native.py 0`，两个正对照使用位置 1、2。其全核 wrapper 时刻和正式首末指令的统计口径不同。
- `python3 development/validation/large20/dram.py` 在正式采集期间保全 DT 各通道的同一日志文件（硬链接），等待正式 CA 资格确认后再汇总。`toolchains/envs/cann-9.2.1/bin/python development/validation/large20/analysis.py --watch` 追踪完成结果，重绘流水和伸缩图，读已有产物即可，去掉 `--watch` 可生成单次快照。

交互采集管理器退出时，可用 `python3 development/validation/large20/supervise.py start` 在宿主恢复已有 GNU time 与模拟器进程，再按原预算接续队列。守护锁和每例锁防止相同采集重复。原进程的消耗由完成标记、run.log 创建时刻、cpu.txt 结束时刻、时间状态和原始 trace 复核；无法恢复原管理器退出前的磁盘峰值时，报告标 ≥。单卡大 case 的首末窗口、逐轮读写与资源模型的图在有新的合格结果时自动刷新。
