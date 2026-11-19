# KDA 十种形状的前向验证

第一批 T80/H1、T145/H2 的四份 DT/PR 回执已经合格。第二批增加 T/H 为 63/4、64/3、65/4、127/1、128/2、130/3、193/2、257/1 的八种形状，每种各采两个 SKU。T=257 跨五个 64 行块，63/64/65 与 127/128/130 覆盖分块变化。所有形状均为 B=1，Hq=Hv=H，K=V=128。两批合计目标每芯片十种形状。第二批的选择使用了第一批已知结果，两批使用同一冻结模型、同一数值参考和同一 CA 整窗口径。

第二批冻结 SHA-256 为 `6a23f270cdb9bb03caa7057d506a8c4731c87b6dc53c13562e64e175e2086b45`。其回执记录每个目标芯片的候选、v0.23 对照、970DT 外推、分阶段时间和槽位时标。运行程序验证旧资源源文件的压缩归档、正式算子代码段、原宿主及四项参考的散列。[冻结登记](register.json) · [实时状态](state.json) · [累计合格、待测及失败](report.md)

## 资源安排

- 会话名 `kda-prefetch-v024`，单一文件锁，按形状依次执行 DT 然后 PR。等待先前登记的 Pro B8/128Ki PR 开始，等待 Post 短作业结束，并等待 RMSNorm 最后一个 PR 回执。Kimi 长跑由原作业管理。
- 每次采集前检查可用磁盘 ≥64 GiB、可用内存 ≥16 GiB、一分钟负载/在线 CPU 数 ≤0.92。会话以 nice +10 执行。每端上限 720 秒、3 GiB 原始目录，全部 16 端合计 9,600 秒 CA 进程时间；排队时间另计。
- 每端通过数值检查、三段正式 ELF、全活跃核、完整物理结束和对应设备窗才计分。首次完成的两种形状独立列于原报告，累计发布表逐端三段给出新旧预测与 CA。数值或采集失败单列。若进程中断且仅留下原始日志，恢复入口保留该次尝试并报出停止原因。

## 状态与恢复

以下命令在仓库目录运行。作业正常时读取状态即可。每收到一个端的回执，作业自动重建累计报告。手动重新发布不需要 CA 或 CANN 环境。

```bash
cat results/v0.24/kda-prefetch/supplement/state.json
python3 development/validation/kda_prefetch24/supplement_job.py publish
```

在给定会话已经退出时，使用相同的 tmux 启动命令恢复。单一文件锁防止重复，当同一组已有 cost 而尚无回执时，直接进入资格核查。作业运行日志位于本机忽略目录 `results/v0.24/kda-prefetch/raw/supplement-worker.log`。

```bash
tmux new-session -d -s kda-prefetch-v024 -c /home/l00925865/operator_sim \
  'exec nice -n 10 python3 -u development/validation/kda_prefetch24/supplement_job.py run >>results/v0.24/kda-prefetch/raw/supplement-worker.log 2>&1'
```

恢复入口预先装载当前目录的报告模块。首次挂队时发现旧版 RMS 的同名模块可能遮蔽 KDA 发布器，在第二批尚有零个原始目录和零个回执时已切换入口。冻结预测、宿主、采集程序、数值检查、资源顺序和预算的散列均保留。
