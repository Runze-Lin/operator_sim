# 冻结 Sinkhorn 的后续独立验证

从仓库根目录运行。两类候选的预测沿用原冻结。公共默认路由需要十个不同新形、每端 WAPE≤20%、最大≤40%，以及同集改善。独立 Sinkhorn 给定宿主记录的同家族高水位 H 才能区分冷、扩大活动集合和热态。保留 shape-only 预测独立比较。

| 文件 | 用途 |
|---|---|
| `results/v0.26/sink/register.json` | 独立 FP32 4×4 输入、每次启动真实 H、I、两端和 970DT 预测 |
| `results/v0.27/fusion/register.json` | 融合 BF16 [T,4,4096] 每个冻结候选、原对照和 970DT 预测、预算与散列 |
| `results/v0.27/fusion/scoring-resume.json` | 已结束原 T72 PR 经同名模块装载问题丢失评分，保存其退出状态和未经修改的解析器散列 |
| `results/v0.27/fusion/assessment.json` | 版本计分截点；使用过的各个 raw、回执、模型、单次测量、来源与成对比值 |
| `results/v0.27/fusion/{state,assessment-live}.json` | 实时排队状态与滚动计分，终态前可能部分缺失 |

复核计分和三张图。绘图需要项目的 matplotlib 环境，所有输入来自已有的真实采集。

```bash
python3 development/validation/fusion27/assessment.py
toolchains/envs/cann-9.2.1/bin/python development/validation/fusion27/comparison_figure.py
toolchains/envs/cann-9.2.1/bin/python development/validation/fusion27/fused_figure.py fused-t72-pr
```

正常运行中，独立 Sinkhorn 由原 `sink26` 的单进程跑完各自的同族序列。融合核在 `presink27-score` 中使用原 `results/v0.26/presink/raw/exclusive.lock` 的非阻塞独占锁。已有正式回执优先；已经有成本文件的 raw 直接走原流式解析器、核对所有活动核 EOF 及 3 个全张量输出。已有 raw 而未完结的条目只报告中断。若整个会话退出，先核对上次状态和实际任务，再在仓库根目录用下列命令恢复同一队列。

```bash
OPENBLAS_NUM_THREADS=1 toolchains/envs/cann-9.2.1/bin/python development/validation/fusion27/resume.py run
```

计分进程 `fusion27-scorecard` 仅读小型 JSON；恢复它可执行 `python3 development/validation/fusion27/assessment.py --watch`。融合队列选择空闲 CPU、nice 19 和 SCHED_IDLE；每个 CA 的 2700 秒 / 10 GiB 原上限保留。新的登记最多动用 9 小时采集墙钟和 80 GiB 目录空间，旧长作业以及 FA 保持优先。原 T72 PR 用同一 PID、71 个工作线程和同一 ELF 在现场调整亲和性，逐线程计数分别保留在 `native-before.json`、`native-after.json` 中。

## 单槽后备几何和物理退出

T127 DT 的官方最后核表与原 epilogue 提取器的偏移差为 3.001351 周期，保留原失败。逐核展开原 trace 发现唯一额外退役是 `core15.veccore1` 上的 `MOV_XD_IMM`（END 前 8 字节），在 END 描述符后 3 周期退出。所有真实描述符的全窗偏移散布为 0.003002 周期。独立主表继续保留这次不合格，事后诊断见 [物理指令、PC 与散列](../../../results/v0.27/fusion/fused-t127-dt-physical.json)。

原门禁仅在最早的 T72/T113 对失败时释放 T91，这两对均成功。T127 是扩充主组内的失败。已在 T91 的两端均无 raw 或回执时，单独登记释放此前封存的两项预算、预测及相同 gate。完成的原主队列无须再次启动。[后备启动回执](../../../results/v0.27/fusion/reserve-activation.json)

```bash
OPENBLAS_NUM_THREADS=1 toolchains/envs/cann-9.2.1/bin/python development/validation/fusion27/reserve.py run
python3 development/validation/fusion27/assessment.py
python3 development/validation/fusion27/public_check.py
```
