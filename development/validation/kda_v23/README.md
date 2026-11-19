# FLA V2 小样本 CA 采集

该目录记录默认 Kimi 分派阈值以下强制使用公开三段 V2 接口的开发与冻结验收。目标平台是 950DT 和 950PR；所有运行采用同一个 FLA 26.9.1 A5/aarch64 发布 wheel、CANN 9.2.1 和独立 conda 环境。实际 host 用 dladdr 校验两个公开 API 的共享库身份。Linux 真实 CA 需要本机通信。上游资产准备见[来源和复跑步骤](../chunk_recovery/fla2691/README.md)。

## 从已有采集复核

在仓库根目录执行下列命令。复核不会启动设备仿真。

```bash
python development/validation/kda_v23/replay.py
python development/validation/kda_v23/report.py
python development/validation/kda_v23/mechanisms.py
```

[原合同](../../../results/kda-v23/development-register.json)固定了输入、设备分配、文件哈希（含驱动、ELF 和芯片配置）、判据及预算。原 DT T2/H1 控制复用先前完整采集。其他输入每个芯片每个形状运行一次。运行时每个活动核的分片、合并 Chrome B/E 与 X 事件、官方 duration 表须一致，且必须可辨实 END 和发布 ELF。

[精度修正合同](../../../results/kda-v23/precision-development-register.json)保留原数据、两端 CA 配置、阈值与原数学参考，另将两个中间辅助矩阵与发布的 Prepare 参考按 BF16 分阶段对齐。数值测试的实际输入不依赖 CPU 参考。attention 与最后状态每个元素继续对数学递推。逐形状两份参考的最大绝对/相对误差、数组指纹、每核时钟、下一步冻结验证见[机制与误差](../../../results/kda-v23/mechanisms.md)。没有四张数值回执或不完整的窗不进入统计。

新检出需要恢复旧轮次忽略的大体积原始 trace、动态库、生成的测试宿主，并验证相应注册清单中的哈希。上游独立算法参考来自[固定提交的 CPU 实现](https://github.com/flashserve/flash-linear-attention-npu/blob/5cc699069c7f91a35aabd583032c8720a0b626fa/tests/atk/chunk_kda_fwd_prepare/executor_chunk_kda_fwd_prepare.py)。本地副本置于 `results/kda-v23/raw/vendor-prepare-oracle.py`，注册时与 Git blob 比较。CPU 端 NumPy/SciPy 位于本地 CANN conda 环境。

## 串行采集与恢复

只在空白采集区注册。`series.py register`、`precision_cases.py register` 都拒绝覆盖原登记。历史结果目录中再次调用 `run` 采用现有回执，遇到有日志但没有 cost 的中断尝试时保留原状态。每例预算 900 秒、3 GiB；原队列总预算 5,400 秒。所有 V2 CA 使用同一把文件锁。

```bash
python development/validation/kda_v23/series.py register
python development/validation/kda_v23/series.py run
python development/validation/kda_v23/precision_cases.py register development
python development/validation/kda_v23/sequence.py
```

`sequence.py` 等待原队列释放槽位，先补 T65/H1 的 DT/PR 精度对照。两端都通过后，它验证资源模型文件、参数、预测和精度协议哈希，再登记 T3/H1 与 T5/H2 同形双端独立留出。运行中查看 `results/kda-v23/status.json` 以及 `precision-{development,holdout}-state.json`。采集清单、CA 输出、资格和冻结预测分别留档。原极严数学参考失败退出曾导致 stdout 留在 glibc 缓冲；可选的 `numeric_log.py` 仅匹配本用户、相同宿主、相同登记窗口及全部输出计数的 core，保留四行原始输出和指纹。原失败状态照常保留。

补充八形在首批两形之后单独登记。选择时刻、模型文件哈希、两端资格与扩展队列分别保留，适合在后台继续运行。八形累计 CA 预算 5,400 秒，每项 900 秒、3 GiB，独立形状数按 T/H 算，同形两芯片分别统计。实测与 970DT 外推可查[机制报告](../../../results/kda-v23/mechanisms.md)。

```bash
python development/validation/kda_v23/supplement.py choose
python development/validation/kda_v23/supplement.py run
python development/validation/kda_v23/watch_report.py
```

已存在登记的当前工作区只需 `run`，每项保留单独回执。报告所需的真实 Aqk、Akk 两个小张量保留为 gzip 压缩 BF16。即使原始大 trace 被归档，仍可离线用 CPU 重算全部元素。执行 `OPENBLAS_NUM_THREADS=1 toolchains/envs/cann-9.2.1/bin/python development/validation/kda_v23/check_arrays.py` 可重新生成 [CPU 核对记录](../../../results/kda-v23/actual-array-checks.json)。
