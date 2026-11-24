# 950DT / 950PR 的 MXFP4 候选索引器校准

同为 `Q=1, H=32, D=128, TopK=512`，库中的普通 QLI 与 vLLM-Ascend 在 Flash A5 上调用的融合 QSLI 有不同的输入合同。下表每行使用同一份已发布 QSLI 设备二进制。两次运行仅改变 PA 页表宽度。页表的前 8 列映射到 8 个物理页，多出的 8193 列均填合法页号 0。`seqused_k=1024`；每行 81 个有效八位置候选块，共 648 个键，输入候选容量为 2048 块。

| PA 页表列数 | DT 冻结 / CA (µs) | DT 误差 | PR 冻结 / CA (µs) | PR 误差 | 离线墙钟 DT / PR |
|---:|---:|---:|---:|---:|---:|
| 8 | 6.831 / 11.088 | −38.39% | 6.836 / 11.483 | −40.47% | 74 / 84 s |
| 8201 | 8.072 / 11.352 | −28.89% | 8.077 / 11.225 | −28.05% | 100 / 94 s |

模型预测宽表使两端各增加约 1.24 µs。实测 DT 增加 0.264 µs，PR 减少 0.258 µs。每个 SKU、每种页表仅运行一次。分支切换可在指令层面确认，在完成打分的 4 个 AIV 上，非立即数寻址标量读取 `LD_XD_XN` 合计从 15 增至 271 条，即时偏移寻址 `LD_XD_XN_IMM` 从 576 减至 468 条。`RV_VLDI` 都是 1154 条。标量流水、GM 服务、双发、空泡与 Cube 计算之间可以重叠，额外的 256 条读取对整窗的影响不能直接用单条平均读等待乘算。两张页表上，每个 SKU 都有两个 AIC 各发出一条 `(32,512,128)` 的 `MMAD_MX`。紧表时首个 MMAD 出现在 DT 6.815、PR 7.652 µs；宽表分别为 7.433、7.361 µs。从零点到首发的等待已占整窗过半。这两个短行仅识别了分支与局部关键路径。[窄表冻结](qsli-freeze.json) · [宽表冻结](qsli-wide-table-freeze.json) · [逐端指令与整窗](fused-qsli.json)

**数值与计时资格。** 输入为 E2M1 Q/K、E8M0 分组 scale 和 FP32 头权重，每个 PA 子块按 512 个键字节加 32 个 scale 字节折叠，非候选位置故意置高。每次对 512 个返回的全局位置排序并与 CPU 参考逐项相等，验证 TopK 集合。有并列分数，未要求同分者排序一致。原厂 provider 版本是 `cannbot-arena-net-ops 0.1.0`，对应 `cannbotdsl 0.6.0+g91e7c8f`，AOT 条目 `400228d9…`。启用 require 模式，核验 `installed_hit`、元数据 SHA、完整设备 ELF 在 provider 文件偏移 443904 处逐字匹配、每核 `END` 和 profiler 时长。计时口径为所有物理核第一条指令至最晚退休，错误率使用 `(预测−实测)/实测`。GH 发布的 A5 镜像说明列出版本。[官方 A5 Dockerfile](https://github.com/vllm-project/vllm-ascend/blob/main/Dockerfile.a5) · [核验脚本](fused_qsli_report.py)

**调度与框架限制。** 9.2.1 CA 的驱动桩明确报告 `unsupported task type: AICPU_D`。在两端测试自动 metadata 时均未取得具资格的整窗。可运行的计时从原厂 AOT 融合核开始，排除 AI CPU 的开销。手工给出的两条 LI 边界和一条 FD 边界在总计 1024 个 int32 上与当时同一 wheel 的 `metadata_kernel.run_host` 输出完全一致。后者使用各 SKU 的原始核数。在 msOpProf 下，AOT 发出后，宿主运行库的 profiling range pop 返回 −1。探针只接受这一个明确的异常，保留官方包装器已分配的输出对象，同步流并完整回读；未替换 kernel 或输出数据。因此设备窗、调度和上述 golden 合格，宿主公开接口在 profiler 下的完整成功返回仍未验证。

## 普通 QuantLightningIndexerV2 组件对照

CANN 9.2.1 的 `aclnnGatherV2 × 3 → aclnnQuantLightningIndexerV2 → aclnnGatherV2 → aclnnSort` 对应候选键 compact、带权量化打分、原位置映射与排序。普通 QLI 用一组固定元数据使单个 AIC 打分。每个样本的输入、golden 和 6 个实际 OPP ELF 均核验，物理分片摘要与各段窗口在 JSON 中。每 SKU 三组在同一进程按表中次序执行。Q=4 的四行分别构造独立的打分维度和右对齐可见键集合。

| Q | 源长度 | 紧凑键 | 普通 QLI 冻结 DT / PR (µs) | 普通 QLI CA DT / PR (µs) | 六段设备窗和 DT / PR (µs) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 648 | 8.987 / 7.698 | 11.249 / 10.588 | 24.102 / 23.132 |
| 1 | 2056 | 648 | 8.987 / 7.698 | 5.600 / 5.605 | 14.237 / 14.290 |
| 4 | 2056 | 648 | 11.785 / 10.042 | 6.875 / 6.804 | 16.380 / 16.415 |

Q1、Q4 都收到了 6 条 `MMAD_MX`，其中 5 次 N=128、最后一次 N=8，M 随 Q 从 64 变成 128，K 始终 128。`RV_VMULA` 等归约指令也有独立收据。前两行的打分张量相同，但第二行与第一行进程内运行次序不同，无法将时间差单独归因于总上下文长度。六段是串行的设备窗之和，实际融合 QSLI 的指令、搬运和分片另见上表。每端 18 次 kernel 总计用时约 232 秒。[冻结预测](mx-controls-freeze.json) · [逐形组件窗口](mx-controls.json) · [采集成本与日志摘要](mx-acquisition-cost.json)

## 复现

在同一 CPython 3.12 aarch64 环境加载上述两只官方 wheel、PyTorch 2.14.0 和 torch-npu，并激活 CANN 9.2.1。CA 需要宿主本地 Unix socket。`${PYTHON}` 表示该解释器，`$ARENA_WHEEL` 为同一已发布 wheel，`$RAW` 位于 git 忽略的目录。

```sh
CANNBOTDSL_NATIVE_BINARY_MODE=require CANNBOTDSL_NATIVE_BINARY_REPORT="$RAW/provider.json" \
  msopprof simulator --soc-version=Ascend950DT_9582 --launch-count=1 --dump=false \
  --timeout=10 --output="$RAW/profile" "$PYTHON" calibration/candidate_indexer/fused_qsli_probe.py \
  --static --ratio 1 --table-pages 8201
```

重复双端和两种页表时，按 `fused_qsli_report.py` 中的 `<SKU>-static`、`<SKU>-wide` 目录和 `.log` / `-provider.json` 命名保存，对照使用 `--table-pages 8`。独立编译普通 QLI 探针的方法见本目录 README，改为运行 `candidate-indexer mx-controls`、记录 32 个以内的 dispatch，将每个 SKU 的 `profile/` 与同名 `.log` 分别放在 `$RAW/<SKU>/` 和 `$RAW/`。重放收据的方法如下。

```sh
python -m calibration.candidate_indexer.fused_qsli_report "$RAW_AOT" \
  --wheel "$ARENA_WHEEL" --out /tmp/checked-fused.json
python -m calibration.candidate_indexer.mx_controls_report "$RAW_QLI" \
  --opp "$ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/kernel/ascend950" \
  --out /tmp/checked-ordinary.json
```

## 一整波长上下文 prefill

Q32 留出在 950DT 上使用 B=1、每行 32 头、D=128、逻辑压缩历史 1,048,576、2048 个有效候选块、PA128、8201 列页表。独立 CPU 构造 32 个不同 TopK 集合。第 r 行（从 0 计）只允许位置小于 `1048576 − 32 + r + 1` 的键，未来位置故意置高。16,384 个结果均通过。将原厂 metadata 逻辑在宿主逐指令解释后回放，32 个 AIC 各负责一整行的 32 个 512-key tile，无跨核部分 TopK 合并。该实验覆盖一个满核波；系统里 4096 行的 128 个波之间仍可能受不同的启动、缓存、带宽和发起行为影响。

| 全窗项目 | 950DT |
|---|---:|
| 所有物理核首发至最后退休 | **84.792121 µs**（139,907 cycles） |
| 原资源预测 / 误差 | 52.020856 µs / −38.65% |
| 从单团队指令间隔得到的冻结预测 / 误差 | 75.855501 µs / −10.54% |
| MMAD_MX 总数 / 每行 | 1024 / 32 |
| 实际同时有 MMAD 在执行的 AIC 峰值 | 20 |
| 同核相邻 MMAD 发起间隔的中位数 | 2863.5 cycles |
| 首 MMAD 发起 / 最后 MMAD 退休 | 19,732 / 133,760 cycles |
| 最后退休的物理核 | core9.veccore0 |
| 离线采集、导出墙钟 / 本地原始目录占用 | 758.10 s / 约 4.7 GiB |

20 是将同一核上重叠的 MMAD 指令区间取并集，再在全部 AIC 的统一时间轴上逐时刻统计得到。它与分配到任务的 32 个 AIC 衡量不同的活动。原调度日志标出 TASK_DONE 后，宿主曾在 msOpProf 的 `HandleDumpLogAfterLaunch` 等待数据导出；设备微秒数来自最终的全物理 trace。[原冻结](qsli-1mi-wave-freeze.json) · [间隔版冻结](qsli-interval-wave-prediction.json) · [逐行归属与核验](prefill-wave.json)

使用同一环境与 `CANNBOTDSL_NATIVE_BINARY_MODE=require`，把单行命令的 Python 程序改为 `prefill_wave_probe.py`，将 watchdog 设置为 15 分钟。复核使用 `python -m calibration.candidate_indexer.prefill_wave_report "$RAW" --wheel "$ARENA_WHEEL" --out /tmp/wave.json`。该脚本校验全部 96 个物理核的退休时刻，核对每行 MMAD 的数目、精度、tile 尺寸和并发，并逐项核对两个冻结记录。当窄表与宽表的回放需要不同的预算时，只需单独重放对应的证据文件。
