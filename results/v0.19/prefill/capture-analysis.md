# 长 prefill 的 CA 成本和可复用证据

单卡 batch=8、每请求 32768 个输入 token，在一次无 chunk 的 kernel 中是 `[262144,5120]`，共 **1,342,177,280** 个元素。已有采集用 `msopprof simulator --dump=false --launch-count=1`，保留全核指令与真实设备时间。关闭额外文本 dump 后，完整导出的聚合 trace、逐核 trace 和可视化文件仍会占用磁盘。[实验合同和预算](registration.json) · [冻结的逐芯片外推](forecasts.json)

| 实际单次 shape 和芯片 | CA 全核时间 | 活动 AIV | 实际 CA 墙钟 | 磁盘峰值 / 留存 | 完整指令事件 |
|---|---:|---:|---:|---:|---:|
| RMSNorm [2048,5120] DT | 14.337 µs | 64 | 982.7 s | 10.76 / 5.58 GiB | 4,899,456 |
| Quant [2048,5120] DT | 11.364 µs | 64 | 445.8 s | 4.29 / 2.36 GiB | 2,065,790 |
| RMSNorm [2048,5120] PR 补采 | 17.820 µs | 56 | 891.9 s | ≥9.56 / 8.17 GiB | 4,891,596 |
| Quant [2048,5120] PR | 18.504 µs | 56 | 495.1 s | 3.86 / 2.34 GiB | 2,045,196 |

四次均完成整形状数值校验与厂商逐核时间表核验。例如 RMSNorm 聚合 `trace.json` 和 `visualize_data.bin` 单个文件分别为 2.029 GB、2.031 GB，逐核 JSON 还保留同一批事件。墙钟含初始化、模拟、数值检查与导出；没有各阶段独立的 host profiler，不能把 982.7 秒全部归为导出。[同集采集证据](../sku/records.json)

## 无 chunk 的现场证据

DT RMSNorm 的 900.28 秒墙钟上限先触发，峰值 166,881,264 B (159.15 MiB)，远低于已登记的 3072 MiB 磁盘上限；受控终止的 CPU 秒数未取得。已看到指定 SKU、RmsNorm 库 kernel 和 64 个不同 AIV block 的开始。最后一条 thread 32 heartbeat 为 100000。没有 block_end、输出 golden、逐核 duration 表或可解析的内层/外层 `simulator/trace.json`，32 个 `core*_summary_log` 均为 0 B。因此可验证启动与分核；单次完整 CA **N/A**。不能由此分摊设备算术、内存拥塞、宿主负载各自占了多少墙钟。[RMSNorm 终态](rmsnorm_262144_5120.json)

DT Quant 同样先触发墙钟上限，900.53 秒、峰值和留存均为 257,812,101 B（245.87 MiB）。有 64 个不同 AIV block_start、0 个 block_end，thread 32 最后 heartbeat 为 150000。没有完整指令文件、厂商逐核时间表或正确性结果；全核 CA **N/A**。没有证据表明单纯取消文本导出或压缩 JSON 就能让这两个大样例在相同预算内完成，因为中止时甚至还没有可归约的逐核文件。[Quant 终态](quant_262144_5120.json)

两条大 shape 的启动二进制 SHA256 与各自 2048 行的 DT 校准完全一致（RMSNorm `0926fe47…`，Quant `17d48c74…`）。这能证明同一份 kernel 代码进入仿真器；不同的 tiling 入参、每核轮数、访问交错和末块仍会改变时间。

## 能证明多少轮

两个完整的 DT chunk 中每核分到 32 行。以同一代表核 `core29.veccore1` 的 MTE3 `MOV_SRC_TO_DST_ALIGNv2` 指令复核：

- RMSNorm 的 BF16 行长 10,240 B。六条写回各含 5 行，最后一条 2 行，另有相应的 6×20 B + 8 B 的 FP32 rstd。由此可直接读出 **6×5+2 行、7 轮**。
- Quant 的输出行长 5,120 B。四条写回各含 7 行、最后一条 4 行，还伴随同样行数、每行 160 B 的 scale 搬运。可读出 **4×7+4 行、5 轮**。
- 同一个 2048 行 RMSNorm 调到 PR，内层导出的 core0.veccore0 写回 7×5+2=37 行、8 轮，core29.veccore1 写回 2×5+3=13 行、3 轮。尾核的实际分配不同，单核切片不能无条件复制到其余核。
- 无 chunk 时冻结模型给出每核 4096 行，RMSNorm **820 轮**，Quant **586 轮**。这是模型计划。单次无 chunk 的采集未提供这些轮次的完整核内指令序列，已经完成的轮数不能从 `Thread 32 ... Cycle Count` 或内存通道的累计字节中反推。内存统计没有将请求一一绑定到核、tile 和算子输出。

### 有 JSON 文件仍可能缺写回

PR RMSNorm [2048,5120] 的 `stars_log` 有 TASK_DONE，两组 golden 都通过。1500.81 s 到达墙钟上限，留存 5,983,942,096 B；厂商内部 JSON 合法，56 核的汇总 CSV 中只有 52 核列有 END_LABEL。进一步检查 core12.veccore0，MTE3 确实有 7 次 5 行 BF16 写和末组 8 B rstd，但少了最后的 2 行 BF16 payload。该核最后可见指令为 VF（PC 0x9000d12008），没有对应尾写（0x9000d12128）、END 或 DCI。相同输入 PR core0、正式完成的 DT 同核，以及 PR [408,4416] 同核均可见这一段的尾写及终止。这个缺陷有实际的流水证据。另一组**正常结束**的 DT Quant chunk 中，core25.veccore0/1 也缺少合成 END_LABEL，但两核都保留了全部 32 行结果和 scale 写回，以及同 kernel 的 END、末尾 DCI；厂商完整导出的设备窗口仍可核对。单独缺一个合成标签与缺实际输出写的证据等级不同。

缺尾的 core12 的可见窗口为 17.327 µs，厂商表也打印 17.33 µs。两者吻合说明表格可以与已经截断的流水一致。任务完成标志和正确输出证明算子运行结束，不能补齐缺少的每核结束时刻。首次尝试保留数值资格，严格“全片首条至末条”耗时 **N/A**。单次同输入补采沿用原 shape、宿主 SHA 和冻结预测，开启指令导出并使用被动等待，预算 2400 s、16 GiB。补采 **17.820 µs**，56 核实际首末与完整导出逐核一致。两组 golden 覆盖全部 10,485,760 个 BF16 值和 2048 个 rstd。56 核输出的 2048 行全部被 MTE3 写回，可核对 55 个核各写 37 行，尾核写 13 行。完整官方采集的墙钟 891.86 s、记录 CPU 24,932.45 s；30 秒采样得到的 9.56 GiB 峰值为下界。已有证据不能把此次墙钟改善分别归因到被动等待、导出选项或宿主并发负载。[补采登记](chunk-retry-registration.json) · [补采成本与完整性](chunk-retry-outcome.json) · [物理终止与行数](chunk-retry-tail-witness.json)。恢复程序继续接受具备完整尾部或其他独立终止时钟证据的样本。[全核导出恢复和对照](../sku/window-recovery.json)

## 一个核不能代替全片

设备延迟按 `T=maxᵢ(endᵢ)−minᵢ(startᵢ)` 计，`startᵢ`、`endᵢ` 是同一时钟基准下第 i 个核的首末指令时间。相同输入行数与指令次数不保证所有核同时开始或完成。

| DT [2048,5120] | core0.veccore0 时长 | 最慢单核时长 | 首核至末核开始相差 | 实测全片时间 |
|---|---:|---:|---:|---:|
| RMSNorm | 14.118 µs | 14.118 µs | 0.599 µs | 14.337 µs |
| Quant | 9.919 µs | 11.178 µs | 0.435 µs | 11.364 µs |

RMSNorm 各核都有 4160 次 RV_VCADD、84 次 RV_SEND，逐核实际时间仍跨 **12.892–14.118 µs**。乘以轮数会把每个短调用的启动和尾排空成本重复累加；把一个核复制到其他核还会丢失到达偏移与共享资源的等待。两种算法都不能给无 chunk shape 提供独立 CA 标签。日志 `block_start/end` 与指令窗也使用不同边界，DT RMSNorm 所有 `block_start` 的记录仅相差一个日志 tick，而真实首指令相差约 0.60 µs。

## 降低采集和回放开销

已合格记录中的逐核端点可以直接复算全核时间。下面只读约 **9.8 KB** 的现有收据，在这两个 shape 上与原全量 trace 结果一致。保留 binary/trace 摘要、活动核心集合、逐核端点、计数与两组 golden；需要分析某条流水时再读原始指令。这降低后续回放成本，首次 CA 仍承担设备仿真和当前 msopprof 导出的开销。

```python
import json
from pathlib import Path

for case in ('rmsnorm_2048_5120_Ascend950DT_9582', 'quant_2048_5120_Ascend950DT_9582'):
    rec = json.loads((Path('runs/sku19') / case / 'qualified.json').read_text())
    stage, = rec['stages']
    windows = list(stage['windows'].values())
    assert len(windows) == stage['active_aiv'] == 64
    assert abs(max(end for _, end in windows) - min(start for start, _ in windows) - rec['ca_us']) < 1e-8
```

若后续采集器可从官方每条退休指令的回调直接读取起止时间，可对所有核心流式执行同样的 min/max 归约，额外内存只随核心数增长。另存逐核首末指令标识、事件总数、每类指令计数和少量被选核的完整流水即可兼顾诊断。这个方案仍需在相同 SKU、binary、数据上与正常 msopprof 全量导出做同集对比，还要验证末尾回调和数值检查都已完成。现有残存目录不能补回尚未写出的指令。若只是 profiler 外层导出超限但内层完整 JSON 已生成，可复用已验证的全核窗口恢复步骤，保留导出失败的统计。[恢复门槛和正常导出对照](../workloads/router-window-recovery.json)

### 使用已有厂商原始指令的低成本端点法

在这一次 PR 补采中，每个 AIV 的 `*.instr_popped_log.dump` 首行提供 MOV_XD_SPR 发起 tick，`*.instr_log.dump` 的最后一行提供 END_LABEL 退休 tick。分别从每个已关闭文件读取首行、末端 4096 B，核对 PC、核名和终止事件，跨 56 核取最早 7144、最晚 36547，再除以 1650 cycles/µs，结果是 **17.820000000 µs**。官方全量 JSON 还原 17.820000651 µs，同核逐个检查全部 **112 个端点**，最大绝对差为 0.00000110 µs；每核还与厂商显示到 0.01 µs 的 duration 表一致。两组 golden、完整核集合、相同的配置与 binary、真实尾写都是这个对照的前提。[原始端点和行摘要](chunk-retry-raw-bounds.json) · [全核逐端点差值](chunk-retry-bound-comparison.json)

本次端点读取总量 **244,552 B**、耗时 **10.76 ms**。完整聚合 JSON 为 2,025,474,347 B。端点检查可用来减少已完成同合同 trace 的解码和读盘；本实验并未实现“只写 244 KB”的采集器，底层依然写出了全部指令。这里的正对照限于此路由及 PR 补采。对于无 END_LABEL 的路径需按真实终止指令及其 PC 单独对照，对于缺真实尾写或缺退休时刻的路径仍保留 N/A。

本机已验证的 msopprof 方式有全指令导出和关闭额外指令 dump 两种。`--dump=false` 仍会生成全量可视化 trace，还拿不到上述原始首末行；`--dump=true` 提供这些行，但它会把整段流水写入文件。本次没有验证一种在不修改 CANN、低磁盘预算下只导出全部核心首末并可靠取得 [262144,5120] 实际整批的 CA 时间的厂商配置。即便额外离线读取趋近于零，CA 仍需把该输入的实际工作量完整模拟到终止（模型预计 820/586 轮），不能将 heartbeat 或中间读盘时刻替代全片完成时刻。冻结的两条无 chunk 实验保留为外推，本轮未重复发起。
