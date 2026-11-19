# RMSNorm 同一 ACL stream 的逐次设备窗复核

这次复核只读取已经结束的采集。宿主在循环外创建一个 ACL stream，十次调用逐次同步并核对 Y 和 RSTD。每个合格时延都来自本次调用全部活跃 AIV 的最早指令到最晚端点。一次命令的真实退出码与单个 launch 的可观测性分别列出。

| SKU | 单进程退出 | 原始 profiler 回执 | 实际墙钟 s | 合格 / 注册 | 数值检查通过组 |
|---|---:|---|---:|---:|---:|
| Ascend950DT_9582 | -9 | raw cap | 600.9 | 9/10 | 20 |
| Ascend950PR_957b | -9 | raw cap | 428.7 | 9/10 | 20 |

### 按冻结顺序复核的设备时延

| 索引 | `[token, channel]` | 950DT 全核窗 µs | 950DT 实际 AIV | 950PR 全核窗 µs | 950PR 实际 AIV |
|---:|---|---:|---:|---:|---:|
| 0 | [1, 7040] | 3.7964 | 1 | 3.8764 | 1 |
| 1 | [2, 6688] | 4.6818 | 2 | 4.6182 | 2 |
| 2 | [3, 5440] | 2.9873 | 3 | 2.8539 | 3 |
| 3 | [4, 4800] | 2.5897 | 4 | 2.5109 | 4 |
| 4 | [16, 1776] | 3.9406 | 16 | 4.0836 | 16 |
| 5 | [34, 1904] | 4.9152 | 34 | 5.0436 | 34 |
| 6 | [64, 1984] | N/A | 64 | 3.3091 | 32 |
| 7 | [65, 1808] | 3.2388 | 33 | N/A | 33 |
| 8 | [135, 2992] | 3.4430 | 45 | 4.0618 | 45 |
| 9 | [504, 5056] | 5.6212 | 63 | 7.9758 | 56 |

### 时延、顺序和完整性的证据

1. 冻结文件、宿主 `inputs.txt`、实际 `case=`、`BATCH_BEGIN/END`、同段两个数值检查和单个库名按序一一对应。核实 `runtime_soc`、物理内存模拟器配置、同一 task 的 `TASK_BEGIN/DONE`、全批 `SMOKE_PASS` 和 profiler 的设备任务成功标志。CA 内部 `full_task_id` 对应 launch 序号；该内部日志的整数 `stream_id` 与宿主 ACL 流句柄属于不同接口。
2. 使用权威的 `OPPROF_*/device0/RmsNorm_normal_all_bf16_high_performance_5000/<索引>/`。其中索引 0 的设备文件保存完整，即使 profiler 未将它复制到 `device0` 之外。每一行均验证捕获 ELF `.text` 与已安装 CANN 9.2.1 ops-nn 同名二进制相同，符号表给出设备入口 `.text+0x3e00`。
3. 十份官方逐核 duration 表须与各 launch 的逐核 JSON 名单完全对应。所有 JSON 都解析到文件末。每个活跃 AIV 都必须包含入口 PC 的 `MOV_XD_SPR`，ELF `.text+0x51d0` 的实际 `END` 和末端 `END_LABEL`。逐核起止时长与官方两位小数 duration 表逐行比较（绝对差不超过 0.00501 µs）。聚合 trace 和所有核切片还逐核比较首末、真实 END、指令数和全部非同步指令 `(PC, opcode, start_cycle, end_cycle)` 的可重算多重集指纹。
4. 额外用设备输出验证是哪一个 shape。`.text+0x4f38` 写 FP32 RSTD，`.text+0x5128` 写 BF16 Y。每个 Y 描述符的 burst 长度必须等于 `2×channel`，所有核累计 Y 行数为 `token`，Y 写入字节数为 `2×token×channel`，RSTD 为 `4×token`。这些观测独立于保存目录编号及预测的核数。
5. 与已有 CA 报告一致，`stage.ca_us=(全核最晚 END_LABEL−全核最早设备指令)/1650`，1650 的单位为周期/µs。`physical_ca_us` 同时给出去掉合成标签后的真实指令窗，`cores` 保留每一核的官方 duration、首末 cycle、真实 END 位置和切片哈希。多核存在到达间隔，聚合时不能简单取最长的单核 duration。

**Ascend950DT_9582 索引 6，[64, 1984]。** missing ELF entry/physical END/terminal label on core27.veccore1。官方表列出 64 个核。core27.veccore1 官方时长为 3.95 µs，核切片最后观察到 cycle 65412。该核的切片和合并文件分别记录了 0、0 条 `END`。保留两项数值检查，设备时间记 N/A。

**Ascend950PR_957b 索引 7，[65, 1808]。** missing ELF entry/physical END/terminal label on core7.veccore0。官方表列出 33 个核。core7.veccore0 官方时长为 3.39 µs，核切片最后观察到 cycle 69162。该核的切片和合并文件分别记录了 0、0 条 `END`。保留两项数值检查，设备时间记 N/A。

[Ascend950DT_9582 逐 launch 核对及哈希](verified-ordered-Ascend950DT_9582.json) · [原始回执](ordered-Ascend950DT_9582.json)
[Ascend950PR_957b 逐 launch 核对及哈希](verified-ordered-Ascend950PR_957b.json) · [原始回执](ordered-Ascend950PR_957b.json)

```bash
python development/validation/rms_ordered_verify/verify.py Ascend950DT_9582 Ascend950PR_957b
python development/validation/rms_ordered_verify/report.py
```
