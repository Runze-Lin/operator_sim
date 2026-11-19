# Chunk KDA 的计时资格

| 路线 | 950DT 合格耗时 | 950PR 合格耗时 | WAPE / 最大误差 | 970DT |
|---|---:|---:|---|---|
| CANN 9.2.1 ops-transformer `aclnnChunkKdaFwd` | 0 | 0 | N/A | 工作量可计算，设备时间 N/A |

固定长度、两头、FP32 gate 的历史样例在 DT 和 PR 报出相同的三个 SET_FLAG PC。PR 保留了异常发生前的退休指令。逐事件核对可见 Cube→MTE1 的 0、1、2 号事件在相邻的两次 SET 之间没有匹配的 WAIT。DT 主 kernel 数据库的已提交指令行数为 0。辅助转置的指令窗口、主 kernel 的部分指令以及 raw gate 的工作量均不能作为完整设备耗时。[原始指令审计](pr-flag-audit.json)

本轮在 dense 调用的对照上增加 VK 状态、规范顺序的 cu_seqlens/chunk_indices、dtBias，保留 FP32 gate 和 −0.03125 的对照值。第一次在沙箱内运行时 profiler 的 Unix socket 失败，运行时回报 9589。随后在可使用 IPC 的主机环境复采，标签与实际 SKU 都匹配。两端 msopprof 达到 6 分钟超时并终止宿主，工具自行结束时返回 0，两端均缺少宿主数值通过标志。DT、PR 墙钟分别为 369.7 和 368.6 秒，均只记录到第一个转置。这四次尝试均无合格设备时间。[逐次回执](probe-summary.json)

进一步的 Kimi-K3 合同使用 12 个本地头，raw gate 与隐藏投影的 BF16 一致，beta 为 FP32，gate 下界取模型配置中的 −5。注册输入为 [1,64,12,12,128,128]。PR 使用 msopprof simulator，DT 使用 npusim compact 和一个仿真工作线程。输入和参考输出在 CPU 生成，宿主调用安装包的公开 API。检查项包括注意力输出、两份三角输出、VK 最终状态和初始状态保持；其中注意力输出与最终状态使用独立逐 token 递推参考。注册墙钟上限 1,800 秒，原始文件上限 1 GiB，首个重复 SET_FLAG 触发停止。已完成尝试由报告脚本从终态日志核对实际 SKU、错误 PC 和数值完成标志。[PR 注册](kimi-k3-bf16g-64x12-registration.json) · [DT 注册](kimi-k3-bf16g-64x12-dt-compact-registration.json)

模型入口按每个请求的 token 数分别分块，保留跨块状态序列依赖，并按 gate / beta / 状态 / 临时导出合同计算逻辑 I/O。8 × 32k、128k、256k 的结果是结构计算，尚无完整大规模 KDA 时延。可复现工作量及指令图见[逐算子算例](../../../docs/examples/kda_chunk.md)。

Kimi-K3 的 PR 精确合同已运行 1,755.615 秒。运行时核对为 950PR_957b，profiling 只记录到第一个 `Transpose_float32_int64_high_performance_10001`；msopprof 到自身 29 分钟限时，宿主没有输出数值检查结果。这次不能确定后续主 kernel 会触发哪个事件。[PR 精确合同终态](kimi-k3-bf16g-64x12-Ascend950PR_957b.json)

同组输入的结构算例能在没有 CANN 的 Python 进程里复算。已安装相同版本 CANN 的开发环境还可直接从保留的原始 trace 重做诊断，命令如下。

```bash
python3 development/validation/chunk21/workload.py
python3 development/validation/chunk21/flag_audit.py
toolchains/envs/cann-9.2.1/bin/python development/validation/chunk21/flag_figure.py
python3 development/validation/chunk21/report.py
```

DT 精确合同在主 kernel 之前已关闭两个 compact 数据库。第一个有 31,056 条指令，时间窗 3.674 µs；第二个有 259,584 条，时间窗 6.558 µs。二者都是 helper 记录的 `entry_kernel_2711`。逐库核对了 SQL 指令数、最早/最晚 tick 与 1.65 GHz 时钟，主 kernel 数据库单列。这两个辅助阶段不能代入主 kernel 的建模误差。[辅助阶段窗口](partial-components.json)

另以最短合法序列 [1,1,1,1,128,128] 区分 sub-16 tail 分支。DT compact 在 579.270 秒时复现三个重复 SET_FLAG，注册的首错终止条件生效。此例输入使用预激活 gate，辅助阶段 2.825 µs，主 kernel 无合格窗口。把序列压到一个 token 仍然保留了错误。[独立分支诊断](sub16-Ascend950DT_9582.json)

同一 Kimi-K3 合同的 DT 精确采集在 1,801.966 秒达到注册的墙钟上限。第三个 DB 对应 `entry_kernel_2`，终止时没有已提交指令。该次日志没有重复 SET 报错，主机还没有返回数值结果。两端精确合同均缺少完整时延；历史固定样例和本轮最短尾块的 SET 故障只对各自合同成立。[DT 精确合同终态](kimi-k3-bf16g-64x12-dt-compact-Ascend950DT_9582.json)
