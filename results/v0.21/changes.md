# v0.21 方法与同集变化

Addmm 按目标 AIC 选择分波，并单独计算列偏置读入 L1、送入偏置缓冲的服务。Cast 与 Sigmoid 共用全片缓存、读写端口和发起间隔模型。mHC Sinkhorn 把同一向量发射可容纳的两行合并，PreSinkhorn 按目标核数分 K，AttnResFwd 按活动通道和缓冲访问计算尾段服务。RMSNorm 与 mHC Post 根据前序核驻留定价，系统可按本 rank 的派发次序维护状态。KDA chunk 增加变长序列、gate 和状态的工作量统计。

本版八条有新时延预测的路线，各登记 10 个新 shape，双端共 160 个窗口。当前合格 152 个，形成 75 对。下表只在同一组已经完成的窗口上比较 v0.20 和 v0.21。每端满十个不同 shape 后再判断独立组是否达标。

| 算子 | 芯片 | 已实测 / 登记 | 旧 WAPE / 最大 | 当前 WAPE / 最大 |
|---|---|---:|---:|---:|
| RMSNorm | DT | 10 / 10 | 18.10% / 37.22% | 11.01% / 27.76% |
| RMSNorm | PR | 10 / 10 | 12.36% / 23.42% | 16.41% / 23.99% |
| Addmm | DT | 10 / 10 | 9.85% / 15.88% | 7.69% / 12.82% |
| Addmm | PR | 10 / 10 | 15.50% / 59.60% | 5.60% / 13.45% |
| mHC Sinkhorn | DT | 10 / 10 | 16.48% / 29.23% | 16.48% / 29.23% |
| mHC Sinkhorn | PR | 10 / 10 | 19.25% / 64.37% | 11.68% / 19.28% |
| mHC PreSinkhorn | DT | 8 / 10 | 41.99% / 75.90% | 8.76% / 18.32% |
| mHC PreSinkhorn | PR | 7 / 10 | 43.08% / 101.27% | 9.49% / 14.68% |
| mHC Post | DT | 9 / 10 | 24.93% / 43.22% | 26.55% / 42.02% |
| mHC Post | PR | 8 / 10 | 26.98% / 41.76% | 23.93% / 35.37% |
| Cast BF16→FP32 | DT | 10 / 10 | 28.43% / 38.60% | 10.22% / 22.33% |
| Cast BF16→FP32 | PR | 10 / 10 | 7.92% / 22.53% | 6.67% / 22.53% |
| Kimi KDA beta Sigmoid | DT | 10 / 10 | 31.17% / 46.68% | 5.59% / 12.85% |
| Kimi KDA beta Sigmoid | PR | 10 / 10 | 84.33% / 105.31% | 17.66% / 26.96% |
| Kimi AttnResFwd | DT | 10 / 10 | 30.54% / 100.87% | 4.43% / 8.14% |
| Kimi AttnResFwd | PR | 10 / 10 | 36.99% / 105.39% | 4.55% / 9.53% |

隔离安装的公开 Python 入口共 1126 种输入，中位 0.129 ms，P95 0.510 ms，最慢 4.234 ms。
扩展归档后，独立安装再验 1589 种输入，单次预测中位 0.114 ms，P95 0.505 ms，最大 3.950 ms。[扩展归档开销](coverage/wheel-check.json)
目前收齐的 183 份宿主成本回执（含失败）里，CA 墙钟中位 114.6 秒，最长 2859.4 秒。多队列并行，按单任务计时。

### 大输入的采集范围

B8×32k 的全宽 Cast 输入为 [262144,5120]，读 2.5 GiB、写 5 GiB，预测在 64 个 DT AIV 上每核最多 993 轮。原 v0.20 预测 2014.328 µs 在启动前冻结；v0.21 的 2514.820 µs 是运行中的后续预测。记录全核 wrapper 从开始到完成的设备窗口，另要求 1,342,177,280 个输出逐元素校验。原 4 小时预算按同一进程增加 2.5 小时，原始数据仍限 8 GiB。
同形 Matmul [1280,2048,512] 补做一次 PR 完整采集。预测 14.027 µs，CA 13.218 µs；逐核确认 24 个矩形尾块与 416 次 MMAD。DT 同形先前已经实测。[两卡逐页图与耗时比](matmul-pages-pr/report.md)
RMSNorm [32768,1280] 双端完整。最忙核的 22 / 25 轮与预测一致，PR 满块 rstd 发起比模型多等 0.776 µs（中位数之差）。[大输入的逐轮实测与模型](wide-rms/report.md)
原测试宿主在进入 ACL 前拒绝了七种已冻结的 RMS 列宽。补采沿用同一组 shape、预测、金标和设备库。对照形 [32,1280] 在 PR 上使用同一 ELF，原宿主 6.309 µs，修复后 6.266 µs，两组成对输出均通过。[补采登记与端点范围](rms/host-recovery.json)
按同一实际时间顺序，RMSNorm 候选在十个完全配对的新形上的 WAPE / 最大误差为 DT 6.02% / 15.64%、PR 7.03% / 14.47%。较早的 18 个完整窗口直接复用，另两个只补采缺口；无历史模式的 PR WAPE 略增。[十形条件与逐端结果](../rms-dispatch-audit/report.md)
全宽 Cast 仍在进行，CA 时间待测。
同一低开销队列登记 Kimi 的动态量化 [262144,7168]，独立于已经取到全指令窗口的窄输出 Cast、Sigmoid。[全宽 Cast 续测依据](large/full-hidden-continuation.json) · [Kimi 量化采集合同](../v0.20/full-quant/freeze.json)

另以 Addmm [16,64,256] 检查单 CPU worker 能否替代默认采集。DT 的数值、二进制、核数和 MMAD 计数一致，但全指令窗口由 3.588 µs 变成 3.448 µs，超过事前设定的一个设备时钟容差。PR 单 worker 在 420 秒上限前未完成。本次独立统计继续使用原来的采集口径。[DT 对照](dispatch/host-workers-dt.json) · [PR 对照](dispatch/host-workers-pr.json)

同流连续调用的 mHC Post [3,4,5376] 已取得双端完整窗口。无历史单值的最坏误差下界为 DT 43.34%、PR 41.08%，均超过 40%；流式 API 在这两次调用各自的最大误差是 12.15%、14.37%。KDA chunk 缺少通过校验的主 kernel 设备窗口，所有芯片耗时填 N/A。完整大 hidden 的 B8×32k 和稀疏注意力分支按实际采集状态披露；长序列小宽度的投影、Cast、Sigmoid 有单独的完整 CA。970DT 全部为架构情景外推。

[RMSNorm 扩核的按地址流水验证](../rms-dispatch-audit/report.md) · [Post 连续两次派发的可辨识性与流水](../post-candidate/paired-result.md) · [Post 不同前序的读响应](../post-candidate/issue-response.md) · [大 Cast 的双缓冲与物理写回](../cast-capacity/report.md) · [核心三路线 270 个完整窗口](coverage/report.md) · [Chunk KDA 版本与事件核对](../chunk-recovery/report.md)

[建模方法](../../docs/modeling-strategy.md) · [总体误差与超限样本](../../docs/error-analysis.md) · [按 tile 或阶段上下对照](../../docs/pipeline-comparisons.md) · [双端耗时与加速比](pairs.md) · [预测开销](wheel-check.json) · [CA 成本](collection-cost.json) · [模型指纹](model.json)
