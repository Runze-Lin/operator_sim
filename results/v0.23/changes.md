# v0.23 · 从工作块对齐到真实库实现

| 公开输入合同 | 方法与同集结果 |
|---|---|
| Flash / Pro Cast，Kimi beta Sigmoid | 三模型 14 张上下同轴对照。Flash、Kimi 各有 32k、128k、256k，Pro 有 32k，每种形状都在两端核验。每一栏选首、中、末块，并保留全核时标和该核全部读发起，最多 57 个 UB 块。回放相对 v0.22 时延相同。|
| FLA KDA V2 | 按源代码分别给 Prepare、FwdH、Finalize 的 MMAD、VF、分层 DMA 和依赖定价。公开显式选择后可能返回实验外推。两批 10 个双端冻结独立 shape 的 WAPE / 最大为 DT 3.84% / 7.87%，PR 2.48% / 6.99%。20 窗共花 38.87 分钟 CA，DT/PR 比值 WAPE 为 1.88%，快慢方向命中 4/10。融合 V1 保持无合格时延。 |
| RMSNorm | 每端 10 个新 shape 各运行两个前序，40 个目标窗全部合格；用时 25.31 分钟。默认仅 shape 的 WAPE / 最大 DT 26.41% / 57.94%，PR 25.56% / 53.97%。候选点虽降低 WAPE，最大误差分别升到 61.46%、61.01%，默认保留现有点估计。给实际前序 H 的结果为 DT 5.34% / 15.33%、PR 6.10% / 17.73%。新增可选 `estimate_rms_history_band` 返回各个 H 的模型值。 |
| vLLM HcPost | 从固定 vLLM Ascend 源码重建的真实扩展，双端三种形状，6 个完整窗与 BF16 全输出通过，累计约 7 分钟。本版另用 3 个开发窗定价分片 DMA、算术 VF、固定满片 CAST 及双槽依赖，再以两端新 shape 独立验证。合格域上 10 个独立 shape 的 WAPE / 最大为 DT 2.72% / 5.33%、PR 2.90% / 5.59%。原登记 D=4112 数值失败，额外登记 D=4160 两端通过。公开接口给工作量、设备时间与外推状态，失败短尾仅返回工作量。 |

14 份整批长 prefill CA 的原采集累计 5.46 小时，绘图复用已归档的窗口。970DT 数字均为变更芯片配置后按相同实现计算的外推。公开 wheel 安装到源码树外后，1,698 个不同输入单次查询 P95 0.517 ms、最大 6.053 ms；15 次额外 RMS 历史范围查询最大 10.751 ms。Python 3.9 与 3.12 的 240 个回归用例均通过。3.12 的浮点求和在旧 RMSNorm 小数末位可与归档不同，原有冻结时延测试采用 12 位小数容差；模型本身未因此改价。[wheel 和分算子开销](wheel-check.json)

原有逐算子同形 DT/PR/970 结果在当前 wheel 复核。高风险的历史 RMS 与内置 MhcPost 的无历史状态、融合 V1 的同步失败、三段 V2 长 prefill 的 AIV 搬运分支外推仍单列。Flash 全 hidden [262144,5120] 的首轮在 6 小时 11 分钟后触及 8 GiB 日志预算。64 个 AIV 开始、31 个完成，原生包络仅有下界 2100.106 µs，缺完整 golden。Kimi 全 hidden [262144,7168] 已启动。两者均未计入逐指令图与 WAPE。新的 64 核全输出对照确认仅重定向 128 份周期文本时，原生包络相对同条件全日志低 0.434%，相对历史全日志高 4.961%。Flash 按此采法排在 Kimi 之后，预算为 24 小时与 16 GiB。[原始核级证据](full-hidden/previous-partial.json)。vLLM HcPost 的正确性失败、另外新登记的 64 列尾片和完整组进度单列。[总误差分析](../../docs/error-analysis.md) · [本版逐形记录](shape-errors.md) · [14 张长 prefill 对照图](prefill/README.md) · [V2 独立验证进度](../kda-v23/precision-supplement-state.json) · [全 hidden 的新短控与持久队列](full-hidden/report.md)
