# v0.23 · 从工作块对齐到真实库实现

| 公开输入合同 | 方法与同集结果 |
|---|---|
| Flash / Pro Cast，Kimi beta Sigmoid | 三模型六张上下同轴对照。每一栏选首、中、末块，并保留全核时标和该核全部读发起。两张 Pro B8×32k 图、两张 Kimi B8×256k 图均按当前模型重建。其余两张为 Flash B8×256k，多达 57 个 UB 块。回放相对 v0.22 时延相同。|
| FLA KDA V2 | 按源代码分别给 Prepare、FwdH、Finalize 的 MMAD、VF、分层 DMA 和依赖定价。公开显式选择后可能返回实验外推。首批两个双端冻结独立形状的 WAPE / 最大为 DT 4.09% / 5.75%，PR 0.57% / 0.74%；两次 DT/PR 快慢方向都判反。另八形在第二份登记中串行验证。融合 V1 保持无合格时延。 |
| RMSNorm | 每端 10 个新 shape 各运行两个前序，40 个目标窗全部合格；用时 25.31 分钟。默认仅 shape 的 WAPE / 最大 DT 26.41% / 57.94%，PR 25.56% / 53.97%。候选点虽降低 WAPE，最大误差分别升到 61.46%、61.01%，默认保留现有点估计。给实际前序 H 的结果为 DT 5.34% / 15.33%、PR 6.10% / 17.73%。新增可选 `estimate_rms_history_band` 返回各个 H 的模型值。 |
| vLLM HcPost | 从固定 vLLM Ascend 源码重建的真实扩展，双端三种形状，6 个完整窗与 BF16 全输出通过，累计约 7 分钟。公开工具按完整 token 分核与 4096 通道分块给出工作量。独立计时误差与 970DT 耗时为 N/A。 |

六份整批长 prefill CA 的原采集累计 3.72 小时，绘图复用已归档的窗口。970DT 数字均为变更芯片配置后按相同实现计算的外推。公开 wheel 安装到源码树外后，1,665 个不同输入单次查询 P95 0.544 ms、最大 10.122 ms；15 次额外 RMS 历史范围查询最大 11.847 ms。Python 3.9 与 3.12 的 234 个回归用例均通过。3.12 的浮点求和在旧 RMSNorm 小数末位可与归档不同，原有冻结时延测试采用 12 位小数容差；模型本身未因此改价。[wheel 和分算子开销](wheel-check.json)

原有逐算子同形 DT/PR/970 结果在当前 wheel 复核。高风险的历史 RMS 与内置 MhcPost 的无历史状态、融合 V1 的同步失败、三段 V2 长 prefill 的 AIV 搬运分支外推仍单列。Flash 全 hidden [262144,5120] 与排队的 Kimi [262144,7168] 取低输出量 CA，不计入逐指令图和 WAPE。vLLM HcPost 的更多机制验证与 KDA 补八形可后台执行。[总误差分析](../../docs/error-analysis.md) · [本版逐形记录](shape-errors.md) · [六张长 prefill 对照图](prefill/README.md) · [V2 独立验证进度](../kda-v23/precision-supplement-state.json) · [全 hidden 作业](../v0.22/full-prefill.md)
