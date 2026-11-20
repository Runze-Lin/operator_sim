# 1536 行 MoE Gating TopK 的冻结元数据更正

本次待测输入是 **[1536,896,16]**，对应单卡 8 个请求、本次各 prefill 192 个 token、896 个专家、每 token 选 16 个。冻结记录同时保留原预约的 4096 行合同。注册新任务时沿用了旧任务的 `job.prices` 和 `job.scenario`，其中 B8 × 512 及三个芯片的公开价格都属于 4096 行。引用 1536 行场景时应读顶层 `scenario`。取任何 shape 的预测时，按完整 `shape` 唯一匹配 `cases[]`，再取对应芯片 `forecasts[chip].public_us`（公开模型）或 `.us`（候选）。

| [1536,896,16] 芯片 | 冻结公开预测 µs | 冻结候选预测 µs | 对照用的 4096 行公开预测 µs |
|---|---:|---:|---:|
| 950DT | 48.882727 | 48.735643 | 125.682727 |
| 950PR | 60.086665 | 58.813388 | 154.458284 |
| 970DT（外推） | 25.421735 | 26.263077 | 64.095311 |

采集器直接运行 `build/smoke/moe_router19 1536 896 16`，传给仿真器的 SKU 由 DT / PR 循环分别设置。两端各预留 240 分钟和 16 GiB。完整性检查用 `job.shape` 核对启动日志中的 T/E/K，要求 24,576 个 ID 和 24,576 个权重全部通过，并检查实际 SKU、库 ELF 和完整核窗。流水判定用 `cases[0].forecasts[sku]`，第 0 项恰为 1536 行。计分器按完整 shape 和 SKU 读 `cases[].forecasts`。原始表中 5 个 shape × 3 个芯片的公开和候选价、活动核数、忙核行数已逐项与计分表核对。[注册与执行](../../../development/validation/gating35/campaign.py) · [命令构造](../../../development/validation/large29/vector/acquire.py) · [数据与计时核验](../../../development/validation/large29/vector/verify.py) · [评分器](../../../development/validation/gating35/report.py)

更正时双端均在等待采集，尚无运行命令回执。可执行 `python development/validation/gating35/audit_freeze.py` 重做核对。各端结束后，该命令还会比对真实 `cost.json` 中记录的完整命令、芯片和参数，以及合格收据中的数值检查计数。原冻结文件 SHA-256 为 `e6d46cd411c932b065269157c6b344202c3fc3998b10dc4a9d37301edba230a1`。[原始冻结记录](freeze.json) · [元数据核对脚本](../../../development/validation/gating35/audit_freeze.py) · [逐形评分](score.json)
