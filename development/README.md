# 维护与复现

系统接入请读[工具说明](../README.md)和[接入指南](../docs/integration.md)。本目录保存开发过程、早期设计和验证辅助程序。当前有效的公式与校验结论分别发布在[总建模方法](../docs/modeling-strategy.md)和[总误差分析](../docs/error-analysis.md)。

## 复现跨芯片后端

本轮独立验证包括 DynamicMXQuant、Matmul、FA 三条 PR 路线。固定输入、两端模型与直接复用 DT 的预测后，采集预算按 shape 执行；可从[本轮变更](../results/v0.14/changes.md)进入冻结合同、实际样例和逐点对比。[采集与发布脚本](validation/sku14)

`development/validation/replay_v13.py` 对既有 950DT 计时输入回放两个 SKU，并校验新旧 DT 预测以及两端工作量。两款芯片的 `msopprof` 证据、采集预算与独立冻结合同见[双 SKU 配对校准](../results/v0.13/sku/README.md)。`development/validation/report_v13.py` 使用合格结果生成逐 SKU 误差表与对照图，`wheel_v13.py` 从部署环境核对安装包输出。两款芯片的输入参数及来源保存在 [configs/chips](../configs/chips)。

## 从哪里复现一次校验

1. **确定合同。** 记录 SoC、CANN 版本、ACLNN 入口、dtype / layout 和可变属性，确认使用的库 kernel。平台和环境见 [CANN 包清单](../docs/cann-packages.json)、[安装记录](../docs/cann-installation.json)、[950DT 最小用例与复跑命令](../docs/smoke-950dt.md)。离线校验可用[隔离 Conda 环境](../environment.yml)，通过 `bash scripts/cann-run.sh COMMAND ...` 同时选定 CANN 和 Python；本机还保留了 [CANNBot 安装来源](../docs/cannbot-installation.json)。
2. **先分析开发集。** 优先读取已有 trace 的数据库和已压缩统计。先对照实际 kernel、正确性、核数、端到端窗口和各流水的区间并集。把用于改公式或选参数的 shape 标成开发样本，即使它曾是旧模型的留出样本。记录哪些耗时成分可以识别，哪些只是解释偏差的假设。
3. **在看 CA 答案前冻结。** 选定新 shape、分支和参考采集合同，包含小/大负载和 tiling 边界。写入版本目录的模型参数、预测、代码/证据哈希和预算（单例超时、总时间、可留存文件）。先冻结再采；避免为每次拟合重跑 CA。FA 长上下文的采集取舍见[数据库与开销实验](fa-compact-capture.md)。
4. **同集评估。** 对每个实际库调用检查输出、同步及完整指令窗口。仅使用计时合格的新点计算独立 WAPE / 最大误差；前版和新版对照使用完全相同的点。开发重放、纯结构验证、超时和外推均另列。分别记录 CA 采集墙钟与模型查询墙钟；后者需注明是否计入首次载入、是否命中缓存。
5. **发布证据。** 在 `results/vX.Y/` 增加采集清单、冻结预测、逐点评估与简短 `changes.md`，同步更新[总方法](../docs/modeling-strategy.md)和[总误差表](../docs/error-analysis.md)。旧版数据保留原路径和数值；曾经缺失的证据不补写成当时的结果。新报告生成脚本放在 `development/validation/`。普通调用方无需执行这些步骤。

## 档案索引

| 内容 | 路径 |
|---|---|
| 初始方案、工况定义、规格来源 | [早期设计稿](modeling-design.md) · [输入工况](../configs/workload-plan.json) · [固定模型 config](../configs/models/sources.json) |
| 首批三算子预测与采集 | [v0.1 说明](model-v1.md) / [结果](../results/v0.1/report.md) · [v0.2 说明](model-v2.md) / [误差归因](error-analysis-v2.md) |
| 量化流水和 FA 指令建模 | [v0.3 方法](model-v3.md) / [结果](../results/v0.3/report.md) · [v0.4 方法](model-v4.md) / [结果](../results/v0.4/report.md) |
| FA 长上下文 | [压缩采集方案](fa-compact-capture.md) · [长上下文数据](../results/long-fa/error-analysis.md) · [v0.5 方法](../results/v0.5/modeling-strategy.md) · [v0.6 变更](../results/v0.6/changes.md) |
| 增加算子与独立验证 | [扩展算子早期证据](../docs/extensions/README.md) · [v0.7](../results/v0.7/changes.md) · [v0.8](../results/v0.8/changes.md) · [v0.9](../results/v0.9/changes.md) |
| 少 token、宽通道的 mHC Post / RMSNorm | [v0.10 冻结清单](../results/v0.10/wide/freeze.json) · [模型与采集结果](../results/v0.10/) |
| 写回与上下文、固定宽度放大 | [v0.11 变更](../results/v0.11/changes.md) · [规模实验](../results/v0.11/scaling/summary.md) |
| RMSNorm 分轮写回与 SIMD 重叠、mHC Post 启动状态 | [v0.12 变更](../results/v0.12/changes.md) · [独立 CA 分组与图](../results/v0.12/audit/summary.md) · [汇总发布脚本](validation/publish.py) · [wheel 冻结核对](validation/wheel_check.py) |
| 所有版本 | [版本目录](../results/) |

`results/vX.Y/` 本身就是按版本隔离的实验档案，冻结清单和早期复现脚本会按这些路径查找数据；不要为了整理首页搬动它们。早期脚本留在 [`scripts/`](../scripts/)；新的维护代码和实验说明可放在本目录。`runs/` 的原始 CA trace、安装包、工具链和权重均留在本机。重跑前请核对可用空间和每组采集预算。
