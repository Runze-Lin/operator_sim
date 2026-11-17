# 昇腾算子性能建模

面向模型系统仿真的低开销算子性能模型。以 A5 的 CA Model 为首阶段参考，通过 tile、硬件流水、依赖和分核调度预测耗时，区分流水占用、核覆盖率与算力利用率。A7、A8 外推是后续目标。

已实现 v0.2 建模闭环：dynamicMXquant、matmul、FA 的分核与流水预测、CA 校准、冻结留出评估、误差诊断与查询开销测试；扩展 UB 循环、tile/K 尾块、多 head 和 D=128 分支。项目内安装 CANN 9.2.1 和独立 Conda 环境，目标为 `Ascend950DT_9582`；在线预测仅依赖 Python 标准库。

- [v0.2 实现、使用与采样预算](docs/model-v2.md)
- [最新 CA 留出误差与开销报告](results/v0.2/report.md)
- [冻结的模型](results/v0.2/model.json) / [逐点结果](results/v0.2/evaluation.json)
- [历史 v0.1 实现](docs/model-v1.md) / [历史小 shape 结果](results/v0.1/report.md)

当前新留出 WAPE：quant **13.85%（未达标）**、matmul **6.91%**、FA **4.47%**；核数 7/7 匹配。FA 指令次数仍有 51.28% WAPE，固定耗时基线在 quant/FA 上仍更好。详见 [误差归因与后续验证](docs/error-analysis-v2.md)，不能将这一版作为已验证的通用缩放或跨代模型。

- [建模方案](docs/modeling-design.md)
- [环境核查](docs/environment-audit.json)
- [架构假设草案](configs/architecture-scenarios.json)
- [工作负载与采样计划](configs/workload-plan.json)
- [CANNBOT 安装记录](docs/cannbot-installation.json)
- [950DT 三算子验证与复跑说明](docs/smoke-950dt.md)
- [采集结果摘要](docs/smoke-950dt-results.json)
- [CANN 安装记录](docs/cann-installation.json)
- [模型 config 与每卡 shape 推导](docs/model-configs.md)

A5 平台固定为 `Ascend950DT_9582`；配套 CANN 9.2.1 的目标配置为 32 Cube / 64 Vector、1.65 GHz。MTP 草拟长度覆盖 0–16，主模型每请求校验长度覆盖 1–17，prefill/decode 和用户指定的并行组合均纳入计划。三个模型的官方 config 已固定 revision 并下载；本阶段不依赖权重。

代码版本为 v0.2.0，长期方案版本号独立。两轮选择 32 个不同输入，本轮用 25 个校准点、7 个全新留出点，优先覆盖实现机制变化；仍不代表整网算子或所有算法覆盖。尚未校准独立 ISA 峰值利用率，也未验证 A7/A8 外推。使用 `bash scripts/cann-run.sh COMMAND ...` 启动 CANN 环境；在线预测无需加载 CANN。

```bash
python3 -m operator_model --op matmul --shape 17 96 128
python3 -m operator_model --op fa --shape 16 128 64 1 8
python3 -m unittest discover -s tests -v
```

本机可将 `python3` 换为 `toolchains/envs/cann-9.2.1/bin/python` 使用独立 Conda 环境。

CANNBOT 的 `ops-perf-optimize` 插件已接入 Codex，OpModel 技能文档已安装；其 Python wheel 暂未能从文档指定的两个源安装。知识库已完成 10,550 张跨平台知识卡的索引，另安装 3 个检索技能，并通过 950 平台查询验证。知识检索入口为 `bash scripts/cannbot-query.sh`，安装状态见上述记录。知识卡采用本地检索模式，全量上游源码和 SDK 资源按需另行准备。

GitHub 私有仓库：[Runze-Lin/operator_sim](https://github.com/Runze-Lin/operator_sim)，默认分支 `main`。原始 CA trace、工具链、凭据和权重不进入 Git，只维护代码、配置、来源清单、小规模验证样本和聚合结果。
