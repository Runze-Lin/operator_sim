# 昇腾算子性能建模

面向模型系统仿真的低开销算子性能模型。以 A5 的 CA Model 为首阶段参考，通过 tile、硬件流水、依赖和分核调度预测耗时，并输出明确定义的利用率；A7、A8 采用显式硬件假设进行外推。

已实现 v0.1 建模闭环：dynamicMXquant、matmul、FA 的 shape 预测、少量 CA 校准、冻结留出评估、误差诊断与查询开销测试。项目内安装 CANN 9.2.1 和独立 Conda 环境，目标为 `Ascend950DT_9582`；在线预测仅依赖 Python 标准库。

- [v0.1 实现与使用](docs/model-v1.md)
- [CA 留出误差与开销报告](results/v0.1/report.md)
- [冻结的模型](results/v0.1/model.json) / [逐点结果](results/v0.1/evaluation.json)

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

长期建模方案为 v0.2，首个代码实现版本为 v0.1.0，两者版本号独立。首轮只覆盖少量标准 BF16/FP8 小 shape，不代表整网算子覆盖；未验证 A7/A8 外推。使用 `bash scripts/cann-run.sh COMMAND ...` 启动 CANN 环境；在线预测无需加载 CANN。

```bash
python3 -m operator_model --op matmul --shape 17 96 128
python3 -m unittest discover -s tests -v
```

本机可将 `python3` 换为 `toolchains/envs/cann-9.2.1/bin/python` 使用独立 Conda 环境。

CANNBOT 的 `ops-perf-optimize` 插件已接入 Codex，OpModel 技能文档已安装；其 Python wheel 暂未能从文档指定的两个源安装。知识库已完成 10,550 张跨平台知识卡的索引，另安装 3 个检索技能，并通过 950 平台查询验证。知识检索入口为 `bash scripts/cannbot-query.sh`，安装状态见上述记录。知识卡采用本地检索模式，全量上游源码和 SDK 资源按需另行准备。

GitHub 私有仓库：[Runze-Lin/operator_sim](https://github.com/Runze-Lin/operator_sim)，默认分支 `main`。原始 CA trace、工具链、凭据和权重不进入 Git，只维护代码、配置、来源清单、小规模验证样本和聚合结果。
