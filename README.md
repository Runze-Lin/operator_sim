# 昇腾算子性能建模

面向模型系统仿真的低开销算子性能模型。以 A5 的 CA Model 为首阶段参考，通过 tile、硬件流水、依赖和分核调度预测耗时，并输出明确定义的利用率；A7、A8 采用显式硬件假设进行外推。

已完成环境与三个小 case 验证：项目内安装 CANN 9.2.1 和独立 Conda 环境，dynamicMXquant、matmul、FA 在 `Ascend950DT_9582` CA 仿真中均通过数值校验，并取得指令和流水数据。尚未展开大规模采样或性能模型校准。

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

当前建模方案为 v0.2。首轮三个例子只验证采集链路，不作为整模型算子覆盖或跨代外推准确性的证据。使用 `bash scripts/cann-run.sh COMMAND ...` 启动独立环境；复跑命令见验证说明。

CANNBOT 的 `ops-perf-optimize` 插件已接入 Codex，OpModel 技能文档已安装；其 Python wheel 暂未能从文档指定的两个源安装。知识库已完成 10,550 张跨平台知识卡的索引，另安装 3 个检索技能，并通过 950 平台查询验证。知识检索入口为 `bash scripts/cannbot-query.sh`，安装状态见上述记录。知识卡采用本地检索模式，全量上游源码和 SDK 资源按需另行准备。

本地 Git 分支为 main。用户已于 2026-11-17 授权：进入建模开发阶段时，在其 GitHub 账号下新建仓库并维护版本。当前仍在环境与三个 smoke case 验证阶段，没有远端。原始 CA trace、工具链和权重不进入 Git，只维护代码、配置、来源清单、小规模验证样本和聚合结果。
