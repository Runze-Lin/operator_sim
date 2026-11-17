# 昇腾算子性能建模

面向模型系统仿真的低开销算子性能模型。以 A5 的 CA Model 为首阶段参考，通过 tile、硬件流水、依赖和分核调度预测耗时，区分流水占用、核覆盖率与算力利用率。A7、A8 外推是后续目标。

已实现 v0.3 建模闭环：dynamicMXquant、matmul、FA 的分核与流水预测、CA 校准、冻结留出评估、误差诊断与查询开销测试。本版复用已有 trace 修正 quant 双缓冲 UB 关键路径，以及 FA 外层 KV tile 与内部 MMAD 分段。项目内安装 CANN 9.2.1 和独立 Conda 环境，目标为 `Ascend950DT_9582`；在线预测仅依赖 Python 标准库。

- [v0.3 实现、使用与固定基线分析](docs/model-v3.md)
- [最新 CA 留出误差与开销报告](results/v0.3/report.md)
- [冻结的模型](results/v0.3/model.json) / [逐点结果](results/v0.3/evaluation.json)
- [历史 v0.2 方法](docs/model-v2.md) / [结果与已知问题](docs/error-analysis-v2.md)
- [历史 v0.1 实现](docs/model-v1.md) / [历史小 shape 结果](results/v0.1/report.md)

本轮 6 个新留出点：quant 时延 WAPE **4.70%**，FA MMAD 次数 **3/3 精确匹配**，核数 **6/6 匹配**；新增 CA 采集 **9.29 分钟**，无重试。相同新集上旧 quant 模型为 4.93%、更新固定均值基线为 17.24%；两个大 UB 点改善，但小工作量点回退，整体提升有限。FA 时延未修改，新集 WAPE 5.31%。仍不能作为全范围或跨代准确性保证。

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

代码版本为 v0.3.0，长期方案版本号独立。累计 38 个不同输入；本轮复用旧 32 个作为校准，只新增 6 个独立留出点。在线未缓存查询 P95 约 104 μs、缓存约 0.99 μs；未缓存开销增加，建议复用 Predictor 和 shape 缓存。尚未校准独立 ISA 峰值利用率，也未验证 A7/A8 外推。使用 `bash scripts/cann-run.sh COMMAND ...` 启动 CANN 环境；在线预测无需加载 CANN。

```bash
python3 -m operator_model --op matmul --shape 17 96 128
python3 -m operator_model --op fa --shape 16 128 64 1 8
python3 -m unittest discover -s tests -v
```

本机可将 `python3` 换为 `toolchains/envs/cann-9.2.1/bin/python` 使用独立 Conda 环境。

CANNBOT 的 `ops-perf-optimize` 插件已接入 Codex，OpModel 技能文档已安装；其 Python wheel 暂未能从文档指定的两个源安装。知识库已完成 10,550 张跨平台知识卡的索引，另安装 3 个检索技能，并通过 950 平台查询验证。知识检索入口为 `bash scripts/cannbot-query.sh`，安装状态见上述记录。知识卡采用本地检索模式，全量上游源码和 SDK 资源按需另行准备。

GitHub 私有仓库：[Runze-Lin/operator_sim](https://github.com/Runze-Lin/operator_sim)，默认分支 `main`。原始 CA trace、工具链、凭据和权重不进入 Git，只维护代码、配置、来源清单、小规模验证样本和聚合结果。
