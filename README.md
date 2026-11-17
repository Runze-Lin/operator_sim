# 昇腾算子性能建模

面向模型系统仿真的低开销算子性能模型。以 A5 的 CA Model 为首阶段参考，通过 tile、硬件流水、依赖和分核调度预测耗时，区分流水占用、核覆盖率与算力利用率。A7、A8 外推是后续目标。

已实现 v0.5 建模闭环：保留 dynamicMXquant、matmul 的分核与流水模型，用已有32k数据库将FA读延迟与稳态发起间隔分开；冻结后通过128k与16k+1两个新shape验证。新增 Lightning Indexer、mHC、RMSNorm、KDA、Addmm、MoE 融合链的结构模型与库接口证据，未完成时延校准的入口明确返回 `null`。项目内安装 CANN 9.2.1 和独立 Conda 环境，目标为 `Ascend950DT_9582`；在线预测仅依赖 Python 标准库。

- [当前建模策略、公式及解释](results/v0.5/modeling-strategy.md)
- [当前误差分析：Summary、逐shape、库身份及图表](results/v0.5/error-analysis.md)
- [冻结模型](results/v0.5/model.json) / [独立验证与查询成本](results/v0.5/evaluation.json)
- [新增六类算子：公式、API与校准状态](docs/extensions/README.md)
- [轻量采集复现与证据](docs/fa-compact-capture.md)
- [历史32k采集与47.4%误差](results/long-fa/error-analysis.md)
- [历史v0.4实现与使用](docs/model-v4.md) / [报告](results/v0.4/report.md)
- [场景映射与覆盖](results/v0.4/workload-coverage.json)
- [历史v0.3方法](docs/model-v3.md) / [报告](results/v0.3/report.md)
- [历史v0.2方法](docs/model-v2.md) / [结果](docs/error-analysis-v2.md)
- [历史v0.1实现](docs/model-v1.md) / [结果](results/v0.1/report.md)

v0.5 两个新FA留出点：WAPE **0.13%**，同集冻结v0.4 **52.27%**，固定均值 **94.00%**；128k与16k+1单点误差分别 **0.01% / 0.88%**。仅验证普通非因果FA的 Q9/D64/B1/H1 路径，样本少且每点采集一次。32k已用于本轮校准，不再计为独立验证；256k/512k仍是**未实测外推**。FA两点共15.71分钟，另加RMSNorm/Addmm两个极小调用验证共79.54秒；后两者通过完整数值检查，但尚无泛化时延模型。

quant和matmul保持v0.4，45个旧记录预测未变；历史6个quant留出点的v0.4 WAPE仍为5.54%，不能把新FA准确率用于quant或新增族。普通FA代理不是三个模型实际MLA/KDA/压缩注意力的整网验证。每轮固定交付策略与误差两份独立文档，保留历史结果。

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

代码版本为 v0.5.0，长期方案版本号独立。本轮复用32k完整DB校准，先提交冻结模型再采集两个新FA独立留出点，无自动重试。场景覆盖 prefill 单卡batch=1/4/16，decode=1/4/16/32/128/256，输入32k/128k/256k/512k与输出1k，推测8个时主模型校验Q=9；生成360条场景映射，但仅代表点有CA实测。尚未校准独立 ISA 峰值利用率，也未验证 A7/A8 外推。使用 `bash scripts/cann-run.sh COMMAND ...` 启动 CANN 环境；在线预测无需加载 CANN。

```bash
python3 -m operator_model --op quant --shape 9 5120
python3 -m operator_model --op matmul --shape 17 96 128
python3 -m operator_model --op fa --shape 9 131072 64 1 1
python3 -m operator_model.extensions --op rmsnorm --shape 9 7168
python3 -m unittest discover -s tests -v
```

本机可将 `python3` 换为 `toolchains/envs/cann-9.2.1/bin/python` 使用独立 Conda 环境。

CANNBOT 的 `ops-perf-optimize` 插件已接入 Codex，OpModel 技能文档已安装；其 Python wheel 暂未能从文档指定的两个源安装。知识库已完成 10,550 张跨平台知识卡的索引，另安装 3 个检索技能，并通过 950 平台查询验证。知识检索入口为 `bash scripts/cannbot-query.sh`，安装状态见上述记录。知识卡采用本地检索模式，全量上游源码和 SDK 资源按需另行准备。

GitHub 私有仓库：[Runze-Lin/operator_sim](https://github.com/Runze-Lin/operator_sim)，默认分支 `main`。原始 CA trace、工具链、凭据和权重不进入 Git，只维护代码、配置、来源清单、小规模验证样本和聚合结果。
