# 昇腾算子性能建模

面向模型系统仿真的低开销算子性能模型。以 A5 的 CA Model 为首阶段参考，通过 tile、硬件流水、依赖和分核调度预测耗时，区分流水占用、核覆盖率与算力利用率。A7、A8 外推是后续目标。

当前版本 **v0.6**：dynamicMXquant、matmul、FA 已有局部耗时模型；Lightning Indexer、mHC、RMSNorm、KDA、Addmm、MoE融合链已加入结构模型和库接口证据，未校准耗时返回 `null`。使用项目内 CANN9.2.1 与独立 Conda 环境，目标 `Ascend950DT_9582`；在线预测仅依赖 Python 标准库。

- **[整套当前建模方法：公式、解释、分核、流水及边界](docs/modeling-strategy.md)**
- **[整套当前误差分析：逐算子Summary、全部shape、实际库实现及图表](docs/error-analysis.md)**
- [本轮方法与误差变化](results/v0.6/changes.md)
- [当前参数](results/v0.6/model.json) / [新留出评估与成本](results/v0.6/evaluation.json) / [全量证据快照](results/current/evaluation.json)
- [新增算子公式、API与校准状态](docs/extensions/README.md)
- [轻量采集复现](docs/fa-compact-capture.md) / [场景映射](results/v0.4/workload-coverage.json)
- [历史v0.5 FA误差](results/v0.5/error-analysis.md) / [历史32k诊断](results/long-fa/error-analysis.md)
- [历史v0.4](results/v0.4/report.md) / [v0.3](results/v0.3/report.md) / [v0.2](results/v0.2/report.md) / [v0.1](results/v0.1/report.md)

本轮复用19个quant旧trace，只新增3个冻结shape，CA合计 **4.14分钟**。同集WAPE **3.88%→3.90%**，最大误差 **6.36%→5.22%**：总体平均误差基本持平，不能宣称全域精度提升。未缓存标量查询减少对象构造及重复规划，完整诊断接口没有整体提速；具体基准见报告。旧开发回放、仍有效的冻结验证、仅调用验证和外推分开统计。

FA保留v0.5：128k/16k+1两点WAPE0.13%，但只覆盖普通非因果Q9/D64/B1/H1；256k/512k仍是未实测外推。matmul本轮无新增独立验证。当前模型不能代表所有真实模型专用注意力或全部TP/EP/量化分支。

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

代码版本为 v0.6.0，长期方案版本号独立。本轮模型在采集前冻结，无自动重试。场景覆盖 prefill 单卡batch=1/4/16，decode=1/4/16/32/128/256，输入32k/128k/256k/512k与输出1k，推测8个时主模型校验Q=9；生成360条场景映射，但仅代表点有CA实测。尚未校准独立 ISA 峰值利用率，也未验证 A7/A8 外推。使用 `bash scripts/cann-run.sh COMMAND ...` 启动 CANN 环境；在线预测无需加载 CANN。

```bash
python3 -m operator_model --op quant --shape 9 5120
python3 -m operator_model --op matmul --shape 17 96 128
python3 -m operator_model --op fa --shape 9 131072 64 1 1
python3 -m operator_model.extensions --op rmsnorm --shape 9 7168
python3 -m unittest discover -s tests -v
```

本机可将 `python3` 换为 `toolchains/envs/cann-9.2.1/bin/python` 使用独立 Conda 环境。

系统仿真只需时延时，优先使用流式标量接口（无需CANN）：

```python
from operator_model.fast_query import Predictor
from operator_model.model_v6 import Case
model = Predictor("results/v0.6/model.json")
us = model.predict_us(Case("quant", (288, 7168)), use_cache=False)
```

CANNBOT 的 `ops-perf-optimize` 插件已接入 Codex，OpModel 技能文档已安装；其 Python wheel 暂未能从文档指定的两个源安装。知识库已完成 10,550 张跨平台知识卡的索引，另安装 3 个检索技能，并通过 950 平台查询验证。知识检索入口为 `bash scripts/cannbot-query.sh`，安装状态见上述记录。知识卡采用本地检索模式，全量上游源码和 SDK 资源按需另行准备。

GitHub 私有仓库：[Runze-Lin/operator_sim](https://github.com/Runze-Lin/operator_sim)，默认分支 `main`。原始 CA trace、工具链、凭据和权重不进入 Git，只维护代码、配置、来源清单、小规模验证样本和聚合结果。
