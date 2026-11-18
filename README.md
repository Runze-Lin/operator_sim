# 昇腾算子性能建模

面向模型系统仿真的低开销算子性能模型。以 A5 的 CA Model 为首阶段参考，通过 tile、硬件流水、依赖和分核调度预测耗时，区分流水占用、核覆盖率与算力利用率。A7、A8 外推是后续目标。

当前 v0.8 通过一个入口估算单算子耗时。Vector 路径按核内归约、读算写和状态依赖建模，Cube 路径按矩阵 tile、K 循环和核间波次建模，CV 融合路径按流水交接和重叠建模。每条路径用未参与开发的 shape 校验。总表列出一个当前预测的误差和实测查询开销，KDA chunk 的计时资格单独标明。

- [算子建模方法、三类实施步骤与算例](docs/modeling-strategy.md)
- [逐算子独立验证、全部 shape 和查询开销](docs/error-analysis.md)
- [v0.8 方法及误差变化](results/v0.8/changes.md)
- [冻结参数、采集记录和误差数据](results/v0.8)
- [历史 v0.7 对照与验证](results/v0.7/error-analysis.md)

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

代码版本为 v0.8.0，长期方案版本号独立。各组在 CA 验证前冻结预测。场景覆盖 prefill 单卡batch=1/4/16，decode=1/4/16/32/128/256，输入32k/128k/256k/512k与输出1k，推测8个时主模型校验Q=9；生成360条场景映射，但仅代表点有CA实测。尚未校准独立 ISA 峰值利用率，也未验证 A7/A8 外推。使用 `bash scripts/cann-run.sh COMMAND ...` 启动 CANN 环境；在线预测无需加载 CANN。

```bash
python3 -m operator_model --op quant --shape 9 5120
python3 -m operator_model --op matmul --shape 17 96 128
python3 -m operator_model --op fa --shape 9 131072 64 1 1
python3 -m operator_model --op rmsnorm --shape 9 7168
python3 -m unittest discover -s tests -v
```

本机可将 `python3` 换为 `toolchains/envs/cann-9.2.1/bin/python` 使用独立 Conda 环境。

系统仿真器可在仓库环境中直接调用 Python 接口。返回的 `predicted_us` 单位为微秒。

```python
from operator_model.predictor import predict

result = predict("rmsnorm", (36, 7168))
device_us = result["predicted_us"]
```

CANNBOT 的 `ops-perf-optimize` 插件已接入 Codex，OpModel 技能文档已安装；其 Python wheel 暂未能从文档指定的两个源安装。知识库已完成 10,550 张跨平台知识卡的索引，另安装 3 个检索技能，并通过 950 平台查询验证。知识检索入口为 `bash scripts/cannbot-query.sh`，安装状态见上述记录。知识卡采用本地检索模式，全量上游源码和 SDK 资源按需另行准备。

GitHub 私有仓库：[Runze-Lin/operator_sim](https://github.com/Runze-Lin/operator_sim)，默认分支 `main`。原始 CA trace、工具链、凭据和权重不进入 Git，只维护代码、配置、来源清单、小规模验证样本和聚合结果。
