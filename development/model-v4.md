# v0.4：quant 控制路径与在线开销

本轮修改性能预测模型，不修改 CANN 内核。目标仍为 Ascend950DT_9582、CANN 9.2.1。完整结果及图表见 [独立验证报告](../results/v0.4/report.md)。

## 方法

保留 v0.3 的源代码分核、实际 UB 尾块以及双缓冲 max-plus 递推。量化算子预测为：

```
T = max(T_pipeline, a + b * min(log2(1 + active_AIV), 4))
```

`T_pipeline` 包含启动前缀、首轮 DMA、后续 DMA 发起间隔、向量服务时间、输出槽复用和写回 drain。`a,b` 使用旧数据中每核≤4096元素的控制主导样本拟合；cap=4 是历史 trace 上开发诊断选出的经验饱和项，不是硬件核数上限。较大工作量继续由流水路径决定，不用固定值覆盖 UB 多轮增长。

复用全部38个历史输入，其中13个 quant。10个 phase 归约直接复用 v0.3，另外3个从已有本地 trace 归约，没有新增校准 CA。旧 trace-analysis 顶层 `source_dataset_sha256` 沿用了最初10个归约的 v0.2 数据集哈希；完整13个样本的身份应以每条 trace_sha256、v0.4 imported-calibration 和冻结哈希为准。

5个候选仅在历史数据上进行 leave-one-shape-out 重拟合，用小工作量 WAPE 选择。开发诊断不是无偏泛化估计，独立证据来自之后冻结的10个新输入。新留出测量后不修改模型参数；无法完成的 CA 点保留失败记录。

## 在线路径

v0.3 的 quant 预测先完整执行 v0.2，再重新规划 quant、构建 phase，并重复冻结整个嵌套结果。v0.4 直接生成一次 quant 计划与结果，移除在线历史版本比较。FA/matmul 时延不变，直接复用旧不可变结果；不在调用中读取 trace 或 CANN。

```python
from operator_model.model_v4 import Case, Predictor

model = Predictor('results/v0.4/model.json', cache_size=4096)
case = Case('quant', (9, 5120))
full = model.predict(case, use_cache=False)  # 包含分核、UB 与 phase
latency_us = model.predict_us(case, use_cache=False)  # 可选标量接口
```

完整 predict 的复杂度仍为 O(UB tile数)，并非所有大 shape 都是常数时间。缓存为有界 LRU，结果递归不可变。`use_cache=False` 不查询也不填充缓存。标量接口复用同一计划与调度，省掉不可变诊断结果构造，不宣称与完整 API 是相同成本口径。

## 比较与范围

v0.3 的参数和代码完整保留。比较扩展 shape 时只使用新的 Case 容器绕过旧输入上限，公式没有重拟合。因此宽度5120/7168、FA B>1、长KV的旧模型输出是“冻结公式外推”。两版本在同一全新留出集测误差，查询性能按相同输入与完整 API 交替计时。

固定耗时主基线使用全部旧38个输入的分算子均值，并额外记录 v0.3 原均值。固定耗时没有用新留出数据估计。常数在启动主导的小 shape 上可能更稳，但在 UB 迭代、长 KV 等算量增长路径上会失效；必须分别评估。

新 Case 扩展的是可预测输入域，不代表扩展后的内核选择已验证。FA 仍只支持普通非因果 BF16 SBH 的 D64/128 代理，没有实现 MLA、KDA、DeepSeek D512 或压缩/稀疏注意力。实际模型 config 用于提出候选 shape；确定实际 kernel、融合、cache 布局需要框架运行配置及 dispatch 证据。

## 证据及复现

- [采样及预算](../configs/sampling-v4.json)
- [参数与完整历史基线](../results/v0.4/model.json)
- [冻结预测与文件哈希](../results/v0.4/frozen-predictions.json)
- [历史数据交叉验证](../results/v0.4/calibration-diagnostics.json)
- [已有 trace 归约](../results/v0.4/trace-analysis.json)
- [场景推导脚本](../scripts/workload-v4.py)
- [独立评估脚本](../scripts/evaluate-v4.py)

继续使用此前已全文核对的950 DynamicMxQuant与FlashAttentionScore知识卡、固定版本 ops-nn/ops-transformer 源码，以及 v0.3 的正式知识检索审计。没有依据未检索的新硬件规格新增 ISA 结论。具体来源和版本见 [v0.3证据](evidence-v3.md)。CANN 9.1知识卡与本地9.2.1安装之间的差异继续由实际CA证据确认。

扩大 host 输入限额时保留原数据模式和全输出 golden：普通算子工作乘积≤64M，FA≤512M，FA每个输入张量≤64M元素。模拟超时不是数值或模型失败的自动证明；但没有完整成功采集就不能计入误差验收。
