# 从模型配置导出本卡 prefill shape

`prefill.py` 读取固定修订的模型配置，提供 DeepSeek-V4.1-Flash、DeepSeek-V4-Pro 和 Kimi-K3 的文本 prefill 合同示例。无需权重。配置的来源、修订和 SHA-256 随示例提供；加载时核对摘要。[模型清单](models/sources.json)

```sh
python examples/workloads/prefill.py --model Kimi-K3 --seq 32768 --batch 8 --tp 8 --ep 8 --chunk 2048 --step 0
```

这里 32,768 是每请求的原始长度，batch=8 是一个 attention DP 副本本次处理的请求数。整段共 262,144 个输入 token；chunk=2,048 将该预算均分给 8 个请求，首步每请求新增 256 个 token。去掉 `--chunk` 可生成整段调用，`--step` 选择具体 chunk 以计算真实 KV 前缀。TP 在头维上切分；示例按连续专家区间分片，要求 EP=TP×DP。需要本地专家行数时用 `--expert-counts` 提供实际路由计数。

输出包含每个逻辑计算的层号、rank 本地张量、证据范围和适用的独立算子调用。能匹配计时合同的调用会附 DT、PR 和 970DT 的估计；整网融合、实际路由或内核证据不足的行保留工作描述。这个静态示例用于准备系统输入，在线时应以框架实际提交的 shape、dtype 和实现为准。
