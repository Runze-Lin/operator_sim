# 按模型查找算子、精度与真实 shape

表内的 shape 是完整前向实际捕获的物理输入和输出尺寸元组。相同尺寸的多种精度只计一个 shape；层号、stride、offset 和视图身份也不增加计数。同一张图重复使用的签名按出现次数计入节点数。这里测量工作负载覆盖，完整 CA 校准样本另见总误差分析。

[逐模型、逐算子的完整 CSV](shape-index.csv.gz) 为每个算子列出 shape 数、是否少于十种、实际 tensor dtype、生产者明确记录的语义精度、最小与最大用例。两个用例按所有输入输出的元素数之和排序（含权重），附有对应的特征键及可用的逻辑 shape，可在 features.json.gz 定位参数与全部张量。空的 semantic_formats 表示该调用没有显式精度标签。

| 模型 | 捕获任务 | 调用名 | 节点实例 | 该模型中有至少十种 shape 的调用名 |
|---|---:|---:|---:|---:|
| ALL | 183 | 119 | 766,282 | 116 |
| Qwen/Qwen3.5-27B | 26 | 47 | 58,330 | 36 |
| Qwen/Qwen3.5-397B-A17B | 10 | 54 | 33,826 | 19 |
| Qwen/Qwen3.8-Flash-Next | 13 | 63 | 55,112 | 53 |
| deepseek-ai/DeepSeek-V3.2 | 12 | 50 | 73,434 | 29 |
| deepseek-ai/DeepSeek-V4-Flash | 10 | 56 | 49,323 | 29 |
| deepseek-ai/DeepSeek-V4-Pro | 10 | 57 | 71,532 | 31 |
| deepseek-ai/DeepSeek-V4.1-Flash | 13 | 62 | 49,564 | 56 |
| deepseek_v3 | 18 | 49 | 71,595 | 37 |
| moonshotai/Kimi-K3 | 19 | 57 | 104,419 | 42 |
| moonshotai/Kimi-K3.1-Pro | 14 | 57 | 80,434 | 45 |
| source_default | 12 | 28 | 2,474 | 23 |
| zai-org/GLM-5.2 | 13 | 51 | 75,606 | 47 |
| zai-org/GLM-5.3-Flash | 13 | 62 | 40,633 | 52 |

ALL 在全目录按调用名和尺寸去重。最后一列适用于该行的图样本，单个预设有十份任务也不保证每个调用能得到十种尺寸。

## 全目录仍少于十种物理 shape 的调用

| 调用名 | shape 数 | 覆盖任务 |
|---|---:|---:|
| `aten.arange.start_step` | 1 | 12 |
| `aten.div.Tensor` | 1 | 12 |
| `aten.pow.Scalar` | 1 | 12 |
