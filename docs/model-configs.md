# 模型规格与 rank-local shape

已于 2026-11-17 从三个官方 Hugging Face 仓库下载 config.json，并在 `configs/models/sources.json` 固定 revision、原始 URL 和 SHA256。没有下载权重，也没有执行仓库中的模型代码。

## 已取得的静态规格

| 文本模型 | hidden size | 主干层数 | attention heads | routed experts | 每 token routed experts | MoE intermediate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek V4.1 Flash | 5120 | 40 | 64 | 384 | 6 | 2304 |
| Kimi K3 | 7168 | 93 | 96 | 896 | 16 | 3072 |
| DeepSeek V4 Pro | 7168 | 61 | 128 | 384 | 6 | 3072 |

这些数字直接摘自锁定的 config，不代表已运行模型。Kimi K3 的 routed_expert_hidden_size 为 3584；MoE 专家内部的投影维度不能统一套用 hidden_size=7168。KDA 与 MLA 的层列表、head 维度和低秩投影参数必须分别使用。DeepSeek 的压缩比例、索引器和输出分组投影也要按具体层解释。

## 不依赖权重的部分

逻辑 shape 推导使用：模型 config + 推理实现的算子/融合规则 + 并行 rank 映射 + 请求场景。

- config 提供静态维度、层类型、专家数量、路由 top-k 和已有量化配置。
- 请求场景提供每个 attention replica 的 batch、每请求 q、KV 长度、prefill chunk 与 padding。MTP 校验 q=1..17；prefill q 不受此上限限制。
- 按 head 的 TP 给出每 rank query heads。以 TP=8 为例，三个模型的 query heads 分别为 8、12、16；这不意味着所有 KV 张量也除以 8。实际缓存表示、KV 复制和低秩重组由实现决定。
- token-wise GEMM 的 M 通常是本地 sum(q_i)，N/K 从对应投影公式及 TP 切分获得。head TP 本身不等价于 token TP。
- EP 映射与路由场景给出每 rank 的 expert 集合，以及每 expert 的 M。平均分布可用于场景估计，真实分布需从请求/路由 trace 取得；不必为了第一版模型先下载完整权重。
- 量化打包、scale shape、对齐、grouped GEMM、kernel 融合和 tiling 由选定的软件实现补齐。静态配置不能唯一确定运行时 kernel 分支。

大于专家数或不能整除专家数的 EP 配置，应结合具体实现的专家复制、负载均衡与放置规则检查；不能简单用 expert_count/EP 得到小数后当作本地专家数。

## 本轮范围

三个 CA smoke case 用于确认 950DT 环境、正确性和指令/流水数据，与模型权重无关。完整 checkpoint 不是本阶段前置条件。后续若要确认某套部署实际触发了哪些 kernel，可使用已存在的真实运行 trace，或在具备部署条件时采集；静态推导结果保持可追溯且标注尚未运行确认。
