# 三个模型的大 prefill 如何落到单卡算子

单卡 attention 副本同时接收 8 个等长请求。按完整 prefill 调度时，32k、128k 和 256k 分别给出 **262,144、1,048,576 和 2,097,152 个本地 token**。32k 指每个请求有 32,768 个 token。三个配置的每请求位置上限均为 1,048,576。整批长输入是否适合一次 dispatch，还取决于调度上限、显存、KV 分配、跨卡通信和后端 workspace。导出器只计算维度和工作量，不分配这些张量。[模型配置与源码指纹](../results/v0.20/workloads/source-lock.json)

| 场景参数 | 整批 | 单卡 chunk 预算 2,048 |
|---|---:|---:|
| 每个请求本次追加的 Q | 32,768 / 131,072 / 262,144 | 256 |
| 单卡 token 数 T | 262,144 / 1,048,576 / 2,097,152 | 2,048 |
| 每个请求累计 KV 长度 | 与 Q 相同 | 第一块 256，最后一块 32,768 / 131,072 / 262,144 |
| 每请求分块次数 | 1 | 128 / 512 / 1,024 |

生成结果覆盖三种模型、三个长度、两种并行拓扑和三个调度截面，共 **54 个场景**。每个场景含逐算子输入、参数或工作量、输出、适用层号、映射依据、条件分支，以及能建立同语义单算子定价时的 DT、PR、970DT 设备时间。其余条目给出具体原因或所缺的运行时输入。所有层号从 0 开始。[场景索引](../results/v0.20/workloads/index.csv) · [完整数值张量清单，gzip JSON Lines](../results/v0.20/workloads/contracts.jsonl.gz)

固定版 vLLM Ascend 的外层 decoder 另用符号张量重跑了 27 组完整及分块场景。每组核对 token 数、hidden、各类层数以及路由/残差的实际调用次数。B8×32k 的 Flash 执行 80 次 mHC post、40 次主注意力；Pro 为 122 和 61 次；Kimi 的 decoder 内执行 186 次 AttnResFwd、69 次 KDA 和 24 次 MLA，另有模型末端一次 AttnResFwd。探针运行 decoder 方法，设备 leaf 由记录模块代替，物理 kernel 与调度仍需要 CA 或真实服务 trace。[源码假跑与逐场景断言](../results/v0.20/workloads/decoder-check.json)

## 同一个 T 对应什么矩阵

以下表格采用整批 32k、attention TP8、EP8、单 attention 副本。Q 的头数按 TP 切，所有副本内的 TP rank 看到同一批 token。两种 DeepSeek 的 indexer 都用复制投影，权重和 indexer Q 的头数保持完整。三种模型都包含 KV、投影和专家分支，表中列出能区分其实际算子路径的维度。[DeepSeek Flash](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/models/deepseek_v41/model.py) · [DeepSeek Pro](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/models/deepseek_v4/model.py) · [Kimi decoder](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/models/kimi_k3.py)

| 合同，T=262,144 | DeepSeek V4.1 Flash | DeepSeek V4 Pro | Kimi K3 |
|---|---|---|---|
| 层数与 hidden | 40 × 5,120 | 61 × 7,168 | 93 × 7,168 |
| attention Q | [T,8,512] | [T,16,512] | 69 个 KDA 层 [1,T,12,128]，24 个 MLA 层 [T,12,192] |
| KV / state | SWA 一头 512，源层长 KV 一头 512 | 一头 512，C4 / C128 压缩 | KDA 状态 [8,12,128,128]；MLA 缓存逻辑宽 576 |
| Q 低秩与输出 | Q 低秩 1,280；1 个本地输出组，组内 K=4,096，输出 1,024 | Q 低秩 1,536；2 个本地输出组，各 K=4,096、输出 1,024 | MLA 复制投影 7,168→2,112，Q-B 1,536→2,304，输出投影 1,536→7,168 |
| indexer / beta | Q [T,32,128]，weights [T,32]，top-512 | C4 的 Q [T,64,128]，weights [T,64]，top-1,024 | KDA beta [1,T,12]，一次 FP32 sigmoid |
| 残差 | mHC 两组 pre/post，4 流，Sinkhorn 20 轮 | 同为 4 流，pre kernel 版本与 Flash 不同 | AttnRes bank [T,8,7168]，每 12 层写入新槽 |
| MoE gate | [T,384]，top-6 | [T,384]，top-6；层 0–2 为 hash 分支 | [T,896]，top-16；首层为 dense，后续 92 层 MoE |
| 每 rank 路由专家与 FFN 中间宽 | 48 个，2,304 | 48 个，3,072 | 112 个，3,072；路由专家输入/输出宽 3,584 |
| 平衡路由例的本地 GMM 输入 | [196608,5120] | [196608,7168] | [524288,3584] |

**Flash。** 输入头数是 64，头宽 512，RoPE 后缀 64。第 2、8、14、20 层拥有长 KV，另有 8 个 index 源层。例：第 26 层使用第 20 层的 KV、第 24 层产生的 top-k。第 20 层生成 [T,1,2048] 候选块 ID，每块 8 行。A5 packed cache 的 SWA、长 KV 分别需要 544、320 字节/已存行。index K/scale 分别需要 64、4 字节/行。设压缩率为 r、该请求已有 P 个 token，本步新增的持久化行数为 ⌊(P+Q)/r⌋−⌊P/r⌋；Flash C2 实现为每个 query 提供投影槽位，部分槽位跳过写入。因此投影工作量与持久化写入行数分别导出。Engram 在第 1、14 层取宽 6,144 的行，复制投影 6,144→25,600，启用条件为运行时提供 Engram 配置。[索引与候选](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/models/deepseek_v41/indexer.py) · [A5 cache 格式](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/models/deepseek_v41/cache_config.py)

**Pro。** C128 和 C4 均访问滑窗和压缩历史。默认 C4 有单独的 indexer 与 index K 压缩器。其复制投影生成 [T,64] 的 BF16 权重，乘缩放系数后由 A5 prepare_dsa_indexer_weights 升为 FP32；Flash 的 [T,32] 升精度在乘系数前。以整批 128k 为例，每请求 C128 存 1,024 行，C4 存 32,768 行。C4 最末 query 至多选择其中 1,024 行，加上至多 128 个窗口位置。清单保留窗口与压缩分支的容量，不将两者之和当作精确的唯一 key 计数。采用默认关闭跨层 IndexCache 的路径。开 HF override、改 CP 或 cache block 规格时须重新导出对应物理调度。[Pro 压缩器](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/models/deepseek_v4/compressor.py) · [Pro 索引器](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/models/deepseek_v4/indexer.py)

**Kimi。** KDA 与 MLA 取自配置的两套层号。KDA prefill 通过 cu_seqlens 标记 8 个请求的边界，在 64-token 块上执行，gather/scatter 每个请求的 VK 状态。beta 在普通投影路径经 float().sigmoid()，mixed 路径在辅助流提前计算，主路径收到预处理标记。同一份 beta 只计一次。MLA 父类显式构造 QK 宽 128+64=192 和 KV 低秩宽 512+64=576；本模型关闭实际 RoPE 变换。V 的宽仍是 128。以整批 32k 为例，单本地 MLA 头处理的 causal query-key 对数为 8×32768×32769/2；实际分块和中间张量驻留由硬件后端选择。[KDA 分支](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/ops/kimi_kda.py) · [KDA 算子参数](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/ops/kda.py) · [MLA 投影定义](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/models/kimi_k3/amd/linear.py#L319-L447)

## 并行、路由与资源开销

两种八卡拓扑均取 PP1，关闭 sequence parallel 与 DSA CP，启用 routed expert 的 EP8，不做权重冗余或专家内 TP。TP8×DP1 的 EP 组面对一份 token；TP1×DP8 时每个 attention 副本本身都接 8 个请求，整个 EP 组的独立 token 数为 8T。设 E 为专家总数、k 为每 token 专家数，理想平衡路由分给专家 i 的行数取 ⌊(i+1)A/E⌋−⌊iA/E⌋，其中 A=DP×T×k。每个 EP rank 管理连续 E/EP 个专家。调用方可传实际直方图替换这一假设。真实路由受输入、层和负载均衡影响。shared FFN 按张量并行切分，未合并进 routed GMM。MLP 宽、EP 物理布局或 shared expert 混合放置有变更须重新计算。本导出保留 collective 的逻辑张量，网络耗时由系统仿真器提供。

导出行可以是完整语义算子、融合内部组件、可选分支或缓存容量。举例，若 MLA 预处理和投影被融合，直接累加所有逻辑子项会重复计时；银行容量和滑窗容量也不能当作每步实际 HBM 字节。把 standalone_prediction 用作独立算子成本时应检查 dtype、布局、属性、调用边界及当前校准状态。模型实际使用的 FP8/FP4 专家矩阵、压缩 sparse attention 和 packed KDA 预填充均保留实际几何，等待对应 kernel 的 CA 校准。仅存在数学 GEMM 的地方提供有用计算量 2MNK；分组投影另外乘组数。逐专家 FFN 两次 GEMM 的总计算量为 6×本地接收行数×专家输入宽×专家中间宽。输出 logits 只对每请求最后一个位置定价，假设未请求 prompt logprobs。

## 一次实际形状的设备预测

下表均为独立 kernel 的模型值，单位 µs。TP8、B8、整批 prefill；970DT 使用假设配置。计算条件一致时，64 倍或更多的每核工作量会产生重复 UB tile、DMA 事务与带宽竞争；小 shape 的单次固定开销占比不能直接搬到这里。以 Flash Cast 为例，当前各档预测并不构成固定比例，CA 检查仍需覆盖工作集和访存并发发生变化的位置。Kimi sigmoid 是 beta 的一个子步骤，输入规整为 [T,12] 后按 FP32 读写。下表每个数据点的实测资格请查 CA 校准表。[大 Cast 的 CA 证据](../results/v0.20/large/report.md) · [Sigmoid 的 CA 证据](../results/v0.20/newop/evaluation.json)

| 算子与输入 | 上下文 | 950DT 预测 | 950PR 预测 | 970DT 外推 |
|---|---:|---:|---:|---:|
| Flash indexer Cast [262144,32] | 32k | 8.36 | 13.05 | 6.05 |
| Flash indexer Cast [1048576,32] | 128k | 23.27 | 64.36 | 13.34 |
| Flash indexer Cast [2097152,32] | 256k | 69.52 | 208.16 | 23.06 |
| Kimi KDA beta sigmoid [262144,12] | 32k | 10.28 | 21.73 | 5.43 |
| Kimi KDA beta sigmoid [2097152,12] | 256k | 54.18 | 147.42 | 17.85 |
| Kimi 动态 MX 量化 [2097152,7168]，mixed QKV、scale_alg=0 | 256k | 11,362.75 | 32,472.19 | 4,186.59 |

## 提取与验证

在仓库根目录可导出任意一个场景，标准输出是一个 JSON 对象，model 名与配置文件相同。

```bash
python -m development.workloads.prefill_shapes --check-decoder runs/workloads19/sources/full > decoder-check.json
python -m development.workloads.prefill_shapes --model Kimi-K3 --seq 131072 --tp 8 --dp 1 --ep 8 --chunk 2048 --step 511 > kimi-last.json
python -m development.workloads.prefill_shapes --model DeepSeek-V4-Pro --seq 262144 --tp 1 --dp 8 --ep 8 --ep-rank 3 > pro-whole.json
```

`scenario` 记录本次的输入与步数，`contracts[].layers` 表示发生该项的零基层号；input、output 表示边界。`weight_logical` 是线性层 [out,in] 权重，分组时权重首维是组数；wo_a 输入和输出采用 [T,本地组,宽]，与运行时 transpose batchmatmul 的置换对应。`standalone_call` 可以直接传给系统后端，指定 chip 和 timing_mode=resource。`standalone_prediction.<chip>.us` 的单位为设备微秒；无合格时间时返回 null 与原因。970DT 仅为芯片配置外推。源程序的 term 是否融合及权重布局仍需运行时证据。部署若获得每层路由直方图，可逐层调用同一导出器并只保留相应层的结果。每个 JSON 整数数组必须恰有 E 个元素，所有元素的和必须为全组路由 assignment 总数。

系统仿真器可以流式读取 gzip 归档，先选请求调度，再选与运行图对应的独立调用。下例用 Kimi 的 128k 整批输入重新估算 970DT 上一次 beta sigmoid。

```python
import gzip, json
from operator_model import estimate

with gzip.open("results/v0.20/workloads/contracts.jsonl.gz", "rt") as source:
    cases = (json.loads(line) for line in source)
    run = next(c for c in cases if c["scenario"]["model"] == "Kimi-K3"
               and c["scenario"]["seq"] == 131072 and c["scenario"]["tp"] == 8
               and c["scenario"]["chunk"] == 0)
call = next(r["standalone_call"] for r in run["contracts"] if r["op"] == "kda.beta.sigmoid")
event = estimate(**call, chip="Ascend970DT", timing_mode="resource")
print(event.latency_us, event.status)  # 约 10.66 µs，extrapolated
```

一次量测中，对清单内去重的 140 种算子/shape/属性，三芯片各调用公共 API 三次。覆盖长 sequence、宽 hidden、不同 AttnRes 有效槽数，以及无定价返回。在同一 Python 进程内计时，绕过导出器结果缓存，不含解释器导入、JSON 解压和 CA 仿真。各算子的墙钟中位数、最大值及对应 shape 可在结果中复核。[查询成本](../results/v0.20/workloads/query-cost.json)

发布 wheel 还在源码目录外、隔离解释器内直接调用这批清单和历史算子，20 个公开算子路径、显式无效输入以及三种芯片共 1,089 个不同 API 调用。中位数 0.083 ms，P95 0.548 ms，最大 2.933 ms。检查项含 v0.20 RMSNorm 冻结的双端和外推数值、状态、活动核数；部署包可重放所有已导出的设备预测。[独立安装与兼容性结果](../results/v0.20/wheel-smoke.json)

真实大 shape 的 CA 采集优先选 Flash 的 indexer 输出转 FP32。32k 场景单次复制 8,388,608 个元素，输入 16 MiB、输出 32 MiB。128k 和 256k 将单次元素数增至 4 倍和 8 倍。每条采集状态以及 CA 与设备投影的同口径对照以专项报告为准。两端归档缺失的 shape 继续使用外推标签；覆盖这些 token 数的清单本身只证明维度成立。[大 shape 的实际采集与预算](../results/v0.20/large/report.md) · [总误差分析](error-analysis.md)
