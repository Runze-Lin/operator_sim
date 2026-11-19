# 三个文本模型的无权重 decoder 假跑

这一页回答“模型走到了哪条算子路径”。我们执行固定版 vLLM-Ascend 的 Python decoder 方法，把权重、张量及叶算子替换为只记录调用和形状的对象。本机执行环境没有 PyTorch、torch_npu、vLLM 或可见的 davinci 设备。模型层面的计数来自这个符号执行器。新增的 Kimi 路由、融合残差和 DeepSeek BF16→FP32 转换分别在 CANN 9.2.1 CA 中检查数值、二进制和设备时间。

| 模型 | 文本主干层数 | 每次走完主干时记录到的调用 | 需要单独定价的路径 |
|---|---:|---|---|
| DeepSeek V4.1 Flash | 40 | 80 次 hc_pre_v3、80 次 hc_post；input RMSNorm、post RMSNorm、BF16→FP32、dsa_v41_forward、FP32 路由打分和专家模块各 40 次；Engram WKV 投影和 fused gate 各 2 次 | mHC v3 的 pre_mix、回退路径的 cast、共享索引与压缩 KV。模型外层还在配置的第 1、14 层（零起始）注入 Engram |
| DeepSeek V4 Pro | 61 | 122 次 hc_pre 和 122 次 hc_post；input RMSNorm、post RMSNorm、BF16→FP32、DSA、FP32 路由打分和专家模块各 61 次 | DSA/压缩缓存、回退路径的 cast，以及 sqrtsoftplus/hash 专家选择 |
| Kimi K3 | 93 | 69 层 KDA、24 层 no-RoPE MLA；186 次 attn_res_fwd、1 层 dense MLP、92 层 MoE 的 FP32 gate、sigmoid top-k 及专家模块 | 残差混合与归一化融合、no-RoPE MLA、独立路由选择。新增校准覆盖 sigmoid top-k 与残差的单 token/核驻留分支 |

这张表中的 attention 与 experts 有些是整个模块的一次调用。模块内部的多次设备启动、通信和跨模块融合仍需在目标部署核对。有用算力也不能由调用次数相加代替。Kimi 的两处残差是主干每层进入注意力前与进入 MLP 前各一次；模型最外层的最终物化另计。两种 DeepSeek 均执行了真实的 rms_norm_cast 辅助函数。锁定版本的 A5 capability 表没有 RMS_NORM_CAST，Flash 取普通 post RMSNorm 后对 BF16 结果调用 float()。Pro 的辅助函数只检查 enable_custom_op；本场景显式关闭这一开关，走相同回退。打开开关会请求 npu_rms_norm_cast，但其注册表只列出 910B/910_93，不能计入 950 已校准融合算子。[设备能力表](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/device/hardware_profile.py)

## 实际执行范围与场景

| 输入场景 | 每个 attention DP 副本的请求数 | 每请求这一轮 q | 整层入口的 token 行数 T | 解释 |
|---|---:|---:|---:|---|
| 单次完整 prefill | 8 | 32768 | 262144 | 假设没有 chunk，总 token 预算至少 262144，一次 forward 的压力上界 |
| chunked prefill | 8 | 256 | 2048 | 示例总预算 2048，八请求等分。同样的 32k 输入需要多个调度轮次 |
| 投机校验 | 8 | 9 | 72 | 已有 32768 上下文，每请求 1 个基础 token 加 8 个草稿 token |

上述 9 次假跑使用三个已锁定的模型配置。token 数由输入明确设定，KV 已有长度作为场景元数据保存。层内采用单 rank、无序列并行、无上下文并行；DeepSeek 在外部 FP32 gate 和返回单个张量的专家路径上运行。Kimi 运行上游路由可用性判断、fused 路由方法和 A5 设备适配方法，并截获其 torch_npu 调用。注意力叶保留整段模块边界。Flash 另按模型外层的两个 Engram 层号运行真实 AscendEngram.forward，假定准备好的 lookup 行和未旋转的权重，执行 A5 的 WKV→fused_engram_gate 路径。每个输入行宽 (4−1)×8×256=6144，投影输出 (4+1)×5120=25600。CPU 查表、传输与其可重叠程度需另采；当前 NPU 算子表未校准 Triton gate 成本。以完整 32k×8 prefill 为例，每个 Engram 层若以 BF16 传输所有 lookup 行，逻辑数据量为 3 GiB；之后的 WKV 投影为 82,463,372,083,200 个标量等价运算（FMA 按 2），这两部分必须分开计价。当前计数覆盖文本主干，视觉 encoder、MTP 草稿网络、服务端调度、通信及路由真实负载分布不在这次假跑中。[逐层调用路径和输入](../results/v0.19/workloads/decoder-probe.json) · [配置与 revision](../configs/models/sources.json)

## Kimi 路由与仿真后端

Kimi 每层 router linear 用 hidden=7168 的输入生成 FP32 logits，896 个专家中选 16 个。独立路由的输入 shape 为 `[T,896,16]`。896 表示全局候选专家数；EP 上一张卡拥有的本地专家数影响后续 dispatch 和专家 GEMM，不能用它缩小 gate 的候选维度。gate linear、top-k、专家计算在调用方的时间线上各自占一个位置。若生产路径已将某个组件融合，调用方只应计入该融合分支实际发起的 kernel。[路由完整算例与校准](examples/moe_gating_top_k.md)

三种工作负载的 router-only 压力外推如下，单位 µs。每个单元格中的第二个数是最忙核处理的 token 行数。这里三代的三种大工作量都属于应用场景推演，没有这些完整 shape 的 CA 标签。

| 单次路由 T | 950DT 预测 / 行 | 950PR 预测 / 行 | 970DT 外推 / 行 |
|---:|---:|---:|---:|
| 72 | 7.19 / 2 | 7.29 / 2 | 4.89 / 1 |
| 2048 | 64.24 / 32 | 78.57 / 37 | 34.21 / 18 |
| 262144 | 7876.86 / 4096 | 9605.15 / 4682 | 3848.24 / 2185 |

最后一行的逻辑读写分别为 **939,527,680 B** 和 **33,554,432 B**。当前路由模型假设批内每核重用一次 bias 并按最忙核行数增长；当行数大时该模型自然接近线性。这三个外推值不能证明实际 kernel 在多轮时保持相同斜率，大输入的调度选择和跨行读写重叠会影响结果。更换 chunk 时须重算每次的行数和固定启动成本。

源程序的主链依次经过 `AscendKimiMoE.forward`、专家 runner、`AscendFusedTopKRouter._compute_routing`、`A5DeviceAdaptor.moe_gating_top_k`。探针运行前后两个 forward/设备分支，专家 runner 用转发到所选路由的测试替身。数值测试直接启动 CANN 的 `aclnnMoeGatingTopK`，逐个核对权重和专家 ID；它验证此 FP32 sigmoid 算子的成本。DeepSeek 的 sqrtsoftplus 以及 hash 分支有不同的属性与实现，须保持独立。[Kimi 层与 gate](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/models/kimi_k3.py) · [路由](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/ops/fused_moe/router/fused_topk_router.py) · [A5 适配](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/device/device_op.py)

## 复现

```bash
python3 development/workloads/decoder_probe.py --fetch
```

脚本仅下载锁定版本的若干 Python 源文件，经 SHA256 校验后提取已审阅的方法执行。已有缓存时可省略 `--fetch`；运行一次不分配与 T 成正比的张量。每个模型各场景完整层序列均保存在输出 JSON，中间各叶的重复次数可以沿 `layer_paths` 核对。[探针代码](../development/workloads/decoder_probe.py) · [源码锁](../development/workloads/sources.json) · [DeepSeek V4.1 实现](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/models/deepseek_v41/model.py) · [DeepSeek V4 实现](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/models/deepseek_v4/model.py)

## 后续校准优先级

优先级依据主干调用频次、输入规模和当前已有证据。表中次数均指一次指定文本主干遍历，单个模块后的设备启动数和在整网耗时中的占比尚未知。

| 优先顺序 | 路径与规模 | 与现有模型的关系 | 已锁定的组件与仍缺的设备证据 |
|---|---|---|---|
| 1 | DeepSeek 稀疏/压缩注意力，Flash 40 层、Pro 61 层。两者滑窗均为 128，index_topk 分别为 512、1024；Flash 还存在跨层 KV/索引复用 | 普通 FA 预测连续稠密 Q/K/V 的 QK、softmax 与 PV 流水。DSA 额外涉及压缩、索引生成或复用、非连续 KV gather 以及滑窗与稀疏分支合并。当前普通 FA 和独立 lightning_indexer 不能给出整个模块的设备时间 | 源码定位到 DSA 与 sparse MLA 路径；CANN 安装包提供 aclnnCompressor、aclnnQuantLightningIndexer、aclnnSparseFlashMla 与量化变体。需固定层角色、cache dtype、目标 950 实际分支和 kernel 序列，端到端耗时未知 |
| 2 | Kimi 每次主干有 69 层 KDA。配置给出 96 heads、每 head 128 维、短卷积核宽 4 | KDA 维护跨 token 的递推状态。decode 主要推进状态，prefill 的 chunk 需要块内计算及块间状态传递。其访存与工作量均不同于对整个 KV 长度做稠密 softmax 的 FA | 已有 aclnnRecurrentKda 的单算子双端证据。aclnnChunkKdaFwd 的完整 CA 时间仍缺，短卷积、gate 和组合路径的实际 kernel 顺序也未由模型假跑确认，整个模块耗时未知 |
| 3 | Kimi 每次主干有 24 层 no-RoPE MLA，kv_lora_rank=512，qk_nope_head_dim、v_head_dim 均为 128，另有输出门控 | 层内还涉及低秩投影、潜变量 KV cache 的读写和输出处理。同形状的普通 FA 只描述 attention 核心的一种参照 | 固定源码定位到 AscendKimiMLAAttention 和 MLA backend，CANN 有 MlaProlog 系列。在 prefill、decode、Q=9 校验和不同 cache 表示下实际发起的 950 二进制仍未知 |
| 4 | Kimi 的融合 attn_res_fwd，186 次；每个 token 的残差 bank 至多 8×7168 元素 | 归一化、投影、跨残差分数和加权与有效 bank 数相关，已有结构模型给出工作量与流量 | 锁定 vLLM-Ascend 的 arch35 kernel 源及自有 ACLNN 合同。当前可核验范围和具体计时状态见[融合残差评估](../results/v0.19/workloads/attention-residual.md) |
| 5 | Flash 的 Engram 共 2 处，每层在完整 32k×8、BF16 场景可有 3 GiB 的逻辑 lookup 行 | 即使频次低，Host→NPU 传输、可重叠窗口、WKV GEMM 和 gate 仍各有成本 | 符号执行已定位 WKV 与 triton.fused_engram_gate。还缺真实驻留策略、lookup 命中与传输 trace、Triton 生成的 exact binary 和设备计时 |
| 6 | DeepSeek 的 sqrtsoftplus/hash 路由、Flash hc_pre_v3，以及 FFN 前 post RMSNorm→cast | Kimi 的 FP32 sigmoid top-k 已有独立双端 CA；不同路由属性、hash 表输入以及 pre_mix 跨层依赖须分别锁定。post RMSNorm→cast 已独立拆分，以便在系统后端复用 | DeepSeek 路由和 mHC 有固定源码，但与当前独立 CA 组件是否为同一库 binary 需在部署栈核对；cast 见[950 路径核对与校准](../results/v0.19/workloads/rmsnorm-cast-assessment.md) |

此顺序可以随实跑的设备时间重新排序。若分离式部署采用不同 prefill/decode 算法、专家并行、KV cache dtype、图编译或联算融合，应分别记录该合同下的 kernel 名和形状。[Flash DSA backend](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/attention/dsa_v41.py) · [Pro DSA backend](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/attention/dsa_v1.py) · [当前 CA 与逐形状误差](error-analysis.md)
