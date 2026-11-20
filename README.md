# operator-sim：昇腾算子耗时预测后端

`operator-sim` 给系统仿真器提供单卡算子的**设备耗时（µs）、Cube / Vector 工作量和逻辑读写字节**。调用方传入某个 rank 的算子、shape、dtype、调用属性和目标芯片配置，模型按库实现的分块、分核和流水依赖估时。可以在原来调用 Roofline 的地方接入它；每次预测只运行 Python，无需 NPU、CANN 或调用 CA 仿真器。

| 调用方要了解什么 | 入口 |
|---|---|
| 怎样安装、传入 shape、判断结果可用并接到事件调度器？ | [系统仿真接入指南](docs/integration.md) |
| 某个算子为什么需要这么久？不同 shape 如何改变结果？ | [建模方法：Vector、Cube、CV 三类流程，含逐算子公式与算例](docs/modeling-strategy.md) |
| 哪些算子、shape 和库实现实测过？偏差多大？预测有多快？ | [CA 校验与误差：逐算子摘要、逐 shape 数据及查询墙钟](docs/error-analysis.md) |

第一版已可作为系统仿真的算子耗时后端接入。内置 mHC Post 在只提供 shape 时，冷启动、扩核和紧邻复跑共用一个点预测，这是已接受的精度限制。系统知道每卡每流的调用顺序时，使用 `OperatorStream` 可显著降低这项误差。大 shape 双端实测覆盖与 970DT 外推资格逐项标注。[第一版校准与已知限制](docs/error-analysis.md) · [同形两次调用的算例](docs/examples/mhc_post.md)

各算子在 DT、PR 上分别核验 WAPE 和最大误差；偏快、偏慢用同窗带符号偏差与偏快占比描述。Matmul 按目标 L1、输出几何和工作集自动选择分 K、普通多页或权重常驻的多波排程。[逐算子两端误差与方向](results/v0.35/acceptance.md)

当前 KDA 验收包含 recurrent 单状态链与 chunk FLA V2。整套模型有 21 条验收路线，双端大 shape 覆盖按同一几何、同一实现的完整设备时间单列。[当前误差表](docs/error-analysis.md) · [逐算子大形覆盖](results/v0.35/large/report.md)

短 query、长 KV 的 Lightning Indexer 按实际 query 分组、每页 MMAD 与每行 TopK 排程。真实 Kimi AttnResFwd 可按 UB 容量为驻留分支预测时延；大 prefill 的接口会标注当前证据为外推。[Indexer 流水算例](docs/examples/lightning_indexer.md) · [AttnRes 的输入、输出与适用层](docs/examples/attn_res_fwd.md)

单卡 B8 整批 prefill 的工作负载表按 32Ki、128Ki、256Ki 给出每个算子的输入、dtype、并行切分、两卡 CA 实测和 970DT 外推。矩阵投影、indexer 系数与全 hidden 激活各有自己的宽度。[DeepSeek V4.1 Flash](docs/workloads/deepseek-flash.md) · [DeepSeek V4 Pro](docs/workloads/deepseek-pro.md) · [Kimi K3](docs/workloads/kimi-k3.md)

普通 decode 的每卡 B 为 1、9、11、16 时，M 分别为 1、9、11、16。同一批请求做 MTP8 主模型校验时，M 分别为 9、81、99、144。Flash、Pro 和 Kimi 的 N、K 取各自的投影宽度。同一几何复用设备证据，单卡 batch 和每请求 token 数分别保留。[三模型 decode 的 batch、投影与实测](results/v0.32/matmul/decode.md) · [大 prefill 的分块与实测](results/v0.32/matmul/report.md) · [32Ki 到 256Ki 的斜率](results/v0.32/matmul/scaling.md) · [安装库的 host 分支核验](results/v0.30/matmul-dispatch/report.md)

在 `estimate("matmul", [M,N,K], chip=...)` 中，窄输出、长 K 的投影可按目标核数重选输出块，再计算部分积与 FP32 归并。对 Pro 的 B11×9 校验，950DT 的预测 / CA 是 6.475 / 7.056 µs。对 Kimi 多头 recurrent decode，普通查询会给出独立状态链的目标 AIV 分配、工作量和条件时延，并标明外推状态。[Matmul 逐步计算与指令图](results/v0.31/matmul/device/example.md) · [Kimi 大形算例](docs/examples/kda_recurrent.md)

Flash B8×128Ki 的 Matmul 有 1,048,576 行。950DT 的预测 / 完整 CA 为 **409.270 / 405.617 µs**，950PR 为 **847.926 / 804.867 µs**。PR 的最后 8 个输出块各拆成 3 片，全芯片 65,792 条 MMAD 与 trace 一致。970DT 在相同库策略下外推 **87.608 µs**。这些窗口参与了开发；64Ki、256Ki 及不同投影宽度的冻结样例分别跟踪。[逐波模型与 CA 同轴算例](results/v0.32/matmul/method.md) · [大形覆盖与采集进度](results/v0.32/large/report.md)

逐算子算例从一次调用的输入算到设备耗时。Matmul 算例将两波输出块的各页 K 逐行对齐，可以同时看预测和 CA 的首条 MMAD、重叠和末次写回。[Matmul](docs/examples/matmul.md) · [DynamicMXQuant](docs/examples/quant.md) · [FA](docs/examples/fa.md) · [大 prefill Cast](docs/examples/cast_bf16_fp32.md) · [其余算例](docs/pipeline-comparisons.md)

## 试一次预测

在仓库根目录用系统仿真器的 Python 环境安装（Python 3.10 或更新版本）。安装后可在任意工作目录查询。下面的 `9` 表示这个 rank 上的一次 RMSNorm 同时处理 9 个 token，`7168` 是每个 token 的通道数。

```sh
python3 -m pip install .
operator-sim --op rmsnorm --shape 9 7168
```

要对另一款芯片做同一调用，可以在系统仿真器启动时加载芯片 JSON，然后在同一个进程反复调用后端。可以按内置名加载 950DT、950PR 或 970DT 情景。以下示例输出一次本卡 Matmul 的微秒数、Cube 工作量、逻辑输入字节和计时状态。

```python
from operator_model import estimate, load_chip

chip = load_chip("Ascend950PR_957b")
result = estimate("matmul", (128, 256, 512), dtype="bf16", chip=chip)
print(result.latency_us, result.work["cube_ops"], result.work["logical_read_bytes"], result.status)
```

**比较芯片架构时可显式指定 `timing_mode="resource"`。** Quant、RMSNorm、Matmul 和 FA 会按目标配置重新决定分核、tile、读写字节、计算与流水时间。普通 Matmul 和 FA 在 DT、PR 和自定义配置上都默认走资源排程。RMSNorm 按目标资源和前序选择排程。Quant 在原 DT 校准域可走参考模型，在 PR、自定义配置或超出该域时走资源排程。新增的 sigmoid 专家路由、BF16→FP32 Cast 和 Kimi 融合残差也按目标资源估计。其余路线沿用 DT 阶段投影，或仅返回工作量。相同配置更换名称时，显式资源模型给出相同预测。[资源公式与算例](docs/modeling-strategy.md) · [独立 PR 校验与误差](docs/error-analysis.md)

维护 960、970、980 等架构情景时，每个目标使用一份 JSON，分别填写按位宽区分的两类算力、核数、频率、外存容量和带宽。内置 `Ascend970DT` 情景为 120 Cube / 120 Vector、1.8 GHz、288 GB、14.4 TB/s；其中 16 位 Vector 峰值 110.592 TOP/s 已含双发射。这些是架构假设。调用方通过 `extrapolated` 状态识别外推，并决定是否将其交给事件调度器。[970DT 配置及逐项来源](configs/chips/Ascend970DT.json) · [自定义芯片示例](configs/chips/scenario-example.json) · [字段、单位和返回合同](docs/integration.md)

安装版的 3,958 次进程内调用实测中位 0.174 ms，P95 为 3.692 ms，最大 18.382 ms。计时包含单次算子估计或有序流推进，进程启动与离线 CA 采集分别记录。[查询开销](results/v0.34/wheel-check.json) · [按算子归档的双端误差](results/v0.35/acceptance.md)

## 可预测的算子

| 方法 | 路线 |
|---|---|
| Vector | DynamicMXQuant、RMSNorm、mHC Sinkhorn、mHC Post、KDA recurrent、Kimi beta FP32 Sigmoid、MoE sigmoid Top-K、BF16→FP32 Cast、Kimi AttnResFwd 单 token/核驻留分支 |
| Cube | Matmul、Addmm、MoE GMM2 |
| Cube + Vector（CV） | FA、Lightning Indexer、mHC Pre / PreSinkhorn、mHC 组合、GMM1–SwiGLU–MX 量化及 MoE FFN 组合 |

三个模型的 B8 整批与 chunked prefill 输入可直接查每个 rank 的张量形状、dtype、层号和适用的独立 kernel。覆盖每请求 32k、128k、256k，以及 attention TP8 或 DP8、MoE EP8 两种拓扑。[逐模型真实维度与导出器](docs/prefill-shapes.md) · [场景索引](results/v0.20/workloads/index.csv) · [工作负载覆盖](docs/workload-coverage.md)

Kimi 的 `plan_kimi_kda_prefill` 根据每条请求本次提交的 token 数和 attention TP 选择 V1/V2。FLA 26.9.1 三段 KDA V2 可显式预测阶段耗时、工作量和分核，已在 DT、PR 用新 shape 实测。默认小形使用的融合 V1，完整耗时仍为 `None`。DeepSeek 扩展 `aclnnHcPost` 可显式预测分核、分块与设备耗时；存在数值失败的短尾片仅返回工作量。组合路线与组成它的单个 kernel 在同一调用中只计一次。[KDA 冻结模型与 CA 同轴图](docs/examples/kda_chunk.md) · [DeepSeek HcPost](docs/examples/hc_post_vllm.md) 各路线的 dtype、布局和分支限制见[建模方法](docs/modeling-strategy.md)。模型配置、本地 batch、TP/EP 切分和实际路由共同决定输入 shape，见[如何推导每卡 shape](docs/model-configs.md)。

## 硬件与计时边界

主要校准参考是 **Ascend950DT_9582（A5），CANN 9.2.1，32 AIC / 64 AIV，1.65 GHz**。参考值是安装包实际 kernel 在 CA Model 上从首条到末条设备指令的跨核时间窗；串行组合返回各窗口之和。主机 launch、卡间通信、排队及算子间并发由系统仿真器处理。报告的查询开销是运行预测器花费的真实墙钟时间，与 `predicted_us` 使用的是两条不同的时间轴。

**Ascend950PR_957b** 有 28 AIC / 56 AIV，1.65 GHz，名义片外带宽 1.4 TB/s；DT 为 4.0 TB/s。核数正确时仍可能因每核的发起、写回和共享带宽产生时延偏差。精度按具体库路线与 shape 报告；同一 shape 的两 SKU 实测、开发回放、冻结后留出和 970 外推都标明数据角色。对于 960/970/980 的外推，假设软件实现仍采用当前库路线。[逐算子误差与硬件差异](docs/error-analysis.md) · [970 场景下 FA 的逐 shape 瓶颈](results/v0.17/fa-scenarios.md)

维护模型、复现 CA 或查阅历史版本，请从[开发与验证档案](development/README.md)进入。
