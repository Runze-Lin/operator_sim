# operator-sim：昇腾算子耗时预测后端

`operator-sim` 给系统仿真器提供单卡算子的**设备耗时（µs）、Cube / Vector 工作量和逻辑读写字节**。调用方传入某个 rank 的算子、shape、dtype、调用属性和目标芯片配置，模型按库实现的分块、分核和流水依赖估时。可以在原来调用 Roofline 的地方接入它；每次预测只运行 Python，无需 NPU、CANN 或调用 CA 仿真器。

| 调用方要了解什么 | 入口 |
|---|---|
| 怎样安装、传入 shape、判断结果可用并接到事件调度器？ | [系统仿真接入指南](docs/integration.md) |
| 某个算子为什么需要这么久？不同 shape 如何改变结果？ | [建模方法：Vector、Cube、CV 三类流程，含逐算子公式与算例](docs/modeling-strategy.md) |
| 哪些算子、shape 和库实现实测过？偏差多大？预测有多快？ | [CA 校验与误差：逐算子摘要、逐 shape 数据及查询墙钟](docs/error-analysis.md) |

[RMSNorm 分块与 mHC Post 启动状态的独立校验](results/v0.12/audit/summary.md) 给出大 shape 的实际写回与耗时。固定宽度下逐级增大 token 的数据见[规模实验](results/v0.11/scaling/summary.md)。

## 试一次预测

在仓库根目录用系统仿真器的 Python 环境安装（Python 3.10 或更新版本）。安装后可在任意工作目录查询。下面的 `9` 表示这个 rank 上的一次 RMSNorm 同时处理 9 个 token，`7168` 是每个 token 的通道数。

```sh
python3 -m pip install .
operator-sim --op rmsnorm --shape 9 7168
```

要对另一款芯片做同一调用，可以在系统仿真器启动时加载芯片 JSON，然后在同一个进程反复调用后端。用内置 SKU 名也可以加载 950DT 或 950PR。以下示例输出一次本卡 Matmul 的微秒数、Cube 工作量、逻辑输入字节和计时状态。

```python
from operator_model import estimate, load_chip

chip = load_chip("Ascend950PR_957b")
result = estimate("matmul", (128, 256, 512), dtype="bf16", chip=chip)
print(result.latency_us, result.work["cube_ops"], result.work["logical_read_bytes"], result.status)
```

维护 960、970、980 等架构情景时，每个目标使用一份 JSON，分别填写按位宽区分的两类算力、核数、频率、外存容量和带宽。提供的 120 Cube / 120 Vector 配置用于演示自定义输入。调用方通过 `extrapolated` 状态识别外推，并决定是否将其交给事件调度器。[自定义芯片示例](configs/chips/scenario-example.json) · [字段、单位和返回合同](docs/integration.md)

## 可预测的算子

| 方法 | 路线 |
|---|---|
| Vector | DynamicMXQuant、RMSNorm、mHC Sinkhorn、mHC Post、KDA recurrent |
| Cube | Matmul、Addmm、MoE GMM2 |
| Cube + Vector（CV） | FA、Lightning Indexer、mHC Pre / PreSinkhorn、mHC 组合、GMM1–SwiGLU–MX 量化及 MoE FFN 组合 |

KDA chunk 可查询任务结构；其 CA 同步检查尚未通过，系统层应单独处理该路线的耗时。组合路线和组成它的单算子是两种记账粒度，一条路径上只能选一种。各路线的 dtype、布局和分支限制见[建模方法](docs/modeling-strategy.md)。模型配置、本地 batch、TP/EP 切分和实际路由共同决定输入 shape，见[如何推导每卡 shape](docs/model-configs.md)。

## 硬件与计时边界

主要校准参考是 **Ascend950DT_9582（A5），CANN 9.2.1，32 AIC / 64 AIV，1.65 GHz**。参考值是安装包实际 kernel 在 CA Model 上从首条到末条设备指令的跨核时间窗；串行组合返回各窗口之和。主机 launch、卡间通信、排队及算子间并发由系统仿真器处理。报告的查询开销是运行预测器花费的真实墙钟时间，与 `predicted_us` 使用的是两条不同的时间轴。

**Ascend950PR_957b** 上的 DynamicMXQuant、Matmul 和 FA 各有 10 个冻结后采集的独立 shape。FA 已用正式 PR CA 验到单头 Q=9、KV=32,768；Matmul 包含了触发 24/28 核分块的大矩阵。未校准目标沿用实现假设，用自己的硬件配置重新分块、计价。同一份[误差分析](docs/error-analysis.md)列出两款芯片的核数、分精度算力、频率、存储容量与带宽，并用同一组 PR CA 比较跨芯片预测和直接复用 DT。尚未取得 960、970、980 的 CA 或上板数据，预测采用调用方提供的硬件参数及相近实现假设。逐 SKU 的开发、独立误差和数据缺口见[校验报告](docs/error-analysis.md)。

维护模型、复现 CA 或查阅历史版本，请从[开发与验证档案](development/README.md)进入。
