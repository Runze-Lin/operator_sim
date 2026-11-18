# operator-sim：昇腾算子耗时预测后端

`operator-sim` 给系统仿真器提供单卡算子的**设备耗时估计（µs）**。调用方传入某个 rank 实际执行的算子、shape 和属性，模型按库实现的分块、分核和流水依赖估时。可以在原来调用 Roofline 的地方接入它；每次预测只运行 Python，无需 NPU、CANN 或调用 CA 仿真器。

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

CLI 输出 JSON，`predicted_us` 是这段设备工作的估计微秒数。系统仿真器应在自己的 Python 进程内复用后端，无需为每个事件启动 CLI。下面是从 Roofline 接入的一个最小回调；本例只接受 `estimated`，把外推和缺少合格计时的情况交回原有后端。

```python
from operator_model import estimate

def device_time_us(op, local_shape, attrs, soc, roofline_us):
    result = estimate(op, local_shape, attrs, soc=soc)
    if result.status == "estimated":
        return result.latency_us
    return roofline_us(op, local_shape, attrs)
```

`soc` 传入当前模拟设备的型号；`roofline_us` 由调用方提供，返回值同样使用 µs。若要单独记录外推或区分“算子不支持”和“参考计时不可用”，见[接入指南](docs/integration.md)。

## 可预测的算子

| 方法 | 路线 |
|---|---|
| Vector | DynamicMXQuant、RMSNorm、mHC Sinkhorn、mHC Post、KDA recurrent |
| Cube | Matmul、Addmm、MoE GMM2 |
| Cube + Vector（CV） | FA、Lightning Indexer、mHC Pre / PreSinkhorn、mHC 组合、GMM1–SwiGLU–MX 量化及 MoE FFN 组合 |

KDA chunk 可查询任务结构；其 CA 同步检查尚未通过，系统层应单独处理该路线的耗时。组合路线和组成它的单算子是两种记账粒度，一条路径上只能选一种。各路线的 dtype、布局和分支限制见[建模方法](docs/modeling-strategy.md)。模型配置、本地 batch、TP/EP 切分和实际路由共同决定输入 shape，见[如何推导每卡 shape](docs/model-configs.md)。

## 硬件与计时边界

校准目标是 **Ascend950DT_9582（A5），CANN 9.2.1，32 AIC / 64 AIV，1.65 GHz**。参考值是安装包实际 kernel 在 CA Model 上从首条到末条设备指令的跨核时间窗；串行组合返回各窗口之和。主机 launch、卡间通信、排队及算子间并发由系统仿真器处理。报告的查询开销是运行预测器花费的真实墙钟时间，与 `predicted_us` 使用的是两条不同的时间轴。

误差数据针对已列出的 A5 路线和 CA 参考，不代表整网时延、实际芯片效率或 A7/A8 的精度。移植到另一代芯片需要确认分核、指令与访存成本以及实现分支，再用目标平台数据验证。

维护模型、复现 CA 或查阅历史版本，请从[开发与验证档案](development/README.md)进入。
