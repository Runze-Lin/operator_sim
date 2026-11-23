# operator-sim

**operator-sim** 是面向系统仿真的昇腾算子建模后端。输入一张卡实际执行的算子、shape、dtype 和芯片配置，返回设备耗时、Cube / Vector 工作量、逻辑访存量与活跃核数。模型根据库实现推导分核、分块和流水重叠。在线预测只依赖 Python 3.10+ 标准库，可在没有昇腾硬件和 CANN 的系统仿真主机使用。

## 安装与调用

从本仓库根目录安装当前源码。使用已发布的 wheel 时，在系统框架的 Python 环境中安装对应版本。可配置的发射开销和 `scheduled_us` 接口从 1.1.0 开始提供。[已发布安装包](https://github.com/Runze-Lin/operator_sim/releases)

```sh
python -m pip install .
operator-sim --op matmul --shape 1280 2048 512 --dtype bf16 --chip Ascend950DT_9582
```

在系统框架原来调用 Roofline 的地方，可以直接接入 Python API。下面的矩阵形状依次为 M、N、K。返回值的时间单位是**微秒**。

```python
from operator_model import estimate

r = estimate("matmul", (1280, 2048, 512), dtype="bf16", chip="Ascend950DT_9582")
if r.latency_us is None:
    raise ValueError(f"{r.status}: {r.reason}")
print(r.status, r.latency_us, r.active_aic)
print(r.work["cube_ops"], r.work["logical_read_bytes"], r.work["logical_write_bytes"])
```

批量仿真在同一进程复用模型和芯片配置。已知 kernel 发起顺序时，为每卡每条流建立一个 `OperatorStream`，让 RMSNorm 和 mHC Post 等算子看到同流前序。框架继续负责通信、跨流并发和设备间调度。[接入契约](docs/integration.md) · [可运行的调度器示例](examples/system_sim.py)

## 从哪里读代码

| 你要找的内容 | 入口 |
|---|---|
| 一次预测如何选择实现、计算时间并组装返回值 | [backend.py](src/operator_model/backend.py) |
| Matmul、FA、quant、RMSNorm、mHC、KDA、MoE 等具体算子的公式与实现 | [按算子族组织的源码](src/operator_model/operators/) · [建模方法与逐算子算例](docs/modeling-strategy.md) |
| 同流前序、分核和流水调度、工作量与速率 | [stream.py](src/operator_model/stream.py) · [scheduling/](src/operator_model/scheduling/) · [workload.py](src/operator_model/workload.py) |
| 添加芯片或检查校准参数 | [芯片配置](src/operator_model/data/chips/) · [模型参数](src/operator_model/data/parameters/) · [自定义芯片示例](examples/chip.json) |
| 各算子的耗时误差、逐 shape 真值、验证边界 | [误差分析](docs/error-analysis.md) · [可复算的参考数据](calibration/reference/) |
| FA 的低位 KV cache、量化/反量化与 MoE 不同权重精度 | [精度组合、库入口与验证状态](docs/precision-coverage.md) |
| 从三种模型的 config 推导单卡 prefill shape | [工作负载示例](examples/workloads/) |
| 开发新算子、重新校准、构建和测试 | [维护指南](CONTRIBUTING.md) · [CA 校准工具](calibration/) |

单算子接口的 `latency_us` 可直接与 CA 比较。若系统需要为每次 kernel 发射增加外部开销，配置 `kernel_overhead_us` 后使用 `scheduled_us`。UB、L1、L2 容量可按设备设置。从 1.2.0 起，稠密 Matmul 可通过 `estimate` 选择原生 A4W4，或选择在核内展开权重的 A8W4。970DT 的设备耗时按目标配置外推。[芯片配置和精度支持表](docs/integration.md) · [MXFP4 Matmul 调用与算例](docs/examples/mxfp4-design.md) · [原生窄尾波](docs/examples/mx-native-tail.md) · [混合精度多 K 页交接](docs/examples/mx-handoff.md)

## 如何解读预测

950DT 和 950PR 用于 CA 校准；970DT 是按给定规格和继承的微架构参数进行的情景外推。改变核数、峰值吞吐、频率或可用带宽会改变资源定价，模型沿用已建模的实现策略。对接时请保留返回的 `status` 和 `reason`。含融合、量化、前序或布局约束的算子需要匹配实际调用合同。

参考集含 28 项合同、2,076 个设备窗。稠密 A4W4、A8W4 各有 14 个开发回放几何和 10 个采前冻结的新投影几何，均在 950DT/PR 两端测量。覆盖 M=1–8192、N 至 8192、K 至 7168；耗时、活跃核数和 DT/PR 加速比分别统计。原有 26 项合同中有 24 项具备双端完整窗，其中 23 项在两端同时满足 WAPE ≤25%、最大相对误差 ≤50%。同一物理窗可以分别评价仅 shape 和有序流入口。内置 mHC Post 在仅给 shape 时会混合冷、热启动；已知顺序的系统可使用有序流。大 shape 的实测范围、缺失整窗及 970DT 外推均在误差手册内逐项标明。[逐算子汇总](docs/error-analysis.md)
