# operator-sim

**operator-sim** 是面向系统仿真的昇腾算子建模后端。输入一张卡实际执行的算子、shape、dtype 和芯片配置，返回设备耗时、Cube / Vector 工作量、逻辑访存量与活跃核数。能匹配库实现的路径用 CA 校准；捕获图中的研究算子按明确的分核、分块和流水推演。在线预测只依赖 Python 3.10+ 标准库，可在没有昇腾硬件和 CANN 的系统仿真主机使用。

[v2.0.0 发布包](https://github.com/Runze-Lin/operator_sim/releases/tag/v2.0.0) 含系统接入用的 Python wheel、源码包和 [46 页论文版报告（PDF）](docs/operator-sim-report-zh.pdf)。报告从指令时标推导阶段价格与整窗，给出 950DT/PR 的验证、970DT 外推的敏感性和 33 组系统结果。报告中的 v1.13.1 是本次发布采用的冻结模型版本。[分章正文与复算指南](docs/paper/README.md)

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
print(r.rates["logical_roofline"]["time_factor"])  # 约 0.499，相对逻辑 Roofline 基准
print(r.rates["physical_roofline"]["known_lower_bound_us"])  # 可核实的计算下界，µs
print(r.work["cube_ops"], r.work["logical_read_bytes"], r.work["logical_write_bytes"])
```

批量仿真在同一进程复用模型和芯片配置。已知 kernel 发起顺序时，为每卡每条流建立一个 `OperatorStream`，让 RMSNorm 和 mHC Post 等算子看到同流前序。系统框架负责跨流并发和设备间调度；全图入口还可给 TP/EP 通信提供本卡价格。[接入契约](docs/integration.md) · [可运行的调度器示例](examples/system_sim.py)

在 SIMPlatform 的任务 TOML 中设置 `[prediction]` 下的 `backends = ["operator_sim_full"]`，即可按实际 shape、精度、并行切分和开销估计每张图。当前已核对原 27 组 decode，以及三个模型、四种上下文长度的 prefill 与 decode。新增矩阵覆盖 86 个原始调用名，包括有 dtype/布局要求的计算、跨卡路由、零设备时间的视图及分配。原 Roofline 或按原生合同选择的混合后端仍可单独运行。[三种后端的选法](docs/sim-platform.md) · [27 组比较和逐算子账本](docs/system-full-comparison.md)

## 从哪里读代码

| 你要找的内容 | 入口 |
|---|---|
| 一次预测如何选择实现、计算时间并组装返回值 | [backend.py](src/operator_model/backend.py) |
| Matmul、FA、quant、RMSNorm、mHC、KDA、MoE 等具体算子的公式与实现 | [独立算子源码](src/operator_model/operators/) · [系统捕获算子的分族实现](src/operator_model/system_routes/) · [建模方法与逐算子算例](docs/modeling-strategy.md) |
| 同流前序、分核和流水调度、工作量与速率 | [stream.py](src/operator_model/stream.py) · [scheduling/](src/operator_model/scheduling/) · [workload.py](src/operator_model/workload.py) |
| 添加芯片或检查校准参数 | [芯片配置](src/operator_model/data/chips/) · [模型参数](src/operator_model/data/parameters/) · [自定义芯片示例](examples/chip.json) |
| 各算子的耗时误差、逐 shape 真值、验证边界 | [误差分析、逻辑 Roofline 系数与物理下界](docs/error-analysis.md) · [每个算子的全部 shape](docs/errors/README.md) · [可复算的参考数据](calibration/reference/) |
| FA 的低位 KV cache、量化/反量化与 MoE 不同权重精度 | [精度组合、库入口与验证状态](docs/precision-coverage.md) |
| 三种模型在 128Ki decode、三种并行配置和 B=1/16/64 的实际系统算子覆盖 | [全部调用、几何、证据与定价](docs/system-full-comparison.md) |
| 32Ki–1Mi 输入、1Ki 输出的 prefill/decode 对比，以及原 27 组 decode 复算 | [33 组请求与长上下文结果](docs/context-sweep.md) · [Flash 候选页序、团队流水与逐窗 CA](docs/errors/flash-a5-qsli.md) · [安装与后端选择](docs/sim-platform.md) · [原 27 组审计](docs/roofline-audit.md) |
| 按候选 ID 与 PA 表估算 Flash 原生 MXFP4 Indexer，或只传 shape | [输入、公式与末块算例](docs/examples/flash-a5-qsli.md) · [同形同芯片的地址实验](calibration/candidate_indexer/v113_control.md) |
| 从三种模型的 config 推导单卡 prefill shape | [工作负载示例](examples/workloads/) |
| 开发新算子、重新校准、构建和测试 | [维护指南](CONTRIBUTING.md) · [CA 校准工具](calibration/) |

单算子默认 `memory_policy="ca_window"`，设备时间可与相同计时口径的 CA 比较。系统全图默认 `cold_io`，按冷读并物化写回后的设备时间调度。若系统需要为每次 kernel 发射增加外部开销，配置 `kernel_overhead_us` 后使用 `scheduled_us`。UB、L1、L2 容量可按设备设置。从 1.2.0 起，稠密 Matmul 可通过 `estimate` 选择原生 A4W4，或选择在核内展开权重的 A8W4。970DT 的设备耗时按目标配置外推。[芯片配置和精度支持表](docs/integration.md) · [MXFP4 Matmul 调用与算例](docs/examples/mxfp4-design.md) · [原生窄尾波](docs/examples/mx-native-tail.md) · [混合精度多 K 页交接](docs/examples/mx-handoff.md)

## 如何解读预测

950DT 和 950PR 用于 CA 校准；970DT 是按给定规格和继承的微架构参数进行的情景外推。改变核数、峰值吞吐、频率或可用带宽会改变资源定价，模型沿用已建模的实现策略。对接时请保留返回的 `status` 和 `reason`。含融合、量化、前序或布局约束的算子需要匹配实际调用合同。

参考集登记 36 个合同、2,180 个模型与 CA 对照窗。连续 FP4 量化、逐 token FP8 量化、显式 FP4 cache 展开、原生路由 A4W4 专家乘均在 950DT/PR 各验证 10 个新的独立 shape。稠密 Matmul 和其他已校准算子的误差、未匹配的精度、额外的实测和 970DT 外推可逐项查询。长上下文捕获的 86 个原始调用名和覆盖次数有单独账单。[长上下文覆盖账单](calibration/reference/context-950dt/coverage.csv)[逐算子误差](docs/error-analysis.md) · [系统图到设备实现的对应关系](docs/system-workload-coverage.md)
