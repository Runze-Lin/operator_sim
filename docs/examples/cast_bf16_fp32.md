# Cast · Flash 全 hidden 的 165 MiB 转换

输入 5632×5120=28,835,840 个 BF16，读 55 MiB、写 110 MiB。输入与输出流量比 1:2。

## 单卡输入与调用

```python
from operator_model import estimate

shape = [5632, 5120]
attrs = {}
result = estimate('cast_bf16_fp32', shape, attrs, chip='Ascend950PR_957b')
print(result.latency_us, result.active_aic, result.active_aiv)
print(result.work["logical_read_bytes"], result.work["logical_write_bytes"])
```

## 从工作量到设备时间

在 PR 的 56 个 AIV 上，关键核处理 514,928 个元素，每个 UB 块容量 21,120，合计 25 个块。两个读槽可同时有请求在途。稳态检查读、写、转换和释放缓冲中最慢的条件，再与连到 HBM/L2 的服务下界比较。

公开工作量统计为 Cube **0**、Vector **28,835,840** 个有用标量等效操作。逻辑读 **57,671,680 B**、写 **115,343,360 B**。片外实际流量取决于重用与缓存。

双输入和双输出占用 UB，tile 元素上限按 64 对齐为 64⌊UB/[2×(2+4)×64]⌋。参考核的 64 个转换组占用 113 tick。两个槽的在途响应分别是对应纯读写服务的 1.439 与 1.975 倍，由 DT 的整块描述符中位数换算。

下表取 950PR 的公开阶段分账。各段已处理模型可辨识的重叠，数值保留三位小数。

| 阶段字段 | 暴露时间 µs |
|---|---:|
| `arrival` | 0.548 |
| `control_to_read` | 0.828 |
| `read` | 0.956 |
| `vector` | 0.363 |
| `dependency_to_write` | 0.508 |
| `write` | 0.921 |
| `steady_tiles` | 25.514 |
| `shared_io` | 25.244 |
| **整窗** | **54.881** |

## 本例各阶段怎样定价

启用 56 个 AIV，忙核 514928 个元素。双输入、双输出占 12 B/元素，UB 253952 B 给每块容量 64⌊253952/(12×64)⌋=21120 元素，共 25 块。下面的首块实测截距、端口每 KiB 周期及双槽响应因子均来自 DT 的已标记 Cast/同类型 DMA 事件。

- 到达展宽 (1035/63)×55=903.57 周期，控制至读为 1367。首读 R=817.83+(2×21120/1024)×18.39635=1576.68 周期；首写 W=198.71+(4×21120/1024)×16.01826=1520.21。向量 V=(113/64)×330×1.02717=598.49，写前依赖为 838。
- 每个后续满块的输入槽周转为 (1.43942R+V)/2=1433.99，输出槽为 (1.97505W+V)/2=1800.49 周期。剩余 493808 个元素等价于 23.381 个满块。比较这两个槽约束以及读端口、写端口与向量的持续服务后，`steady_tiles`=23.381×1800.49/1650=25.514 µs。
- 共享链路按 L2 可留的写回和必经外存的数据定价，`shared_io`=max(0,90553.24/1650−已计各段时间)=25.244 µs。所有读写对 DRAM 的需求约 75.35 MB。整窗 **54.881 µs**，其中包含共享约束暴露出来的时间。

[Cast 首块与双槽表达式](../../src/operator_model/operators/vector/cast.py) · [缓存保留和公共带宽](../../src/operator_model/scheduling/memory.py)

## 与完整 CA 核对

| 目标 | 公开预测 µs | 完整 CA µs | 有符号误差 | 预测有效 AIC / AIV |
|---|---:|---:|---:|---:|
| Ascend950DT_9582 | 27.431 | 29.062 | -5.61% | 0 / 64 |
| Ascend950PR_957b | 54.881 | 55.220 | -0.61% | 0 / 56 |
| Ascend970DT（外推） | 14.688 | N/A | N/A | 0 / 120 |

PR 的有效持续写带宽与大容量脏输出在 L2 中的驻留决定出口。该例总量 165 MiB，不能从逻辑写量直接断言所有 110 MiB 已在设备退休前写入 HBM。

示例收据的调用为 `aclnnCast`，provider 为 CANN 9.2.1 installed OPP，实际记录的 kernel 为 `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0`。

复核时，在逐窗记录中筛选 `contract=cast_bf16_fp32`、此页 shape 和属性。有多个前序的样本逐窗保留。示例用于解释该次计算。校准统计、冻结后独立样本数及完整误差范围以总报告为准。[逐窗数据](../../calibration/reference/windows.jsonl) · [模型实现](../../src/operator_model/operators/vector/cast.py) · [总建模方法](../modeling-strategy.md) · [误差分析](../error-analysis.md)
