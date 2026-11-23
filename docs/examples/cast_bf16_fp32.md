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

双输入和双输出占用 UB，tile 元素上限按 64 对齐为 64⌊UB/[2×(2+4)×64]⌋。参考核的 64 个转换组占用 113 cycles。两个槽的在途响应分别是对应纯读写服务的 1.439 与 1.975 倍，由 DT 的整块描述符中位数换算。

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

**从元素到块。** [5632,5120] 共 n=28,835,840 个元素。一个转换读 2 B、写 4 B。2048 元素的最小分核粒度使 56 个 AIV 全部工作。以 16 元素（32 B 输入）对齐，忙核分到 16⌈⌈n/16⌉/56⌉=514928 个。双输入、双输出占 2×(2+4)=12 B/元素，UB 每块容量 C=64⌊253952/(12×64)⌋=21120 个。每核 25 块，第一块以后还剩 m=493808 个。以 950DT 的参考 FP32 向量吞吐作基准，PR 时间倍率 s=[27000/(64×1.65)]/[23000/(56×1.65)]=1.027174。

| 成本 | 推导与代入，单位为 PR cycles |
|---|---|
| 首发及控制 | 从 950DT 多核末核到达差提取 1035/63 cycles/额外核。到达 (56−1)×1035/63=903.571，首读前的控制另计 1367。 |
| 首块读 R | 参考 Cast 的 8 KiB 首读窗为 965，扣除传输得固定价 965−8×18.39635=817.8292。R=817.8292+(2C/1024)×18.39635=1576.679。 |
| 首块算 V | 开发 trace 做 64 个寄存器组占 113 cycles，每组 64 元素。V=(113/64)×⌈21120/64⌉×s=598.489。 |
| 首块写 W | 计算至可写的依赖预算 838；参考 16 KiB 写窗为 455，得固定价 455−16×16.01826=198.70784。W=198.70784+(4C/1024)×16.01826=1520.214。 |
| 后续服务 | 端口可连续工作，剩余读、写和向量分别为 (2m/1024)×18.39635、(4m/1024)×16.01826、⌈m/64⌉×113/64×s，分别为 17742.70、30898.22、13993.77 cycles。双缓冲最多同时持有两个输入槽、两个输出槽；故每块至少要 max((1.439418R+V)/2,(1.975050W+V)/2)=max(1433.995,1800.495) cycles。四个下界取最大值，此处由槽限制给出 (m/C)×1800.495=42097.474。两个无量纲系数来自 950DT 满块 DMA 响应与持续端口成本的比值。 |

**共享外存。** 读 I=2n=57,671,680 B，写 O=4n=115,343,360 B。按 950DT 流式混合读写的 30 B/cycle/AIV 标定，本例可发起 56×30×1.65=2772 GB/s，PR 的外存带宽取 min(2772,1400)=1400 GB/s。缓存可用容量 L=min(I+O,117440512)=117,440,512 B；当前 Cast 采用压力相关的脏输出保留近似。可保留的输入 L×I/(I+O)×min(1,1400/2772)=19,771,130 B，其余 97,669,382 B 可容纳输出。需写出 O−97,669,382=17,673,978 B；外存总流量 I+17,673,978=75,345,658 B。PR 无 L2 带宽价，此处全片下界为 1753 cycles 启动释放+75,345,658×1.65/1400=90553.24 cycles。

首块与后续局部阶段合计 48901.43 cycles，因此 shared_io=max(0,90553.24−48901.43)/1650=25.244 µs。整窗 max(48901.43,90553.24)/1650=**54.881 µs**。本例的留存是缓存占用估计，实际外存流量需用控制器数据辨识。[Cast 分块、价格来由与周转](../../src/operator_model/operators/vector/cast.py) · [共享缓存及发行限制](../../src/operator_model/scheduling/memory.py) · [芯片配置](../../src/operator_model/data/chips/Ascend950PR_957b.json)

## 如何从 CA 得到这些参数

**规格和假设。** 2 B→4 B 和寄存器组来自 Cast；2048 元素的最小分核粒度由 16384 元素实际启用 8 AIV 反推。UB、总线和峰值读目标配置。18.39635/16.01826 cycles/KiB 沿用 RMSNorm 对多种长短 DMA 的参数，Cast 的固定响应则来自本身的终止核。

**从一个 950DT 窗口重算首块。** [64,4096]，64 个 AIV，选 retire 最晚者，剔除同步/等待和头部元数据。该核 8192 B 读 [1.45575738,2.04060584] µs，持续×1650=965 cycles；16384 B 写 [2.61696982,2.89272740] 给 455 cycles。扣除持续服务，得 965−8×18.39635=817.8292 和 455−16×16.01826=198.70784 cycles。尾核起点 0.62727284×1650=1035；全片首个 payload 发起 1.06242442×1650=1753；尾核启动至有效读 (1.45575738−0.62727284)×1650=1367。64 组向量的 SIMD 并集 0.06848481×1650=113。读就绪至写发起扣除此并集，(2.61696982−2.04060584)×1650−113=838 cycles。控制/依赖是这些事件之间的剩余时间。[终止核描述符与 trace 摘要](https://github.com/Runze-Lin/operator_sim/blob/2059edce197af60ddee6cfbc7c78d40c715ed76f/results/v0.19/workloads/cast-phases.json) · [终止核筛选及 SIMD 并集](https://github.com/Runze-Lin/operator_sim/blob/2059edce197af60ddee6cfbc7c78d40c715ed76f/development/workloads/cast_phases.py)

**双槽与缓存。** 950DT [262144,32] 末核满块读/写描述符在途中位数约 1.3755/1.8197 µs，除以相同满块的原模型读/写服务 0.95556/0.92134 µs，得发布比值 1.43942 和 1.97505。外存混合发起 30 B/cycle/AIV 是 DT 控制器流量与满块发起间隔折算的圆整上限。PR 压力留存公式、额外给脏输出腾出缓存以及向量峰值重定价是推断；本算例的外存字节数应当按该假设解释。[原始在途/服务及实际控制器口径](https://github.com/Runze-Lin/operator_sim/blob/2059edce197af60ddee6cfbc7c78d40c715ed76f/results/cast-capacity/report.md) · [冻结比值和训练 shape](https://github.com/Runze-Lin/operator_sim/blob/2059edce197af60ddee6cfbc7c78d40c715ed76f/results/cast-capacity/two-slot-freeze.json) · [控制器按区间字节数](https://github.com/Runze-Lin/operator_sim/blob/2059edce197af60ddee6cfbc7c78d40c715ed76f/results/cast-capacity/controller-periods.json)

## 与完整 CA 核对

| 目标 | 公开预测 µs | 完整 CA µs | 有符号误差 | 预测有效 AIC / AIV |
|---|---:|---:|---:|---:|
| Ascend950DT_9582 | 27.431 | 29.062 | -5.61% | 0 / 64 |
| Ascend950PR_957b | 54.881 | 55.220 | -0.61% | 0 / 56 |
| Ascend970DT（外推） | 14.688 | N/A | N/A | 0 / 120 |

PR 的有效持续写带宽与大容量脏输出在 L2 中的驻留决定出口。该例总量 165 MiB，不能从逻辑写量直接断言所有 110 MiB 已在设备退休前写入 HBM。

示例收据的调用为 `aclnnCast`，provider 为 CANN 9.2.1 installed OPP，实际记录的 kernel 为 `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0`。

复核时，在逐窗记录中筛选 `contract=cast_bf16_fp32`、此页 shape 和属性。有多个前序的样本逐窗保留。示例用于解释该次计算。校准统计、冻结后独立样本数及完整误差范围以总报告为准。[逐窗数据](../../calibration/reference/windows.jsonl) · [模型实现](../../src/operator_model/operators/vector/cast.py) · [总建模方法](../modeling-strategy.md) · [误差分析](../error-analysis.md)
