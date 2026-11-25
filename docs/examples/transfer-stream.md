# 如何从发起与返回两种价格还原一段流水

一次搬运有两个时间。**发起间隔**是端口接纳下一条相同请求前要等的周期；**返回延迟**是这条请求从发起到数据可用的周期。对同一条指令、布局、负载，在无依赖的连续请求上取相邻发起差，在逐次排空的请求上取完成减发起。前者约束连续吞吐，后者决定首次依赖何时解除。保留测量来源、芯片与负载条件。已有 FA 的 950DT 单 AIC、连续 BF16 [9,64] ND→NZ 读探针给出 160.6458 和 638 周期。理想地连续发起 3 条，末次返回约在 959.29 周期；若必须等前次返回才发下一条，则需 1914 周期。这个数只描述该读端口。[探针的几何、两端价格与来源](../../src/operator_model/data/parameters/fa_port_scenarios.json)

下面是可以复算的**设计示例**。假设单核在两个缓冲槽上分别完成读、计算、写，读和写各用独立端口。同形的 5 个 tile，读的发起/返回为 70/130 周期，写的发起/返回为 80/150 周期，每个 tile 独占计算资源 110 周期。前三个 tile 的一组最早排程如下。

| tile | 读发起 → 返回 | 计算开始 → 结束 | 写发起 → 返回 |
|---|---:|---:|---:|
| 0 | 0 → 130 | 130 → 240 | 240 → 390 |
| 1 | 70 → 200 | 240 → 350 | 350 → 500 |
| 2 | 390 → 520 | 520 → 630 | 630 → 780 |

**列说明**　tile 从 0 编号；其余三列是同一忙核、同一零点的设备周期。第三次读等到第一块的写退休才复用缓冲槽。

记 $D=L_r+C+L_w$，其中 $L_r,L_w$ 是读写的返回延迟，$C$ 是每块计算服务；记 $I=\max(I_r,C,I_w)$，其中 $I_r,I_w$ 是读写各自的发起间隔。若缓冲槽数为 $S$，同形 tile 数为 $n$，末次写返回时刻为

$$
T=D+(n-1)I+\left\lfloor\frac{n-1}{S}\right\rfloor\max(0,D-SI).
$$

单块从读到写退休需要 $D$ 周期；连续出块至少相隔 $I$。每隔 $S$ 块要再次占用同一缓冲槽，当旧块尚未退休时，再等 $D-SI$ 周期。本例 $D=390$、$I=110$、$S=2$，末块在 1170 周期退休，1.65 GHz 下为 0.7091 µs。对首尾尺寸不同的块，离线逐块重放各自的描述符。校准时按同一设备零点核对首发、相邻稳态的间隔、尾写退休和核间最后完成者；仅有整窗无法同时反推两种端口价格。

```python
from dataclasses import replace
from operator_model import estimate, load_chip
from operator_model.scheduling.transfer_pipeline import replay_tiles

def descriptor(direction, opcode, ii, response):
    return dict(direction=direction, opcode=opcode, layout="ND contiguous",
                dtype="bf16", bytes=4096, issue_cycles=ii,
                response_cycles=response, source="worked design example")

chip = replace(load_chip(), transfer_profiles={
    "read4k": descriptor("GM→UB", "read", 70, 130),
    "write4k": descriptor("UB→GM", "write", 80, 150),
})
program = dict(read_profile="read4k", write_profile="write4k",
               tiles=5, slots=2, compute_cycles=110)
events = list(replay_tiles(chip.transfer_profiles["read4k"],
                           chip.transfer_profiles["write4k"], 110, 5, 2))
assert events[-1]["write_retire"] == 1170
# 示范如何把一个已识别的忙核子流水作为整窗下界。
r = estimate("rmsnorm", [640, 2048], {"transfer_pipeline": program}, chip=chip)
assert r.projection["transfer_pipeline"]["cycles"] == 1170
```

在线只算闭式下界。若已有专用模型报 8 µs，0.7091 µs 的子流水完全可以隐藏，整窗仍为 8 µs。配置者负责将描述符几何、核内 tile 数、可用缓冲槽和依赖关系对应到真实代码。各算子仍使用已有的首发、分核、跨引擎调度；命中与边界由具体程序决定。整窗下界不能据此把一个偏慢的旧模型修快。独立端口假设不适用于同端口串行服务、跨核归并或存在额外全局屏障的阶段。切换芯片须在目标配置声明对应 profile 和来源。
