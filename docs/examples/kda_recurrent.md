# Recurrent KDA 算例

## 模型与实际库流水对照

![950PR kda_recurrent 的模型与 CA 同轴对照](figures/kda_recurrent-pr-pipeline.png)

本卡 shape 为 `[1, 1, 1, 1, 128, 128]`，950PR 模型 **15.277 µs**，完整 CA **14.622 µs**，有符号误差 +4.48%。上栏画出当前预测器能够辨识的阶段时长。只返回一个阶段的路线保留整段预测；多个暴露阶段按贡献累计，模型尚未辨识它们的具体指令发起点。下栏用实际库 trace 的最后完成核逐队列展开，黑色刻度来自真实 DMA 或 MMAD 发起。 [图上每段的时间与 trace 指纹](../../results/v0.20/examples/kda_recurrent-pr.json)

输入轴顺序为 **B,T,Hq,Hv,K,V**。本例比较同一输入契约在 950DT、950PR 与假设的 970DT 上的设备指令窗口。

```python
from operator_model import estimate

shape = [1, 1, 1, 1, 128, 128]
attrs = {'layout': 'BSND', 'dtype': 'BF16', 'gate_dtype': 'FP32', 'state_dtype': 'FP32', 'state_v_first': False, 'use_gate_in_kernel': False, 'chunk_size': None, 'output_final_state': True, 'export_gk': False}
for chip in ('Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT'):
    r = estimate('kda_recurrent', shape, attrs, chip=chip, dtype='BF16', timing_mode='auto')
    print(r.latency_us, r.status, r.active_aic, r.active_aiv, r.work)
```

## 输入到工作量

该路线调用 CANN 9.2.1 的 ops-transformer 算子族。实际调用接口为 `aclnnRecurrentKda`。
工作量接口给出的 Cube 操作为 0，Vector 计数为 114,944。逻辑读/写分别为 66820 / 65792 B。此处 None 表示缺少计数；片外实际流量须另行观测，逻辑字节不含中间层重复搬运。

## 分块、资源和关键路径

一条状态链占一个 AIV。每 token 更新 128×128 状态，1 个 token 顺序执行。128×128 作为一个工作单位，U=1×128×128/16384=1。除 arrival=log₂(1)=0，计算和控制阶段都用 U，搬运用 U·log₂(1+1)=U。

三个 DT 开发 DB 取临界 AIV 暴露时间，分别拟合非负截距 a 和斜率 b；a 单位 µs，b 单位 µs/状态工作单位，arrival 用 µs/log₂(核数)。此模型的截距包含在整次调用内，只支付一次。来源 results/v0.8/vector/profile.json。

| 阶段 | a µs | b µs/单位 | X | a+bX µs |
|---|---|---|---|---|
| arrival | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| SCALAR | 4.11182 | 0.06377 | 1.00000 | 4.17558 |
| SIMD | 4.30242 | 2.93740 | 1.00000 | 7.23983 |
| MTE2 | 1.41485 | 0.02602 | 1.00000 | 1.44087 |
| MTE3 | 0.24121 | 0.44823 | 1.00000 | 0.68944 |
| other | 1.52586 | 0.00000 | 1.00000 | 1.52586 |

970DT 的算子实现与分片延用 DT 模板。先算 DT 各段的暴露时间，共 15.07157 µs。设备配置取 configs/chips/Ascend970DT.json。目标有 120 个 AIV、1.8 GHz，本例预计用 1 核。控制与到达按频率比 1.65/1.8=0.91667；SIMD 按实际 dtype 的每核峰值比及波数，倍率 0.91553。

每次调用逻辑总读写 132612 B，芯片级带宽占比代理 p=min(1,132612/(1000×4000×15.07157))=0.00220。MTE2/MTE3 倍率是 (1−p)×局部价格比+p×4000/14400，得到 0.91526/0.91526；局部部分在未指定两端 MTE 速率时取频率比×波数比。该项使用全片逻辑读写量作代理。

| 阶段 | DT 暴露 µs | 目标价格倍率 | 970 外推 µs |
|---|---|---|---|
| arrival | 0.00000 | 0.91667 | 0.00000 |
| SCALAR | 4.17558 | 0.91667 | 3.82762 |
| SIMD | 7.23983 | 0.91553 | 6.62826 |
| MTE2 | 1.44087 | 0.91526 | 1.31877 |
| MTE3 | 0.68944 | 0.91526 | 0.63102 |
| other | 1.52586 | 0.91667 | 1.39870 |

六段相加得到 DT 15.07157 µs、970DT 13.80437 µs。arrival 是最晚结束核的到达偏移，other 是未归入已知流水的区间；DT 原始重叠已按 SIMD→MTE3→MTE2→SCALAR 唯一归属，合计时只算一次。

## 设备时间及误差

| 平台 | 模型 (µs) | CA (µs) | 有符号误差 | 预测 AIC / AIV | CA AIC / AIV |
|---|---:|---:|---:|---|---|
| 950DT | 15.07157 | 14.83394 | +1.60% | N/A / 1 | 0 / 1 |
| 950PR | 15.27701 | 14.62242 | +4.48% | 0 / 1 | 0 / 1 |
| 970DT 外推 | 13.80437 | N/A | N/A | 0 / 1 | N/A |

DT/PR 加速比取同形耗时之比，模型 0.9866×，CA 1.0145×。正时延误差为高估。

**950DT CA**　msopprof official full trace，角色 new paired DT CA；核名 `RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0`。该宿主的数值检查共 3 组；子算子切片共享宿主检查。各设备窗以最早到最晚核指令计，多 kernel 取串行窗之和。

**950PR CA**　msopprof official full trace，角色 frozen PR holdout；核名 `RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0`。该宿主的数值检查共 3 组；子算子切片共享宿主检查。各设备窗以最早到最晚核指令计，多 kernel 取串行窗之和。

## CA 流水与模型差距

![Recurrent KDA 实测队列](figures/kda_recurrent.png)

横轴为每次 kernel 自身的设备时间，两端各自从零开始。每种核取最晚结束的一个核。每行按该队列有效指令区间的并集着色，100 个等宽时间格，深蓝表示格内占满。不同队列可重叠，行长不能相加。DMA/FIX 行表示描述符的在途窗口，可能包含等待依赖的时间；颜色深度不能直接换算 HBM 总线利用率。

950DT 第 1 段 14.8339 µs；所选核上占用最长的队列为 AIV/Scalar 9.7545 µs、AIV/PUSHQ 6.9442 µs、AIV/MTE2 1.8042 µs。
950PR 第 1 段 14.6224 µs；所选核上占用最长的队列为 AIV/Scalar 9.6685 µs、AIV/PUSHQ 6.8624 µs、AIV/MTE2 1.9000 µs。

PR 的模型与 CA 相差 +0.6546 µs。时序状态更新串行，关键核 Vector 和标量控制不随整芯片核数成比例加速。上图给出可观测的队列活动；一次端到端差值无法唯一分摊到这些因素。

[总建模方法](../modeling-strategy.md) · [全部 shape 的误差](../error-analysis.md) · [970DT 配置与来源](../../configs/chips/Ascend970DT.json)
