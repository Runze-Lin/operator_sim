# 稠密 MX Matmul · 原生 A4W4 与在线展开 A8W4

同一个无 bias 的线性层，系统侧传本卡 `(M,N,K)`。M 是本次发给 kernel 的 token 行数，N 是本卡输出宽，K 是本卡点积长度。下面的 M=9 可以表示单请求、推测 8 个 token 后的一次主模型校验；M=2048 可以表示一次 prefill chunk。张量从已量化、已打包的状态开始计时。

```python
from operator_model import estimate, load_chip

pr = load_chip("Ascend950PR_957b")
for m in (9, 2048):
    native = estimate("matmul", (m, 128, 256), dtype="mxfp4", chip=pr)
    mixed = estimate("matmul", (m, 128, 256), {"weight_dtype": "mxfp4"},
                     dtype="fp8_e4m3fn", chip=pr)
    for result in (native, mixed):
        print(result.status, result.latency_us, result.active_aic, result.active_aiv,
              result.work["cube_ops"], result.work["logical_read_bytes"])
```

### 从打包数据到工作量

计算式为 $Y_{ij}=\mathrm{BF16}(\sum_g s^A_{ig}s^W_{jg}\sum_{t\in g} A_{it}W_{jt})$。每个 K 组最多 32 项，组内值用 E2M1 或 E4M3FN，两个尺度均为 E8M0，累加按 FP32 建模。A 的物理形状为 [M,K]，W 以 [N,K] 的 ND 存储并在乘法中转置。两份 scale 的物理形状分别为 [M,⌈K/64⌉,2] 与 [N,⌈K/64⌉,2]；K 尾部不足两个组仍为该行分配两个槽。

以 M=9、N=128、K=256 为例，两种实现均有 $2MNK=589,824$ 个有用 Cube 运算，输出写 $2MN=2,304$ B。原生路线 A 的 payload 占 1,152 B，混合路线占 2,304 B。两者 W 都为 16,384 B，A/W 两份 scale 分别占 72 B、1,024 B，因此总逻辑读分别为 **18,632 B** 与 **19,784 B**。这些是张量读写量；不同输出块重复取 W、L2 命中以及 UB 中间交接另算。

| 路线 | 首个 Cube 操作之前的依赖 | 资源定价依据 |
|---|---|---|
| 原生 A4W4 | 将打包的 A、W 和 scale 搬入 L1，装载 L0 后发射带 scale 的 MX 乘加 | 4-bit Cube、MTE2/MTE1、写回 |
| A8W4 | AIV 取打包 W，在片上展开为 FP8 并交接到 L1；AIC 等待对应页后用 FP8 乘加 | 8-bit Cube、AIV 转换、UB→L1 端口、交接与写回 |

两路都只计一次融合 kernel 发射。A8W4 的片上中间权重占用局部传输和缓冲；无需在框架计算图中再加一次独立反量化。950 的稠密实现使用 ops-nn 的 QuantBatchMatmulV3 和 QuantBatchMatmulV4，入口同为 `aclnnQuantMatmulV5`。不同量化粒度、权重的 NZ 布局、带 bias 或 MoE 的 grouped 算子是不同合同。[上游 V5 参数与精度组合](https://gitcode.com/cann/ops-nn/blob/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v4/docs/aclnnQuantMatmulV5.md)

### 逐页算例与完整设备窗

以 950DT 的 M=9 为例。下表从本次公开查询的分块计划、整窗关键路径和冻结后对同一 shape 做的 CA 独立测量取数。周期价格按 1.65 GHz 换成设备微秒。

| 设备或流水量 | 原生 A4W4 | 核内展开 A8W4 |
|---|---:|---:|
| 忙核输出块 M×N、K 页深 | 16 × 16 × 256 | 16 × 16 × 256 |
| 活动 AIC / 已分配 AIV | 8 / 0 | 8 / 16 |
| 忙核输出块数 × 每块 K 页数 | 1 × 1 | 1 × 1 |
| 每个 AIC 每周期的 Cube 操作数 | 32,765 | 16,383 |
| 算子边界读 / 写 B | 18,632 / 2,304 | 19,784 / 2,304 |
| 首发 µs | 2.4385 | 2.2479 |
| 核间到达差 µs | 0.1896 | 0.1670 |
| 读响应和页服务 µs | 0.7193 | 0.7586 |
| AIV 净展开 µs | 0.0000 | 0.1371 |
| UB→L1 交接 µs | 0.0000 | 0.1113 |
| 末页装载 µs | 0.0347 | 0.0439 |
| 末页乘加 µs | 0.0673 | 0.0551 |
| 输出完成 µs | 0.4909 | 0.4248 |
| 预测整窗 µs | 3.9403 | 3.9457 |
| 真实 CA 整窗 µs | 4.1539 | 4.0115 |
| 预测误差 | -5.14% | -1.64% |

![950DT 两条 MX Matmul 同形的阶段预算与真实 CA 描述符](assets/mx-matmul.png)

上栏按模型给出的首发、读响应、交接、乘加和写尾分解整窗，横向位置表示可加的关键路径预算。下栏按同一设备零点、同一刻度画最晚退休 AIC 与同核簇执行权重交接的 AIV，实线标全核完成时间。下栏只画有载荷的搬运、计算和写回，标量控制段留白。短竖线是指令发起，色带包括到退休的依赖等待；FIXP 可以在 MMAD 完成前就登记写描述符。比较两栏时，应按同类阶段核对发起、交接和最后退休。原生 FP4×FP4 的 Cube 直接消耗打包权重，混合路径的 AIV 展开和片上交接在图中单列。[绘图数值与 trace 指纹](assets/mx-matmul.json) · [分块与周期价格的来源](../modeling-strategy.md#稠密-mx-matmul--乘加与权重展开)

### 大 M 与 970DT 外推

保持 N=128、K=256，将 M 从单请求校验扩大到 2048 行的 prefill chunk。下表的 950 结果在限定几何内返回 `estimated`，970DT 返回 `extrapolated`。

| 路线、M | DT 预测 / CA µs | PR 预测 / CA µs | 970DT 外推 µs |
|---|---:|---:|---:|
| A4W4，9 | 3.940 / 4.154 | 3.803 / 3.835 | 3.521 |
| A8W4，9 | 3.946 / 4.012 | 3.616 / 3.946 | 3.457 |
| A4W4，2048 | 4.608 / 4.336 | 4.595 / 4.219 | 3.986 |
| A8W4，2048 | 5.965 / 5.845 | 5.786 / 5.792 | 5.660 |

以整卡 8×32Ki token、不分 chunk 的 [262144,128,512] 投影形状做精度情景推演。若这一投影改用上述打包格式，970DT 上 A4W4 和 A8W4 分别预测 **15.40 / 26.18 µs**。线上框架采用 chunk 时应按每次真实发射的 M 查询。这个大几何未做 MX 的 CA 计时，970DT 的数值还依赖 1:1 核配比、假设的实现和片上端口。[逐形误差及证据](../error-analysis.md#稠密-mx-matmul--双端同形验证)
