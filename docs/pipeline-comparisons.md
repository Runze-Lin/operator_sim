# 算例中的模型与 CA 流水对照

每幅图及其可重绘的时间数据都放在对应算例。读图时先核对 shape、SKU、实际 kernel 和设备窗资格，再比较工作单元、资源发起与最后退休。描淡的推断区间、描述符等待和不能拆开的阶段按各图图注解读。

| 算例 | 可以对照什么 |
|---|---|
| [逐 token FP8 量化](examples/quant-token-fp8.md) | 9×1024 的最慢核，逐项比较首读、整行归约、写载荷和 scale 退休 |
| [FP4 CacheLoad](examples/cache_load.md) | 同一缓存形状在双端的模型阶段与末核事件 |
| [Matmul](examples/matmul.md) | 两个输出块、每块两个 K 页，模型关键路径与最后退休核的真实指令时间线 |
| [稠密 MX Matmul](examples/mxfp4-design.md) | 同形 A4W4 与 A8W4 的阶段预算、FP4/FP8 MMAD、AIV 权重展开与 UB→L1 交接 |
| [950PR A8W4 的 26 个行块](examples/mxfp4-design.md#为什么-950pr-上-2048128256-是-26-核) | 输出地址定位每个行块，比较两版分核、实际覆盖和 26 个核簇的权重交接 |
| [A4W4 原生窄尾波](examples/mx-native-tail.md) | 对照两端预算和全片宽块、窄块的实际 MMAD 窗口 |
| [A8W4 长 K 页交接](examples/mx-handoff.md) | 按 K 页对照 950DT/PR 模型与 CA 的 UB→L1 退休和首次 MMAD 发起；另附关键路径预算和指令区间 |
| [Quant](examples/quant.md) | UB 块的读、量化、数据写和 scale 写 |
| [RMSNorm](examples/rmsnorm.md) | 归约后写出和多块重叠 |
| [FA](examples/fa.md) | 首块和 KV 循环的 Cube/Vector 阶段 |
| [GMM1–SwiGLU–MXQuant](examples/grouped_matmul_swiglu_quant_v2.md) | 计算与搬运重叠、融合阶段和设备窗口径 |

各算子的覆盖和全量误差另见[逐算子校准表](error-analysis.md)。
