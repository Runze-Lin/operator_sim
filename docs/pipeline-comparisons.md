# 算例中的模型与 CA 流水对照

每幅图及其可重绘的时间数据都放在对应算例。读图时先核对 shape、SKU、实际 kernel 和设备窗资格，再比较工作单元、资源发起与最后退休。描淡的推断区间、描述符等待和不能拆开的阶段按各图图注解读。

| 算例 | 可以对照什么 |
|---|---|
| [Matmul](examples/matmul.md) | 两个输出块、每块两个 K 页，模型关键路径与最后退休核的真实指令时间线 |
| [稠密 MX Matmul](examples/mxfp4-design.md) | 同形 A4W4 与 A8W4 的阶段预算、FP4/FP8 MMAD、AIV 权重展开与 UB→L1 交接 |
| [Quant](examples/quant.md) | UB 块的读、量化、数据写和 scale 写 |
| [RMSNorm](examples/rmsnorm.md) | 归约后写出和多块重叠 |
| [FA](examples/fa.md) | 首块和 KV 循环的 Cube/Vector 阶段 |
| [GMM1–SwiGLU–MXQuant](examples/grouped_matmul_swiglu_quant_v2.md) | 计算与搬运重叠、融合阶段和设备窗口径 |

各算子的覆盖和全量误差另见[逐算子校准表](error-analysis.md)。
