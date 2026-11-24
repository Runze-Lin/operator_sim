# 摘要

更换芯片会同时改变可用核数、每核算力、片上搬运能力和可并发的任务数。一个在 950DT 上测得的利用率，很难回答“同一层在更多 Cube 核、更少相对 Vector 资源的芯片上要跑多久”。本文面向大模型系统仿真，把一次算子调用还原为本卡实际执行的工作。由本卡张量、数据格式和算子语义决定工作量，由实现策略决定 tile、分核和重复的轮数，由目标芯片决定计算和搬运的阶段时间，最后沿数据依赖合成设备时间。在线只计算决定完成时刻的工作单元和重复间隔，以尽可能低的查询开销支持整网排程。

我们用三个可以手工复算的例子解释方法。动态量化说明为何每核的有效负载、scale 写回和启动影响小张量。矩阵乘说明为何尾波的块数与形状会改变跨核重叠。融合注意力贯穿校准与验证。在 950DT 的 32Ki 上下文 CA 指令记录中，按页配对 K/V 描述符，分别求首响应、持续服务率和后续页发起间隔。将参数用于 950PR 上 128Ki 单头验证，预测 274.988 µs，对应完整 CA 窗为 265.893 µs，相差 +3.42%。进一步将同一算法映射到假设的 970DT，得到 252.093 µs。最后一个数值取决于控制节拍、搬运与实现的继承假设。

基础校准集含 36 项执行合同的 2,180 条可比记录。30 项可在两种 950 芯片上计分，28 项达到项目门槛，其中 5 项有双端冻结后新形状验证。误差较大的两项说明了两种不同的问题。mHC Post 的本卡形状没有描述前序驻留，补充有序调用后，同一批 CA 窗上的误差显著降低。A8W4 Matmul 在 950PR 的总体加权误差较小，但一个独立形状仍高估 56.67%。我们给出逐项结果、最差形状及推断的适用边界。

三个模型的 33 组系统情景涵盖 32Ki 至 1Mi 输入以及 prefill、推测 decode 两个阶段。18 组精细预测快于历史 Roofline；把计算引擎、精度、内存口径和依赖对齐后，没有一组快于对应峰值参考。端到端数字是系统仿真的结果。本文将 CA 设备实测、已有输入的开发回放、冻结验证及无目标实现的外推分别标明，为将来选择微基准和更新芯片配置提供依据。

**关键词**　算子性能建模；昇腾；指令流水；校准；跨芯片外推；大模型系统仿真

## Abstract

A system simulator needs a prediction for a particular invocation on a particular chip. A utilization factor measured on one accelerator hides the work assignment, memory hierarchy and pipeline dependencies that can change on another. We derive rank-local work from tensor shapes, precision and semantics, map tiles to cores, price each stage with target resources, and schedule repeated work on the critical path. The online estimator summarizes repeated tiles and waves to keep query overhead low.

We work through vector quantization, matrix multiplication and fused attention. In the attention case, K/V issue timestamps in an archived 32 Ki-token Ascend 950DT run identify a steady page interval distinct from individual descriptor latency. For a 128 Ki-token 950PR run, the resulting estimate is 274.988 µs against 265.893 µs in the complete CA window. Applying the same implementation hypothesis to a prospective 970DT yields 252.093 µs. Sensitivity experiments identify which inherited parameters can change that extrapolation.

The base evaluation contains 2,180 comparable records. Of 30 contracts scored on both 950 variants, 28 meet the study thresholds on their respective evaluation sets, including five with dual-device held-out validation. A shape-only mHC case and an A8W4 projection expose limitations of input observability and tail prediction. Across 33 simulated system configurations, 18 predictions are faster than the historical Roofline backend, while all respect the precision- and implementation-aligned peak reference at request level. The results distinguish measured CA windows, validation on unseen shapes and cross-generation extrapolations.
