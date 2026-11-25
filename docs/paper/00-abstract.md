# 摘要

芯片换代会改变核数、每核吞吐和存储带宽。950DT 上的一个利用率，无法说明相同算子在另一种 Cube 与 Vector 配比下为什么会变慢。本文从本卡张量和精度计算工作量，由实现规则选择分块与分核，再用目标芯片资源给各阶段定价，按依赖求出设备完成时间。在线用稳态间隔概括重复块，以尽可能低的查询开销服务于整网仿真。

我们用动态量化解释核间负载与 scale 写回，用矩阵乘解释尾波分块与跨核重叠，用融合注意力演示如何校准。按页配对 950DT 的 32Ki CA 指令记录中的 K/V 描述符，可分别估计首次响应、服务速率和后续发起间隔。对 950PR 的 128Ki 单头样例，预测 274.988 µs，完整 CA 窗 265.893 µs，偏差 +3.42%。沿用该实现映射到假设的 970DT 得 252.093 µs；这只是继承控制与搬运参数的外推。

基础档案含 36 个执行合同、2,180 条可比记录。30 个合同可在两种 950 芯片上计分，28 个达项目门槛，其中 5 个有双端冻结后新形状验证。mHC Post 暴露了仅给 shape 时无法辨别前序驻留的问题，A8W4 Matmul 在 950PR 的一个独立形状上仍高估 56.67%。

对有 Vector 计算的实现，保留两种计价方法：一种由已有库内核的资源与循环记录约束服务时间，另一种把算法展开为按类别计数的寄存器程序。前者需要足够接近的实现，后者可用于新精度，但依赖指令映射和发射能力的假设。按相同 CA 窗对照，两种方法各有胜负。为检查局部差异会不会传到请求，本文在三个模型上比较原法、指令法与默认混用策略，分别考察 27 组 128Ki 单轮 decode 和 33 组 32Ki 至 1Mi 的 prefill 加 decode 请求。在 33 组中，默认策略相对原法的请求时间变化是 −5.52% 至 +1.30%，强制指令法是 −22.77% 至 −2.78%。三种策略都不短于同精度峰值参考的配置有 33/33 组。这些是整网离散事件仿真（DES）结果。

在 v2.5 的 60 个新 CA 窗口上，逐段修正后，BF16 Indexer LayerNorm 的 DT/PR WAPE 降至 8.92%/5.76%，双侧 MXFP8 Matmul 降至 7.73%/7.90%，BF16 AddRmsNorm 降至 11.12%/9.53%。跨芯片时间比的误差另计。

**版本口径**　三模型端到端实验采用 v2.3。附录 I 对应 v2.4 的文本模型目录；附录 J 报告 v2.5 同形状独立 CA 对照。基础 CA 和历史系统冻结分别保留原有时间口径，整网实机延迟尚待验证。

**关键词**　算子性能建模；昇腾；指令流水；校准；跨芯片外推；大模型系统仿真

## Abstract

A system simulator needs a prediction for a particular invocation on a particular chip. A utilization factor measured on one accelerator hides the work assignment, memory hierarchy and pipeline dependencies that can change on another. We derive rank-local work from tensor shapes, precision and semantics, map tiles to cores, price each stage with target resources, and schedule repeated work on the critical path. The online estimator summarizes repeated tiles and waves to keep query overhead low.

We work through vector quantization, matrix multiplication and fused attention. In the attention case, K/V issue timestamps in an archived 32 Ki-token Ascend 950DT run identify a steady page interval distinct from individual descriptor latency. For a 128 Ki-token 950PR run, the resulting estimate is 274.988 µs against 265.893 µs in the complete CA window. Applying the same implementation hypothesis to a prospective 970DT yields 252.093 µs. Sensitivity experiments identify which inherited parameters can change that extrapolation.

The base evaluation contains 2,180 comparable records. Of 30 contracts scored on both 950 variants, 28 meet the study thresholds on their respective evaluation sets, including five with dual-device held-out validation. A shape-only mHC case and an A8W4 projection expose limitations of input observability and tail prediction. Two Vector pricing rules are evaluated on matching device windows: a mechanism calibrated against an implementation's loop timing, and an explicit count of instruction classes priced by resource. To test how differences propagate, we replay 27 speculative decode configurations and 33 prefill-plus-decode requests across three models. We compare both forced pricing policies, the per-implementation default, the historical Roofline backend and a precision-matched peak reference. Across the 33 requests, the default changes predicted duration by −5.52% to +1.30% relative to the original rule, and forcing instruction pricing changes it by −22.77% to −2.78%. All three predictions respect the matched peak reference for 33 of 33 requests. The system figures are discrete-event predictions; neither policy has been validated against full-request device timings.

In a subsequent paired validation, we froze the revised models and collected 60 new windows across both 950 variants. Accounting for strided materialization, first matrix issue, and shared statistics writes reduced WAPE on DT/PR to 8.92%/5.76% for Indexer LayerNorm, 7.73%/7.90% for native MXFP8 matrix multiplication, and 11.12%/9.53% for residual RMS normalization. Small inter-chip speed differences remain harder to resolve than per-chip latency.
