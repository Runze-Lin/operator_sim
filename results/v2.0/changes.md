# operator-sim v2.0.0

面向系统仿真的昇腾算子建模后端。Python 3.10+ 可以直接使用 wheel，通过 `estimate` 查询设备时间、计算量、逻辑读写量和活动核数。系统适配器支持精细建模与原 Roofline 的同图比较。

- 收录 46 页中文论文及中英文摘要。正文以量化、矩阵乘和融合注意力说明分核、阶段定价、流水重叠及 CA trace 校准。附录列出逐合同误差与 33 组系统预测。
- 950DT、950PR 的基础归档有 2,180 个可比窗，其中 120 个为冻结独立窗。报告区分开发回放、独立验证以及 970DT 架构外推。
- v2.0.0 沿用报告中的 v1.13.1 模型与参数。本次发布复用既有预测及 CA 数据，未新增采集。4,360 次归档 CPU 查询中位数为 0.173 ms，P95 为 3.058 ms。

Release 附件包括 Python wheel、含报告与复算材料的源码包，以及可单独下载的 PDF。[论文源稿与复算指南](https://github.com/Runze-Lin/operator_sim/tree/v2.0.0/docs/paper) · [系统接入](https://github.com/Runze-Lin/operator_sim/blob/v2.0.0/docs/integration.md) · [逐算子误差](https://github.com/Runze-Lin/operator_sim/blob/v2.0.0/docs/error-analysis.md)
