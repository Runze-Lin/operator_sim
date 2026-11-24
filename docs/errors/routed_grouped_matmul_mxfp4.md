# routed_grouped_matmul_mxfp4 · 逐窗误差

[全部合同](README.md) · [总误差分析](../error-analysis.md) · [原始逐窗记录](../../calibration/reference/windows.jsonl)

950DT、950PR 均有 0 个合格完整窗。模型对照、完整窗误差和物理利用率为 N/A，970DT 无对应校准窗。

显式 GroupedMatmulV5 有[双端校准](routed_grouped_matmul_mxfp4_v5.md)。系统捕获了路由专家的 A4W4 分组乘法，工作量取决于专家行分布。稠密 Matmul 的 CA 窗属于另一合同。[路由专家 GMM 算例](../examples/routed-grouped-matmul.md)
