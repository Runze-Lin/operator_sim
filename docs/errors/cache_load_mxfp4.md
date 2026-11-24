# cache_load_mxfp4 · 逐窗误差

[全部合同](README.md) · [总误差分析](../error-analysis.md) · [原始逐窗记录](../../calibration/reference/windows.jsonl)

950DT、950PR 均有 0 个合格完整窗。模型对照、完整窗误差和物理利用率为 N/A，970DT 无对应校准窗。

显式 AntiMxQuant 有[独立双端校准](cache_load_mxfp4_antimx.md)。系统捕获了 packed E2M1 值、按 32 元素分组并补齐的 E8M0 scale 和 BF16 物化输出。[缓存读取算例](../examples/cache_load.md)
