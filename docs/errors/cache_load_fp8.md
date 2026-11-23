# cache_load_fp8 · 逐窗误差

[全部合同](README.md) · [总误差分析](../error-analysis.md) · [原始逐窗记录](../../calibration/reference/windows.jsonl)

950DT、950PR 均有 0 个合格完整窗。模型对照、CA 误差和等效利用率为 N/A，970DT 无对应校准窗。

默认调用没有派发身份。显式 AscendAntiQuant 有原生实测，[FP8 的双端设备时间](measured-only.md)。系统捕获了 E4M3FN 值、逐行 FP32 scale 和 BF16 物化输出。[缓存读取算例](../examples/cache_load.md)
