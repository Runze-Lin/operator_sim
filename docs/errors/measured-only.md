# 已测量、尚无通用价格的调用

[总误差分析](../error-analysis.md) · [系统捕获的全部 80 个名字](../system-workload-coverage.md)

每一行均已核对有效输出、实际安装 ELF 和每个物理核的退休。模型时间、预测误差、等效 Roofline 利用率和 970DT 外推均为 N/A。两个端点各一例只能证明这些输入能够运行。单 kernel 采用全芯片首发至末退，多 kernel 使用同一个设备时钟并计入发射之间的间隔。

| 调用 | 精度 | 主输入 | 芯片 | ACLNN | 实际 kernel | 每阶段 AIV | CA µs | 离线秒 | 证据 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cache_load | E4M3FN / FP32 → BF16 | 1×128×512 | Ascend950DT_9582 | `aclnnAscendAntiQuant` | `AscendAntiQuantV2_650baea70767c4bd7838a0d2f6bb5823_high_performance_2` | 64 | 4.5491 | 64.68 | [收据](../../calibration/cache_load/fp8-smoke.json.gz) |
| cache_load | E4M3FN / FP32 → BF16 | 1×128×512 | Ascend950PR_957b | `aclnnAscendAntiQuant` | `AscendAntiQuantV2_650baea70767c4bd7838a0d2f6bb5823_high_performance_2` | 56 | 4.0424 | 61.46 | [收据](../../calibration/cache_load/fp8-smoke.json.gz) |
| aten.add.Tensor | bf16 | 9×7168 | Ascend950DT_9582 | `aclnnAdd` | `Add_BFLOAT16_high_performance_8` | 36 | 2.8412 | 55.13 | [收据](../../calibration/elementwise/reference.json.gz) |
| aten.add.Tensor | bf16 | 9×7168 | Ascend950PR_957b | `aclnnAdd` | `Add_BFLOAT16_high_performance_8` | 28 | 2.5339 | 57.16 | [收据](../../calibration/elementwise/reference.json.gz) |
| aten.mul.Tensor | fp32 | 9×4 | Ascend950DT_9582 | `aclnnMul` | `Mul_ee98c6628030785f610b924ab1557b31_high_performance_0` | 1 | 3.7200 | 51.71 | [收据](../../calibration/elementwise/reference.json.gz) |
| aten.mul.Tensor | fp32 | 9×4 | Ascend950PR_957b | `aclnnMul` | `Mul_ee98c6628030785f610b924ab1557b31_high_performance_0` | 1 | 3.3812 | 49.82 | [收据](../../calibration/elementwise/reference.json.gz) |
| quant_mxfp4 | BF16 → E2M1 + E8M0 | 9×128 | Ascend950DT_9582 | `aclnnDynamicMxQuant` | `Slice_ee29dafe4fe21e12bb8d56ae91626cba_high_performance_150` → `DynamicMxQuant_bf16toe2m1_high_performance_33_mix_aiv` | 3 / 9 | 7.1030 | 65.12 | [收据](../../calibration/quant_update/strided-950dt.json) |
| quant_mxfp4 | BF16 → E2M1 + E8M0 | 9×128 | Ascend950PR_957b | `aclnnDynamicMxQuant` | `Slice_ee29dafe4fe21e12bb8d56ae91626cba_high_performance_150` → `DynamicMxQuant_bf16toe2m1_high_performance_33_mix_aiv` | 3 / 9 | 6.9800 | 56.84 | [收据](../../calibration/quant_update/strided-950pr.json) |

Add 的两输入都是连续 [9,7168]，alpha=1；Mul 的另一路为 [9,1]，沿末轴广播。FP8 缓存的每行配一个 FP32 scale，按 [128,1] 传入，输出 BF16。FP4 量化输入是 stride=[6288,1]、起始偏移 6144 个 BF16 元素的 Kimi 视图。单独运行时的 Slice 可能由系统的上游物化或融合消掉。输入变化应重新核对派发。

从仓根运行 `python -m calibration.lookup cache_load --contract cache_load_fp8_antiquant --shape 1 128 512` 可读取 CA、源收据、数值与布局。
