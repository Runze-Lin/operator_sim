# mhc · 逐窗误差

[全部合同](README.md) · [总误差分析](../error-analysis.md) · [原始逐窗记录](../../calibration/reference/windows.jsonl)

模型和 CA 均为设备微秒（µs）。有符号误差为 `100 × (模型 / CA − 1)`。等效 Roofline 利用率 U 为同一输入的理想时间除以模型设备时间。表内同时列出理想时间的控制项；合同汇总只对可比窗按 `Σ理想时间 / Σ模型时间` 聚合。查询列为归档 CPU 墙钟毫秒（ms），顺序为 950 / 970。970DT 的时间与 U 都是外推。

## 完整窗与误差

| 芯片 | 数据角色 | 不同 shape | 完整窗 | WAPE % | 最大绝对误差 % | U 可比窗 | 汇总 U |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ascend950DT_9582 | 开发回放 | 23 | 24 | 2.61 | 7.20 | 0 | N/A |
| Ascend950DT_9582 | 冻结独立 | 0 | 0 | N/A | N/A | 0 | N/A |
| Ascend950PR_957b | 开发回放 | 10 | 10 | 4.19 | 9.37 | 0 | N/A |
| Ascend950PR_957b | 冻结独立 | 0 | 0 | N/A | N/A | 0 | N/A |

下表的行号链接到 JSONL 原记录，其中包含全长 trace 指纹和已归档的收据字段。A、H、K 分别链接到本页的属性、前序和实现。前序列的“未传”表示本次调用未提供流历史，“空流”表示已知从空的 OperatorStream 开始；显式 resident_aiv 归在属性内。

## Ascend950DT_9582 · 开发回放

| 原始行 | 本卡 shape | 主输入 dtype | 属性 / 前序 | 实现 | 模型 µs | CA µs | 误差 % | 理想 µs / 控制项 | U | 970 外推 µs | 970 外推 U | 查询 ms（950 / 970） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [380](../../calibration/reference/windows.jsonl#L380) | 1×4×64 | bf16 | [A1](#a1) / 未传 | [K1](#k1) | 18.0022 | 17.1430 | +5.01 | N/A / 原因 1 | N/A | 16.5017 | N/A（原因 1） | 0.285 / 0.542 |
| [381](../../calibration/reference/windows.jsonl#L381) | 1×4×80 | bf16 | [A1](#a1) / 未传 | [K1](#k1) | 18.2296 | 17.5594 | +3.82 | N/A / 原因 1 | N/A | 16.7099 | N/A（原因 1） | 0.246 / 0.482 |
| [382](../../calibration/reference/windows.jsonl#L382) | 1×4×128 | bf16 | [A1](#a1) / 未传 | [K2](#k2) | 16.9034 | 15.7685 | +7.20 | N/A / 原因 1 | N/A | 15.4944 | N/A（原因 1） | 0.240 / 0.476 |
| [383](../../calibration/reference/windows.jsonl#L383) | 1×4×128 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 16.9034 | 15.9703 | +5.84 | N/A / 原因 1 | N/A | 15.4944 | N/A（原因 1） | 0.244 / 0.486 |
| [384](../../calibration/reference/windows.jsonl#L384) | 1×4×512 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 17.4568 | 17.3430 | +0.66 | N/A / 原因 1 | N/A | 15.9999 | N/A（原因 1） | 0.249 / 0.484 |
| [385](../../calibration/reference/windows.jsonl#L385) | 2×4×64 | bf16 | [A1](#a1) / 未传 | [K1](#k1) | 18.0275 | 17.9352 | +0.52 | N/A / 原因 1 | N/A | 16.5249 | N/A（原因 1） | 0.242 / 0.523 |
| [386](../../calibration/reference/windows.jsonl#L386) | 2×4×128 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 16.9229 | 16.4158 | +3.09 | N/A / 原因 1 | N/A | 15.5123 | N/A（原因 1） | 0.254 / 0.475 |
| [387](../../calibration/reference/windows.jsonl#L387) | 3×4×128 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 16.9424 | 16.8345 | +0.64 | N/A / 原因 1 | N/A | 15.5302 | N/A（原因 1） | 0.241 / 0.490 |
| [388](../../calibration/reference/windows.jsonl#L388) | 4×4×96 | bf16 | [A1](#a1) / 未传 | [K1](#k1) | 18.4516 | 17.5697 | +5.02 | N/A / 原因 1 | N/A | 16.9132 | N/A（原因 1） | 0.237 / 0.475 |
| [389](../../calibration/reference/windows.jsonl#L389) | 4×4×256 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 17.3001 | 17.3442 | -0.25 | N/A / 原因 1 | N/A | 15.8574 | N/A（原因 1） | 0.239 / 0.472 |
| [390](../../calibration/reference/windows.jsonl#L390) | 4×4×1024 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 18.1265 | 18.2200 | -0.51 | N/A / 原因 1 | N/A | 16.6078 | N/A（原因 1） | 0.240 / 0.475 |
| [391](../../calibration/reference/windows.jsonl#L391) | 5×4×256 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 17.3263 | 17.1255 | +1.17 | N/A / 原因 1 | N/A | 15.8814 | N/A（原因 1） | 0.327 / 0.496 |
| [392](../../calibration/reference/windows.jsonl#L392) | 6×4×128 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 17.0009 | 17.0164 | -0.09 | N/A / 原因 1 | N/A | 15.5838 | N/A（原因 1） | 0.247 / 0.480 |
| [393](../../calibration/reference/windows.jsonl#L393) | 6×4×1024 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 18.2593 | 18.3612 | -0.56 | N/A / 原因 1 | N/A | 16.7283 | N/A（原因 1） | 0.242 / 0.487 |
| [394](../../calibration/reference/windows.jsonl#L394) | 8×4×128 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 17.0399 | 16.7909 | +1.48 | N/A / 原因 1 | N/A | 15.6195 | N/A（原因 1） | 0.242 / 0.482 |
| [395](../../calibration/reference/windows.jsonl#L395) | 8×4×512 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 17.7339 | 18.6515 | -4.92 | N/A / 原因 1 | N/A | 16.2529 | N/A（原因 1） | 0.242 / 0.476 |
| [396](../../calibration/reference/windows.jsonl#L396) | 9×4×1024 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 18.4584 | 18.4824 | -0.13 | N/A / 原因 1 | N/A | 16.9086 | N/A（原因 1） | 0.238 / 0.509 |
| [397](../../calibration/reference/windows.jsonl#L397) | 12×4×384 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 17.6008 | 18.4861 | -4.79 | N/A / 原因 1 | N/A | 16.1316 | N/A（原因 1） | 0.240 / 0.488 |
| [398](../../calibration/reference/windows.jsonl#L398) | 12×4×768 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 18.1748 | 18.7873 | -3.26 | N/A / 原因 1 | N/A | 16.6517 | N/A（原因 1） | 0.245 / 0.481 |
| [399](../../calibration/reference/windows.jsonl#L399) | 16×4×128 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 17.1960 | 18.2024 | -5.53 | N/A / 原因 1 | N/A | 15.7624 | N/A（原因 1） | 0.245 / 0.481 |
| [400](../../calibration/reference/windows.jsonl#L400) | 16×4×512 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 18.0507 | 18.6909 | -3.43 | N/A / 原因 1 | N/A | 16.5411 | N/A（原因 1） | 0.243 / 0.483 |
| [401](../../calibration/reference/windows.jsonl#L401) | 17×4×512 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 19.3429 | 19.3279 | +0.08 | N/A / 原因 1 | N/A | 17.7257 | N/A（原因 1） | 0.247 / 0.508 |
| [402](../../calibration/reference/windows.jsonl#L402) | 24×4×1536 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 22.5952 | 21.9861 | +2.77 | N/A / 原因 1 | N/A | 19.7623 | N/A（原因 1） | 0.248 / 0.479 |
| [403](../../calibration/reference/windows.jsonl#L403) | 32×4×1024 | bf16 | [A1](#a1) / 未传 | [K4](#k4) | 19.5583 | 20.0697 | -2.55 | N/A / 原因 1 | N/A | 17.9128 | N/A（原因 1） | 0.246 / 0.476 |

## Ascend950PR_957b · 开发回放

| 原始行 | 本卡 shape | 主输入 dtype | 属性 / 前序 | 实现 | 模型 µs | CA µs | 误差 % | 理想 µs / 控制项 | U | 970 外推 µs | 970 外推 U | 查询 ms（950 / 970） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [404](../../calibration/reference/windows.jsonl#L404) | 1×4×64 | bf16 | [A1](#a1) / 未传 | [K5](#k5) | 18.0055 | 16.4636 | +9.37 | N/A / 原因 1 | N/A | 16.5017 | N/A（原因 1） | 0.482 / 0.481 |
| [405](../../calibration/reference/windows.jsonl#L405) | 1×4×128 | bf16 | [A1](#a1) / 未传 | [K2](#k2) | 16.9092 | 15.4964 | +9.12 | N/A / 原因 1 | N/A | 15.4944 | N/A（原因 1） | 0.476 / 0.477 |
| [406](../../calibration/reference/windows.jsonl#L406) | 1×4×512 | bf16 | [A1](#a1) / 未传 | [K2](#k2) | 17.4725 | 16.7939 | +4.04 | N/A / 原因 1 | N/A | 15.9999 | N/A（原因 1） | 0.499 / 0.478 |
| [407](../../calibration/reference/windows.jsonl#L407) | 2×4×128 | bf16 | [A1](#a1) / 未传 | [K2](#k2) | 16.9287 | 16.0539 | +5.45 | N/A / 原因 1 | N/A | 15.5123 | N/A（原因 1） | 0.481 / 0.479 |
| [408](../../calibration/reference/windows.jsonl#L408) | 3×4×128 | bf16 | [A1](#a1) / 未传 | [K2](#k2) | 16.9482 | 16.4061 | +3.30 | N/A / 原因 1 | N/A | 15.5302 | N/A（原因 1） | 0.472 / 0.479 |
| [409](../../calibration/reference/windows.jsonl#L409) | 4×4×256 | bf16 | [A1](#a1) / 未传 | [K2](#k2) | 17.3125 | 17.0606 | +1.48 | N/A / 原因 1 | N/A | 15.8574 | N/A（原因 1） | 0.482 / 0.479 |
| [410](../../calibration/reference/windows.jsonl#L410) | 4×4×1024 | bf16 | [A1](#a1) / 未传 | [K2](#k2) | 18.1599 | 18.5255 | -1.97 | N/A / 原因 1 | N/A | 16.6078 | N/A（原因 1） | 0.504 / 0.481 |
| [411](../../calibration/reference/windows.jsonl#L411) | 8×4×128 | bf16 | [A1](#a1) / 未传 | [K2](#k2) | 17.0459 | 16.7382 | +1.84 | N/A / 原因 1 | N/A | 15.6195 | N/A（原因 1） | 0.489 / 0.480 |
| [412](../../calibration/reference/windows.jsonl#L412) | 9×4×1024 | bf16 | [A1](#a1) / 未传 | [K2](#k2) | 18.5019 | 19.1988 | -3.63 | N/A / 原因 1 | N/A | 16.9086 | N/A（原因 1） | 0.481 / 0.481 |
| [413](../../calibration/reference/windows.jsonl#L413) | 16×4×128 | bf16 | [A1](#a1) / 未传 | [K2](#k2) | 17.2025 | 17.6770 | -2.68 | N/A / 原因 1 | N/A | 15.7624 | N/A（原因 1） | 0.480 / 0.473 |

## 实际调用的库实现

| 编号 | ACLNN 接口 | 实现提供方 | 设备 kernel，按执行顺序 |
| --- | --- | --- | --- |
| <a id="k1"></a>K1 | aclnnMhcPre + aclnnMhcSinkhorn + aclnnMhcPost | CANN 9.2.1 ops-transformer | `entry_kernel_1` → `entry_kernel_0` → `entry_kernel_5` |
| <a id="k2"></a>K2 | aclnnMhcPre + aclnnMhcSinkhorn + aclnnMhcPost | CANN 9.2.1 ops-transformer | `MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic` → `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` → `MhcPost_59130ccee6717ff2c0f15b485e731162_5` |
| <a id="k3"></a>K3 | aclnnMhcPre + aclnnMhcSinkhorn + aclnnMhcPost | CANN 9.2.1 ops-transformer | `entry_kernel_2` → `entry_kernel_0` → `entry_kernel_5` |
| <a id="k4"></a>K4 | aclnnMhcPre + aclnnMhcSinkhorn + aclnnMhcPost | CANN 9.2.1 ops-transformer | `entry_kernel_2` → `entry_kernel_0` → `entry_kernel_6` |
| <a id="k5"></a>K5 | aclnnMhcPre + aclnnMhcSinkhorn + aclnnMhcPost | CANN 9.2.1 ops-transformer | `MhcPre_52e042c374f575edbe357a32523dba2c_1_mix_aic` → `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` → `MhcPost_59130ccee6717ff2c0f15b485e731162_5` |

## 调用属性

| 编号 | 原始 attrs |
| --- | --- |
| <a id="a1"></a>A1 | `{"dtype":"bf16","implementation":"split","num_iters":20}` |

## Roofline 缺项说明

| 编号 | N/A 原因 |
| --- | --- |
| 1 | Missing peak or logical byte count: Cube |
