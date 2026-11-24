# mhc_pre · 逐窗误差

[全部合同](README.md) · [总误差分析](../error-analysis.md) · [原始逐窗记录](../../calibration/reference/windows.jsonl)

模型和 CA 均为设备微秒（µs）。有符号误差为 `100 × (模型 / CA − 1)`。逻辑系数 U 等于按逻辑张量流量和名义峰值折算的基准时间除以模型设备时间。这里没有同窗 HBM 控制器流量，物理利用率为 N/A。表内同时列出逻辑基准的控制项；合同汇总只对可比窗按 `Σ逻辑基准时间 / Σ模型时间` 聚合。查询列为归档 CPU 墙钟毫秒（ms），顺序为 950 / 970。970DT 的时间与 U 都是外推。

## 完整窗与误差

| 芯片 | 数据角色 | 不同 shape | 完整窗 | WAPE % | 最大绝对误差 % | U 可比窗 | 汇总逻辑系数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ascend950DT_9582 | 开发回放 | 23 | 24 | 3.21 | 9.62 | 0 | N/A |
| Ascend950DT_9582 | 冻结独立 | 0 | 0 | N/A | N/A | 0 | N/A |
| Ascend950PR_957b | 开发回放 | 10 | 10 | 6.10 | 10.75 | 0 | N/A |
| Ascend950PR_957b | 冻结独立 | 0 | 0 | N/A | N/A | 0 | N/A |

下表的行号链接到 JSONL 原记录，其中包含全长 trace 指纹和已归档的收据字段。A、H、K 分别链接到本页的属性、前序和实现。前序列的“未传”表示本次调用未提供流历史，“空流”表示已知从空的 OperatorStream 开始；显式 resident_aiv 归在属性内。

## Ascend950DT_9582 · 开发回放

| 原始行 | 本卡 shape | 主输入 dtype | 属性 / 前序 | 实现 | 模型 µs | CA µs | 误差 % | 逻辑基准 µs / 控制项 | 逻辑系数 | 已知物理下界 µs | 970 外推 µs | 970 外推逻辑系数 | 查询 ms（950 / 970） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [414](../../calibration/reference/windows.jsonl#L414) | 1×4×64 | bf16 | [A1](#a1) / 未传 | [K1](#k1) | 9.5788 | 9.4818 | +1.02 | N/A / 原因 1 | N/A | N/A | 8.7803 | N/A（原因 1） | 0.116 / 0.207 |
| [415](../../calibration/reference/windows.jsonl#L415) | 1×4×80 | bf16 | [A1](#a1) / 未传 | [K1](#k1) | 9.7055 | 9.8024 | -0.99 | N/A / 原因 1 | N/A | N/A | 8.8963 | N/A（原因 1） | 0.114 / 0.259 |
| [416](../../calibration/reference/windows.jsonl#L416) | 1×4×128 | bf16 | [A1](#a1) / 未传 | [K2](#k2) | 8.3776 | 7.8788 | +6.33 | N/A / 原因 1 | N/A | N/A | 7.6794 | N/A（原因 1） | 0.113 / 0.229 |
| [417](../../calibration/reference/windows.jsonl#L417) | 1×4×128 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 8.3776 | 8.2709 | +1.29 | N/A / 原因 1 | N/A | N/A | 7.6794 | N/A（原因 1） | 0.129 / 0.239 |
| [418](../../calibration/reference/windows.jsonl#L418) | 1×4×512 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 8.7174 | 8.5442 | +2.03 | N/A / 原因 1 | N/A | N/A | 7.9893 | N/A（原因 1） | 0.120 / 0.217 |
| [419](../../calibration/reference/windows.jsonl#L419) | 2×4×64 | bf16 | [A1](#a1) / 未传 | [K1](#k1) | 9.5891 | 9.9618 | -3.74 | N/A / 原因 1 | N/A | N/A | 8.7898 | N/A（原因 1） | 0.124 / 0.216 |
| [420](../../calibration/reference/windows.jsonl#L420) | 2×4×128 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 8.3799 | 8.0055 | +4.68 | N/A / 原因 1 | N/A | N/A | 7.6815 | N/A（原因 1） | 0.116 / 0.218 |
| [421](../../calibration/reference/windows.jsonl#L421) | 3×4×128 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 8.3822 | 8.3121 | +0.84 | N/A / 原因 1 | N/A | N/A | 7.6836 | N/A（原因 1） | 0.114 / 0.209 |
| [422](../../calibration/reference/windows.jsonl#L422) | 4×4×96 | bf16 | [A1](#a1) / 未传 | [K1](#k1) | 9.8787 | 9.3115 | +6.09 | N/A / 原因 1 | N/A | N/A | 9.0549 | N/A（原因 1） | 0.115 / 0.218 |
| [423](../../calibration/reference/windows.jsonl#L423) | 4×4×256 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 8.5047 | 8.4206 | +1.00 | N/A / 原因 1 | N/A | N/A | 7.7955 | N/A（原因 1） | 0.116 / 0.212 |
| [424](../../calibration/reference/windows.jsonl#L424) | 4×4×1024 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 9.2260 | 8.7297 | +5.68 | N/A / 原因 1 | N/A | N/A | 8.4499 | N/A（原因 1） | 0.111 / 0.205 |
| [425](../../calibration/reference/windows.jsonl#L425) | 5×4×256 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 8.5094 | 8.2509 | +3.13 | N/A / 原因 1 | N/A | N/A | 7.7997 | N/A（原因 1） | 0.115 / 0.204 |
| [426](../../calibration/reference/windows.jsonl#L426) | 6×4×128 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 8.3892 | 8.4097 | -0.24 | N/A / 原因 1 | N/A | N/A | 7.6899 | N/A（原因 1） | 0.111 / 0.208 |
| [427](../../calibration/reference/windows.jsonl#L427) | 6×4×1024 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 9.2629 | 8.9733 | +3.23 | N/A / 原因 1 | N/A | N/A | 8.4832 | N/A（原因 1） | 0.148 / 0.214 |
| [428](../../calibration/reference/windows.jsonl#L428) | 8×4×128 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 8.3938 | 8.5424 | -1.74 | N/A / 原因 1 | N/A | N/A | 7.6942 | N/A（原因 1） | 0.113 / 0.212 |
| [429](../../calibration/reference/windows.jsonl#L429) | 8×4×512 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 8.7821 | 8.8661 | -0.95 | N/A / 原因 1 | N/A | N/A | 8.0480 | N/A（原因 1） | 0.112 / 0.210 |
| [430](../../calibration/reference/windows.jsonl#L430) | 9×4×1024 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 9.3184 | 9.0242 | +3.26 | N/A / 原因 1 | N/A | N/A | 8.5331 | N/A（原因 1） | 0.111 / 0.210 |
| [431](../../calibration/reference/windows.jsonl#L431) | 12×4×384 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 8.6804 | 9.1370 | -5.00 | N/A / 原因 1 | N/A | N/A | 7.9556 | N/A（原因 1） | 0.111 / 0.209 |
| [432](../../calibration/reference/windows.jsonl#L432) | 12×4×768 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 9.0965 | 9.0582 | +0.42 | N/A / 原因 1 | N/A | N/A | 8.3328 | N/A（原因 1） | 0.112 / 0.211 |
| [433](../../calibration/reference/windows.jsonl#L433) | 16×4×128 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 8.4123 | 9.3079 | -9.62 | N/A / 原因 1 | N/A | N/A | 7.7111 | N/A（原因 1） | 0.113 / 0.208 |
| [434](../../calibration/reference/windows.jsonl#L434) | 16×4×512 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 8.8561 | 9.3855 | -5.64 | N/A / 原因 1 | N/A | N/A | 8.1151 | N/A（原因 1） | 0.114 / 0.207 |
| [435](../../calibration/reference/windows.jsonl#L435) | 17×4×512 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 10.1180 | 10.0945 | +0.23 | N/A / 原因 1 | N/A | N/A | 9.2721 | N/A（原因 1） | 0.111 / 0.205 |
| [436](../../calibration/reference/windows.jsonl#L436) | 24×4×1536 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 11.5141 | 12.2006 | -5.63 | N/A / 原因 1 | N/A | N/A | 9.6317 | N/A（原因 1） | 0.112 / 0.210 |
| [437](../../calibration/reference/windows.jsonl#L437) | 32×4×1024 | bf16 | [A1](#a1) / 未传 | [K3](#k3) | 10.9963 | 11.3685 | -3.27 | N/A / 原因 1 | N/A | N/A | 10.0644 | N/A（原因 1） | 0.110 / 0.209 |

## Ascend950PR_957b · 开发回放

| 原始行 | 本卡 shape | 主输入 dtype | 属性 / 前序 | 实现 | 模型 µs | CA µs | 误差 % | 逻辑基准 µs / 控制项 | 逻辑系数 | 已知物理下界 µs | 970 外推 µs | 970 外推逻辑系数 | 查询 ms（950 / 970） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [438](../../calibration/reference/windows.jsonl#L438) | 1×4×64 | bf16 | [A1](#a1) / 未传 | [K4](#k4) | 9.5794 | 8.9515 | +7.01 | N/A / 原因 1 | N/A | N/A | 8.7803 | N/A（原因 1） | 0.215 / 0.244 |
| [439](../../calibration/reference/windows.jsonl#L439) | 1×4×128 | bf16 | [A1](#a1) / 未传 | [K2](#k2) | 8.3779 | 7.6376 | +9.69 | N/A / 原因 1 | N/A | N/A | 7.6794 | N/A（原因 1） | 0.215 / 0.206 |
| [440](../../calibration/reference/windows.jsonl#L440) | 1×4×512 | bf16 | [A1](#a1) / 未传 | [K2](#k2) | 8.7223 | 7.8758 | +10.75 | N/A / 原因 1 | N/A | N/A | 7.9893 | N/A（原因 1） | 0.214 / 0.203 |
| [441](../../calibration/reference/windows.jsonl#L441) | 2×4×128 | bf16 | [A1](#a1) / 未传 | [K2](#k2) | 8.3803 | 7.6436 | +9.64 | N/A / 原因 1 | N/A | N/A | 7.6815 | N/A（原因 1） | 0.209 / 0.210 |
| [442](../../calibration/reference/windows.jsonl#L442) | 3×4×128 | bf16 | [A1](#a1) / 未传 | [K2](#k2) | 8.3826 | 7.9212 | +5.82 | N/A / 原因 1 | N/A | N/A | 7.6836 | N/A（原因 1） | 0.211 / 0.208 |
| [443](../../calibration/reference/windows.jsonl#L443) | 4×4×256 | bf16 | [A1](#a1) / 未传 | [K2](#k2) | 8.5062 | 8.0848 | +5.21 | N/A / 原因 1 | N/A | N/A | 7.7955 | N/A（原因 1） | 0.211 / 0.206 |
| [444](../../calibration/reference/windows.jsonl#L444) | 4×4×1024 | bf16 | [A1](#a1) / 未传 | [K2](#k2) | 9.2470 | 9.4200 | -1.84 | N/A / 原因 1 | N/A | N/A | 8.4499 | N/A（原因 1） | 0.211 / 0.205 |
| [445](../../calibration/reference/windows.jsonl#L445) | 8×4×128 | bf16 | [A1](#a1) / 未传 | [K2](#k2) | 8.3942 | 8.0715 | +4.00 | N/A / 原因 1 | N/A | N/A | 7.6942 | N/A（原因 1） | 0.213 / 0.210 |
| [446](../../calibration/reference/windows.jsonl#L446) | 9×4×1024 | bf16 | [A1](#a1) / 未传 | [K2](#k2) | 9.3440 | 9.8648 | -5.28 | N/A / 原因 1 | N/A | N/A | 8.5331 | N/A（原因 1） | 0.211 / 0.211 |
| [447](../../calibration/reference/windows.jsonl#L447) | 16×4×128 | bf16 | [A1](#a1) / 未传 | [K2](#k2) | 8.4129 | 8.6964 | -3.26 | N/A / 原因 1 | N/A | N/A | 7.7111 | N/A（原因 1） | 0.216 / 0.231 |

## 实际调用的库实现

| 编号 | ACLNN 接口 | 实现提供方 | 设备 kernel，按执行顺序 |
| --- | --- | --- | --- |
| <a id="k1"></a>K1 | aclnnMhcPre | CANN 9.2.1 ops-transformer | `entry_kernel_1` |
| <a id="k2"></a>K2 | aclnnMhcPre | CANN 9.2.1 ops-transformer | `MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic` |
| <a id="k3"></a>K3 | aclnnMhcPre | CANN 9.2.1 ops-transformer | `entry_kernel_2` |
| <a id="k4"></a>K4 | aclnnMhcPre | CANN 9.2.1 ops-transformer | `MhcPre_52e042c374f575edbe357a32523dba2c_1_mix_aic` |

## 调用属性

| 编号 | 原始 attrs |
| --- | --- |
| <a id="a1"></a>A1 | `{"alpha_count":3,"dtype":"bf16","has_gamma":false,"return_intermediates":true}` |

## Roofline 缺项说明

| 编号 | N/A 原因 |
| --- | --- |
| 1 | Missing peak or logical byte count: Cube |
