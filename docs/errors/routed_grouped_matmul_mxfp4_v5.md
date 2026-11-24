# routed_grouped_matmul_mxfp4_v5 · 逐窗误差

[全部合同](README.md) · [总误差分析](../error-analysis.md) · [原始逐窗记录](../../calibration/reference/windows.jsonl)

模型和 CA 均为设备微秒（µs）。有符号误差为 `100 × (模型 / CA − 1)`。逻辑系数 U 等于按逻辑张量流量和名义峰值折算的基准时间除以模型设备时间。这里没有同窗 HBM 控制器流量，物理利用率为 N/A。表内同时列出逻辑基准的控制项；合同汇总只对可比窗按 `Σ逻辑基准时间 / Σ模型时间` 聚合。查询列为归档 CPU 墙钟毫秒（ms），顺序为 950 / 970。970DT 的时间与 U 都是外推。

## 完整窗与误差

| 芯片 | 数据角色 | 不同 shape | 完整窗 | WAPE % | 最大绝对误差 % | U 可比窗 | 汇总逻辑系数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ascend950DT_9582 | 开发回放 | 3 | 3 | 2.96 | 5.12 | 3 | 1.344% |
| Ascend950DT_9582 | 冻结独立 | 10 | 10 | 9.82 | 21.07 | 10 | 15.676% |
| Ascend950PR_957b | 开发回放 | 3 | 3 | 1.86 | 3.21 | 3 | 3.839% |
| Ascend950PR_957b | 冻结独立 | 10 | 10 | 6.30 | 10.59 | 10 | 35.287% |

下表的行号链接到 JSONL 原记录，其中包含全长 trace 指纹和已归档的收据字段。A、H、K 分别链接到本页的属性、前序和实现。前序列的“未传”表示本次调用未提供流历史，“空流”表示已知从空的 OperatorStream 开始；显式 resident_aiv 归在属性内。

## Ascend950DT_9582 · 开发回放

| 原始行 | 本卡 shape | 主输入 dtype | 属性 / 前序 | 实现 | 模型 µs | CA µs | 误差 % | 逻辑基准 µs / 控制项 | 逻辑系数 | 已知物理下界 µs | 970 外推 µs | 970 外推逻辑系数 | 查询 ms（950 / 970） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [2155](../../calibration/reference/windows.jsonl#L2155) | 3×128×256 | mxfp4 | [A1](#a1) / 未传 | [K1](#k1) | 5.3603 | 5.6497 | -5.12 | 0.008998 / HBM | 0.168% | N/A | 4.9136 | 0.051% | 4.503 / 0.439 |
| [2157](../../calibration/reference/windows.jsonl#L2157) | 8×256×512 | mxfp4 | [A2](#a2) / 未传 | [K1](#k1) | 5.7218 | 5.9442 | -3.74 | 0.071200 / HBM | 1.244% | N/A | 5.2450 | 0.377% | 0.157 / 0.147 |
| [2159](../../calibration/reference/windows.jsonl#L2159) | 34×512×1024 | mxfp4 | [A3](#a3) / 未传 | [K1](#k1) | 6.2434 | 6.2594 | -0.26 | 0.152592 / HBM | 2.444% | N/A | 5.7231 | 0.741% | 0.150 / 0.254 |

## Ascend950DT_9582 · 冻结独立

| 原始行 | 本卡 shape | 主输入 dtype | 属性 / 前序 | 实现 | 模型 µs | CA µs | 误差 % | 逻辑基准 µs / 控制项 | 逻辑系数 | 已知物理下界 µs | 970 外推 µs | 970 外推逻辑系数 | 查询 ms（950 / 970） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [2161](../../calibration/reference/windows.jsonl#L2161) | 3×64×128 | mxfp4 | [A4](#a4) / 未传 | [K1](#k1) | 5.3603 | 5.5709 | -3.78 | 0.003411 / HBM | 0.064% | N/A | 4.9136 | 0.019% | 0.115 / 0.094 |
| [2163](../../calibration/reference/windows.jsonl#L2163) | 3×160×256 | mxfp4 | [A5](#a5) / 未传 | [K1](#k1) | 5.3873 | 5.4739 | -1.58 | 0.011222 / HBM | 0.208% | N/A | 4.9384 | 0.063% | 0.122 / 0.101 |
| [2165](../../calibration/reference/windows.jsonl#L2165) | 9×128×320 | mxfp4 | [A6](#a6) / 未传 | [K1](#k1) | 5.4864 | 5.8048 | -5.49 | 0.017278 / HBM | 0.315% | N/A | 5.0292 | 0.095% | 0.133 / 0.151 |
| [2167](../../calibration/reference/windows.jsonl#L2167) | 54×5120×2304 | mxfp4 | [A7](#a7) / 未传 | [K1](#k1) | 14.7893 | 18.7370 | -21.07 | 9.555084 / HBM | 64.608% | N/A | 7.4522 | 35.616% | 0.176 / 0.149 |
| [2169](../../calibration/reference/windows.jsonl#L2169) | 6×128×768 | mxfp4 | [A8](#a8) / 未传 | [K1](#k1) | 5.6124 | 5.9873 | -6.26 | 0.079332 / HBM | 1.414% | N/A | 5.1447 | 0.428% | 0.181 / 0.110 |
| [2171](../../calibration/reference/windows.jsonl#L2171) | 22×192×384 | mxfp4 | [A9](#a9) / 未传 | [K1](#k1) | 5.5191 | 5.8691 | -5.96 | 0.042402 / HBM | 0.768% | N/A | 5.0592 | 0.233% | 0.153 / 0.168 |
| [2173](../../calibration/reference/windows.jsonl#L2173) | 32×256×640 | mxfp4 | [A10](#a10) / 未传 | [K1](#k1) | 5.8768 | 6.1618 | -4.63 | 0.050336 / HBM | 0.857% | N/A | 5.3871 | 0.260% | 0.144 / 0.190 |
| [2175](../../calibration/reference/windows.jsonl#L2175) | 13×384×1024 | mxfp4 | [A11](#a11) / 未传 | [K1](#k1) | 5.9940 | 6.1982 | -3.29 | 0.160936 / HBM | 2.685% | N/A | 5.4945 | 0.814% | 0.148 / 0.129 |
| [2177](../../calibration/reference/windows.jsonl#L2177) | 12×128×1536 | mxfp4 | [A12](#a12) / 未传 | [K1](#k1) | 5.9906 | 7.1370 | -16.06 | 0.316560 / HBM | 5.284% | N/A | 5.4914 | 1.601% | 0.140 / 0.121 |
| [2179](../../calibration/reference/windows.jsonl#L2179) | 160×256×1024 | mxfp4 | [A13](#a13) / 未传 | [K1](#k1) | 6.4438 | 6.7576 | -4.64 | 0.181504 / HBM | 2.817% | N/A | 5.9068 | 0.854% | 0.142 / 0.204 |

## Ascend950PR_957b · 开发回放

| 原始行 | 本卡 shape | 主输入 dtype | 属性 / 前序 | 实现 | 模型 µs | CA µs | 误差 % | 逻辑基准 µs / 控制项 | 逻辑系数 | 已知物理下界 µs | 970 外推 µs | 970 外推逻辑系数 | 查询 ms（950 / 970） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [2156](../../calibration/reference/windows.jsonl#L2156) | 3×128×256 | mxfp4 | [A1](#a1) / 未传 | [K1](#k1) | 5.3603 | 5.5382 | -3.21 | 0.025709 / HBM | 0.480% | N/A | 4.9136 | 0.051% | 0.362 / 0.114 |
| [2158](../../calibration/reference/windows.jsonl#L2158) | 8×256×512 | mxfp4 | [A2](#a2) / 未传 | [K1](#k1) | 5.7218 | 5.7442 | -0.39 | 0.203429 / HBM | 3.555% | N/A | 5.2450 | 0.377% | 0.142 / 0.143 |
| [2160](../../calibration/reference/windows.jsonl#L2160) | 34×512×1024 | mxfp4 | [A3](#a3) / 未传 | [K1](#k1) | 6.2434 | 6.1200 | +2.02 | 0.435977 / HBM | 6.983% | N/A | 5.7231 | 0.741% | 0.230 / 0.269 |

## Ascend950PR_957b · 冻结独立

| 原始行 | 本卡 shape | 主输入 dtype | 属性 / 前序 | 实现 | 模型 µs | CA µs | 误差 % | 逻辑基准 µs / 控制项 | 逻辑系数 | 已知物理下界 µs | 970 外推 µs | 970 外推逻辑系数 | 查询 ms（950 / 970） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [2162](../../calibration/reference/windows.jsonl#L2162) | 3×64×128 | mxfp4 | [A4](#a4) / 未传 | [K1](#k1) | 5.3603 | 5.3085 | +0.98 | 0.009746 / HBM | 0.182% | N/A | 4.9136 | 0.019% | 0.130 / 0.093 |
| [2164](../../calibration/reference/windows.jsonl#L2164) | 3×160×256 | mxfp4 | [A5](#a5) / 未传 | [K1](#k1) | 5.3873 | 5.3873 | +0.000895 | 0.032063 / HBM | 0.595% | N/A | 4.9384 | 0.063% | 0.123 / 0.100 |
| [2166](../../calibration/reference/windows.jsonl#L2166) | 9×128×320 | mxfp4 | [A6](#a6) / 未传 | [K1](#k1) | 5.4864 | 5.6842 | -3.48 | 0.049367 / HBM | 0.900% | N/A | 5.0292 | 0.095% | 0.138 / 0.152 |
| [2168](../../calibration/reference/windows.jsonl#L2168) | 54×5120×2304 | mxfp4 | [A7](#a7) / 未传 | [K1](#k1) | 32.5345 | 29.4182 | +10.59 | 27.300240 / HBM | 83.912% | N/A | 7.4522 | 35.616% | 0.156 / 0.204 |
| [2170](../../calibration/reference/windows.jsonl#L2170) | 6×128×768 | mxfp4 | [A8](#a8) / 未传 | [K1](#k1) | 5.6124 | 5.9400 | -5.51 | 0.226663 / HBM | 4.039% | N/A | 5.1447 | 0.428% | 0.131 / 0.109 |
| [2172](../../calibration/reference/windows.jsonl#L2172) | 22×192×384 | mxfp4 | [A9](#a9) / 未传 | [K1](#k1) | 5.5191 | 5.9055 | -6.54 | 0.121149 / HBM | 2.195% | N/A | 5.0592 | 0.233% | 0.151 / 0.181 |
| [2174](../../calibration/reference/windows.jsonl#L2174) | 32×256×640 | mxfp4 | [A10](#a10) / 未传 | [K1](#k1) | 5.8768 | 5.7909 | +1.48 | 0.143817 / HBM | 2.447% | N/A | 5.3871 | 0.260% | 0.136 / 0.189 |
| [2176](../../calibration/reference/windows.jsonl#L2176) | 13×384×1024 | mxfp4 | [A11](#a11) / 未传 | [K1](#k1) | 5.9940 | 6.0570 | -1.04 | 0.459817 / HBM | 7.671% | N/A | 5.4945 | 0.814% | 0.147 / 0.130 |
| [2178](../../calibration/reference/windows.jsonl#L2178) | 12×128×1536 | mxfp4 | [A12](#a12) / 未传 | [K1](#k1) | 6.1387 | 6.8024 | -9.76 | 0.904457 / HBM | 14.734% | N/A | 5.4914 | 1.601% | 1.264 / 0.134 |
| [2180](../../calibration/reference/windows.jsonl#L2180) | 160×256×1024 | mxfp4 | [A13](#a13) / 未传 | [K1](#k1) | 6.4438 | 6.7830 | -5.00 | 0.518583 / HBM | 8.048% | N/A | 5.9068 | 0.854% | 0.158 / 0.223 |

## 实际调用的库实现

| 编号 | ACLNN 接口 | 实现提供方 | 设备 kernel，按执行顺序 |
| --- | --- | --- | --- |
| <a id="k1"></a>K1 | aclnnGroupedMatmulV5 | CANN 9.2.1 ops-transformer | `GroupedMatmul_40ca02eaeb01067b879dbc1dc766aa58_1` |

## 调用属性

| 编号 | 原始 attrs |
| --- | --- |
| <a id="a1"></a>A1 | `{"expert_counts":[1,2],"implementation":"aclnnGroupedMatmulV5"}` |
| <a id="a2"></a>A2 | `{"expert_counts":[2,2,2,2],"implementation":"aclnnGroupedMatmulV5"}` |
| <a id="a3"></a>A3 | `{"expert_counts":[0,11,0,23],"implementation":"aclnnGroupedMatmulV5"}` |
| <a id="a4"></a>A4 | `{"expert_counts":[1,1,1],"implementation":"aclnnGroupedMatmulV5"}` |
| <a id="a5"></a>A5 | `{"expert_counts":[2,0,1],"implementation":"aclnnGroupedMatmulV5"}` |
| <a id="a6"></a>A6 | `{"expert_counts":[3,1,0,5],"implementation":"aclnnGroupedMatmulV5"}` |
| <a id="a7"></a>A7 | `{"expert_counts":[9,9,9,9,9,9],"implementation":"aclnnGroupedMatmulV5"}` |
| <a id="a8"></a>A8 | `{"expert_counts":[1,1,1,1,1,1],"implementation":"aclnnGroupedMatmulV5"}` |
| <a id="a9"></a>A9 | `{"expert_counts":[7,2,9,4],"implementation":"aclnnGroupedMatmulV5"}` |
| <a id="a10"></a>A10 | `{"expert_counts":[16,16],"implementation":"aclnnGroupedMatmulV5"}` |
| <a id="a11"></a>A11 | `{"expert_counts":[9,3,0,1],"implementation":"aclnnGroupedMatmulV5"}` |
| <a id="a12"></a>A12 | `{"expert_counts":[1,1,1,1,1,1,1,1,1,1,1,1],"implementation":"aclnnGroupedMatmulV5"}` |
| <a id="a13"></a>A13 | `{"expert_counts":[32,64,48,16],"implementation":"aclnnGroupedMatmulV5"}` |
