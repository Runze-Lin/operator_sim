# 逐个 shape 的当前预测和完整 CA

以下 854 个历史指令窗按各自真实库合同、芯片、输入属性重新调用本版。设备时间单位均为 µs，偏差取 (本版/CA−1) × 100%。同形出现多次时保留各自的实际观测。每条 970DT 数值均为该输入的资源外推。

冻结后的独立样本另外提供[FA](../v0.25/coverage/supplement/score.json)、[有序 Post](post/report.md)、[Sinkhorn 与 PreSinkhorn](fusion/report.md)、[RMSNorm 两种前序](rms/report.md) 和 [Pro 大 Cast](memory/README.md)。内置 Post 和 RMSNorm 的旧前序协议见[总表](../../docs/error-analysis.md)。
逐窗原始指纹、直接调用参数和来源可在[JSON](replay.json)及[去重账本](../v0.25/coverage/audit.json)核查。

## addmm

| Shape | 目标 SKU | 输入属性 | 旧模型 | 本版 | CA | 偏差 | 970DT 外推 | 实测 kernel |
|---|---|---|---:|---:|---:|---:|---:|---|
| [1, 32, 16] | Ascend950DT_9582 | `{"self_shape":[32],"alpha":1,"beta":1}` | 3.092 | 3.092 | 3.288 | -5.96% | 2.834 | `entry_kernel_2` |
| [1, 32, 16] | Ascend950PR_957b | `{"self_shape":[32],"alpha":1,"beta":1}` | 3.092 | 3.092 | 3.002 | +3.00% | 2.834 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [1, 48, 80] | Ascend950DT_9582 | `{"self_shape":[48],"alpha":1,"beta":1}` | 3.208 | 3.208 | 3.078 | +4.23% | 2.941 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [1, 48, 80] | Ascend950PR_957b | `{"self_shape":[48],"alpha":1,"beta":1}` | 3.208 | 3.208 | 2.828 | +13.45% | 2.941 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [1, 64, 32] | Ascend950DT_9582 | `{"self_shape":[64],"alpha":1,"beta":1}` | 3.147 | 3.147 | 3.145 | +0.04% | 2.884 | `entry_kernel_2` |
| [2, 16, 16] | Ascend950DT_9582 | `{"self_shape":[16],"alpha":1,"beta":1}` | 3.083 | 3.083 | 2.992 | +3.02% | 2.826 | `entry_kernel_2` |
| [2, 16, 16] | Ascend950PR_957b | `{"self_shape":[16],"alpha":1,"beta":1}` | 3.083 | 3.083 | 2.728 | +13.00% | 2.826 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [2, 32, 32] | Ascend950DT_9582 | `{"self_shape":[32],"alpha":1,"beta":1}` | 3.110 | 3.110 | 3.408 | -8.76% | 2.851 | `entry_kernel_2` |
| [2, 64, 32] | Ascend950DT_9582 | `{"self_shape":[64],"alpha":1,"beta":1}` | 3.147 | 3.147 | 3.423 | -8.06% | 2.885 | `entry_kernel_2` |
| [3, 48, 96] | Ascend950DT_9582 | `{"self_shape":[48],"alpha":1,"beta":1}` | 3.236 | 3.236 | 3.352 | -3.45% | 2.966 | `entry_kernel_2` |
| [4, 80, 48] | Ascend950DT_9582 | `{"self_shape":[80],"alpha":1,"beta":1}` | 3.239 | 3.239 | 3.625 | -10.66% | 2.969 | `entry_kernel_2` |
| [4, 80, 48] | Ascend950PR_957b | `{"self_shape":[80],"alpha":1,"beta":1}` | 3.239 | 3.239 | 3.307 | -2.04% | 2.969 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [8, 96, 192] | Ascend950DT_9582 | `{"self_shape":[96],"alpha":1,"beta":1}` | 3.562 | 3.562 | 3.687 | -3.41% | 3.265 | `entry_kernel_2` |
| [9, 64, 128] | Ascend950DT_9582 | `{"self_shape":[64],"alpha":1,"beta":1}` | 3.364 | 3.364 | 3.342 | +0.66% | 3.084 | `entry_kernel_2` |
| [9, 64, 128] | Ascend950PR_957b | `{"self_shape":[64],"alpha":1,"beta":1}` | 3.364 | 3.364 | 3.059 | +9.97% | 3.084 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [9, 128, 256] | Ascend950DT_9582 | `{"self_shape":[128],"alpha":1,"beta":1}` | 3.705 | 3.705 | 3.756 | -1.38% | 3.396 | `entry_kernel_2` |
| [16, 64, 256] | Ascend950DT_9582 | `{"self_shape":[64],"alpha":1,"beta":1}` | 3.653 | 3.653 | 3.447 | +5.98% | 3.348 | `entry_kernel_2` |
| [16, 64, 256] | Ascend950DT_9582 | `{"self_shape":[64],"alpha":1,"beta":1}` | 3.653 | 3.653 | 3.588 | +1.81% | 3.348 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [16, 64, 256] | Ascend950PR_957b | `{"self_shape":[64],"alpha":1,"beta":1}` | 3.653 | 3.653 | 3.215 | +13.61% | 3.348 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [16, 192, 384] | Ascend950DT_9582 | `{"self_shape":[192],"alpha":1,"beta":1}` | 4.042 | 4.042 | 3.877 | +4.24% | 3.705 | `entry_kernel_2` |
| [16, 192, 384] | Ascend950PR_957b | `{"self_shape":[192],"alpha":1,"beta":1}` | 4.042 | 4.042 | 3.741 | +8.04% | 3.705 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [17, 80, 96] | Ascend950DT_9582 | `{"self_shape":[80],"alpha":1,"beta":1}` | 3.407 | 3.407 | 3.557 | -4.21% | 3.123 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [17, 80, 96] | Ascend950PR_957b | `{"self_shape":[80],"alpha":1,"beta":1}` | 3.407 | 3.407 | 3.410 | -0.09% | 3.123 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [17, 128, 128] | Ascend950DT_9582 | `{"self_shape":[128],"alpha":1,"beta":1}` | 3.478 | 3.478 | 3.393 | +2.52% | 3.188 | `entry_kernel_2` |
| [31, 64, 257] | Ascend950DT_9582 | `{"self_shape":[64],"alpha":1,"beta":1}` | 3.783 | 3.783 | 4.253 | -11.04% | 3.468 | `entry_kernel_2` |
| [31, 144, 160] | Ascend950DT_9582 | `{"self_shape":[144],"alpha":1,"beta":1}` | 3.582 | 3.582 | 4.082 | -12.25% | 3.283 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [31, 144, 160] | Ascend950PR_957b | `{"self_shape":[144],"alpha":1,"beta":1}` | 3.582 | 3.582 | 3.818 | -6.17% | 3.283 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [32, 80, 272] | Ascend950DT_9582 | `{"self_shape":[80],"alpha":1,"beta":1}` | 3.863 | 3.863 | 4.228 | -8.63% | 3.541 | `entry_kernel_2` |
| [48, 256, 384] | Ascend950DT_9582 | `{"self_shape":[256],"alpha":1,"beta":1}` | 4.152 | 4.152 | 4.027 | +3.12% | 3.806 | `entry_kernel_2` |
| [64, 128, 512] | Ascend950DT_9582 | `{"self_shape":[128],"alpha":1,"beta":1}` | 4.350 | 4.350 | 4.519 | -3.75% | 3.987 | `entry_kernel_2` |
| [64, 128, 512] | Ascend950PR_957b | `{"self_shape":[128],"alpha":1,"beta":1}` | 4.350 | 4.350 | 4.293 | +1.33% | 3.987 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [96, 176, 320] | Ascend950DT_9582 | `{"self_shape":[176],"alpha":1,"beta":1}` | 4.069 | 4.069 | 4.580 | -11.15% | 3.730 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [96, 176, 320] | Ascend950PR_957b | `{"self_shape":[176],"alpha":1,"beta":1}` | 4.069 | 4.069 | 4.499 | -9.56% | 3.730 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [96, 192, 256] | Ascend950DT_9582 | `{"self_shape":[192],"alpha":1,"beta":1}` | 3.883 | 3.883 | 3.925 | -1.06% | 3.560 | `entry_kernel_2` |
| [128, 192, 256] | Ascend950DT_9582 | `{"self_shape":[192],"alpha":1,"beta":1}` | 3.908 | 3.908 | 3.763 | +3.85% | 3.581 | `entry_kernel_2` |
| [128, 192, 256] | Ascend950PR_957b | `{"self_shape":[192],"alpha":1,"beta":1}` | 3.909 | 3.909 | 3.753 | +4.15% | 3.581 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [128, 256, 384] | Ascend950DT_9582 | `{"self_shape":[256],"alpha":1,"beta":1}` | 4.235 | 4.235 | 4.275 | -0.94% | 3.878 | `entry_kernel_2` |
| [192, 384, 192] | Ascend950DT_9582 | `{"self_shape":[384],"alpha":1,"beta":1}` | 3.842 | 3.842 | 3.840 | +0.06% | 3.530 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [192, 384, 192] | Ascend950PR_957b | `{"self_shape":[384],"alpha":1,"beta":1}` | 3.846 | 3.846 | 3.794 | +1.38% | 3.530 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [256, 384, 256] | Ascend950DT_9582 | `{"self_shape":[384],"alpha":1,"beta":1}` | 4.052 | 4.052 | 4.381 | -7.50% | 3.682 | `entry_kernel_2` |
| [256, 512, 256] | Ascend950DT_9582 | `{"self_shape":[512],"alpha":1,"beta":1}` | 4.088 | 4.088 | 4.327 | -5.54% | 3.695 | `entry_kernel_2` |
| [256, 512, 256] | Ascend950PR_957b | `{"self_shape":[512],"alpha":1,"beta":1}` | 4.392 | 4.392 | 4.230 | +3.83% | 3.695 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [257, 256, 192] | Ascend950DT_9582 | `{"self_shape":[256],"alpha":1,"beta":1}` | 3.839 | 3.839 | 3.645 | +5.33% | 3.526 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [257, 256, 192] | Ascend950PR_957b | `{"self_shape":[256],"alpha":1,"beta":1}` | 3.843 | 3.843 | 3.750 | +2.48% | 3.526 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [272, 480, 192] | Ascend950DT_9582 | `{"self_shape":[480],"alpha":1,"beta":1}` | 4.044 | 4.044 | 4.538 | -10.87% | 3.564 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [272, 480, 192] | Ascend950PR_957b | `{"self_shape":[480],"alpha":1,"beta":1}` | 4.175 | 4.175 | 4.410 | -5.33% | 3.564 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [384, 480, 256] | Ascend950DT_9582 | `{"self_shape":[480],"alpha":1,"beta":1}` | 4.428 | 4.428 | 5.079 | -12.82% | 3.725 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [384, 480, 256] | Ascend950PR_957b | `{"self_shape":[480],"alpha":1,"beta":1}` | 4.731 | 4.731 | 5.075 | -6.78% | 3.725 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [511, 384, 192] | Ascend950DT_9582 | `{"self_shape":[384],"alpha":1,"beta":1}` | 4.333 | 4.333 | 4.158 | +4.22% | 3.585 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [511, 384, 192] | Ascend950PR_957b | `{"self_shape":[384],"alpha":1,"beta":1}` | 4.448 | 4.448 | 4.253 | +4.57% | 3.585 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [512, 512, 256] | Ascend950DT_9582 | `{"self_shape":[512],"alpha":1,"beta":1}` | 4.773 | 4.773 | 4.479 | +6.57% | 3.761 | `entry_kernel_2` |
| [512, 512, 256] | Ascend950PR_957b | `{"self_shape":[512],"alpha":1,"beta":1}` | 4.856 | 4.856 | 4.687 | +3.62% | 3.761 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [1024, 192, 256] | Ascend950DT_9582 | `{"self_shape":[192],"alpha":1,"beta":1}` | 4.585 | 4.585 | 4.252 | +7.85% | 3.725 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [1024, 192, 256] | Ascend950PR_957b | `{"self_shape":[192],"alpha":1,"beta":1}` | 4.736 | 4.736 | 4.449 | +6.45% | 3.725 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |

## attn_res_fwd

| Shape | 目标 SKU | 输入属性 | 旧模型 | 本版 | CA | 偏差 | 970DT 外推 | 实测 kernel |
|---|---|---|---:|---:|---:|---:|---:|---|
| [1, 256, 8, 1] | Ascend950DT_9582 | `{"addend":true,"materialize":true,"write_idx":1}` | 6.668 | 6.668 | 6.325 | +5.42% | 6.112 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [1, 256, 8, 2] | Ascend950DT_9582 | `{"addend":true,"materialize":true,"write_idx":2}` | 7.250 | 7.250 | 7.004 | +3.51% | 6.645 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [1, 256, 8, 2] | Ascend950PR_957b | `{"addend":true,"materialize":true,"write_idx":2}` | 7.253 | 7.253 | 6.583 | +10.17% | 6.645 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [1, 512, 8, 2] | Ascend950PR_957b | `{"addend":true,"materialize":true,"write_idx":2}` | 7.410 | 7.410 | 7.039 | +5.27% | 6.787 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [1, 1024, 8, 7] | Ascend950DT_9582 | `{"addend":true,"materialize":true,"write_idx":7}` | 11.009 | 11.009 | 11.573 | -4.88% | 10.091 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [1, 1024, 8, 7] | Ascend950PR_957b | `{"addend":true,"materialize":true,"write_idx":7}` | 11.031 | 11.031 | 10.764 | +2.48% | 10.091 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [1, 7168, 8, 1] | Ascend950DT_9582 | `{"addend":true,"materialize":true,"write_idx":1}` | 10.152 | 10.152 | 9.505 | +6.80% | 9.303 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [1, 7168, 8, 1] | Ascend950PR_957b | `{"addend":true,"materialize":true,"write_idx":1}` | 10.221 | 10.221 | 9.499 | +7.59% | 9.303 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [1, 7168, 8, 3] | Ascend950DT_9582 | `{"addend":true,"materialize":true,"write_idx":3}` | 12.701 | 12.701 | 11.572 | +9.76% | 11.639 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [1, 7168, 8, 3] | Ascend950PR_957b | `{"addend":true,"materialize":true,"write_idx":3}` | 12.799 | 12.799 | 11.488 | +11.42% | 11.639 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [2, 512, 8, 1] | Ascend950DT_9582 | `{"addend":true,"materialize":true,"write_idx":1}` | 6.797 | 6.797 | 7.122 | -4.57% | 6.230 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [2, 512, 8, 1] | Ascend950PR_957b | `{"addend":true,"materialize":true,"write_idx":1}` | 6.802 | 6.802 | 6.598 | +3.09% | 6.230 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [2, 1024, 8, 4] | Ascend950DT_9582 | `{"addend":true,"materialize":true,"write_idx":4}` | 9.032 | 9.032 | 9.178 | -1.59% | 8.278 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [2, 1024, 8, 4] | Ascend950PR_957b | `{"addend":true,"materialize":true,"write_idx":4}` | 9.048 | 9.048 | 8.973 | +0.83% | 8.278 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [2, 2048, 8, 3] | Ascend950DT_9582 | `{"addend":true,"materialize":true,"write_idx":3}` | 9.094 | 9.094 | 9.490 | -4.17% | 8.335 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [2, 2048, 8, 3] | Ascend950PR_957b | `{"addend":true,"materialize":true,"write_idx":3}` | 9.122 | 9.122 | 9.085 | +0.40% | 8.335 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [2, 4096, 8, 1] | Ascend950DT_9582 | `{"addend":true,"materialize":true,"write_idx":1}` | 8.603 | 8.603 | 8.113 | +6.04% | 7.885 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [2, 4096, 8, 1] | Ascend950PR_957b | `{"addend":true,"materialize":true,"write_idx":1}` | 8.643 | 8.643 | 7.984 | +8.24% | 7.885 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [2, 4096, 8, 2] | Ascend950DT_9582 | `{"addend":true,"materialize":true,"write_idx":2}` | 9.570 | 9.570 | 9.398 | +1.83% | 8.771 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [2, 4096, 8, 2] | Ascend950PR_957b | `{"addend":true,"materialize":true,"write_idx":2}` | 9.618 | 9.618 | 9.007 | +6.78% | 8.771 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [2, 7168, 8, 3] | Ascend950DT_9582 | `{"addend":true,"materialize":true,"write_idx":3}` | 12.701 | 12.701 | 11.745 | +8.14% | 11.639 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [2, 7168, 8, 3] | Ascend950PR_957b | `{"addend":true,"materialize":true,"write_idx":3}` | 12.799 | 12.799 | 11.686 | +9.53% | 11.639 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [3, 512, 8, 1] | Ascend950DT_9582 | `{"addend":true,"materialize":true,"write_idx":1}` | 6.797 | 6.797 | 7.010 | -3.04% | 6.230 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [3, 512, 8, 1] | Ascend950PR_957b | `{"addend":true,"materialize":true,"write_idx":1}` | 6.802 | 6.802 | 6.697 | +1.56% | 6.230 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [3, 512, 8, 3] | Ascend950DT_9582 | `{"addend":true,"materialize":true,"write_idx":3}` | 8.012 | 8.012 | 8.479 | -5.51% | 7.344 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [3, 512, 8, 3] | Ascend950PR_957b | `{"addend":true,"materialize":true,"write_idx":3}` | 8.019 | 8.019 | 7.776 | +3.13% | 7.344 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [3, 2048, 8, 4] | Ascend950DT_9582 | `{"addend":true,"materialize":true,"write_idx":4}` | 9.856 | 9.856 | 9.824 | +0.33% | 9.033 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [3, 2048, 8, 4] | Ascend950PR_957b | `{"addend":true,"materialize":true,"write_idx":4}` | 9.888 | 9.888 | 9.579 | +3.23% | 9.033 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [3, 7168, 8, 1] | Ascend950DT_9582 | `{"addend":true,"materialize":true,"write_idx":1}` | 10.152 | 10.152 | 9.712 | +4.53% | 9.303 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [3, 7168, 8, 1] | Ascend950PR_957b | `{"addend":true,"materialize":true,"write_idx":1}` | 10.221 | 10.221 | 9.474 | +7.88% | 9.303 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [4, 256, 8, 3] | Ascend950DT_9582 | `{"addend":true,"materialize":true,"write_idx":3}` | 7.832 | 7.832 | 7.585 | +3.25% | 7.179 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [4, 256, 8, 3] | Ascend950PR_957b | `{"addend":true,"materialize":true,"write_idx":3}` | 7.835 | 7.835 | 7.368 | +6.34% | 7.179 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [4, 256, 8, 6] | Ascend950DT_9582 | `{"addend":true,"materialize":true,"write_idx":6}` | 9.578 | 9.578 | 9.449 | +1.36% | 8.780 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [4, 256, 8, 6] | Ascend950PR_957b | `{"addend":true,"materialize":true,"write_idx":6}` | 9.583 | 9.583 | 9.181 | +4.38% | 8.780 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [4, 1024, 8, 2] | Ascend950DT_9582 | `{"addend":true,"materialize":true,"write_idx":2}` | 7.714 | 7.714 | 7.852 | -1.76% | 7.070 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [4, 1024, 8, 2] | Ascend950PR_957b | `{"addend":true,"materialize":true,"write_idx":2}` | 7.726 | 7.726 | 7.494 | +3.09% | 7.070 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [5, 512, 8, 4] | Ascend950DT_9582 | `{"addend":true,"materialize":true,"write_idx":4}` | 8.620 | 8.620 | 9.237 | -6.68% | 7.901 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [5, 512, 8, 4] | Ascend950PR_957b | `{"addend":true,"materialize":true,"write_idx":4}` | 8.628 | 8.628 | 8.473 | +1.83% | 7.901 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [5, 2048, 8, 1] | Ascend950DT_9582 | `{"addend":true,"materialize":true,"write_idx":1}` | 7.571 | 7.571 | 7.574 | -0.04% | 6.939 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [5, 2048, 8, 1] | Ascend950PR_957b | `{"addend":true,"materialize":true,"write_idx":1}` | 7.591 | 7.591 | 7.433 | +2.12% | 6.939 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [8, 512, 8, 2] | Ascend950DT_9582 | `{"addend":true,"materialize":true,"write_idx":2}` | 7.404 | 7.404 | 7.652 | -3.24% | 6.787 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |
| [8, 512, 8, 2] | Ascend950PR_957b | `{"addend":true,"materialize":true,"write_idx":2}` | 7.410 | 7.410 | 7.223 | +2.59% | 6.787 | `AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020` |

## cast_bf16_fp32

| Shape | 目标 SKU | 输入属性 | 旧模型 | 本版 | CA | 偏差 | 970DT 外推 | 实测 kernel |
|---|---|---|---:|---:|---:|---:|---:|---|
| [1, 64] | Ascend950DT_9582 | `{}` | 1.957 | 1.957 | 2.172 | -9.89% | 1.794 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [1, 64] | Ascend950PR_957b | `{}` | 1.957 | 1.957 | 1.996 | -1.92% | 1.794 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [1, 3584] | Ascend950DT_9582 | `{}` | 2.099 | 2.099 | 2.216 | -5.28% | 1.924 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [1, 3584] | Ascend950PR_957b | `{}` | 2.100 | 2.100 | 2.104 | -0.17% | 1.924 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [3, 7168] | Ascend950DT_9582 | `{}` | 2.203 | 2.203 | 2.379 | -7.43% | 2.019 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [3, 7168] | Ascend950PR_957b | `{}` | 2.204 | 2.204 | 2.441 | -9.73% | 2.019 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [9, 4096] | Ascend950DT_9582 | `{}` | 2.278 | 2.278 | 2.325 | -2.03% | 2.088 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [9, 4096] | Ascend950PR_957b | `{}` | 2.279 | 2.279 | 2.220 | +2.66% | 2.088 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [9, 5120] | Ascend950DT_9582 | `{}` | 2.326 | 2.326 | 2.412 | -3.57% | 2.132 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [9, 5120] | Ascend950PR_957b | `{}` | 2.327 | 2.327 | 2.613 | -10.96% | 2.132 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [16, 7168] | Ascend950DT_9582 | `{}` | 2.657 | 2.657 | 2.722 | -2.40% | 2.435 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [16, 7168] | Ascend950PR_957b | `{}` | 2.658 | 2.658 | 2.830 | -6.09% | 2.435 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [17, 32] | Ascend950DT_9582 | `{}` | 1.995 | 1.995 | 2.406 | -17.10% | 1.828 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [17, 32] | Ascend950PR_957b | `{}` | 1.995 | 1.995 | 1.953 | +2.16% | 1.828 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [32, 4096] | Ascend950DT_9582 | `{}` | 2.736 | 2.736 | 2.771 | -1.25% | 2.508 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [32, 4096] | Ascend950PR_957b | `{}` | 2.681 | 2.681 | 2.867 | -6.49% | 2.508 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [48, 7168] | Ascend950DT_9582 | `{}` | 2.991 | 2.991 | 2.892 | +3.42% | 3.078 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [48, 7168] | Ascend950PR_957b | `{}` | 2.972 | 2.972 | 2.914 | +2.01% | 3.078 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [64, 4096] | Ascend950DT_9582 | `{}` | 2.893 | 2.893 | 2.893 | -0.00% | 3.030 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [65, 5120] | Ascend950DT_9582 | `{}` | 2.978 | 2.978 | 3.050 | -2.37% | 3.071 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [65, 5120] | Ascend950PR_957b | `{}` | 2.958 | 2.958 | 2.936 | +0.72% | 3.071 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [72, 4096] | Ascend950DT_9582 | `{}` | 2.932 | 2.932 | 2.851 | +2.84% | 3.049 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [72, 4096] | Ascend950PR_957b | `{}` | 2.907 | 2.907 | 2.975 | -2.31% | 3.049 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [96, 7168] | Ascend950DT_9582 | `{}` | 3.401 | 3.401 | 3.096 | +9.87% | 3.278 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [96, 7168] | Ascend950PR_957b | `{}` | 3.445 | 3.445 | 2.875 | +19.83% | 3.278 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [111, 7168] | Ascend950DT_9582 | `{}` | 3.531 | 3.531 | 3.172 | +11.30% | 3.341 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [111, 7168] | Ascend950PR_957b | `{}` | 3.592 | 3.592 | 3.744 | -4.06% | 3.341 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [128, 5120] | Ascend950DT_9582 | `{}` | 3.362 | 3.362 | 3.064 | +9.75% | 3.260 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [128, 5120] | Ascend950PR_957b | `{}` | 3.400 | 3.400 | 3.562 | -4.55% | 3.260 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [144, 6144] | Ascend950DT_9582 | `{}` | 3.636 | 3.636 | 3.522 | +3.25% | 3.393 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [144, 6144] | Ascend950PR_957b | `{}` | 3.715 | 3.715 | 4.111 | -9.62% | 3.393 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [255, 7168] | Ascend950DT_9582 | `{}` | 4.576 | 4.576 | 4.784 | -4.34% | 3.943 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [255, 7168] | Ascend950PR_957b | `{}` | 4.719 | 4.719 | 5.884 | -19.81% | 3.943 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [479, 5120] | Ascend950DT_9582 | `{}` | 5.078 | 5.078 | 4.902 | +3.59% | 4.308 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [479, 5120] | Ascend950PR_957b | `{}` | 5.296 | 5.296 | 6.532 | -18.92% | 4.308 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [511, 5120] | Ascend950DT_9582 | `{}` | 5.210 | 5.210 | 5.380 | -3.16% | 4.387 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [511, 5120] | Ascend950PR_957b | `{}` | 5.446 | 5.446 | 6.576 | -17.18% | 4.387 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [3072, 5120] | Ascend950DT_9582 | `{}` | 15.749 | 15.749 | 16.367 | -3.77% | 9.538 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [3072, 5120] | Ascend950PR_957b | `{}` | 23.532 | 23.532 | 23.237 | +1.27% | 9.538 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [4096, 7168] | Ascend950DT_9582 | `{}` | 28.424 | 28.424 | 29.724 | -4.37% | 14.894 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [4096, 7168] | Ascend950PR_957b | `{}` | 70.967 | 70.967 | 54.612 | +29.95% | 14.894 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [8192, 5120] | Ascend950DT_9582 | `{}` | 52.256 | 52.256 | 55.286 | -5.48% | 19.838 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [8192, 5120] | Ascend950PR_957b | `{}` | 124.894 | 124.894 | 114.128 | +9.43% | 19.838 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [32769, 32] | Ascend950DT_9582 | `{}` | 3.834 | 3.834 | 3.676 | +4.30% | 3.489 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [32769, 32] | Ascend950PR_957b | `{}` | 3.941 | 3.941 | 4.479 | -12.02% | 3.489 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [98305, 64] | Ascend950DT_9582 | `{}` | 8.165 | 8.165 | 8.234 | -0.84% | 5.831 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [98305, 64] | Ascend950PR_957b | `{}` | 10.050 | 10.050 | 11.218 | -10.41% | 5.831 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [131073, 64] | Ascend950DT_9582 | `{}` | 9.850 | 9.850 | 10.518 | -6.34% | 6.656 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [131073, 64] | Ascend950PR_957b | `{}` | 13.046 | 13.046 | 14.119 | -7.60% | 6.656 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [262144, 32] | Ascend950DT_9582 | `{}` | 9.850 | 9.850 | 9.970 | -1.20% | 6.656 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [262144, 32] | Ascend950PR_957b | `{}` | 13.046 | 13.046 | 14.080 | -7.34% | 6.656 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [262144, 64] | Ascend950DT_9582 | `{}` | 16.592 | 16.592 | 17.041 | -2.64% | 9.951 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [262144, 64] | Ascend950PR_957b | `{}` | 25.030 | 25.030 | 23.762 | +5.34% | 9.951 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [393216, 32] | Ascend950DT_9582 | `{}` | 13.221 | 13.221 | 13.728 | -3.69% | 8.303 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [393216, 32] | Ascend950PR_957b | `{}` | 19.038 | 19.038 | 18.826 | +1.13% | 8.303 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [1048576, 32] | Ascend950DT_9582 | `{}` | 36.368 | 36.368 | 35.535 | +2.35% | 16.542 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [1048576, 32] | Ascend950PR_957b | `{}` | 88.943 | 88.943 | 70.711 | +25.78% | 16.542 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [1048576, 64] | Ascend950DT_9582 | `{}` | 99.918 | 99.918 | 105.374 | -5.18% | 42.536 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [1048576, 64] | Ascend950PR_957b | `{}` | 232.748 | 232.748 | 224.719 | +3.57% | 42.536 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [1572864, 32] | Ascend950DT_9582 | `{}` | 68.143 | 68.143 | 72.238 | -5.67% | 27.001 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [1572864, 32] | Ascend950PR_957b | `{}` | 160.845 | 160.845 | 149.959 | +7.26% | 27.001 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [2097152, 32] | Ascend950DT_9582 | `{}` | 99.918 | 99.918 | 106.757 | -6.41% | 42.536 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |
| [2097152, 32] | Ascend950PR_957b | `{}` | 232.748 | 232.748 | 223.493 | +4.14% | 42.536 | `Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0` |

## fa

| Shape | 目标 SKU | 输入属性 | 旧模型 | 本版 | CA | 偏差 | 970DT 外推 | 实测 kernel |
|---|---|---|---:|---:|---:|---:|---:|---|
| [1, 64, 64, 1, 1] | Ascend950DT_9582 | `{}` | 8.616 | 8.616 | 8.937 | -3.60% | 7.898 | `entry_kernel_2050000040100220` |
| [1, 64, 64, 1, 1] | Ascend950PR_957b | `{}` | 8.616 | 8.616 | 8.535 | +0.94% | 7.898 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [1, 128, 64, 1, 1] | Ascend950DT_9582 | `{}` | 8.687 | 8.687 | 8.784 | -1.10% | 7.963 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [1, 128, 64, 1, 1] | Ascend950PR_957b | `{}` | 8.687 | 8.687 | 8.190 | +6.07% | 7.963 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [1, 224, 64, 1, 1] | Ascend950DT_9582 | `{}` | 8.946 | 8.946 | 8.826 | +1.36% | 8.200 | `entry_kernel_2050000040100220` |
| [1, 224, 64, 1, 1] | Ascend950PR_957b | `{}` | 8.946 | 8.946 | 8.634 | +3.62% | 8.200 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [3, 192, 64, 1, 2] | Ascend950DT_9582 | `{}` | 8.955 | 8.955 | 9.167 | -2.31% | 8.214 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [3, 192, 64, 1, 2] | Ascend950PR_957b | `{}` | 8.955 | 8.955 | 8.766 | +2.15% | 8.214 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [4, 160, 64, 1, 1] | Ascend950DT_9582 | `{}` | 8.956 | 8.956 | 9.461 | -5.34% | 8.220 | `entry_kernel_2050000040100220` |
| [4, 160, 64, 1, 1] | Ascend950PR_957b | `{}` | 8.956 | 8.956 | 8.657 | +3.46% | 8.220 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [4, 256, 128, 1, 1] | Ascend950DT_9582 | `{}` | 9.352 | 9.352 | 9.367 | -0.16% | 8.594 | `entry_kernel_2050000070100220` |
| [4, 256, 128, 1, 1] | Ascend950PR_957b | `{}` | 9.353 | 9.353 | 8.520 | +9.78% | 8.594 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic` |
| [8, 128, 128, 1, 1] | Ascend950DT_9582 | `{}` | 8.889 | 8.889 | 9.136 | -2.70% | 8.190 | `entry_kernel_2050000070100220` |
| [8, 128, 128, 1, 1] | Ascend950PR_957b | `{}` | 8.890 | 8.890 | 8.317 | +6.89% | 8.190 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic` |
| [8, 192, 64, 1, 1] | Ascend950DT_9582 | `{}` | 8.973 | 8.973 | 8.805 | +1.91% | 8.247 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [8, 192, 64, 1, 1] | Ascend950PR_957b | `{}` | 8.974 | 8.974 | 8.552 | +4.93% | 8.247 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [9, 256, 64, 4, 2] | Ascend950PR_957b | `{}` | 8.981 | 8.981 | 9.164 | -1.99% | 8.253 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [9, 256, 128, 4, 1] | Ascend950DT_9582 | `{}` | 9.399 | 9.399 | 9.690 | -3.01% | 8.658 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic` |
| [9, 256, 128, 4, 1] | Ascend950PR_957b | `{}` | 9.401 | 9.401 | 9.058 | +3.79% | 8.658 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic` |
| [9, 384, 64, 1, 1] | Ascend950DT_9582 | `{}` | 9.239 | 9.239 | 9.455 | -2.28% | 8.490 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [9, 384, 64, 1, 1] | Ascend950PR_957b | `{}` | 9.240 | 9.240 | 8.834 | +4.60% | 8.490 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [9, 384, 128, 2, 2] | Ascend950PR_957b | `{}` | 9.897 | 9.897 | 9.421 | +5.05% | 9.113 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic` |
| [9, 1008, 64, 1, 1] | Ascend950DT_9582 | `{}` | 10.532 | 10.532 | 11.415 | -7.73% | 9.676 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [9, 1008, 64, 1, 1] | Ascend950PR_957b | `{}` | 10.533 | 10.533 | 10.915 | -3.50% | 9.676 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [9, 1023, 64, 1, 1] | Ascend950DT_9582 | `{}` | 10.532 | 10.532 | 11.287 | -6.68% | 9.676 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [9, 1023, 64, 1, 1] | Ascend950PR_957b | `{}` | 10.533 | 10.533 | 10.985 | -4.11% | 9.676 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [9, 2048, 64, 1, 1] | Ascend950PR_957b | `{}` | 14.312 | 14.312 | 13.273 | +7.83% | 13.139 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic` |
| [9, 2049, 64, 1, 1] | Ascend950PR_957b | `{}` | 14.570 | 14.570 | 13.146 | +10.83% | 13.376 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic` |
| [9, 3072, 64, 1, 1] | Ascend950DT_9582 | `{}` | 16.380 | 16.380 | 16.010 | +2.31% | 15.036 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic` |
| [9, 3072, 64, 1, 1] | Ascend950PR_957b | `{}` | 16.380 | 16.380 | 15.156 | +8.08% | 15.036 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic` |
| [9, 4096, 64, 1, 1] | Ascend950DT_9582 | `{}` | 18.448 | 18.448 | 18.226 | +1.22% | 16.932 | `entry_kernel_2050000040140220` |
| [9, 4096, 64, 1, 1] | Ascend950PR_957b | `{}` | 18.449 | 18.449 | 16.880 | +9.30% | 16.932 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic` |
| [9, 8191, 64, 1, 1] | Ascend950DT_9582 | `{}` | 26.724 | 26.724 | 26.269 | +1.73% | 24.518 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic` |
| [9, 8191, 64, 1, 1] | Ascend950PR_957b | `{}` | 26.725 | 26.725 | 25.197 | +6.06% | 24.518 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic` |
| [9, 8192, 64, 1, 1] | Ascend950PR_957b | `{}` | 26.725 | 26.725 | 25.149 | +6.27% | 24.518 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic` |
| [9, 12289, 64, 1, 1] | Ascend950PR_957b | `{}` | 35.259 | 35.259 | 33.039 | +6.72% | 32.341 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic` |
| [9, 16384, 64, 1, 1] | Ascend950DT_9582 | `{}` | 43.275 | 43.275 | 42.932 | +0.80% | 39.690 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic` |
| [9, 16384, 64, 1, 1] | Ascend950PR_957b | `{}` | 43.276 | 43.276 | 41.556 | +4.14% | 39.690 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic` |
| [9, 16385, 64, 1, 1] | Ascend950DT_9582 | `{}` | 43.533 | 43.533 | 43.219 | +0.73% | 39.927 | `entry_kernel_2050000040140220` |
| [9, 16385, 64, 1, 1] | Ascend950PR_957b | `{}` | 43.534 | 43.534 | 41.293 | +5.43% | 39.927 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic` |
| [9, 32767, 64, 1, 1] | Ascend950PR_957b | `{}` | 76.377 | 76.377 | 73.035 | +4.58% | 70.033 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic` |
| [9, 32768, 64, 1, 1] | Ascend950DT_9582 | `{}` | 76.377 | 76.377 | 75.980 | +0.52% | 70.033 | `entry_kernel_2050000040140220` |
| [9, 32768, 64, 1, 1] | Ascend950PR_957b | `{}` | 76.377 | 76.377 | 72.856 | +4.83% | 70.033 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic` |
| [9, 131072, 64, 1, 1] | Ascend950DT_9582 | `{}` | 274.987 | 274.987 | 274.821 | +0.06% | 252.093 | `entry_kernel_2050000040140220` |
| [16, 64, 64, 1, 1] | Ascend950DT_9582 | `{}` | 8.678 | 8.678 | 8.856 | -2.01% | 7.997 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [16, 64, 64, 1, 1] | Ascend950DT_9582 | `{}` | 8.678 | 8.678 | 9.052 | -4.14% | 7.997 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [16, 64, 64, 1, 1] | Ascend950PR_957b | `{}` | 8.679 | 8.679 | 8.472 | +2.45% | 7.997 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [16, 128, 64, 1, 4] | Ascend950DT_9582 | `{}` | 8.749 | 8.749 | 8.758 | -0.10% | 8.063 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [16, 128, 64, 1, 4] | Ascend950PR_957b | `{}` | 8.751 | 8.751 | 8.539 | +2.48% | 8.063 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [16, 128, 64, 1, 8] | Ascend950DT_9582 | `{}` | 8.749 | 8.749 | 9.402 | -6.94% | 8.063 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [16, 128, 64, 1, 8] | Ascend950PR_957b | `{}` | 8.751 | 8.751 | 8.731 | +0.23% | 8.063 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [16, 128, 128, 1, 1] | Ascend950DT_9582 | `{}` | 8.955 | 8.955 | 8.715 | +2.75% | 8.293 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic` |
| [16, 128, 128, 1, 1] | Ascend950PR_957b | `{}` | 8.957 | 8.957 | 8.224 | +8.91% | 8.293 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic` |
| [16, 1025, 64, 1, 1] | Ascend950DT_9582 | `{}` | 12.528 | 12.528 | 12.384 | +1.16% | 12.046 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic` |
| [16, 1025, 64, 1, 1] | Ascend950PR_957b | `{}` | 12.529 | 12.529 | 11.365 | +10.24% | 12.046 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic` |
| [17, 96, 64, 1, 1] | Ascend950DT_9582 | `{}` | 8.735 | 8.735 | 9.162 | -4.66% | 8.050 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [17, 96, 64, 1, 1] | Ascend950PR_957b | `{}` | 8.737 | 8.737 | 8.736 | +0.01% | 8.050 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [17, 192, 128, 1, 1] | Ascend950DT_9582 | `{}` | 9.266 | 9.266 | 9.050 | +2.38% | 8.670 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic` |
| [17, 192, 128, 1, 1] | Ascend950PR_957b | `{}` | 9.269 | 9.269 | 8.833 | +4.94% | 8.670 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic` |
| [17, 257, 128, 1, 1] | Ascend950DT_9582 | `{}` | 9.762 | 9.762 | 9.551 | +2.21% | 9.057 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic` |
| [17, 257, 128, 1, 1] | Ascend950PR_957b | `{}` | 9.765 | 9.765 | 9.144 | +6.79% | 9.057 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic` |
| [17, 384, 64, 1, 2] | Ascend950DT_9582 | `{}` | 9.293 | 9.293 | 9.757 | -4.75% | 8.865 | `entry_kernel_2050000040100220` |
| [17, 384, 64, 1, 2] | Ascend950PR_957b | `{}` | 9.295 | 9.295 | 9.347 | -0.56% | 8.865 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [17, 512, 128, 1, 1] | Ascend950DT_9582 | `{}` | 10.496 | 10.496 | 10.595 | -0.94% | 9.777 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic` |
| [17, 512, 128, 1, 1] | Ascend950PR_957b | `{}` | 10.498 | 10.498 | 10.058 | +4.38% | 9.777 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic` |
| [17, 895, 64, 1, 2] | Ascend950DT_9582 | `{}` | 10.328 | 10.328 | 11.455 | -9.84% | 10.417 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [17, 895, 64, 1, 2] | Ascend950PR_957b | `{}` | 10.329 | 10.329 | 11.462 | -9.88% | 10.417 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [17, 1153, 64, 1, 1] | Ascend950DT_9582 | `{}` | 12.833 | 12.833 | 12.024 | +6.73% | 12.447 | `entry_kernel_2050000040140220` |
| [17, 1153, 64, 1, 1] | Ascend950DT_9582 | `{}` | 12.833 | 12.833 | 12.150 | +5.62% | 12.447 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic` |
| [17, 1153, 64, 1, 1] | Ascend950DT_9582 | `{}` | 12.833 | 12.833 | 12.487 | +2.77% | 12.447 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic` |
| [17, 1153, 64, 1, 1] | Ascend950DT_9582 | `{}` | 12.833 | 12.833 | 12.553 | +2.23% | 12.447 | `entry_kernel_2050000040140220` |
| [17, 1153, 64, 1, 1] | Ascend950DT_9582 | `{}` | 12.833 | 12.833 | 12.556 | +2.21% | 12.447 | `entry_kernel_2050000040140220` |
| [17, 1153, 64, 1, 1] | Ascend950PR_957b | `{}` | 12.834 | 12.834 | 11.489 | +11.71% | 12.447 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic` |
| [32, 129, 64, 1, 4] | Ascend950DT_9582 | `{}` | 9.097 | 9.097 | 9.150 | -0.58% | 8.423 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [32, 129, 64, 1, 4] | Ascend950PR_957b | `{}` | 9.099 | 9.099 | 8.850 | +2.82% | 8.423 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [32, 512, 64, 1, 1] | Ascend950DT_9582 | `{}` | 10.037 | 10.037 | 9.722 | +3.24% | 10.382 | `entry_kernel_2050000040100220` |
| [32, 512, 64, 1, 1] | Ascend950PR_957b | `{}` | 10.072 | 10.072 | 9.345 | +7.79% | 10.382 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [33, 319, 128, 1, 1] | Ascend950DT_9582 | `{}` | 10.057 | 10.057 | 9.610 | +4.64% | 10.202 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic` |
| [33, 319, 128, 1, 1] | Ascend950PR_957b | `{}` | 10.086 | 10.086 | 9.252 | +9.01% | 10.202 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic` |
| [64, 256, 64, 1, 1] | Ascend950DT_9582 | `{}` | 9.821 | 9.821 | 9.409 | +4.38% | 10.464 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [64, 256, 64, 1, 1] | Ascend950PR_957b | `{}` | 9.865 | 9.865 | 8.935 | +10.40% | 10.464 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [64, 320, 128, 1, 1] | Ascend950DT_9582 | `{}` | 11.208 | 11.208 | 9.968 | +12.43% | 12.691 | `entry_kernel_2050000070100220` |
| [64, 320, 128, 1, 1] | Ascend950PR_957b | `{}` | 11.280 | 11.280 | 9.582 | +17.72% | 12.691 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic` |
| [128, 128, 64, 1, 1] | Ascend950DT_9582 | `{}` | 9.022 | 9.022 | 9.498 | -5.02% | 9.001 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [128, 128, 64, 1, 1] | Ascend950PR_957b | `{}` | 9.043 | 9.043 | 8.793 | +2.84% | 9.001 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [129, 128, 64, 1, 1] | Ascend950DT_9582 | `{}` | 9.022 | 9.022 | 8.871 | +1.70% | 9.001 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |
| [129, 128, 64, 1, 1] | Ascend950PR_957b | `{}` | 9.043 | 9.043 | 8.508 | +6.30% | 9.001 | `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic` |

## fla_kda_v2

| Shape | 目标 SKU | 输入属性 | 旧模型 | 本版 | CA | 偏差 | 970DT 外推 | 实测 kernel |
|---|---|---|---:|---:|---:|---:|---:|---|
| [1, 2, 1, 1, 128, 128] | Ascend950DT_9582 | `{"implementation":"fla_26_9_1_v2"}` | 21.746 | 21.746 | 23.241 | -6.43% | 19.932 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 2, 1, 1, 128, 128] | Ascend950PR_957b | `{"implementation":"fla_26_9_1_v2"}` | 21.774 | 21.774 | 22.213 | -1.98% | 19.932 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 3, 1, 1, 128, 128] | Ascend950DT_9582 | `{"implementation":"fla_26_9_1_v2"}` | 21.839 | 21.839 | 23.422 | -6.76% | 20.017 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 3, 1, 1, 128, 128] | Ascend950PR_957b | `{"implementation":"fla_26_9_1_v2"}` | 21.868 | 21.868 | 22.270 | -1.81% | 20.017 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 5, 2, 2, 128, 128] | Ascend950DT_9582 | `{"implementation":"fla_26_9_1_v2"}` | 22.024 | 22.024 | 22.822 | -3.49% | 20.187 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 5, 2, 2, 128, 128] | Ascend950PR_957b | `{"implementation":"fla_26_9_1_v2"}` | 22.056 | 22.056 | 22.397 | -1.52% | 20.187 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 16, 1, 1, 128, 128] | Ascend950DT_9582 | `{"implementation":"fla_26_9_1_v2"}` | 23.049 | 23.049 | 24.125 | -4.46% | 21.126 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 16, 1, 1, 128, 128] | Ascend950PR_957b | `{"implementation":"fla_26_9_1_v2"}` | 23.093 | 23.093 | 23.421 | -1.40% | 21.126 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 17, 1, 1, 128, 128] | Ascend950DT_9582 | `{"implementation":"fla_26_9_1_v2"}` | 23.395 | 23.395 | 24.547 | -4.70% | 21.443 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 17, 1, 1, 128, 128] | Ascend950PR_957b | `{"implementation":"fla_26_9_1_v2"}` | 23.443 | 23.443 | 23.802 | -1.51% | 21.443 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 31, 2, 2, 128, 128] | Ascend950DT_9582 | `{"implementation":"fla_26_9_1_v2"}` | 24.709 | 24.709 | 25.276 | -2.24% | 22.647 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 31, 2, 2, 128, 128] | Ascend950PR_957b | `{"implementation":"fla_26_9_1_v2"}` | 24.773 | 24.773 | 24.796 | -0.09% | 22.647 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 32, 1, 1, 128, 128] | Ascend950DT_9582 | `{"implementation":"fla_26_9_1_v2"}` | 24.803 | 24.803 | 24.936 | -0.53% | 22.733 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 32, 1, 1, 128, 128] | Ascend950PR_957b | `{"implementation":"fla_26_9_1_v2"}` | 24.869 | 24.869 | 24.924 | -0.22% | 22.733 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 33, 2, 2, 128, 128] | Ascend950DT_9582 | `{"implementation":"fla_26_9_1_v2"}` | 25.709 | 25.709 | 25.521 | +0.74% | 23.563 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 33, 2, 2, 128, 128] | Ascend950PR_957b | `{"implementation":"fla_26_9_1_v2"}` | 25.781 | 25.781 | 25.819 | -0.15% | 23.563 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 63, 4, 4, 128, 128] | Ascend950DT_9582 | `{"implementation":"fla_26_9_1_v2"}` | 29.045 | 29.045 | 29.621 | -1.94% | 26.619 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 63, 4, 4, 128, 128] | Ascend950PR_957b | `{"implementation":"fla_26_9_1_v2"}` | 29.159 | 29.159 | 29.487 | -1.11% | 26.619 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 64, 2, 2, 128, 128] | Ascend950DT_9582 | `{"implementation":"fla_26_9_1_v2"}` | 29.141 | 29.141 | 28.839 | +1.05% | 26.707 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 64, 2, 2, 128, 128] | Ascend950PR_957b | `{"implementation":"fla_26_9_1_v2"}` | 29.257 | 29.257 | 28.999 | +0.89% | 26.707 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 64, 3, 3, 128, 128] | Ascend950DT_9582 | `{"implementation":"fla_26_9_1_v2"}` | 29.141 | 29.141 | 28.744 | +1.38% | 26.707 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 64, 3, 3, 128, 128] | Ascend950PR_957b | `{"implementation":"fla_26_9_1_v2"}` | 29.257 | 29.257 | 28.820 | +1.51% | 26.707 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 65, 1, 1, 128, 128] | Ascend950DT_9582 | `{"implementation":"fla_26_9_1_v2"}` | 31.611 | 31.611 | 30.878 | +2.38% | 28.971 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 65, 1, 1, 128, 128] | Ascend950PR_957b | `{"implementation":"fla_26_9_1_v2"}` | 31.731 | 31.731 | 30.764 | +3.15% | 28.971 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 65, 2, 2, 128, 128] | Ascend950DT_9582 | `{"implementation":"fla_26_9_1_v2"}` | 31.611 | 31.611 | 30.795 | +2.65% | 28.971 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 65, 2, 2, 128, 128] | Ascend950PR_957b | `{"implementation":"fla_26_9_1_v2"}` | 31.731 | 31.731 | 31.164 | +1.82% | 28.971 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 65, 4, 4, 128, 128] | Ascend950DT_9582 | `{"implementation":"fla_26_9_1_v2"}` | 31.611 | 31.611 | 31.675 | -0.20% | 28.971 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 65, 4, 4, 128, 128] | Ascend950PR_957b | `{"implementation":"fla_26_9_1_v2"}` | 31.731 | 31.731 | 31.970 | -0.75% | 28.971 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 80, 1, 1, 128, 128] | Ascend950DT_9582 | `{"implementation":"fla_26_9_1_v2"}` | 31.751 | 31.751 | 30.882 | +2.82% | 29.099 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 80, 1, 1, 128, 128] | Ascend950PR_957b | `{"implementation":"fla_26_9_1_v2"}` | 31.872 | 31.872 | 31.285 | +1.88% | 29.099 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 127, 1, 1, 128, 128] | Ascend950DT_9582 | `{"implementation":"fla_26_9_1_v2"}` | 32.464 | 32.464 | 31.728 | +2.32% | 29.752 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 129, 3, 3, 128, 128] | Ascend950DT_9582 | `{"implementation":"fla_26_9_1_v2"}` | 34.944 | 34.944 | 34.128 | +2.39% | 32.025 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 129, 3, 3, 128, 128] | Ascend950PR_957b | `{"implementation":"fla_26_9_1_v2"}` | 35.072 | 35.072 | 34.527 | +1.58% | 32.025 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 145, 2, 2, 128, 128] | Ascend950DT_9582 | `{"implementation":"fla_26_9_1_v2"}` | 35.185 | 35.185 | 34.586 | +1.73% | 32.246 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |
| [1, 145, 2, 2, 128, 128] | Ascend950PR_957b | `{"implementation":"fla_26_9_1_v2"}` | 35.313 | 35.313 | 35.012 | +0.86% | 32.246 | `ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic, ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic, ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0` |

## gmm2

| Shape | 目标 SKU | 输入属性 | 旧模型 | 本版 | CA | 偏差 | 970DT 外推 | 实测 kernel |
|---|---|---|---:|---:|---:|---:|---:|---|
| [1, 64, 64, 1] | Ascend950DT_9582 | `{"expert_counts":[1]}` | 3.410 | 3.410 | 3.151 | +8.21% | 3.126 | `entry_kernel_0` |
| [1, 64, 64, 1] | Ascend950PR_957b | `{"expert_counts":[1]}` | 3.410 | 3.410 | 3.210 | +6.24% | 3.126 | `GroupedMatmul_037cec40685a17f15b75759e999767c3_0` |
| [2, 64, 64, 1] | Ascend950DT_9582 | `{"expert_counts":[2]}` | 3.410 | 3.410 | 3.090 | +10.34% | 3.126 | `GroupedMatmul_037cec40685a17f15b75759e999767c3_0` |
| [2, 64, 64, 1] | Ascend950DT_9582 | `{"expert_counts":[2]}` | 3.410 | 3.410 | 3.248 | +4.98% | 3.126 | `entry_kernel_0` |
| [2, 64, 64, 1] | Ascend950PR_957b | `{"expert_counts":[2]}` | 3.410 | 3.410 | 3.107 | +9.76% | 3.126 | `GroupedMatmul_037cec40685a17f15b75759e999767c3_0` |
| [2, 64, 128, 1] | Ascend950DT_9582 | `{"expert_counts":[2]}` | 3.695 | 3.695 | 3.545 | +4.22% | 3.387 | `entry_kernel_0` |
| [2, 64, 128, 1] | Ascend950PR_957b | `{"expert_counts":[2]}` | 3.696 | 3.696 | 3.065 | +20.59% | 3.387 | `GroupedMatmul_037cec40685a17f15b75759e999767c3_0` |
| [3, 64, 64, 2] | Ascend950DT_9582 | `{"expert_counts":[1,2]}` | 3.410 | 3.410 | 3.771 | -9.58% | 3.125 | `entry_kernel_0` |
| [3, 64, 64, 2] | Ascend950PR_957b | `{"expert_counts":[1,2]}` | 3.410 | 3.410 | 3.404 | +0.17% | 3.125 | `GroupedMatmul_037cec40685a17f15b75759e999767c3_0` |
| [4, 128, 64, 2] | Ascend950DT_9582 | `{"expert_counts":[1,3]}` | 3.410 | 3.410 | 3.707 | -8.01% | 3.125 | `entry_kernel_0` |
| [4, 128, 64, 2] | Ascend950PR_957b | `{"expert_counts":[1,3]}` | 3.410 | 3.410 | 3.664 | -6.92% | 3.125 | `GroupedMatmul_037cec40685a17f15b75759e999767c3_0` |
| [6, 128, 128, 2] | Ascend950DT_9582 | `{"expert_counts":[1,5]}` | 3.695 | 3.695 | 3.697 | -0.05% | 3.386 | `entry_kernel_0` |
| [6, 128, 128, 2] | Ascend950PR_957b | `{"expert_counts":[1,5]}` | 3.698 | 3.698 | 3.812 | -2.98% | 3.386 | `GroupedMatmul_037cec40685a17f15b75759e999767c3_0` |
| [8, 64, 128, 1] | Ascend950DT_9582 | `{"expert_counts":[8]}` | 3.695 | 3.695 | 3.228 | +14.48% | 3.387 | `entry_kernel_0` |
| [8, 64, 128, 1] | Ascend950PR_957b | `{"expert_counts":[8]}` | 3.696 | 3.696 | 3.164 | +16.81% | 3.387 | `GroupedMatmul_037cec40685a17f15b75759e999767c3_0` |
| [8, 256, 128, 2] | Ascend950DT_9582 | `{"expert_counts":[1,7]}` | 3.695 | 3.695 | 3.630 | +1.79% | 3.385 | `entry_kernel_0` |
| [8, 256, 128, 2] | Ascend950PR_957b | `{"expert_counts":[1,7]}` | 3.701 | 3.701 | 3.663 | +1.05% | 3.385 | `GroupedMatmul_037cec40685a17f15b75759e999767c3_0` |
| [16, 128, 128, 2] | Ascend950DT_9582 | `{"expert_counts":[1,15]}` | 3.695 | 3.695 | 3.838 | -3.71% | 3.386 | `entry_kernel_0` |
| [16, 128, 128, 2] | Ascend950PR_957b | `{"expert_counts":[1,15]}` | 3.699 | 3.699 | 3.979 | -7.05% | 3.386 | `GroupedMatmul_037cec40685a17f15b75759e999767c3_0` |
| [16, 256, 256, 1] | Ascend950DT_9582 | `{"expert_counts":[16]}` | 4.516 | 4.516 | 3.560 | +26.86% | 4.136 | `entry_kernel_0` |
| [17, 128, 64, 2] | Ascend950DT_9582 | `{"expert_counts":[8,9]}` | 3.410 | 3.410 | 3.921 | -13.04% | 3.125 | `entry_kernel_0` |
| [17, 128, 64, 2] | Ascend950PR_957b | `{"expert_counts":[8,9]}` | 3.410 | 3.410 | 3.812 | -10.54% | 3.125 | `GroupedMatmul_037cec40685a17f15b75759e999767c3_0` |
| [32, 128, 128, 4] | Ascend950DT_9582 | `{"expert_counts":[2,6,8,16]}` | 3.695 | 3.695 | 3.885 | -4.88% | 3.385 | `entry_kernel_0` |
| [32, 256, 128, 2] | Ascend950DT_9582 | `{"expert_counts":[1,31]}` | 3.695 | 3.695 | 3.905 | -5.37% | 3.385 | `entry_kernel_0` |
| [64, 128, 256, 4] | Ascend950DT_9582 | `{"expert_counts":[8,8,16,32]}` | 4.516 | 4.516 | 4.142 | +9.04% | 4.132 | `entry_kernel_0` |

## grouped_matmul_swiglu_quant_v2

| Shape | 目标 SKU | 输入属性 | 旧模型 | 本版 | CA | 偏差 | 970DT 外推 | 实测 kernel |
|---|---|---|---:|---:|---:|---:|---:|---|
| [1, 64, 64, 1] | Ascend950DT_9582 | `{"expert_counts":[1]}` | 7.272 | 7.272 | 7.044 | +3.24% | 6.666 | `entry_kernel_0` |
| [1, 64, 64, 1] | Ascend950PR_957b | `{"expert_counts":[1]}` | 7.272 | 7.272 | 6.785 | +7.18% | 6.666 | `GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic` |
| [2, 64, 64, 1] | Ascend950DT_9582 | `{"expert_counts":[2]}` | 7.272 | 7.272 | 7.005 | +3.82% | 6.666 | `entry_kernel_0` |
| [2, 64, 64, 1] | Ascend950DT_9582 | `{"expert_counts":[2]}` | 7.272 | 7.272 | 7.105 | +2.35% | 6.666 | `GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic` |
| [2, 64, 64, 1] | Ascend950PR_957b | `{"expert_counts":[2]}` | 7.272 | 7.272 | 6.885 | +5.63% | 6.666 | `GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic` |
| [2, 64, 128, 1] | Ascend950DT_9582 | `{"expert_counts":[2]}` | 7.272 | 7.272 | 7.280 | -0.11% | 6.666 | `entry_kernel_0` |
| [2, 64, 128, 1] | Ascend950PR_957b | `{"expert_counts":[2]}` | 7.272 | 7.272 | 6.755 | +7.66% | 6.666 | `GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic` |
| [3, 64, 64, 2] | Ascend950DT_9582 | `{"expert_counts":[1,2]}` | 7.272 | 7.272 | 7.081 | +2.70% | 6.666 | `entry_kernel_0` |
| [3, 64, 64, 2] | Ascend950PR_957b | `{"expert_counts":[1,2]}` | 7.272 | 7.272 | 6.862 | +5.97% | 6.666 | `GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic` |
| [4, 128, 64, 2] | Ascend950DT_9582 | `{"expert_counts":[1,3]}` | 7.858 | 7.858 | 7.364 | +6.71% | 7.202 | `entry_kernel_0` |
| [4, 128, 64, 2] | Ascend950PR_957b | `{"expert_counts":[1,3]}` | 7.862 | 7.862 | 7.064 | +11.30% | 7.202 | `GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic` |
| [6, 128, 128, 2] | Ascend950DT_9582 | `{"expert_counts":[1,5]}` | 7.858 | 7.858 | 7.390 | +6.34% | 7.201 | `entry_kernel_0` |
| [6, 128, 128, 2] | Ascend950PR_957b | `{"expert_counts":[1,5]}` | 7.865 | 7.865 | 7.030 | +11.88% | 7.201 | `GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic` |
| [8, 64, 128, 1] | Ascend950DT_9582 | `{"expert_counts":[8]}` | 7.272 | 7.272 | 7.405 | -1.79% | 6.666 | `entry_kernel_0` |
| [8, 64, 128, 1] | Ascend950PR_957b | `{"expert_counts":[8]}` | 7.272 | 7.272 | 7.159 | +1.58% | 6.666 | `GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic` |
| [8, 256, 128, 2] | Ascend950DT_9582 | `{"expert_counts":[1,7]}` | 9.941 | 9.941 | 7.489 | +32.74% | 9.104 | `entry_kernel_0` |
| [8, 256, 128, 2] | Ascend950PR_957b | `{"expert_counts":[1,7]}` | 9.964 | 9.964 | 7.393 | +34.79% | 9.104 | `GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic` |
| [16, 128, 128, 2] | Ascend950DT_9582 | `{"expert_counts":[1,15]}` | 7.858 | 7.858 | 7.372 | +6.60% | 7.201 | `entry_kernel_0` |
| [16, 128, 128, 2] | Ascend950PR_957b | `{"expert_counts":[1,15]}` | 7.866 | 7.866 | 7.073 | +11.20% | 7.201 | `GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic` |
| [16, 256, 256, 1] | Ascend950DT_9582 | `{"expert_counts":[16]}` | 9.941 | 9.941 | 8.193 | +21.34% | 9.104 | `entry_kernel_0` |
| [17, 128, 64, 2] | Ascend950DT_9582 | `{"expert_counts":[8,9]}` | 7.858 | 7.858 | 7.414 | +5.99% | 7.202 | `entry_kernel_0` |
| [17, 128, 64, 2] | Ascend950PR_957b | `{"expert_counts":[8,9]}` | 7.862 | 7.862 | 7.210 | +9.05% | 7.202 | `GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic` |
| [32, 128, 128, 4] | Ascend950DT_9582 | `{"expert_counts":[2,6,8,16]}` | 7.858 | 7.858 | 7.686 | +2.24% | 7.199 | `entry_kernel_0` |
| [32, 256, 128, 2] | Ascend950DT_9582 | `{"expert_counts":[1,31]}` | 9.941 | 9.941 | 8.214 | +21.02% | 9.104 | `entry_kernel_0` |
| [64, 128, 256, 4] | Ascend950DT_9582 | `{"expert_counts":[8,8,16,32]}` | 7.858 | 7.858 | 8.355 | -5.94% | 7.195 | `entry_kernel_0` |

## kda_recurrent

| Shape | 目标 SKU | 输入属性 | 旧模型 | 本版 | CA | 偏差 | 970DT 外推 | 实测 kernel |
|---|---|---|---:|---:|---:|---:|---:|---|
| [1, 1, 1, 1, 128, 128] | Ascend950DT_9582 | `{"layout":"BSND","dtype":"BF16","gate_dtype":"FP32","state_dtype":"FP32","state_v_first":false,"use_gate_in_kernel":false,"chunk_size":null,"output_final_state":true,"export_gk":false}` | 15.072 | 15.072 | 14.834 | +1.60% | 13.804 | `RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0` |
| [1, 1, 1, 1, 128, 128] | Ascend950DT_9582 | `{"layout":"BSND","dtype":"BF16","gate_dtype":"FP32","state_dtype":"FP32","state_v_first":false,"use_gate_in_kernel":false,"chunk_size":null,"output_final_state":true,"export_gk":false}` | 15.072 | 15.072 | 15.219 | -0.97% | 13.804 | `entry_kernel_0` |
| [1, 1, 1, 1, 128, 128] | Ascend950PR_957b | `{"layout":"BSND","dtype":"BF16","gate_dtype":"FP32","state_dtype":"FP32","state_v_first":false,"use_gate_in_kernel":false,"chunk_size":null,"output_final_state":true,"export_gk":false}` | 15.277 | 15.277 | 14.622 | +4.48% | 13.804 | `RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0` |
| [1, 1, 1, 1, 128, 256] | Ascend950DT_9582 | `{"layout":"BSND","dtype":"BF16","gate_dtype":"FP32","state_dtype":"FP32","state_v_first":false,"use_gate_in_kernel":false,"chunk_size":null,"output_final_state":true,"export_gk":false}` | 18.547 | 18.547 | 17.434 | +6.38% | 16.984 | `entry_kernel_0` |
| [1, 1, 1, 1, 128, 256] | Ascend950PR_957b | `{"layout":"BSND","dtype":"BF16","gate_dtype":"FP32","state_dtype":"FP32","state_v_first":false,"use_gate_in_kernel":false,"chunk_size":null,"output_final_state":true,"export_gk":false}` | 18.841 | 18.841 | 21.030 | -10.41% | 16.984 | `RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0` |
| [1, 2, 1, 1, 128, 128] | Ascend950DT_9582 | `{"layout":"BSND","dtype":"BF16","gate_dtype":"FP32","state_dtype":"FP32","state_v_first":false,"use_gate_in_kernel":false,"chunk_size":null,"output_final_state":true,"export_gk":false}` | 18.547 | 18.547 | 18.467 | +0.43% | 16.987 | `entry_kernel_0` |
| [1, 2, 1, 1, 128, 128] | Ascend950PR_957b | `{"layout":"BSND","dtype":"BF16","gate_dtype":"FP32","state_dtype":"FP32","state_v_first":false,"use_gate_in_kernel":false,"chunk_size":null,"output_final_state":true,"export_gk":false}` | 18.832 | 18.832 | 18.048 | +4.34% | 16.987 | `RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0` |
| [1, 2, 1, 1, 128, 256] | Ascend950DT_9582 | `{"layout":"BSND","dtype":"BF16","gate_dtype":"FP32","state_dtype":"FP32","state_v_first":false,"use_gate_in_kernel":false,"chunk_size":null,"output_final_state":true,"export_gk":false}` | 25.498 | 25.498 | 22.684 | +12.41% | 23.349 | `entry_kernel_0` |
| [1, 2, 1, 1, 128, 256] | Ascend950PR_957b | `{"layout":"BSND","dtype":"BF16","gate_dtype":"FP32","state_dtype":"FP32","state_v_first":false,"use_gate_in_kernel":false,"chunk_size":null,"output_final_state":true,"export_gk":false}` | 25.951 | 25.951 | 28.402 | -8.63% | 23.349 | `RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0` |
| [1, 3, 1, 1, 128, 128] | Ascend950DT_9582 | `{"layout":"BSND","dtype":"BF16","gate_dtype":"FP32","state_dtype":"FP32","state_v_first":false,"use_gate_in_kernel":false,"chunk_size":null,"output_final_state":true,"export_gk":false}` | 22.022 | 22.022 | 21.932 | +0.41% | 20.169 | `entry_kernel_0` |
| [1, 3, 1, 1, 128, 128] | Ascend950PR_957b | `{"layout":"BSND","dtype":"BF16","gate_dtype":"FP32","state_dtype":"FP32","state_v_first":false,"use_gate_in_kernel":false,"chunk_size":null,"output_final_state":true,"export_gk":false}` | 22.388 | 22.388 | 21.221 | +5.50% | 20.169 | `RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0` |
| [1, 3, 1, 1, 128, 256] | Ascend950DT_9582 | `{"layout":"BSND","dtype":"BF16","gate_dtype":"FP32","state_dtype":"FP32","state_v_first":false,"use_gate_in_kernel":false,"chunk_size":null,"output_final_state":true,"export_gk":false}` | 32.449 | 32.449 | 31.071 | +4.43% | 29.714 | `entry_kernel_0` |
| [1, 4, 1, 1, 128, 128] | Ascend950DT_9582 | `{"layout":"BSND","dtype":"BF16","gate_dtype":"FP32","state_dtype":"FP32","state_v_first":false,"use_gate_in_kernel":false,"chunk_size":null,"output_final_state":true,"export_gk":false}` | 25.498 | 25.498 | 25.430 | +0.27% | 23.352 | `entry_kernel_0` |
| [1, 4, 1, 1, 128, 128] | Ascend950PR_957b | `{"layout":"BSND","dtype":"BF16","gate_dtype":"FP32","state_dtype":"FP32","state_v_first":false,"use_gate_in_kernel":false,"chunk_size":null,"output_final_state":true,"export_gk":false}` | 25.943 | 25.943 | 25.084 | +3.42% | 23.352 | `RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0` |
| [1, 4, 1, 1, 128, 256] | Ascend950DT_9582 | `{"layout":"BSND","dtype":"BF16","gate_dtype":"FP32","state_dtype":"FP32","state_v_first":false,"use_gate_in_kernel":false,"chunk_size":null,"output_final_state":true,"export_gk":false}` | 39.399 | 39.399 | 38.342 | +2.76% | 36.079 | `entry_kernel_0` |
| [1, 5, 1, 1, 128, 128] | Ascend950DT_9582 | `{"layout":"BSND","dtype":"BF16","gate_dtype":"FP32","state_dtype":"FP32","state_v_first":false,"use_gate_in_kernel":false,"chunk_size":null,"output_final_state":true,"export_gk":false}` | 28.973 | 28.973 | 24.648 | +17.55% | 26.534 | `entry_kernel_0` |
| [1, 5, 1, 1, 128, 128] | Ascend950PR_957b | `{"layout":"BSND","dtype":"BF16","gate_dtype":"FP32","state_dtype":"FP32","state_v_first":false,"use_gate_in_kernel":false,"chunk_size":null,"output_final_state":true,"export_gk":false}` | 29.498 | 29.498 | 28.819 | +2.36% | 26.534 | `RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0` |
| [1, 6, 1, 1, 128, 128] | Ascend950DT_9582 | `{"layout":"BSND","dtype":"BF16","gate_dtype":"FP32","state_dtype":"FP32","state_v_first":false,"use_gate_in_kernel":false,"chunk_size":null,"output_final_state":true,"export_gk":false}` | 32.449 | 32.449 | 28.441 | +14.09% | 29.716 | `entry_kernel_0` |
| [1, 6, 1, 1, 128, 128] | Ascend950PR_957b | `{"layout":"BSND","dtype":"BF16","gate_dtype":"FP32","state_dtype":"FP32","state_v_first":false,"use_gate_in_kernel":false,"chunk_size":null,"output_final_state":true,"export_gk":false}` | 33.054 | 33.054 | 31.955 | +3.44% | 29.716 | `RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0` |
| [1, 6, 1, 1, 128, 256] | Ascend950DT_9582 | `{"layout":"BSND","dtype":"BF16","gate_dtype":"FP32","state_dtype":"FP32","state_v_first":false,"use_gate_in_kernel":false,"chunk_size":null,"output_final_state":true,"export_gk":false}` | 53.301 | 53.301 | 52.691 | +1.16% | 48.808 | `entry_kernel_0` |
| [1, 7, 1, 1, 128, 128] | Ascend950DT_9582 | `{"layout":"BSND","dtype":"BF16","gate_dtype":"FP32","state_dtype":"FP32","state_v_first":false,"use_gate_in_kernel":false,"chunk_size":null,"output_final_state":true,"export_gk":false}` | 35.924 | 35.924 | 31.195 | +15.16% | 32.899 | `entry_kernel_0` |
| [1, 7, 1, 1, 128, 128] | Ascend950PR_957b | `{"layout":"BSND","dtype":"BF16","gate_dtype":"FP32","state_dtype":"FP32","state_v_first":false,"use_gate_in_kernel":false,"chunk_size":null,"output_final_state":true,"export_gk":false}` | 36.609 | 36.609 | 35.718 | +2.49% | 32.899 | `RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0` |
| [1, 8, 1, 1, 128, 128] | Ascend950DT_9582 | `{"layout":"BSND","dtype":"BF16","gate_dtype":"FP32","state_dtype":"FP32","state_v_first":false,"use_gate_in_kernel":false,"chunk_size":null,"output_final_state":true,"export_gk":false}` | 39.399 | 39.399 | 35.396 | +11.31% | 36.081 | `entry_kernel_0` |
| [1, 8, 1, 1, 128, 128] | Ascend950PR_957b | `{"layout":"BSND","dtype":"BF16","gate_dtype":"FP32","state_dtype":"FP32","state_v_first":false,"use_gate_in_kernel":false,"chunk_size":null,"output_final_state":true,"export_gk":false}` | 40.164 | 40.164 | 39.450 | +1.81% | 36.081 | `RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0` |

## lightning_indexer

| Shape | 目标 SKU | 输入属性 | 旧模型 | 本版 | CA | 偏差 | 970DT 外推 | 实测 kernel |
|---|---|---|---:|---:|---:|---:|---:|---|
| [1, 1, 512, 8, 128] | Ascend950DT_9582 | `{"dtype":"bf16","layout":"BSND","sparse_count":512,"sparse_mode":3,"return_values":false}` | 5.771 | 5.771 | 5.862 | -1.56% | 5.288 | `LightningIndexer_2517851d7f53b28f971ff94bfa4b7037_203547_mix_aic` |
| [1, 1, 512, 8, 128] | Ascend950DT_9582 | `{"dtype":"bf16","layout":"BSND","sparse_count":512,"sparse_mode":3,"return_values":false}` | 5.771 | 5.771 | 6.093 | -5.29% | 5.288 | `entry_kernel_31b1b` |
| [1, 1, 512, 8, 128] | Ascend950PR_957b | `{"dtype":"bf16","layout":"BSND","sparse_count":512,"sparse_mode":3,"return_values":false}` | 5.790 | 5.790 | 5.779 | +0.20% | 5.288 | `LightningIndexer_2517851d7f53b28f971ff94bfa4b7037_203547_mix_aic` |
| [1, 1, 640, 8, 128] | Ascend950DT_9582 | `{"dtype":"bf16","layout":"BSND","sparse_count":512,"sparse_mode":3,"return_values":false}` | 6.652 | 6.652 | 6.200 | +7.28% | 6.093 | `entry_kernel_31b1b` |
| [1, 1, 640, 8, 128] | Ascend950PR_957b | `{"dtype":"bf16","layout":"BSND","sparse_count":512,"sparse_mode":3,"return_values":false}` | 6.700 | 6.700 | 5.935 | +12.90% | 6.093 | `LightningIndexer_2517851d7f53b28f971ff94bfa4b7037_203547_mix_aic` |
| [1, 1, 896, 8, 128] | Ascend950DT_9582 | `{"dtype":"bf16","layout":"BSND","sparse_count":512,"sparse_mode":3,"return_values":false}` | 6.711 | 6.711 | 6.439 | +4.23% | 6.146 | `entry_kernel_31b1b` |
| [1, 1, 896, 8, 128] | Ascend950PR_957b | `{"dtype":"bf16","layout":"BSND","sparse_count":512,"sparse_mode":3,"return_values":false}` | 6.765 | 6.765 | 6.129 | +10.38% | 6.146 | `LightningIndexer_2517851d7f53b28f971ff94bfa4b7037_203547_mix_aic` |
| [1, 1, 1024, 8, 128] | Ascend950DT_9582 | `{"dtype":"bf16","layout":"BSND","sparse_count":512,"sparse_mode":3,"return_values":false}` | 6.755 | 6.755 | 6.741 | +0.21% | 6.178 | `entry_kernel_31b1b` |
| [1, 1, 1024, 8, 128] | Ascend950PR_957b | `{"dtype":"bf16","layout":"BSND","sparse_count":512,"sparse_mode":3,"return_values":false}` | 6.822 | 6.822 | 6.493 | +5.07% | 6.178 | `LightningIndexer_2517851d7f53b28f971ff94bfa4b7037_203547_mix_aic` |
| [1, 1, 2048, 8, 128] | Ascend950DT_9582 | `{"dtype":"bf16","layout":"BSND","sparse_count":512,"sparse_mode":3,"return_values":false}` | 10.051 | 10.051 | 8.614 | +16.68% | 9.178 | `entry_kernel_31b1b` |
| [1, 1, 2048, 8, 128] | Ascend950PR_957b | `{"dtype":"bf16","layout":"BSND","sparse_count":512,"sparse_mode":3,"return_values":false}` | 10.220 | 10.220 | 8.455 | +20.87% | 9.178 | `LightningIndexer_2517851d7f53b28f971ff94bfa4b7037_203547_mix_aic` |
| [1, 1, 8192, 8, 128] | Ascend950DT_9582 | `{"dtype":"bf16","layout":"BSND","sparse_count":512,"sparse_mode":3,"return_values":false}` | 29.909 | 29.909 | 22.709 | +31.71% | 27.231 | `entry_kernel_31b1b` |
| [1, 2, 768, 8, 128] | Ascend950DT_9582 | `{"dtype":"bf16","layout":"BSND","sparse_count":512,"sparse_mode":3,"return_values":false}` | 6.682 | 6.682 | 6.365 | +4.97% | 6.120 | `entry_kernel_31b1b` |
| [1, 2, 768, 8, 128] | Ascend950PR_957b | `{"dtype":"bf16","layout":"BSND","sparse_count":512,"sparse_mode":3,"return_values":false}` | 6.733 | 6.733 | 6.171 | +9.11% | 6.120 | `LightningIndexer_2517851d7f53b28f971ff94bfa4b7037_203547_mix_aic` |
| [1, 2, 1024, 8, 128] | Ascend950DT_9582 | `{"dtype":"bf16","layout":"BSND","sparse_count":512,"sparse_mode":3,"return_values":false}` | 6.755 | 6.755 | 6.516 | +3.66% | 6.178 | `entry_kernel_31b1b` |
| [1, 2, 1024, 8, 128] | Ascend950PR_957b | `{"dtype":"bf16","layout":"BSND","sparse_count":512,"sparse_mode":3,"return_values":false}` | 6.823 | 6.823 | 6.570 | +3.85% | 6.178 | `LightningIndexer_2517851d7f53b28f971ff94bfa4b7037_203547_mix_aic` |
| [1, 2, 3072, 8, 128] | Ascend950DT_9582 | `{"dtype":"bf16","layout":"BSND","sparse_count":512,"sparse_mode":3,"return_values":false}` | 13.361 | 13.361 | 11.101 | +20.35% | 12.188 | `entry_kernel_31b1b` |
| [1, 2, 3072, 8, 128] | Ascend950PR_957b | `{"dtype":"bf16","layout":"BSND","sparse_count":512,"sparse_mode":3,"return_values":false}` | 13.633 | 13.633 | 10.776 | +26.51% | 12.188 | `LightningIndexer_2517851d7f53b28f971ff94bfa4b7037_203547_mix_aic` |
| [1, 4, 1024, 8, 128] | Ascend950DT_9582 | `{"dtype":"bf16","layout":"BSND","sparse_count":512,"sparse_mode":3,"return_values":false}` | 6.755 | 6.755 | 8.043 | -16.01% | 6.178 | `entry_kernel_31b1b` |
| [1, 4, 1024, 8, 128] | Ascend950PR_957b | `{"dtype":"bf16","layout":"BSND","sparse_count":512,"sparse_mode":3,"return_values":false}` | 6.824 | 6.824 | 8.030 | -15.01% | 6.178 | `LightningIndexer_2517851d7f53b28f971ff94bfa4b7037_203547_mix_aic` |
| [1, 4, 4096, 8, 128] | Ascend950DT_9582 | `{"dtype":"bf16","layout":"BSND","sparse_count":512,"sparse_mode":3,"return_values":false}` | 16.670 | 16.670 | 15.674 | +6.36% | 15.196 | `entry_kernel_31b1b` |
| [1, 8, 1536, 8, 128] | Ascend950DT_9582 | `{"dtype":"bf16","layout":"BSND","sparse_count":512,"sparse_mode":3,"return_values":false}` | 8.396 | 8.396 | 8.862 | -5.26% | 7.671 | `entry_kernel_31b1b` |
| [1, 8, 1536, 8, 128] | Ascend950PR_957b | `{"dtype":"bf16","layout":"BSND","sparse_count":512,"sparse_mode":3,"return_values":false}` | 8.521 | 8.521 | 8.829 | -3.49% | 7.671 | `LightningIndexer_2517851d7f53b28f971ff94bfa4b7037_203547_mix_aic` |
| [1, 9, 4096, 8, 128] | Ascend950DT_9582 | `{"dtype":"bf16","layout":"BSND","sparse_count":512,"sparse_mode":3,"return_values":false}` | 16.670 | 16.670 | 15.334 | +8.72% | 15.195 | `entry_kernel_31b1b` |

## matmul

| Shape | 目标 SKU | 输入属性 | 旧模型 | 本版 | CA | 偏差 | 970DT 外推 | 实测 kernel |
|---|---|---|---:|---:|---:|---:|---:|---|
| [1, 32, 16] | Ascend950DT_9582 | `{}` | 3.091 | 3.091 | 3.030 | +2.02% | 2.833 | `entry_kernel_2` |
| [1, 32, 16] | Ascend950PR_957b | `{}` | 3.091 | 3.091 | 2.734 | +13.06% | 2.833 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [1, 128, 128] | Ascend950DT_9582 | `{}` | 3.414 | 3.414 | 3.356 | +1.71% | 3.129 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [1, 128, 128] | Ascend950PR_957b | `{}` | 3.414 | 3.414 | 3.138 | +8.78% | 3.129 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [1, 256, 1024] | Ascend950DT_9582 | `{}` | 5.127 | 5.127 | 4.624 | +10.88% | 4.700 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [1, 256, 1024] | Ascend950PR_957b | `{}` | 5.127 | 5.127 | 4.612 | +11.19% | 4.700 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [2, 32, 32] | Ascend950DT_9582 | `{}` | 3.109 | 3.109 | 3.244 | -4.17% | 2.850 | `entry_kernel_2` |
| [2, 32, 32] | Ascend950PR_957b | `{}` | 3.109 | 3.109 | 2.813 | +10.51% | 2.850 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [3, 64, 64] | Ascend950DT_9582 | `{}` | 3.217 | 3.217 | 3.065 | +4.94% | 2.949 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [3, 64, 64] | Ascend950PR_957b | `{}` | 3.217 | 3.217 | 2.858 | +12.57% | 2.949 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [4, 80, 48] | Ascend950DT_9582 | `{}` | 3.237 | 3.237 | 3.524 | -8.14% | 2.967 | `entry_kernel_2` |
| [4, 80, 48] | Ascend950PR_957b | `{}` | 3.237 | 3.237 | 3.341 | -3.10% | 2.967 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [8, 96, 256] | Ascend950DT_9582 | `{}` | 3.702 | 3.702 | 3.755 | -1.41% | 3.393 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [8, 96, 256] | Ascend950PR_957b | `{}` | 3.702 | 3.702 | 3.620 | +2.27% | 3.393 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [9, 128, 256] | Ascend950DT_9582 | `{}` | 3.703 | 3.703 | 4.019 | -7.87% | 3.394 | `entry_kernel_2` |
| [9, 128, 256] | Ascend950PR_957b | `{}` | 3.703 | 3.703 | 3.595 | +2.99% | 3.394 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [16, 192, 384] | Ascend950PR_957b | `{}` | 4.039 | 4.039 | 3.724 | +8.48% | 3.703 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [17, 96, 128] | Ascend950DT_9582 | `{}` | 3.476 | 3.476 | 3.494 | -0.51% | 3.187 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [17, 96, 128] | Ascend950PR_957b | `{}` | 3.476 | 3.476 | 3.414 | +1.83% | 3.187 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [17, 256, 1024] | Ascend950DT_9582 | `{}` | 5.190 | 5.190 | 4.742 | +9.45% | 4.758 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [17, 256, 1024] | Ascend950PR_957b | `{}` | 5.190 | 5.190 | 4.687 | +10.74% | 4.758 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [31, 64, 257] | Ascend950DT_9582 | `{}` | 3.781 | 3.781 | 4.091 | -7.57% | 3.466 | `entry_kernel_2` |
| [31, 64, 257] | Ascend950PR_957b | `{}` | 3.781 | 3.781 | 3.801 | -0.53% | 3.466 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [32, 64, 64] | Ascend950DT_9582 | `{}` | 3.278 | 3.278 | 3.281 | -0.08% | 3.005 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [32, 64, 64] | Ascend950DT_9582 | `{}` | 3.278 | 3.278 | 3.292 | -0.40% | 3.005 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [32, 64, 64] | Ascend950PR_957b | `{}` | 3.278 | 3.278 | 3.158 | +3.83% | 3.005 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [33, 128, 129] | Ascend950DT_9582 | `{}` | 3.513 | 3.513 | 3.671 | -4.30% | 3.220 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [33, 128, 129] | Ascend950PR_957b | `{}` | 3.513 | 3.513 | 3.679 | -4.50% | 3.220 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [48, 128, 192] | Ascend950DT_9582 | `{}` | 3.651 | 3.651 | 3.575 | +2.13% | 3.347 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [48, 128, 192] | Ascend950PR_957b | `{}` | 3.651 | 3.651 | 3.481 | +4.88% | 3.347 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [64, 128, 512] | Ascend950DT_9582 | `{}` | 4.348 | 4.348 | 4.437 | -2.01% | 3.985 | `entry_kernel_2` |
| [64, 128, 512] | Ascend950PR_957b | `{}` | 4.348 | 4.348 | 4.275 | +1.71% | 3.985 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [65, 128, 129] | Ascend950DT_9582 | `{}` | 3.554 | 3.554 | 3.712 | -4.25% | 3.258 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [65, 128, 129] | Ascend950PR_957b | `{}` | 3.554 | 3.554 | 3.733 | -4.81% | 3.258 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [127, 384, 256] | Ascend950DT_9582 | `{}` | 3.954 | 3.954 | 3.819 | +3.52% | 3.630 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [127, 384, 256] | Ascend950PR_957b | `{}` | 3.956 | 3.956 | 3.793 | +4.31% | 3.630 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [128, 192, 256] | Ascend950DT_9582 | `{}` | 3.906 | 3.906 | 3.798 | +2.85% | 3.579 | `entry_kernel_2` |
| [128, 192, 256] | Ascend950PR_957b | `{}` | 3.907 | 3.907 | 3.815 | +2.41% | 3.579 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [128, 256, 512] | Ascend950DT_9582 | `{}` | 4.463 | 4.463 | 4.393 | +1.58% | 4.087 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [128, 256, 512] | Ascend950PR_957b | `{}` | 4.477 | 4.477 | 4.368 | +2.48% | 4.087 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [128, 512, 512] | Ascend950DT_9582 | `{}` | 4.541 | 4.541 | 4.382 | +3.64% | 4.137 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [128, 512, 512] | Ascend950PR_957b | `{}` | 4.600 | 4.600 | 4.604 | -0.10% | 4.137 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [129, 256, 256] | Ascend950DT_9582 | `{}` | 3.936 | 3.936 | 3.664 | +7.43% | 3.608 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [129, 256, 256] | Ascend950PR_957b | `{}` | 3.936 | 3.936 | 3.609 | +9.07% | 3.608 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [192, 384, 256] | Ascend950DT_9582 | `{}` | 4.002 | 4.002 | 3.925 | +1.95% | 3.659 | `entry_kernel_2` |
| [192, 384, 256] | Ascend950PR_957b | `{}` | 4.006 | 4.006 | 4.040 | -0.84% | 3.659 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [224, 512, 256] | Ascend950PR_957b | `{}` | 4.225 | 4.225 | 4.115 | +2.68% | 3.692 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [225, 512, 256] | Ascend950DT_9582 | `{}` | 4.079 | 4.079 | 4.113 | -0.81% | 3.697 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [225, 512, 256] | Ascend950PR_957b | `{}` | 4.225 | 4.225 | 4.116 | +2.64% | 3.697 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [255, 512, 256] | Ascend950PR_957b | `{}` | 4.390 | 4.390 | 4.183 | +4.94% | 3.693 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [256, 512, 256] | Ascend950DT_9582 | `{}` | 4.086 | 4.086 | 4.192 | -2.53% | 3.693 | `entry_kernel_2` |
| [256, 512, 256] | Ascend950PR_957b | `{}` | 4.390 | 4.390 | 4.239 | +3.55% | 3.693 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [257, 512, 256] | Ascend950PR_957b | `{}` | 4.390 | 4.390 | 4.197 | +4.60% | 3.702 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [384, 512, 256] | Ascend950PR_957b | `{}` | 4.734 | 4.734 | 4.504 | +5.11% | 3.723 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [448, 512, 256] | Ascend950PR_957b | `{}` | 4.760 | 4.760 | 4.515 | +5.45% | 3.735 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [449, 512, 256] | Ascend950DT_9582 | `{}` | 4.759 | 4.759 | 4.361 | +9.12% | 3.740 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [449, 512, 256] | Ascend950PR_957b | `{}` | 4.826 | 4.826 | 4.595 | +5.04% | 3.740 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [511, 512, 256] | Ascend950PR_957b | `{}` | 4.852 | 4.852 | 4.612 | +5.22% | 3.759 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [512, 384, 256] | Ascend950PR_957b | `{}` | 4.734 | 4.734 | 4.541 | +4.26% | 3.723 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [512, 512, 256] | Ascend950DT_9582 | `{}` | 4.771 | 4.771 | 4.799 | -0.59% | 3.759 | `entry_kernel_2` |
| [512, 512, 256] | Ascend950PR_957b | `{}` | 4.852 | 4.852 | 4.656 | +4.21% | 3.759 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [513, 384, 256] | Ascend950PR_957b | `{}` | 4.743 | 4.743 | 4.515 | +5.07% | 3.727 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [768, 512, 128] | Ascend950PR_957b | `{}` | 4.312 | 4.312 | 4.068 | +6.00% | 3.737 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [1024, 1792, 32] | Ascend950DT_9582 | `{}` | 4.610 | 4.610 | 4.260 | +8.22% | 3.477 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [1024, 1792, 32] | Ascend950PR_957b | `{}` | 4.709 | 4.709 | 4.347 | +8.33% | 3.477 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [1024, 2048, 32] | Ascend950PR_957b | `{}` | 4.823 | 4.823 | 4.464 | +8.03% | 3.547 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [1024, 2304, 16] | Ascend950DT_9582 | `{}` | 4.531 | 4.531 | 4.258 | +6.43% | 3.450 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [1024, 2304, 16] | Ascend950PR_957b | `{}` | 4.648 | 4.648 | 4.419 | +5.19% | 3.450 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [1280, 2048, 16] | Ascend950DT_9582 | `{}` | 4.630 | 4.630 | 4.397 | +5.31% | 3.481 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [1280, 2048, 16] | Ascend950DT_9582 | `{}` | 4.630 | 4.630 | 4.472 | +3.54% | 3.481 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [1280, 2048, 16] | Ascend950PR_957b | `{}` | 4.762 | 4.762 | 4.633 | +2.78% | 3.481 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [1280, 2048, 512] | Ascend950DT_9582 | `{}` | 12.449 | 12.449 | 12.219 | +1.89% | 6.350 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [1280, 2048, 512] | Ascend950PR_957b | `{}` | 14.027 | 14.027 | 13.218 | +6.12% | 6.350 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [1536, 1792, 16] | Ascend950DT_9582 | `{}` | 4.680 | 4.680 | 4.438 | +5.45% | 3.490 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [1536, 1792, 16] | Ascend950PR_957b | `{}` | 4.818 | 4.818 | 4.540 | +6.13% | 3.490 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [1536, 2048, 16] | Ascend950PR_957b | `{}` | 4.969 | 4.969 | 4.755 | +4.50% | 3.540 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2` |
| [2048, 32, 5120] | Ascend950DT_9582 | `{}` | 12.123 | 12.123 | 13.023 | -6.91% | 7.287 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_4098_mix_aic` |
| [2048, 32, 5120] | Ascend950PR_957b | `{}` | 19.016 | 19.016 | 20.822 | -8.67% | 7.287 | `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_4098_mix_aic` |

## mhc

| Shape | 目标 SKU | 输入属性 | 旧模型 | 本版 | CA | 偏差 | 970DT 外推 | 实测 kernel |
|---|---|---|---:|---:|---:|---:|---:|---|
| [1, 4, 64] | Ascend950DT_9582 | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 18.002 | 18.002 | 17.143 | +5.01% | 16.502 | `entry_kernel_1+entry_kernel_0+entry_kernel_5` |
| [1, 4, 64] | Ascend950PR_957b | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 18.006 | 18.006 | 16.464 | +9.37% | 16.502 | `MhcPre_52e042c374f575edbe357a32523dba2c_1_mix_aic+MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0+MhcPost_59130ccee6717ff2c0f15b485e731162_5` |
| [1, 4, 80] | Ascend950DT_9582 | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 18.230 | 18.230 | 17.559 | +3.82% | 16.710 | `entry_kernel_1+entry_kernel_0+entry_kernel_5` |
| [1, 4, 128] | Ascend950DT_9582 | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 16.903 | 16.903 | 15.768 | +7.20% | 15.494 | `MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic+MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0+MhcPost_59130ccee6717ff2c0f15b485e731162_5` |
| [1, 4, 128] | Ascend950DT_9582 | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 16.903 | 16.903 | 15.970 | +5.84% | 15.494 | `entry_kernel_2 + entry_kernel_0 + entry_kernel_5` |
| [1, 4, 128] | Ascend950PR_957b | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 16.909 | 16.909 | 15.496 | +9.12% | 15.494 | `MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic+MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0+MhcPost_59130ccee6717ff2c0f15b485e731162_5` |
| [1, 4, 512] | Ascend950DT_9582 | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 17.457 | 17.457 | 17.343 | +0.66% | 16.000 | `entry_kernel_2+entry_kernel_0+entry_kernel_5` |
| [1, 4, 512] | Ascend950PR_957b | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 17.473 | 17.473 | 16.794 | +4.04% | 16.000 | `MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic+MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0+MhcPost_59130ccee6717ff2c0f15b485e731162_5` |
| [2, 4, 64] | Ascend950DT_9582 | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 18.028 | 18.028 | 17.935 | +0.52% | 16.525 | `entry_kernel_1+entry_kernel_0+entry_kernel_5` |
| [2, 4, 128] | Ascend950DT_9582 | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 16.923 | 16.923 | 16.416 | +3.09% | 15.512 | `entry_kernel_2+entry_kernel_0+entry_kernel_5` |
| [2, 4, 128] | Ascend950PR_957b | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 16.929 | 16.929 | 16.054 | +5.45% | 15.512 | `MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic+MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0+MhcPost_59130ccee6717ff2c0f15b485e731162_5` |
| [3, 4, 128] | Ascend950DT_9582 | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 16.942 | 16.942 | 16.835 | +0.64% | 15.530 | `entry_kernel_2 + entry_kernel_0 + entry_kernel_5` |
| [3, 4, 128] | Ascend950PR_957b | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 16.948 | 16.948 | 16.406 | +3.30% | 15.530 | `MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic+MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0+MhcPost_59130ccee6717ff2c0f15b485e731162_5` |
| [4, 4, 96] | Ascend950DT_9582 | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 18.452 | 18.452 | 17.570 | +5.02% | 16.913 | `entry_kernel_1+entry_kernel_0+entry_kernel_5` |
| [4, 4, 256] | Ascend950DT_9582 | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 17.300 | 17.300 | 17.344 | -0.25% | 15.857 | `entry_kernel_2+entry_kernel_0+entry_kernel_5` |
| [4, 4, 256] | Ascend950PR_957b | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 17.312 | 17.312 | 17.061 | +1.48% | 15.857 | `MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic+MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0+MhcPost_59130ccee6717ff2c0f15b485e731162_5` |
| [4, 4, 1024] | Ascend950DT_9582 | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 18.127 | 18.127 | 18.220 | -0.51% | 16.608 | `entry_kernel_2+entry_kernel_0+entry_kernel_5` |
| [4, 4, 1024] | Ascend950PR_957b | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 18.160 | 18.160 | 18.525 | -1.97% | 16.608 | `MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic+MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0+MhcPost_59130ccee6717ff2c0f15b485e731162_5` |
| [5, 4, 256] | Ascend950DT_9582 | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 17.326 | 17.326 | 17.125 | +1.17% | 15.881 | `entry_kernel_2+entry_kernel_0+entry_kernel_5` |
| [6, 4, 128] | Ascend950DT_9582 | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 17.001 | 17.001 | 17.016 | -0.09% | 15.584 | `entry_kernel_2+entry_kernel_0+entry_kernel_5` |
| [6, 4, 1024] | Ascend950DT_9582 | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 18.259 | 18.259 | 18.361 | -0.56% | 16.728 | `entry_kernel_2+entry_kernel_0+entry_kernel_5` |
| [8, 4, 128] | Ascend950DT_9582 | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 17.040 | 17.040 | 16.791 | +1.48% | 15.620 | `entry_kernel_2 + entry_kernel_0 + entry_kernel_5` |
| [8, 4, 128] | Ascend950PR_957b | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 17.046 | 17.046 | 16.738 | +1.84% | 15.620 | `MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic+MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0+MhcPost_59130ccee6717ff2c0f15b485e731162_5` |
| [8, 4, 512] | Ascend950DT_9582 | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 17.734 | 17.734 | 18.652 | -4.92% | 16.253 | `entry_kernel_2+entry_kernel_0+entry_kernel_5` |
| [9, 4, 1024] | Ascend950DT_9582 | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 18.458 | 18.458 | 18.482 | -0.13% | 16.909 | `entry_kernel_2+entry_kernel_0+entry_kernel_5` |
| [9, 4, 1024] | Ascend950PR_957b | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 18.502 | 18.502 | 19.199 | -3.63% | 16.909 | `MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic+MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0+MhcPost_59130ccee6717ff2c0f15b485e731162_5` |
| [12, 4, 384] | Ascend950DT_9582 | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 17.601 | 17.601 | 18.486 | -4.79% | 16.132 | `entry_kernel_2+entry_kernel_0+entry_kernel_5` |
| [12, 4, 768] | Ascend950DT_9582 | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 18.175 | 18.175 | 18.787 | -3.26% | 16.652 | `entry_kernel_2+entry_kernel_0+entry_kernel_5` |
| [16, 4, 128] | Ascend950DT_9582 | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 17.196 | 17.196 | 18.202 | -5.53% | 15.762 | `entry_kernel_2+entry_kernel_0+entry_kernel_5` |
| [16, 4, 128] | Ascend950PR_957b | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 17.202 | 17.202 | 17.677 | -2.68% | 15.762 | `MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic+MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0+MhcPost_59130ccee6717ff2c0f15b485e731162_5` |
| [16, 4, 512] | Ascend950DT_9582 | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 18.051 | 18.051 | 18.691 | -3.43% | 16.541 | `entry_kernel_2+entry_kernel_0+entry_kernel_5` |
| [17, 4, 512] | Ascend950DT_9582 | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 19.343 | 19.343 | 19.328 | +0.08% | 17.726 | `entry_kernel_2+entry_kernel_0+entry_kernel_5` |
| [24, 4, 1536] | Ascend950DT_9582 | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 22.595 | 22.595 | 21.986 | +2.77% | 19.762 | `entry_kernel_2+entry_kernel_0+entry_kernel_5` |
| [32, 4, 1024] | Ascend950DT_9582 | `{"dtype":"bf16","implementation":"split","num_iters":20}` | 19.558 | 19.558 | 20.070 | -2.55% | 17.913 | `entry_kernel_2+entry_kernel_0+entry_kernel_6` |

## mhc_pre

| Shape | 目标 SKU | 输入属性 | 旧模型 | 本版 | CA | 偏差 | 970DT 外推 | 实测 kernel |
|---|---|---|---:|---:|---:|---:|---:|---|
| [1, 4, 64] | Ascend950DT_9582 | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 9.579 | 9.579 | 9.482 | +1.02% | 8.780 | `entry_kernel_1` |
| [1, 4, 64] | Ascend950PR_957b | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 9.579 | 9.579 | 8.952 | +7.01% | 8.780 | `MhcPre_52e042c374f575edbe357a32523dba2c_1_mix_aic` |
| [1, 4, 80] | Ascend950DT_9582 | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 9.705 | 9.705 | 9.802 | -0.99% | 8.896 | `entry_kernel_1` |
| [1, 4, 128] | Ascend950DT_9582 | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 8.378 | 8.378 | 7.879 | +6.33% | 7.679 | `MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic` |
| [1, 4, 128] | Ascend950DT_9582 | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 8.378 | 8.378 | 8.271 | +1.29% | 7.679 | `entry_kernel_2` |
| [1, 4, 128] | Ascend950PR_957b | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 8.378 | 8.378 | 7.638 | +9.69% | 7.679 | `MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic` |
| [1, 4, 512] | Ascend950DT_9582 | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 8.717 | 8.717 | 8.544 | +2.03% | 7.989 | `entry_kernel_2` |
| [1, 4, 512] | Ascend950PR_957b | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 8.722 | 8.722 | 7.876 | +10.75% | 7.989 | `MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic` |
| [2, 4, 64] | Ascend950DT_9582 | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 9.589 | 9.589 | 9.962 | -3.74% | 8.790 | `entry_kernel_1` |
| [2, 4, 128] | Ascend950DT_9582 | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 8.380 | 8.380 | 8.005 | +4.68% | 7.681 | `entry_kernel_2` |
| [2, 4, 128] | Ascend950PR_957b | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 8.380 | 8.380 | 7.644 | +9.64% | 7.681 | `MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic` |
| [3, 4, 128] | Ascend950DT_9582 | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 8.382 | 8.382 | 8.312 | +0.84% | 7.684 | `entry_kernel_2` |
| [3, 4, 128] | Ascend950PR_957b | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 8.383 | 8.383 | 7.921 | +5.82% | 7.684 | `MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic` |
| [4, 4, 96] | Ascend950DT_9582 | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 9.879 | 9.879 | 9.312 | +6.09% | 9.055 | `entry_kernel_1` |
| [4, 4, 256] | Ascend950DT_9582 | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 8.505 | 8.505 | 8.421 | +1.00% | 7.796 | `entry_kernel_2` |
| [4, 4, 256] | Ascend950PR_957b | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 8.506 | 8.506 | 8.085 | +5.21% | 7.796 | `MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic` |
| [4, 4, 1024] | Ascend950DT_9582 | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 9.226 | 9.226 | 8.730 | +5.68% | 8.450 | `entry_kernel_2` |
| [4, 4, 1024] | Ascend950PR_957b | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 9.247 | 9.247 | 9.420 | -1.84% | 8.450 | `MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic` |
| [5, 4, 256] | Ascend950DT_9582 | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 8.509 | 8.509 | 8.251 | +3.13% | 7.800 | `entry_kernel_2` |
| [6, 4, 128] | Ascend950DT_9582 | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 8.389 | 8.389 | 8.410 | -0.24% | 7.690 | `entry_kernel_2` |
| [6, 4, 1024] | Ascend950DT_9582 | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 9.263 | 9.263 | 8.973 | +3.23% | 8.483 | `entry_kernel_2` |
| [8, 4, 128] | Ascend950DT_9582 | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 8.394 | 8.394 | 8.542 | -1.74% | 7.694 | `entry_kernel_2` |
| [8, 4, 128] | Ascend950PR_957b | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 8.394 | 8.394 | 8.072 | +4.00% | 7.694 | `MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic` |
| [8, 4, 512] | Ascend950DT_9582 | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 8.782 | 8.782 | 8.866 | -0.95% | 8.048 | `entry_kernel_2` |
| [9, 4, 1024] | Ascend950DT_9582 | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 9.318 | 9.318 | 9.024 | +3.26% | 8.533 | `entry_kernel_2` |
| [9, 4, 1024] | Ascend950PR_957b | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 9.344 | 9.344 | 9.865 | -5.28% | 8.533 | `MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic` |
| [12, 4, 384] | Ascend950DT_9582 | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 8.680 | 8.680 | 9.137 | -5.00% | 7.956 | `entry_kernel_2` |
| [12, 4, 768] | Ascend950DT_9582 | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 9.097 | 9.097 | 9.058 | +0.42% | 8.333 | `entry_kernel_2` |
| [16, 4, 128] | Ascend950DT_9582 | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 8.412 | 8.412 | 9.308 | -9.62% | 7.711 | `entry_kernel_2` |
| [16, 4, 128] | Ascend950PR_957b | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 8.413 | 8.413 | 8.696 | -3.26% | 7.711 | `MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic` |
| [16, 4, 512] | Ascend950DT_9582 | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 8.856 | 8.856 | 9.385 | -5.64% | 8.115 | `entry_kernel_2` |
| [17, 4, 512] | Ascend950DT_9582 | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 10.118 | 10.118 | 10.095 | +0.23% | 9.272 | `entry_kernel_2` |
| [24, 4, 1536] | Ascend950DT_9582 | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 11.514 | 11.514 | 12.201 | -5.63% | 9.632 | `entry_kernel_2` |
| [32, 4, 1024] | Ascend950DT_9582 | `{"dtype":"bf16","alpha_count":3,"has_gamma":false,"return_intermediates":true}` | 10.996 | 10.996 | 11.368 | -3.27% | 10.064 | `entry_kernel_2` |

## mhc_pre_sinkhorn

| Shape | 目标 SKU | 输入属性 | 旧模型 | 本版 | CA | 偏差 | 970DT 外推 | 实测 kernel |
|---|---|---|---:|---:|---:|---:|---:|---|
| [1, 4, 4096] | Ascend950DT_9582 | `{"dtype":"bf16","num_iters":20,"need_backward":false}` | 9.939 | 9.939 | 9.701 | +2.46% | 9.109 | `entry_kernel_3e8` |
| [1, 4, 4096] | Ascend950DT_9582 | `{"dtype":"bf16","num_iters":20,"need_backward":false}` | 9.939 | 9.939 | 10.108 | -1.67% | 9.109 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [1, 4, 4096] | Ascend950PR_957b | `{"dtype":"bf16","num_iters":20,"need_backward":false}` | 9.990 | 9.990 | 8.763 | +14.00% | 9.109 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [2, 4, 4096] | Ascend950DT_9582 | `{"dtype":"bf16","num_iters":20,"need_backward":false}` | 9.939 | 9.939 | 10.648 | -6.66% | 9.109 | `entry_kernel_3e8` |
| [2, 4, 4096] | Ascend950PR_957b | `{"dtype":"bf16","num_iters":20,"need_backward":false}` | 9.990 | 9.990 | 10.198 | -2.04% | 9.109 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [3, 4, 4096] | Ascend950DT_9582 | `{"dtype":"bf16","num_iters":20,"need_backward":false}` | 9.939 | 9.971 | 10.373 | -3.88% | 9.109 | `entry_kernel_3e8` |
| [3, 4, 4096] | Ascend950PR_957b | `{"dtype":"bf16","num_iters":20,"need_backward":false}` | 9.990 | 10.038 | 9.250 | +8.52% | 9.109 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [4, 4, 4096] | Ascend950DT_9582 | `{"dtype":"bf16","num_iters":20,"need_backward":false}` | 9.939 | 9.971 | 9.939 | +0.31% | 9.109 | `entry_kernel_3e8` |
| [4, 4, 4096] | Ascend950PR_957b | `{"dtype":"bf16","num_iters":20,"need_backward":false}` | 9.990 | 10.038 | 9.470 | +5.99% | 9.109 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [5, 4, 4096] | Ascend950DT_9582 | `{"dtype":"bf16","num_iters":20,"need_backward":false}` | 9.939 | 10.002 | 9.761 | +2.47% | 9.109 | `entry_kernel_3e8` |
| [5, 4, 4096] | Ascend950PR_957b | `{"dtype":"bf16","num_iters":20,"need_backward":false}` | 9.990 | 10.086 | 9.495 | +6.23% | 9.109 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [6, 4, 4096] | Ascend950DT_9582 | `{"dtype":"bf16","num_iters":20,"need_backward":false}` | 9.939 | 10.002 | 9.785 | +2.21% | 9.109 | `entry_kernel_3e8` |
| [6, 4, 4096] | Ascend950PR_957b | `{"dtype":"bf16","num_iters":20,"need_backward":false}` | 9.990 | 10.086 | 9.405 | +7.24% | 9.109 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [7, 4, 4096] | Ascend950DT_9582 | `{"dtype":"bf16","num_iters":20,"need_backward":false}` | 9.939 | 10.033 | 11.518 | -12.89% | 9.109 | `entry_kernel_3e8` |
| [7, 4, 4096] | Ascend950PR_957b | `{"dtype":"bf16","num_iters":20,"need_backward":false}` | 9.990 | 10.134 | 9.388 | +7.94% | 9.109 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [8, 4, 4096] | Ascend950DT_9582 | `{"dtype":"bf16","num_iters":20,"need_backward":false}` | 9.939 | 10.033 | 10.529 | -4.71% | 9.109 | `entry_kernel_3e8` |
| [8, 4, 4096] | Ascend950PR_957b | `{"dtype":"bf16","num_iters":20,"need_backward":false}` | 9.990 | 10.134 | 9.713 | +4.33% | 9.109 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [9, 4, 4096] | Ascend950DT_9582 | `{"dtype":"bf16","num_iters":20,"need_backward":false}` | 9.939 | 10.064 | 10.202 | -1.35% | 9.109 | `entry_kernel_3e8` |
| [9, 4, 4096] | Ascend950PR_957b | `{"dtype":"bf16","num_iters":20,"need_backward":false}` | 9.990 | 10.182 | 9.390 | +8.44% | 9.109 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [12, 4, 4096] | Ascend950DT_9582 | `{"dtype":"bf16","num_iters":20,"need_backward":false}` | 9.939 | 10.096 | 10.952 | -7.82% | 9.109 | `entry_kernel_3e8` |
| [12, 4, 4096] | Ascend950PR_957b | `{"dtype":"bf16","num_iters":20,"need_backward":false}` | 9.990 | 10.230 | 9.686 | +5.62% | 9.109 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [16, 4, 4096] | Ascend950DT_9582 | `{"dtype":"bf16","num_iters":20,"need_backward":false}` | 9.939 | 10.158 | 11.560 | -12.13% | 9.109 | `entry_kernel_3e8` |
| [20, 4, 4096] | Ascend950DT_9582 | `{"num_iters":20,"need_backward":false}` | 9.939 | 10.298 | 11.519 | -10.61% | 9.109 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [20, 4, 4096] | Ascend950PR_957b | `{"num_iters":20,"need_backward":false}` | 10.115 | 10.539 | 11.231 | -6.16% | 9.109 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [22, 4, 4096] | Ascend950DT_9582 | `{"num_iters":20,"need_backward":false}` | 9.939 | 10.329 | 11.058 | -6.59% | 9.109 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [22, 4, 4096] | Ascend950PR_957b | `{"num_iters":20,"need_backward":false}` | 10.115 | 10.587 | 11.103 | -4.65% | 9.109 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [24, 4, 4096] | Ascend950DT_9582 | `{"dtype":"bf16","num_iters":20,"need_backward":false}` | 9.939 | 10.360 | 11.583 | -10.56% | 9.109 | `entry_kernel_3e8` |
| [26, 4, 4096] | Ascend950DT_9582 | `{"num_iters":20,"need_backward":false}` | 9.939 | 10.391 | 11.182 | -7.07% | 9.109 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [26, 4, 4096] | Ascend950PR_957b | `{"num_iters":20,"need_backward":false}` | 10.115 | 10.683 | 10.893 | -1.92% | 9.109 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [30, 4, 4096] | Ascend950DT_9582 | `{"num_iters":20,"need_backward":false}` | 9.939 | 10.454 | 11.050 | -5.39% | 9.109 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [30, 4, 4096] | Ascend950PR_957b | `{"num_iters":20,"need_backward":false}` | 10.115 | 10.779 | 10.511 | +2.55% | 9.109 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [32, 4, 4096] | Ascend950DT_9582 | `{"dtype":"bf16","num_iters":20,"need_backward":false}` | 9.939 | 10.485 | 11.114 | -5.66% | 9.109 | `entry_kernel_3e8` |
| [33, 4, 4096] | Ascend950DT_9582 | `{"num_iters":20,"need_backward":false}` | 11.739 | 10.686 | 11.301 | -5.44% | 10.754 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [33, 4, 4096] | Ascend950PR_957b | `{"num_iters":20,"need_backward":false}` | 12.272 | 11.084 | 10.855 | +2.11% | 10.754 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [36, 4, 4096] | Ascend950DT_9582 | `{"num_iters":20,"need_backward":false}` | 11.739 | 10.718 | 11.522 | -6.98% | 10.754 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [36, 4, 4096] | Ascend950PR_957b | `{"num_iters":20,"need_backward":false}` | 12.272 | 11.132 | 10.701 | +4.03% | 10.754 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [48, 4, 4096] | Ascend950DT_9582 | `{"num_iters":20,"need_backward":false}` | 11.739 | 10.905 | 11.890 | -8.28% | 10.754 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [48, 4, 4096] | Ascend950PR_957b | `{"num_iters":20,"need_backward":false}` | 12.272 | 11.421 | 11.271 | +1.33% | 10.754 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [56, 4, 4096] | Ascend950DT_9582 | `{"num_iters":20,"need_backward":false}` | 13.718 | 11.107 | 11.595 | -4.20% | 12.566 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [56, 4, 4096] | Ascend950PR_957b | `{"num_iters":20,"need_backward":false}` | 14.428 | 11.776 | 12.215 | -3.59% | 12.566 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [64, 4, 4096] | Ascend950DT_9582 | `{"num_iters":20,"need_backward":false}` | 13.718 | 11.232 | 12.468 | -9.92% | 12.566 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [64, 4, 4096] | Ascend950PR_957b | `{"num_iters":20,"need_backward":false}` | 14.890 | 14.022 | 13.152 | +6.61% | 12.566 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [80, 4, 4096] | Ascend950DT_9582 | `{"num_iters":20,"need_backward":false}` | 15.752 | 13.651 | 13.782 | -0.95% | 14.378 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |
| [80, 4, 4096] | Ascend950PR_957b | `{"num_iters":20,"need_backward":false}` | 16.640 | 14.616 | 13.653 | +7.05% | 14.378 | `MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic` |

## mhc_sinkhorn

| Shape | 目标 SKU | 输入属性 | 旧模型 | 本版 | CA | 偏差 | 970DT 外推 | 实测 kernel |
|---|---|---|---:|---:|---:|---:|---:|---|
| [1, 4] | Ascend950DT_9582 | `{"dtype":"fp32","num_iters":20,"return_intermediates":false}` | 5.786 | 5.786 | 6.149 | -5.91% | 5.300 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [1, 4] | Ascend950DT_9582 | `{"dtype":"fp32","num_iters":20,"return_intermediates":false}` | 5.786 | 5.786 | 6.296 | -8.11% | 5.300 | `entry_kernel_0` |
| [1, 4] | Ascend950PR_957b | `{"dtype":"fp32","num_iters":20,"return_intermediates":false}` | 5.869 | 5.869 | 5.948 | -1.34% | 5.300 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [2, 4] | Ascend950DT_9582 | `{"dtype":"fp32","return_intermediates":false,"num_iters":2}` | 3.077 | 3.077 | 3.593 | -14.35% | 2.820 | `entry_kernel_0` |
| [2, 4] | Ascend950PR_957b | `{"dtype":"fp32","return_intermediates":false,"num_iters":2}` | 3.085 | 3.085 | 3.697 | -16.54% | 2.820 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [3, 4] | Ascend950DT_9582 | `{"dtype":"fp32","return_intermediates":false,"num_iters":2}` | 3.102 | 3.102 | 3.730 | -16.83% | 2.843 | `entry_kernel_0` |
| [4, 4] | Ascend950DT_9582 | `{"dtype":"fp32","return_intermediates":false,"num_iters":5}` | 3.578 | 3.578 | 2.945 | +21.47% | 3.279 | `entry_kernel_0` |
| [4, 4] | Ascend950PR_957b | `{"dtype":"fp32","return_intermediates":false,"num_iters":5}` | 3.599 | 3.599 | 3.938 | -8.62% | 3.279 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [5, 4] | Ascend950DT_9582 | `{"dtype":"fp32","return_intermediates":false,"num_iters":6}` | 3.744 | 3.744 | 4.066 | -7.91% | 3.431 | `entry_kernel_0` |
| [6, 4] | Ascend950DT_9582 | `{"num_iters":2,"return_intermediates":false}` | 3.144 | 3.144 | 3.580 | -12.17% | 2.882 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [6, 4] | Ascend950PR_957b | `{"num_iters":2,"return_intermediates":false}` | 3.153 | 3.153 | 3.438 | -8.31% | 2.882 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [7, 4] | Ascend950DT_9582 | `{"dtype":"fp32","return_intermediates":false,"num_iters":12}` | 4.682 | 4.682 | 5.142 | -8.94% | 4.290 | `entry_kernel_0` |
| [8, 4] | Ascend950DT_9582 | `{"dtype":"fp32","return_intermediates":false,"num_iters":10}` | 4.385 | 4.385 | 4.995 | -12.22% | 4.017 | `entry_kernel_0` |
| [8, 4] | Ascend950PR_957b | `{"dtype":"fp32","return_intermediates":false,"num_iters":10}` | 4.426 | 4.426 | 4.719 | -6.21% | 4.017 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [9, 4] | Ascend950DT_9582 | `{"dtype":"fp32","return_intermediates":false,"num_iters":20}` | 5.920 | 5.920 | 6.212 | -4.70% | 5.423 | `entry_kernel_0` |
| [9, 4] | Ascend950PR_957b | `{"dtype":"fp32","return_intermediates":false,"num_iters":20}` | 6.003 | 6.003 | 6.268 | -4.23% | 5.423 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [11, 4] | Ascend950DT_9582 | `{"dtype":"fp32","return_intermediates":false,"num_iters":20}` | 5.932 | 5.932 | 6.406 | -7.39% | 5.435 | `entry_kernel_0` |
| [14, 4] | Ascend950DT_9582 | `{"num_iters":7,"return_intermediates":false}` | 3.960 | 3.960 | 4.636 | -14.58% | 3.629 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [14, 4] | Ascend950PR_957b | `{"num_iters":7,"return_intermediates":false}` | 3.989 | 3.989 | 4.395 | -9.22% | 3.629 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [15, 4] | Ascend950DT_9582 | `{"dtype":"fp32","return_intermediates":false,"num_iters":24}` | 6.563 | 6.563 | 6.967 | -5.80% | 6.012 | `entry_kernel_0` |
| [16, 4] | Ascend950DT_9582 | `{"dtype":"fp32","return_intermediates":false,"num_iters":32}` | 7.789 | 7.789 | 8.156 | -4.50% | 7.135 | `entry_kernel_0` |
| [16, 4] | Ascend950PR_957b | `{"dtype":"fp32","return_intermediates":false,"num_iters":32}` | 7.922 | 7.922 | 8.328 | -4.87% | 7.135 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [17, 4] | Ascend950DT_9582 | `{"dtype":"fp32","num_iters":20,"return_intermediates":false}` | 5.959 | 5.959 | 6.532 | -8.77% | 5.459 | `entry_kernel_0` |
| [17, 4] | Ascend950PR_957b | `{"dtype":"fp32","num_iters":20,"return_intermediates":false}` | 6.042 | 6.042 | 6.280 | -3.79% | 5.459 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [21, 4] | Ascend950DT_9582 | `{"dtype":"fp32","return_intermediates":false,"num_iters":16}` | 5.361 | 5.361 | 5.597 | -4.22% | 4.911 | `entry_kernel_0` |
| [28, 4] | Ascend950DT_9582 | `{"num_iters":15,"return_intermediates":false}` | 5.225 | 5.225 | 5.886 | -11.22% | 4.787 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [28, 4] | Ascend950PR_957b | `{"num_iters":15,"return_intermediates":false}` | 5.288 | 5.288 | 5.691 | -7.08% | 4.787 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [31, 4] | Ascend950DT_9582 | `{"dtype":"fp32","return_intermediates":false,"num_iters":20}` | 5.996 | 5.996 | 6.333 | -5.32% | 5.493 | `entry_kernel_0` |
| [31, 4] | Ascend950PR_957b | `{"dtype":"fp32","return_intermediates":false,"num_iters":20}` | 6.079 | 6.079 | 6.634 | -8.37% | 5.493 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [32, 4] | Ascend950DT_9582 | `{"dtype":"fp32","return_intermediates":false,"num_iters":20}` | 5.998 | 5.998 | 5.336 | +12.39% | 5.494 | `entry_kernel_0` |
| [32, 4] | Ascend950PR_957b | `{"dtype":"fp32","return_intermediates":false,"num_iters":20}` | 6.081 | 6.081 | 6.616 | -8.09% | 5.494 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [33, 4] | Ascend950DT_9582 | `{"dtype":"fp32","return_intermediates":false,"num_iters":20}` | 6.000 | 6.000 | 6.377 | -5.92% | 5.496 | `entry_kernel_0` |
| [47, 4] | Ascend950DT_9582 | `{"dtype":"fp32","return_intermediates":false,"num_iters":4}` | 3.576 | 3.576 | 3.996 | -10.52% | 3.277 | `entry_kernel_0` |
| [54, 4] | Ascend950DT_9582 | `{"num_iters":20,"return_intermediates":false}` | 6.030 | 6.030 | 7.551 | -20.14% | 5.524 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [54, 4] | Ascend950PR_957b | `{"num_iters":20,"return_intermediates":false}` | 6.113 | 6.113 | 7.287 | -16.11% | 5.524 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [55, 4] | Ascend950DT_9582 | `{"num_iters":32,"return_intermediates":false}` | 7.865 | 7.865 | 9.274 | -15.19% | 7.204 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [55, 4] | Ascend950PR_957b | `{"num_iters":32,"return_intermediates":false}` | 7.998 | 7.998 | 9.111 | -12.22% | 7.204 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [56, 4] | Ascend950DT_9582 | `{"num_iters":4,"return_intermediates":false}` | 3.587 | 3.587 | 5.068 | -29.23% | 3.287 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [56, 4] | Ascend950PR_957b | `{"num_iters":4,"return_intermediates":false}` | 3.603 | 3.603 | 4.464 | -19.28% | 3.287 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [57, 4] | Ascend950DT_9582 | `{"num_iters":20,"return_intermediates":false}` | 6.033 | 6.033 | 7.321 | -17.59% | 5.527 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [57, 4] | Ascend950PR_957b | `{"num_iters":20,"return_intermediates":false}` | 6.075 | 6.075 | 6.410 | -5.22% | 5.527 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [63, 4] | Ascend950DT_9582 | `{"dtype":"fp32","return_intermediates":false,"num_iters":5}` | 3.747 | 3.747 | 4.101 | -8.64% | 3.434 | `entry_kernel_0` |
| [64, 4] | Ascend950DT_9582 | `{"dtype":"fp32","num_iters":20,"return_intermediates":false}` | 6.040 | 6.040 | 7.598 | -20.50% | 5.533 | `entry_kernel_0` |
| [64, 4] | Ascend950PR_957b | `{"dtype":"fp32","num_iters":20,"return_intermediates":false}` | 6.081 | 6.081 | 6.533 | -6.92% | 5.533 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [65, 4] | Ascend950DT_9582 | `{"dtype":"fp32","return_intermediates":false,"num_iters":10}` | 4.471 | 4.471 | 3.258 | +37.23% | 4.135 | `entry_kernel_0` |
| [66, 4] | Ascend950DT_9582 | `{"dtype":"fp32","return_intermediates":false,"num_iters":10}` | 4.471 | 4.471 | 3.212 | +39.20% | 4.136 | `entry_kernel_0` |
| [68, 4] | Ascend950DT_9582 | `{"num_iters":10,"return_intermediates":false}` | 4.473 | 4.473 | 4.922 | -9.13% | 4.138 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [68, 4] | Ascend950PR_957b | `{"num_iters":10,"return_intermediates":false}` | 4.515 | 4.515 | 4.937 | -8.55% | 4.138 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [96, 4] | Ascend950DT_9582 | `{"dtype":"fp32","return_intermediates":false,"num_iters":20}` | 6.023 | 6.023 | 5.334 | +12.91% | 5.556 | `entry_kernel_0` |
| [113, 4] | Ascend950DT_9582 | `{"num_iters":20,"return_intermediates":false}` | 6.033 | 6.033 | 6.996 | -13.77% | 5.565 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [113, 4] | Ascend950PR_957b | `{"num_iters":20,"return_intermediates":false}` | 6.091 | 6.091 | 6.827 | -10.77% | 5.565 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [128, 4] | Ascend950DT_9582 | `{"dtype":"fp32","return_intermediates":false,"num_iters":20}` | 6.040 | 6.040 | 5.475 | +10.33% | 5.533 | `entry_kernel_0` |
| [193, 4] | Ascend950DT_9582 | `{"num_iters":12,"return_intermediates":false}` | 4.801 | 4.801 | 6.014 | -20.17% | 4.437 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |
| [193, 4] | Ascend950PR_957b | `{"num_iters":12,"return_intermediates":false}` | 4.851 | 4.851 | 5.953 | -18.51% | 4.437 | `MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0` |

## moe_ffn

| Shape | 目标 SKU | 输入属性 | 旧模型 | 本版 | CA | 偏差 | 970DT 外推 | 实测 kernel |
|---|---|---|---:|---:|---:|---:|---:|---|
| [1, 64, 64, 1] | Ascend950DT_9582 | `{"expert_counts":[1]}` | 10.682 | 10.682 | 10.195 | +4.78% | 9.792 | `entry_kernel_0+entry_kernel_0` |
| [1, 64, 64, 1] | Ascend950PR_957b | `{"expert_counts":[1]}` | 10.682 | 10.682 | 9.995 | +6.88% | 9.792 | `GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic+GroupedMatmul_037cec40685a17f15b75759e999767c3_0` |
| [2, 64, 64, 1] | Ascend950DT_9582 | `{"expert_counts":[2]}` | 10.682 | 10.682 | 10.195 | +4.77% | 9.792 | `GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic+GroupedMatmul_037cec40685a17f15b75759e999767c3_0` |
| [2, 64, 64, 1] | Ascend950DT_9582 | `{"expert_counts":[2]}` | 10.682 | 10.682 | 10.253 | +4.19% | 9.792 | `entry_kernel_0 + entry_kernel_0` |
| [2, 64, 64, 1] | Ascend950PR_957b | `{"expert_counts":[2]}` | 10.682 | 10.682 | 9.992 | +6.91% | 9.792 | `GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic+GroupedMatmul_037cec40685a17f15b75759e999767c3_0` |
| [2, 64, 128, 1] | Ascend950DT_9582 | `{"expert_counts":[2]}` | 10.967 | 10.967 | 10.825 | +1.31% | 10.053 | `entry_kernel_0 + entry_kernel_0` |
| [2, 64, 128, 1] | Ascend950PR_957b | `{"expert_counts":[2]}` | 10.968 | 10.968 | 9.819 | +11.70% | 10.053 | `GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic+GroupedMatmul_037cec40685a17f15b75759e999767c3_0` |
| [3, 64, 64, 2] | Ascend950DT_9582 | `{"expert_counts":[1,2]}` | 10.682 | 10.682 | 10.852 | -1.56% | 9.792 | `entry_kernel_0+entry_kernel_0` |
| [3, 64, 64, 2] | Ascend950PR_957b | `{"expert_counts":[1,2]}` | 10.682 | 10.682 | 10.267 | +4.05% | 9.792 | `GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic+GroupedMatmul_037cec40685a17f15b75759e999767c3_0` |
| [4, 128, 64, 2] | Ascend950DT_9582 | `{"expert_counts":[1,3]}` | 11.268 | 11.268 | 11.070 | +1.78% | 10.328 | `entry_kernel_0 + entry_kernel_0` |
| [4, 128, 64, 2] | Ascend950PR_957b | `{"expert_counts":[1,3]}` | 11.272 | 11.272 | 10.727 | +5.08% | 10.328 | `GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic+GroupedMatmul_037cec40685a17f15b75759e999767c3_0` |
| [6, 128, 128, 2] | Ascend950DT_9582 | `{"expert_counts":[1,5]}` | 11.553 | 11.553 | 11.087 | +4.21% | 10.588 | `entry_kernel_0 + entry_kernel_0` |
| [6, 128, 128, 2] | Ascend950PR_957b | `{"expert_counts":[1,5]}` | 11.564 | 11.564 | 10.842 | +6.65% | 10.588 | `GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic+GroupedMatmul_037cec40685a17f15b75759e999767c3_0` |
| [8, 64, 128, 1] | Ascend950DT_9582 | `{"expert_counts":[8]}` | 10.967 | 10.967 | 10.633 | +3.15% | 10.053 | `entry_kernel_0+entry_kernel_0` |
| [8, 64, 128, 1] | Ascend950PR_957b | `{"expert_counts":[8]}` | 10.968 | 10.968 | 10.324 | +6.25% | 10.053 | `GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic+GroupedMatmul_037cec40685a17f15b75759e999767c3_0` |
| [8, 256, 128, 2] | Ascend950DT_9582 | `{"expert_counts":[1,7]}` | 13.636 | 13.636 | 11.119 | +22.63% | 12.490 | `entry_kernel_0+entry_kernel_0` |
| [8, 256, 128, 2] | Ascend950PR_957b | `{"expert_counts":[1,7]}` | 13.666 | 13.666 | 11.056 | +23.61% | 12.490 | `GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic+GroupedMatmul_037cec40685a17f15b75759e999767c3_0` |
| [16, 128, 128, 2] | Ascend950DT_9582 | `{"expert_counts":[1,15]}` | 11.553 | 11.553 | 11.209 | +3.07% | 10.587 | `entry_kernel_0+entry_kernel_0` |
| [16, 128, 128, 2] | Ascend950PR_957b | `{"expert_counts":[1,15]}` | 11.564 | 11.564 | 11.053 | +4.63% | 10.587 | `GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic+GroupedMatmul_037cec40685a17f15b75759e999767c3_0` |
| [16, 256, 256, 1] | Ascend950DT_9582 | `{"expert_counts":[16]}` | 14.457 | 14.457 | 11.753 | +23.01% | 13.240 | `entry_kernel_0+entry_kernel_0` |
| [17, 128, 64, 2] | Ascend950DT_9582 | `{"expert_counts":[8,9]}` | 11.268 | 11.268 | 11.335 | -0.60% | 10.327 | `entry_kernel_0+entry_kernel_0` |
| [17, 128, 64, 2] | Ascend950PR_957b | `{"expert_counts":[8,9]}` | 11.272 | 11.272 | 11.022 | +2.27% | 10.327 | `GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic+GroupedMatmul_037cec40685a17f15b75759e999767c3_0` |
| [32, 128, 128, 4] | Ascend950DT_9582 | `{"expert_counts":[2,6,8,16]}` | 11.553 | 11.553 | 11.571 | -0.15% | 10.585 | `entry_kernel_0+entry_kernel_0` |
| [32, 256, 128, 2] | Ascend950DT_9582 | `{"expert_counts":[1,31]}` | 13.636 | 13.636 | 12.119 | +12.52% | 12.489 | `entry_kernel_0+entry_kernel_0` |
| [64, 128, 256, 4] | Ascend950DT_9582 | `{"expert_counts":[8,8,16,32]}` | 12.374 | 12.374 | 12.496 | -0.98% | 11.327 | `entry_kernel_0+entry_kernel_0` |

## moe_gating_top_k

| Shape | 目标 SKU | 输入属性 | 旧模型 | 本版 | CA | 偏差 | 970DT 外推 | 实测 kernel |
|---|---|---|---:|---:|---:|---:|---:|---|
| [1, 128, 8] | Ascend950DT_9582 | `{}` | 3.825 | 3.825 | 3.822 | +0.09% | 3.506 | `MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005` |
| [1, 384, 6] | Ascend950DT_9582 | `{}` | 4.330 | 4.330 | 4.239 | +2.14% | 3.968 | `MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005` |
| [1, 384, 6] | Ascend950PR_957b | `{}` | 4.353 | 4.353 | 4.094 | +6.33% | 3.968 | `MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005` |
| [1, 896, 16] | Ascend950DT_9582 | `{}` | 5.340 | 5.340 | 5.344 | -0.06% | 4.893 | `MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005` |
| [2, 896, 16] | Ascend950DT_9582 | `{}` | 5.340 | 5.340 | 5.472 | -2.40% | 4.893 | `MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005` |
| [2, 896, 16] | Ascend950PR_957b | `{}` | 5.391 | 5.391 | 5.446 | -1.02% | 4.893 | `MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005` |
| [4, 256, 8] | Ascend950DT_9582 | `{}` | 4.078 | 4.078 | 4.865 | -16.18% | 3.737 | `MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005` |
| [4, 256, 8] | Ascend950PR_957b | `{}` | 4.094 | 4.094 | 4.410 | -7.17% | 3.737 | `MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005` |
| [7, 896, 16] | Ascend950DT_9582 | `{}` | 5.340 | 5.340 | 5.576 | -4.23% | 4.893 | `MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005` |
| [7, 896, 16] | Ascend950PR_957b | `{}` | 5.391 | 5.391 | 5.567 | -3.17% | 4.893 | `MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005` |
| [9, 384, 8] | Ascend950DT_9582 | `{}` | 4.330 | 4.330 | 4.717 | -8.20% | 3.968 | `MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005` |
| [9, 384, 8] | Ascend950PR_957b | `{}` | 4.353 | 4.353 | 4.535 | -4.01% | 3.968 | `MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005` |
| [12, 896, 16] | Ascend950DT_9582 | `{}` | 5.340 | 5.340 | 6.121 | -12.76% | 4.893 | `MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005` |
| [12, 896, 16] | Ascend950PR_957b | `{}` | 5.391 | 5.391 | 5.720 | -5.76% | 4.893 | `MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005` |
| [16, 512, 8] | Ascend950DT_9582 | `{}` | 4.583 | 4.583 | 5.278 | -13.18% | 4.200 | `MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005` |
| [16, 512, 8] | Ascend950PR_957b | `{}` | 4.612 | 4.612 | 4.861 | -5.11% | 4.200 | `MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005` |
| [24, 896, 16] | Ascend950DT_9582 | `{}` | 5.340 | 5.340 | 5.915 | -9.71% | 4.893 | `MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005` |
| [24, 896, 16] | Ascend950PR_957b | `{}` | 5.391 | 5.391 | 5.682 | -5.12% | 4.893 | `MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005` |
| [33, 384, 8] | Ascend950DT_9582 | `{}` | 4.330 | 4.330 | 4.903 | -11.68% | 3.968 | `MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005` |
| [33, 384, 8] | Ascend950PR_957b | `{}` | 4.353 | 4.353 | 4.831 | -9.89% | 3.968 | `MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005` |
| [65, 128, 8] | Ascend950DT_9582 | `{}` | 4.162 | 4.162 | 4.598 | -9.47% | 3.506 | `MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005` |
| [65, 128, 8] | Ascend950PR_957b | `{}` | 4.180 | 4.180 | 4.698 | -11.02% | 3.506 | `MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005` |

## quant

| Shape | 目标 SKU | 输入属性 | 旧模型 | 本版 | CA | 偏差 | 970DT 外推 | 实测 kernel |
|---|---|---|---:|---:|---:|---:|---:|---|
| [1, 64] | Ascend950DT_9582 | `{}` | 4.231 | 4.231 | 4.326 | -2.21% | 3.692 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [1, 64] | Ascend950PR_957b | `{}` | 3.591 | 3.591 | 3.946 | -8.99% | 3.692 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [1, 128] | Ascend950DT_9582 | `{}` | 4.231 | 4.231 | 4.325 | -2.18% | 3.693 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [1, 128] | Ascend950PR_957b | `{}` | 3.592 | 3.592 | 3.828 | -6.19% | 3.693 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [1, 192] | Ascend950DT_9582 | `{}` | 4.231 | 4.231 | 4.129 | +2.46% | 3.693 | `entry_kernel_21` |
| [1, 192] | Ascend950PR_957b | `{}` | 3.592 | 3.592 | 3.833 | -6.30% | 3.693 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [1, 256] | Ascend950DT_9582 | `{}` | 4.231 | 4.231 | 4.493 | -5.83% | 3.693 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [1, 256] | Ascend950PR_957b | `{}` | 3.592 | 3.592 | 3.859 | -6.90% | 3.693 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [1, 704] | Ascend950DT_9582 | `{}` | 4.630 | 4.630 | 4.332 | +6.88% | 4.076 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [1, 704] | Ascend950PR_957b | `{}` | 4.010 | 4.010 | 3.964 | +1.18% | 4.076 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [1, 4096] | Ascend950DT_9582 | `{}` | 5.430 | 5.430 | 5.686 | -4.51% | 4.532 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [1, 4096] | Ascend950PR_957b | `{}` | 4.507 | 4.507 | 4.945 | -8.86% | 4.532 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [2, 832] | Ascend950DT_9582 | `{}` | 5.098 | 5.098 | 4.819 | +5.78% | 4.459 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [2, 832] | Ascend950PR_957b | `{}` | 4.428 | 4.428 | 4.421 | +0.16% | 4.459 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [3, 1920] | Ascend950DT_9582 | `{}` | 5.430 | 5.430 | 5.788 | -6.19% | 4.657 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [3, 1920] | Ascend950PR_957b | `{}` | 4.643 | 4.643 | 5.315 | -12.64% | 4.657 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [4, 256] | Ascend950DT_9582 | `{}` | 4.759 | 4.759 | 4.588 | +3.72% | 4.268 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [4, 256] | Ascend950DT_9582 | `{}` | 4.759 | 4.759 | 4.620 | +3.01% | 4.268 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [4, 256] | Ascend950PR_957b | `{}` | 4.219 | 4.219 | 4.157 | +1.50% | 4.268 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [4, 1472] | Ascend950DT_9582 | `{}` | 5.430 | 5.430 | 5.685 | -4.49% | 4.722 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [4, 1472] | Ascend950PR_957b | `{}` | 4.715 | 4.715 | 5.398 | -12.65% | 4.722 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [5, 320] | Ascend950DT_9582 | `{}` | 5.214 | 5.214 | 4.662 | +11.82% | 4.842 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [5, 320] | Ascend950PR_957b | `{}` | 4.846 | 4.846 | 4.592 | +5.54% | 4.842 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [7, 320] | Ascend950DT_9582 | `{}` | 5.393 | 5.393 | 5.146 | +4.79% | 4.987 | `entry_kernel_21` |
| [7, 320] | Ascend950PR_957b | `{}` | 5.004 | 5.004 | 5.042 | -0.76% | 4.987 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [7, 4480] | Ascend950DT_9582 | `{}` | 5.430 | 5.430 | 5.748 | -5.54% | 5.840 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [7, 4480] | Ascend950PR_957b | `{}` | 4.865 | 4.865 | 4.559 | +6.71% | 5.840 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [8, 384] | Ascend950DT_9582 | `{}` | 5.430 | 5.430 | 5.760 | -5.73% | 4.871 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [8, 384] | Ascend950PR_957b | `{}` | 4.877 | 4.877 | 5.136 | -5.04% | 4.871 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [9, 4096] | Ascend950DT_9582 | `{}` | 5.430 | 5.430 | 5.465 | -0.64% | 6.127 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [9, 4096] | Ascend950PR_957b | `{}` | 5.072 | 5.072 | 4.825 | +5.12% | 6.127 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [9, 5120] | Ascend950DT_9582 | `{}` | 4.980 | 4.980 | 4.867 | +2.31% | 7.084 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [9, 5120] | Ascend950PR_957b | `{}` | 4.377 | 4.377 | 5.025 | -12.90% | 7.084 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [11, 3136] | Ascend950DT_9582 | `{}` | 5.430 | 5.430 | 5.507 | -1.40% | 6.353 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [11, 3136] | Ascend950PR_957b | `{}` | 5.319 | 5.319 | 4.986 | +6.67% | 6.353 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [17, 512] | Ascend950DT_9582 | `{}` | 5.430 | 5.430 | 5.790 | -6.23% | 4.876 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [17, 512] | Ascend950PR_957b | `{}` | 4.882 | 4.882 | 5.432 | -10.12% | 4.876 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [17, 1024] | Ascend950DT_9582 | `{}` | 5.430 | 5.430 | 5.772 | -5.92% | 5.982 | `entry_kernel_21` |
| [17, 1024] | Ascend950PR_957b | `{}` | 4.966 | 4.966 | 5.311 | -6.50% | 5.982 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [19, 1024] | Ascend950DT_9582 | `{}` | 5.430 | 5.430 | 6.230 | -12.84% | 6.256 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [19, 1024] | Ascend950PR_957b | `{}` | 5.102 | 5.102 | 5.983 | -14.72% | 6.256 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [26, 704] | Ascend950DT_9582 | `{}` | 5.430 | 5.430 | 5.640 | -3.73% | 6.430 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [26, 704] | Ascend950PR_957b | `{}` | 5.636 | 5.636 | 5.452 | +3.39% | 6.430 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [27, 576] | Ascend950DT_9582 | `{}` | 5.430 | 5.430 | 6.130 | -11.42% | 6.561 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [27, 576] | Ascend950PR_957b | `{}` | 5.711 | 5.711 | 5.547 | +2.95% | 6.561 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [30, 768] | Ascend950DT_9582 | `{}` | 5.430 | 5.430 | 5.472 | -0.77% | 6.734 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [30, 768] | Ascend950PR_957b | `{}` | 4.770 | 4.770 | 4.917 | -2.98% | 6.734 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [33, 768] | Ascend950DT_9582 | `{}` | 5.430 | 5.430 | 5.210 | +4.22% | 7.042 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [33, 768] | Ascend950PR_957b | `{}` | 4.845 | 4.845 | 4.915 | -1.41% | 7.042 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [36, 512] | Ascend950DT_9582 | `{}` | 5.430 | 5.430 | 5.646 | -3.83% | 6.119 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [36, 512] | Ascend950PR_957b | `{}` | 4.906 | 4.906 | 5.795 | -15.35% | 6.119 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [36, 3072] | Ascend950DT_9582 | `{}` | 5.156 | 5.156 | 4.971 | +3.72% | 7.373 | `entry_kernel_21` |
| [36, 3072] | Ascend950PR_957b | `{}` | 4.448 | 4.448 | 5.170 | -13.95% | 7.373 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [36, 5120] | Ascend950DT_9582 | `{}` | 5.292 | 5.292 | 5.584 | -5.22% | 4.769 | `entry_kernel_21` |
| [36, 5120] | Ascend950PR_957b | `{}` | 4.547 | 4.547 | 5.052 | -10.00% | 4.769 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [53, 1984] | Ascend950DT_9582 | `{}` | 5.156 | 5.156 | 5.616 | -8.19% | 7.506 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [53, 1984] | Ascend950PR_957b | `{}` | 4.657 | 4.657 | 5.105 | -8.77% | 7.506 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [55, 1152] | Ascend950DT_9582 | `{}` | 5.180 | 5.180 | 5.900 | -12.20% | 6.542 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [55, 1152] | Ascend950PR_957b | `{}` | 4.873 | 4.873 | 5.336 | -8.67% | 6.542 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [56, 4096] | Ascend950DT_9582 | `{}` | 5.430 | 5.430 | 6.285 | -13.61% | 4.581 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [56, 4096] | Ascend950PR_957b | `{}` | 4.554 | 4.554 | 4.513 | +0.92% | 4.581 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [57, 3200] | Ascend950DT_9582 | `{}` | 5.331 | 5.331 | 4.945 | +7.80% | 4.970 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [57, 3200] | Ascend950PR_957b | `{}` | 4.776 | 4.776 | 4.430 | +7.81% | 4.970 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [59, 2880] | Ascend950DT_9582 | `{}` | 5.292 | 5.292 | 5.651 | -6.35% | 5.152 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [59, 2880] | Ascend950PR_957b | `{}` | 4.734 | 4.734 | 4.695 | +0.82% | 5.152 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [64, 1024] | Ascend950DT_9582 | `{}` | 5.214 | 5.214 | 5.532 | -5.73% | 5.870 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [64, 1024] | Ascend950PR_957b | `{}` | 4.599 | 4.599 | 5.373 | -14.40% | 5.870 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [65, 256] | Ascend950DT_9582 | `{}` | 5.430 | 5.430 | 4.898 | +10.85% | 5.846 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [65, 256] | Ascend950PR_957b | `{}` | 4.757 | 4.757 | 4.609 | +3.20% | 5.846 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [65, 2624] | Ascend950DT_9582 | `{}` | 5.194 | 5.194 | 5.306 | -2.12% | 4.942 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [65, 2624] | Ascend950PR_957b | `{}` | 4.753 | 4.753 | 4.736 | +0.36% | 4.942 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [97, 2752] | Ascend950DT_9582 | `{}` | 5.316 | 5.316 | 5.279 | +0.71% | 4.781 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [97, 2752] | Ascend950PR_957b | `{}` | 4.831 | 4.831 | 5.107 | -5.40% | 4.781 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [128, 5120] | Ascend950DT_9582 | `{}` | 5.794 | 5.794 | 5.885 | -1.54% | 4.904 | `entry_kernel_21` |
| [128, 5120] | Ascend950PR_957b | `{}` | 5.192 | 5.192 | 5.730 | -9.38% | 4.904 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [144, 5120] | Ascend950DT_9582 | `{}` | 5.886 | 5.886 | 5.905 | -0.32% | 4.728 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [144, 5120] | Ascend950PR_957b | `{}` | 5.180 | 5.180 | 5.984 | -13.42% | 4.728 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [144, 7168] | Ascend950DT_9582 | `{}` | 6.207 | 6.207 | 6.332 | -1.96% | 5.022 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [144, 7168] | Ascend950PR_957b | `{}` | 5.434 | 5.434 | 6.415 | -15.29% | 5.022 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [255, 5440] | Ascend950DT_9582 | `{}` | 6.522 | 6.522 | 5.747 | +13.49% | 5.612 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [255, 5440] | Ascend950PR_957b | `{}` | 6.694 | 6.694 | 6.868 | -2.53% | 5.612 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [257, 4416] | Ascend950DT_9582 | `{}` | 6.528 | 6.528 | 6.467 | +0.95% | 5.083 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [257, 4416] | Ascend950PR_957b | `{}` | 6.069 | 6.069 | 6.130 | -1.00% | 5.083 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [288, 7168] | Ascend950DT_9582 | `{}` | 7.181 | 7.181 | 6.876 | +4.42% | 5.526 | `entry_kernel_21` |
| [288, 7168] | Ascend950PR_957b | `{}` | 7.203 | 7.203 | 8.009 | -10.07% | 5.526 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [384, 4096] | Ascend950DT_9582 | `{}` | 6.695 | 6.695 | 6.673 | +0.33% | 5.091 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [384, 4096] | Ascend950PR_957b | `{}` | 6.397 | 6.397 | 7.332 | -12.75% | 5.091 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [448, 6592] | Ascend950DT_9582 | `{}` | 6.726 | 6.726 | 6.300 | +6.76% | 6.381 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [448, 6592] | Ascend950PR_957b | `{}` | 7.560 | 7.560 | 8.222 | -8.05% | 6.381 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [512, 256] | Ascend950DT_9582 | `{}` | 5.430 | 5.430 | 5.208 | +4.26% | 4.734 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [512, 256] | Ascend950PR_957b | `{}` | 4.667 | 4.667 | 5.090 | -8.31% | 4.734 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [512, 5120] | Ascend950DT_9582 | `{}` | 6.364 | 6.364 | 6.299 | +1.04% | 5.727 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [512, 5120] | Ascend950PR_957b | `{}` | 6.870 | 6.870 | 7.690 | -10.67% | 5.727 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [513, 5504] | Ascend950DT_9582 | `{}` | 6.845 | 6.845 | 6.751 | +1.39% | 6.009 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [513, 5504] | Ascend950PR_957b | `{}` | 6.845 | 6.845 | 8.230 | -16.83% | 6.009 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [576, 4096] | Ascend950DT_9582 | `{}` | 7.470 | 7.470 | 6.948 | +7.51% | 5.440 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [576, 4096] | Ascend950PR_957b | `{}` | 6.425 | 6.425 | 7.692 | -16.47% | 5.440 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [576, 7168] | Ascend950DT_9582 | `{}` | 7.577 | 7.577 | 7.758 | -2.33% | 6.553 | `entry_kernel_21` |
| [576, 7168] | Ascend950PR_957b | `{}` | 9.614 | 9.614 | 10.007 | -3.93% | 6.553 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [640, 4096] | Ascend950DT_9582 | `{}` | 6.364 | 6.364 | 6.218 | +2.36% | 5.727 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [640, 4096] | Ascend950PR_957b | `{}` | 6.912 | 6.912 | 8.108 | -14.75% | 5.727 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [641, 5120] | Ascend950DT_9582 | `{}` | 7.001 | 7.001 | 7.244 | -3.34% | 6.040 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [641, 5120] | Ascend950PR_957b | `{}` | 7.813 | 7.813 | 9.092 | -14.06% | 6.040 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [768, 4096] | Ascend950DT_9582 | `{}` | 6.743 | 6.743 | 7.202 | -6.37% | 5.984 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [1023, 7168] | Ascend950DT_9582 | `{}` | 9.811 | 9.811 | 9.311 | +5.37% | 6.713 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [1023, 7168] | Ascend950PR_957b | `{}` | 14.026 | 14.026 | 14.080 | -0.38% | 6.713 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [1024, 4096] | Ascend950DT_9582 | `{}` | 7.502 | 7.502 | 8.047 | -6.78% | 6.352 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [1024, 6144] | Ascend950DT_9582 | `{}` | 8.935 | 8.935 | 8.704 | +2.65% | 6.245 | `entry_kernel_21` |
| [1024, 7168] | Ascend950DT_9582 | `{}` | 9.586 | 9.586 | 9.332 | +2.73% | 6.713 | `entry_kernel_21` |
| [1152, 5120] | Ascend950DT_9582 | `{}` | 8.942 | 8.942 | 8.684 | +2.97% | 5.893 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [2048, 5120] | Ascend950DT_9582 | `{}` | 11.531 | 11.531 | 11.364 | +1.47% | 7.733 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |
| [2048, 5120] | Ascend950PR_957b | `{}` | 17.834 | 17.834 | 18.504 | -3.62% | 7.733 | `DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv` |

## sigmoid_fp32

| Shape | 目标 SKU | 输入属性 | 旧模型 | 本版 | CA | 偏差 | 970DT 外推 | 实测 kernel |
|---|---|---|---:|---:|---:|---:|---:|---|
| [1, 12] | Ascend950DT_9582 | `{}` | 2.689 | 2.689 | 2.724 | -1.25% | 2.465 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [3, 96] | Ascend950DT_9582 | `{}` | 2.722 | 2.722 | 3.035 | -10.33% | 2.495 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [3, 96] | Ascend950PR_957b | `{}` | 2.723 | 2.723 | 2.628 | +3.60% | 2.495 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [9, 96] | Ascend950DT_9582 | `{}` | 2.790 | 2.790 | 3.032 | -7.95% | 2.558 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [9, 96] | Ascend950PR_957b | `{}` | 2.793 | 2.793 | 2.658 | +5.08% | 2.558 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [72, 12] | Ascend950DT_9582 | `{}` | 2.790 | 2.790 | 3.114 | -10.39% | 2.558 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [72, 12] | Ascend950PR_957b | `{}` | 2.793 | 2.793 | 2.672 | +4.53% | 2.558 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [129, 12] | Ascend950DT_9582 | `{}` | 2.925 | 2.925 | 2.926 | -0.03% | 2.681 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [129, 12] | Ascend950PR_957b | `{}` | 2.928 | 2.928 | 2.716 | +7.80% | 2.681 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [300, 12] | Ascend950DT_9582 | `{}` | 3.042 | 3.042 | 3.085 | -1.38% | 2.789 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [2048, 12] | Ascend950DT_9582 | `{}` | 3.345 | 3.345 | 3.176 | +5.30% | 3.066 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [2048, 12] | Ascend950PR_957b | `{}` | 3.347 | 3.347 | 2.793 | +19.84% | 3.066 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [3072, 96] | Ascend950DT_9582 | `{}` | 3.938 | 3.938 | 3.621 | +8.74% | 3.477 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [3073, 12] | Ascend950DT_9582 | `{}` | 3.418 | 3.418 | 3.066 | +11.47% | 3.133 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [3073, 12] | Ascend950PR_957b | `{}` | 3.420 | 3.420 | 2.932 | +16.66% | 3.133 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [4096, 24] | Ascend950DT_9582 | `{}` | 3.571 | 3.571 | 3.425 | +4.25% | 3.280 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [4096, 24] | Ascend950PR_957b | `{}` | 3.587 | 3.587 | 3.187 | +12.54% | 3.280 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [4097, 96] | Ascend950DT_9582 | `{}` | 4.171 | 4.171 | 3.696 | +12.85% | 3.581 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [4097, 96] | Ascend950PR_957b | `{}` | 4.338 | 4.338 | 4.248 | +2.11% | 3.581 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [4545, 24] | Ascend950DT_9582 | `{}` | 3.604 | 3.604 | 3.267 | +10.30% | 3.297 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [4545, 24] | Ascend950PR_957b | `{}` | 3.607 | 3.607 | 3.178 | +13.51% | 3.297 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [16384, 48] | Ascend950DT_9582 | `{}` | 4.854 | 4.854 | 4.855 | -0.00% | 3.929 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [16384, 48] | Ascend950PR_957b | `{}` | 6.015 | 6.015 | 4.823 | +24.72% | 3.929 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [16385, 48] | Ascend950DT_9582 | `{}` | 4.910 | 4.910 | 4.769 | +2.96% | 3.929 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [16385, 48] | Ascend950PR_957b | `{}` | 6.015 | 6.015 | 5.723 | +5.11% | 3.929 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [32768, 12] | Ascend950DT_9582 | `{}` | 4.121 | 4.121 | 3.698 | +11.44% | 3.581 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [32768, 12] | Ascend950PR_957b | `{}` | 4.338 | 4.338 | 4.128 | +5.10% | 3.581 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [65537, 24] | Ascend950DT_9582 | `{}` | 5.768 | 5.768 | 5.867 | -1.70% | 4.657 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [65537, 24] | Ascend950PR_957b | `{}` | 8.114 | 8.114 | 7.585 | +6.97% | 4.657 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [98304, 12] | Ascend950DT_9582 | `{}` | 5.476 | 5.476 | 5.523 | -0.86% | 4.317 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [98304, 12] | Ascend950PR_957b | `{}` | 6.685 | 6.685 | 6.867 | -2.66% | 4.317 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [131072, 12] | Ascend950DT_9582 | `{}` | 5.709 | 5.709 | 5.987 | -4.64% | 4.657 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [131072, 12] | Ascend950PR_957b | `{}` | 8.114 | 8.114 | 7.882 | +2.94% | 4.657 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [262144, 12] | Ascend950DT_9582 | `{}` | 8.002 | 8.002 | 7.759 | +3.14% | 5.553 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [262144, 12] | Ascend950PR_957b | `{}` | 12.344 | 12.344 | 11.428 | +8.01% | 5.553 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [524288, 12] | Ascend950DT_9582 | `{}` | 11.908 | 11.908 | 11.230 | +6.04% | 7.556 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [524288, 12] | Ascend950PR_957b | `{}` | 21.815 | 21.815 | 19.424 | +12.31% | 7.556 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [1048576, 12] | Ascend950DT_9582 | `{}` | 19.720 | 19.720 | 18.553 | +6.29% | 11.321 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [1048576, 12] | Ascend950PR_957b | `{}` | 39.630 | 39.630 | 34.285 | +15.59% | 11.321 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [1572864, 12] | Ascend950DT_9582 | `{}` | 30.140 | 30.140 | 28.561 | +5.53% | 15.408 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [1572864, 12] | Ascend950PR_957b | `{}` | 69.041 | 69.041 | 54.379 | +26.96% | 15.408 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [2097152, 12] | Ascend950DT_9582 | `{}` | 45.897 | 45.897 | 42.686 | +7.52% | 19.353 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |
| [2097152, 12] | Ascend950PR_957b | `{}` | 104.717 | 104.717 | 88.593 | +18.20% | 19.353 | `Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100` |

## vllm_hc_post

| Shape | 目标 SKU | 输入属性 | 旧模型 | 本版 | CA | 偏差 | 970DT 外推 | 实测 kernel |
|---|---|---|---:|---:|---:|---:|---:|---|
| [1, 1, 4, 512] | Ascend950DT_9582 | `{"implementation":"vllm_hc_post"}` | 3.184 | 3.184 | 3.364 | -5.33% | 2.918 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 1, 4, 512] | Ascend950PR_957b | `{"implementation":"vllm_hc_post"}` | 3.196 | 3.196 | 3.154 | +1.33% | 2.918 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 1, 4, 1024] | Ascend950DT_9582 | `{"implementation":"vllm_hc_post"}` | 3.404 | 3.404 | 3.479 | -2.15% | 3.120 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 1, 4, 1024] | Ascend950PR_957b | `{"implementation":"vllm_hc_post"}` | 3.419 | 3.419 | 3.451 | -0.92% | 3.120 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 1, 4, 16384] | Ascend950DT_9582 | `{"implementation":"vllm_hc_post"}` | 8.661 | 8.661 | 8.379 | +3.36% | 7.933 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 1, 4, 16384] | Ascend950PR_957b | `{"implementation":"vllm_hc_post"}` | 8.807 | 8.807 | 8.341 | +5.59% | 7.933 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 2, 4, 1280] | Ascend950DT_9582 | `{"implementation":"vllm_hc_post"}` | 3.531 | 3.531 | 3.570 | -1.10% | 3.236 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 2, 4, 1280] | Ascend950PR_957b | `{"implementation":"vllm_hc_post"}` | 3.548 | 3.548 | 3.499 | +1.41% | 3.236 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 2, 4, 8192] | Ascend950DT_9582 | `{"implementation":"vllm_hc_post"}` | 6.058 | 6.058 | 5.795 | +4.53% | 5.550 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 2, 4, 8192] | Ascend950PR_957b | `{"implementation":"vllm_hc_post"}` | 6.131 | 6.131 | 5.878 | +4.29% | 5.550 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 2, 4, 16448] | Ascend950DT_9582 | `{"implementation":"vllm_hc_post"}` | 8.869 | 8.869 | 8.636 | +2.70% | 8.124 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 3, 4, 5120] | Ascend950DT_9582 | `{"implementation":"vllm_hc_post"}` | 5.084 | 5.084 | 5.122 | -0.73% | 4.658 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 3, 4, 5120] | Ascend950PR_957b | `{"implementation":"vllm_hc_post"}` | 5.136 | 5.136 | 5.024 | +2.22% | 4.658 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 4, 4, 4080] | Ascend950DT_9582 | `{"implementation":"vllm_hc_post"}` | 4.771 | 4.771 | 4.743 | +0.59% | 4.372 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 4, 4, 4080] | Ascend950PR_957b | `{"implementation":"vllm_hc_post"}` | 4.807 | 4.807 | 4.901 | -1.90% | 4.372 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 5, 4, 7168] | Ascend950DT_9582 | `{"implementation":"vllm_hc_post"}` | 5.779 | 5.779 | 5.729 | +0.88% | 5.295 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 5, 4, 7168] | Ascend950PR_957b | `{"implementation":"vllm_hc_post"}` | 5.845 | 5.845 | 5.798 | +0.82% | 5.295 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 8, 4, 4096] | Ascend950DT_9582 | `{"implementation":"vllm_hc_post"}` | 4.844 | 4.844 | 4.813 | +0.64% | 4.439 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 8, 4, 4096] | Ascend950PR_957b | `{"implementation":"vllm_hc_post"}` | 4.881 | 4.881 | 4.923 | -0.86% | 4.439 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 9, 4, 4160] | Ascend950DT_9582 | `{"implementation":"vllm_hc_post"}` | 5.060 | 5.060 | 4.956 | +2.09% | 4.637 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 9, 4, 4160] | Ascend950PR_957b | `{"implementation":"vllm_hc_post"}` | 5.097 | 5.097 | 4.843 | +5.23% | 4.637 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 33, 4, 7168] | Ascend950DT_9582 | `{"implementation":"vllm_hc_post"}` | 6.265 | 6.265 | 6.101 | +2.69% | 5.740 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 33, 4, 7168] | Ascend950PR_957b | `{"implementation":"vllm_hc_post"}` | 6.331 | 6.331 | 6.508 | -2.72% | 5.740 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 57, 4, 192] | Ascend950DT_9582 | `{"implementation":"vllm_hc_post"}` | 4.019 | 4.019 | 3.941 | +1.97% | 3.683 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 57, 4, 192] | Ascend950PR_957b | `{"implementation":"vllm_hc_post"}` | 4.011 | 4.011 | 4.024 | -0.33% | 3.683 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 65, 4, 256] | Ascend950DT_9582 | `{"implementation":"vllm_hc_post"}` | 4.168 | 4.168 | 4.202 | -0.83% | 3.836 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 65, 4, 256] | Ascend950PR_957b | `{"implementation":"vllm_hc_post"}` | 4.039 | 4.039 | 4.054 | -0.37% | 3.836 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 129, 4, 128] | Ascend950DT_9582 | `{"implementation":"vllm_hc_post"}` | 4.412 | 4.412 | 4.612 | -4.32% | 4.660 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
| [1, 129, 4, 128] | Ascend950PR_957b | `{"implementation":"vllm_hc_post"}` | 4.292 | 4.292 | 4.364 | -1.67% | 4.660 | `HcPost_d2fe6614722b27afa6f8d964c215447e_1` |
