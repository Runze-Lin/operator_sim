# 现版对历史全窗的逐形回放

每个 shape 的设备耗时单位为 µs。相同几何但不同前序可以出现多次；这些行作为回放。新冻结的计时独立列在总误差分析。

| 算子 · 实际 kernel | 芯片 | shape · 属性 | v0.27 | 当前 | CA | 当前偏差 | 970DT 外推 |
|---|---|---|---:|---:|---:|---:|---:|
| addmm · entry_kernel_2 | Ascend950DT_9582 | [1, 32, 16] · {'self_shape': [32], 'alpha': 1, 'beta': 1} | 3.092 | 3.092 | 3.288 | -6.0% | 2.834 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [1, 48, 80] · {'self_shape': [48], 'alpha': 1, 'beta': 1} | 3.208 | 3.208 | 3.078 | +4.2% | 2.941 |
| addmm · ['entry_kernel_2'] | Ascend950DT_9582 | [1, 64, 32] · {'self_shape': [64], 'alpha': 1, 'beta': 1} | 3.147 | 3.147 | 3.145 | +0.0% | 2.884 |
| addmm · entry_kernel_2 | Ascend950DT_9582 | [2, 16, 16] · {'self_shape': [16], 'alpha': 1, 'beta': 1} | 3.083 | 3.083 | 2.992 | +3.0% | 2.826 |
| addmm · ['entry_kernel_2'] | Ascend950DT_9582 | [2, 32, 32] · {'self_shape': [32], 'alpha': 1, 'beta': 1} | 3.110 | 3.110 | 3.408 | -8.8% | 2.851 |
| addmm · ['entry_kernel_2'] | Ascend950DT_9582 | [2, 64, 32] · {'self_shape': [64], 'alpha': 1, 'beta': 1} | 3.147 | 3.147 | 3.423 | -8.1% | 2.885 |
| addmm · ['entry_kernel_2'] | Ascend950DT_9582 | [3, 48, 96] · {'self_shape': [48], 'alpha': 1, 'beta': 1} | 3.236 | 3.236 | 3.352 | -3.5% | 2.966 |
| addmm · entry_kernel_2 | Ascend950DT_9582 | [4, 80, 48] · {'self_shape': [80], 'alpha': 1, 'beta': 1} | 3.239 | 3.239 | 3.625 | -10.7% | 2.969 |
| addmm · ['entry_kernel_2'] | Ascend950DT_9582 | [8, 96, 192] · {'self_shape': [96], 'alpha': 1, 'beta': 1} | 3.562 | 3.562 | 3.687 | -3.4% | 3.265 |
| addmm · entry_kernel_2 | Ascend950DT_9582 | [9, 64, 128] · {'self_shape': [64], 'alpha': 1, 'beta': 1} | 3.364 | 3.364 | 3.342 | +0.7% | 3.084 |
| addmm · ['entry_kernel_2'] | Ascend950DT_9582 | [9, 128, 256] · {'self_shape': [128], 'alpha': 1, 'beta': 1} | 3.705 | 3.705 | 3.756 | -1.4% | 3.396 |
| addmm · entry_kernel_2 | Ascend950DT_9582 | [16, 64, 256] · {'self_shape': [64], 'alpha': 1, 'beta': 1} | 3.653 | 3.653 | 3.447 | +6.0% | 3.348 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [16, 64, 256] · {'self_shape': [64], 'alpha': 1, 'beta': 1} | 3.653 | 3.653 | 3.588 | +1.8% | 3.348 |
| addmm · entry_kernel_2 | Ascend950DT_9582 | [16, 192, 384] · {'self_shape': [192], 'alpha': 1, 'beta': 1} | 4.042 | 4.042 | 3.877 | +4.2% | 3.705 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [17, 80, 96] · {'self_shape': [80], 'alpha': 1, 'beta': 1} | 3.407 | 3.407 | 3.557 | -4.2% | 3.123 |
| addmm · ['entry_kernel_2'] | Ascend950DT_9582 | [17, 128, 128] · {'self_shape': [128], 'alpha': 1, 'beta': 1} | 3.478 | 3.478 | 3.393 | +2.5% | 3.188 |
| addmm · ['entry_kernel_2'] | Ascend950DT_9582 | [31, 64, 257] · {'self_shape': [64], 'alpha': 1, 'beta': 1} | 3.783 | 3.783 | 4.253 | -11.0% | 3.468 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [31, 144, 160] · {'self_shape': [144], 'alpha': 1, 'beta': 1} | 3.582 | 3.582 | 4.082 | -12.2% | 3.283 |
| addmm · ['entry_kernel_2'] | Ascend950DT_9582 | [32, 80, 272] · {'self_shape': [80], 'alpha': 1, 'beta': 1} | 3.863 | 3.863 | 4.228 | -8.6% | 3.541 |
| addmm · ['entry_kernel_2'] | Ascend950DT_9582 | [48, 256, 384] · {'self_shape': [256], 'alpha': 1, 'beta': 1} | 4.152 | 4.152 | 4.027 | +3.1% | 3.806 |
| addmm · entry_kernel_2 | Ascend950DT_9582 | [64, 128, 512] · {'self_shape': [128], 'alpha': 1, 'beta': 1} | 4.350 | 4.350 | 4.519 | -3.8% | 3.987 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [96, 176, 320] · {'self_shape': [176], 'alpha': 1, 'beta': 1} | 4.069 | 4.069 | 4.580 | -11.1% | 3.730 |
| addmm · ['entry_kernel_2'] | Ascend950DT_9582 | [96, 192, 256] · {'self_shape': [192], 'alpha': 1, 'beta': 1} | 3.883 | 3.883 | 3.925 | -1.1% | 3.560 |
| addmm · entry_kernel_2 | Ascend950DT_9582 | [128, 192, 256] · {'self_shape': [192], 'alpha': 1, 'beta': 1} | 3.908 | 3.908 | 3.763 | +3.8% | 3.581 |
| addmm · ['entry_kernel_2'] | Ascend950DT_9582 | [128, 256, 384] · {'self_shape': [256], 'alpha': 1, 'beta': 1} | 4.235 | 4.235 | 4.275 | -0.9% | 3.878 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [192, 384, 192] · {'self_shape': [384], 'alpha': 1, 'beta': 1} | 3.842 | 3.842 | 3.840 | +0.1% | 3.530 |
| addmm · ['entry_kernel_2'] | Ascend950DT_9582 | [256, 384, 256] · {'self_shape': [384], 'alpha': 1, 'beta': 1} | 4.052 | 4.052 | 4.381 | -7.5% | 3.682 |
| addmm · entry_kernel_2 | Ascend950DT_9582 | [256, 512, 256] · {'self_shape': [512], 'alpha': 1, 'beta': 1} | 4.088 | 4.088 | 4.327 | -5.5% | 3.695 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [257, 256, 192] · {'self_shape': [256], 'alpha': 1, 'beta': 1} | 3.839 | 3.839 | 3.645 | +5.3% | 3.526 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [272, 480, 192] · {'self_shape': [480], 'alpha': 1, 'beta': 1} | 4.044 | 4.044 | 4.538 | -10.9% | 3.564 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [384, 480, 256] · {'self_shape': [480], 'alpha': 1, 'beta': 1} | 4.428 | 4.428 | 5.079 | -12.8% | 3.725 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [511, 384, 192] · {'self_shape': [384], 'alpha': 1, 'beta': 1} | 4.333 | 4.333 | 4.158 | +4.2% | 3.585 |
| addmm · entry_kernel_2 | Ascend950DT_9582 | [512, 512, 256] · {'self_shape': [512], 'alpha': 1, 'beta': 1} | 4.773 | 4.773 | 4.479 | +6.6% | 3.761 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [1024, 192, 256] · {'self_shape': [192], 'alpha': 1, 'beta': 1} | 4.585 | 4.585 | 4.252 | +7.9% | 3.725 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [1, 32, 16] · {'self_shape': [32], 'alpha': 1, 'beta': 1} | 3.092 | 3.092 | 3.002 | +3.0% | 2.834 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [1, 48, 80] · {'self_shape': [48], 'alpha': 1, 'beta': 1} | 3.208 | 3.208 | 2.828 | +13.5% | 2.941 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [2, 16, 16] · {'self_shape': [16], 'alpha': 1, 'beta': 1} | 3.083 | 3.083 | 2.728 | +13.0% | 2.826 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [4, 80, 48] · {'self_shape': [80], 'alpha': 1, 'beta': 1} | 3.239 | 3.239 | 3.307 | -2.0% | 2.969 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [9, 64, 128] · {'self_shape': [64], 'alpha': 1, 'beta': 1} | 3.364 | 3.364 | 3.059 | +10.0% | 3.084 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [16, 64, 256] · {'self_shape': [64], 'alpha': 1, 'beta': 1} | 3.653 | 3.653 | 3.215 | +13.6% | 3.348 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [16, 192, 384] · {'self_shape': [192], 'alpha': 1, 'beta': 1} | 4.042 | 4.042 | 3.741 | +8.0% | 3.705 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [17, 80, 96] · {'self_shape': [80], 'alpha': 1, 'beta': 1} | 3.407 | 3.407 | 3.410 | -0.1% | 3.123 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [31, 144, 160] · {'self_shape': [144], 'alpha': 1, 'beta': 1} | 3.582 | 3.582 | 3.818 | -6.2% | 3.283 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [64, 128, 512] · {'self_shape': [128], 'alpha': 1, 'beta': 1} | 4.350 | 4.350 | 4.293 | +1.3% | 3.987 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [96, 176, 320] · {'self_shape': [176], 'alpha': 1, 'beta': 1} | 4.069 | 4.069 | 4.499 | -9.6% | 3.730 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [128, 192, 256] · {'self_shape': [192], 'alpha': 1, 'beta': 1} | 3.909 | 3.909 | 3.753 | +4.2% | 3.581 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [192, 384, 192] · {'self_shape': [384], 'alpha': 1, 'beta': 1} | 3.846 | 3.846 | 3.794 | +1.4% | 3.530 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [256, 512, 256] · {'self_shape': [512], 'alpha': 1, 'beta': 1} | 4.392 | 4.392 | 4.230 | +3.8% | 3.695 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [257, 256, 192] · {'self_shape': [256], 'alpha': 1, 'beta': 1} | 3.843 | 3.843 | 3.750 | +2.5% | 3.526 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [272, 480, 192] · {'self_shape': [480], 'alpha': 1, 'beta': 1} | 4.175 | 4.175 | 4.410 | -5.3% | 3.564 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [384, 480, 256] · {'self_shape': [480], 'alpha': 1, 'beta': 1} | 4.731 | 4.731 | 5.075 | -6.8% | 3.725 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [511, 384, 192] · {'self_shape': [384], 'alpha': 1, 'beta': 1} | 4.448 | 4.448 | 4.253 | +4.6% | 3.585 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [512, 512, 256] · {'self_shape': [512], 'alpha': 1, 'beta': 1} | 4.856 | 4.856 | 4.687 | +3.6% | 3.761 |
| addmm · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [1024, 192, 256] · {'self_shape': [192], 'alpha': 1, 'beta': 1} | 4.736 | 4.736 | 4.449 | +6.4% | 3.725 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950DT_9582 | [1, 256, 8, 1] · {'addend': True, 'materialize': True, 'write_idx': 1} | 6.668 | 6.668 | 6.325 | +5.4% | 6.112 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950DT_9582 | [1, 256, 8, 2] · {'addend': True, 'materialize': True, 'write_idx': 2} | 7.250 | 7.250 | 7.004 | +3.5% | 6.645 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950DT_9582 | [1, 1024, 8, 7] · {'addend': True, 'materialize': True, 'write_idx': 7} | 11.009 | 11.009 | 11.573 | -4.9% | 10.091 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950DT_9582 | [1, 7168, 8, 1] · {'addend': True, 'materialize': True, 'write_idx': 1} | 10.152 | 10.152 | 9.505 | +6.8% | 9.303 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950DT_9582 | [1, 7168, 8, 3] · {'addend': True, 'materialize': True, 'write_idx': 3} | 12.701 | 12.701 | 11.572 | +9.8% | 11.639 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950DT_9582 | [2, 512, 8, 1] · {'addend': True, 'materialize': True, 'write_idx': 1} | 6.797 | 6.797 | 7.122 | -4.6% | 6.230 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950DT_9582 | [2, 1024, 8, 4] · {'addend': True, 'materialize': True, 'write_idx': 4} | 9.032 | 9.032 | 9.178 | -1.6% | 8.278 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950DT_9582 | [2, 2048, 8, 3] · {'addend': True, 'materialize': True, 'write_idx': 3} | 9.094 | 9.094 | 9.490 | -4.2% | 8.335 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950DT_9582 | [2, 4096, 8, 1] · {'addend': True, 'materialize': True, 'write_idx': 1} | 8.603 | 8.603 | 8.113 | +6.0% | 7.885 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950DT_9582 | [2, 4096, 8, 2] · {'addend': True, 'materialize': True, 'write_idx': 2} | 9.570 | 9.570 | 9.398 | +1.8% | 8.771 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950DT_9582 | [2, 7168, 8, 3] · {'addend': True, 'materialize': True, 'write_idx': 3} | 12.701 | 12.701 | 11.745 | +8.1% | 11.639 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950DT_9582 | [3, 512, 8, 1] · {'addend': True, 'materialize': True, 'write_idx': 1} | 6.797 | 6.797 | 7.010 | -3.0% | 6.230 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950DT_9582 | [3, 512, 8, 3] · {'addend': True, 'materialize': True, 'write_idx': 3} | 8.012 | 8.012 | 8.479 | -5.5% | 7.344 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950DT_9582 | [3, 2048, 8, 4] · {'addend': True, 'materialize': True, 'write_idx': 4} | 9.856 | 9.856 | 9.824 | +0.3% | 9.033 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950DT_9582 | [3, 7168, 8, 1] · {'addend': True, 'materialize': True, 'write_idx': 1} | 10.152 | 10.152 | 9.712 | +4.5% | 9.303 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950DT_9582 | [4, 256, 8, 3] · {'addend': True, 'materialize': True, 'write_idx': 3} | 7.832 | 7.832 | 7.585 | +3.2% | 7.179 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950DT_9582 | [4, 256, 8, 6] · {'addend': True, 'materialize': True, 'write_idx': 6} | 9.578 | 9.578 | 9.449 | +1.4% | 8.780 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950DT_9582 | [4, 1024, 8, 2] · {'addend': True, 'materialize': True, 'write_idx': 2} | 7.714 | 7.714 | 7.852 | -1.8% | 7.070 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950DT_9582 | [5, 512, 8, 4] · {'addend': True, 'materialize': True, 'write_idx': 4} | 8.620 | 8.620 | 9.237 | -6.7% | 7.901 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950DT_9582 | [5, 2048, 8, 1] · {'addend': True, 'materialize': True, 'write_idx': 1} | 7.571 | 7.571 | 7.574 | -0.0% | 6.939 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950DT_9582 | [8, 512, 8, 2] · {'addend': True, 'materialize': True, 'write_idx': 2} | 7.404 | 7.404 | 7.652 | -3.2% | 6.787 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950PR_957b | [1, 256, 8, 2] · {'addend': True, 'materialize': True, 'write_idx': 2} | 7.253 | 7.253 | 6.583 | +10.2% | 6.645 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950PR_957b | [1, 512, 8, 2] · {'addend': True, 'materialize': True, 'write_idx': 2} | 7.410 | 7.410 | 7.039 | +5.3% | 6.787 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950PR_957b | [1, 1024, 8, 7] · {'addend': True, 'materialize': True, 'write_idx': 7} | 11.031 | 11.031 | 10.764 | +2.5% | 10.091 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950PR_957b | [1, 7168, 8, 1] · {'addend': True, 'materialize': True, 'write_idx': 1} | 10.221 | 10.221 | 9.499 | +7.6% | 9.303 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950PR_957b | [1, 7168, 8, 3] · {'addend': True, 'materialize': True, 'write_idx': 3} | 12.799 | 12.799 | 11.488 | +11.4% | 11.639 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950PR_957b | [2, 512, 8, 1] · {'addend': True, 'materialize': True, 'write_idx': 1} | 6.802 | 6.802 | 6.598 | +3.1% | 6.230 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950PR_957b | [2, 1024, 8, 4] · {'addend': True, 'materialize': True, 'write_idx': 4} | 9.048 | 9.048 | 8.973 | +0.8% | 8.278 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950PR_957b | [2, 2048, 8, 3] · {'addend': True, 'materialize': True, 'write_idx': 3} | 9.122 | 9.122 | 9.085 | +0.4% | 8.335 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950PR_957b | [2, 4096, 8, 1] · {'addend': True, 'materialize': True, 'write_idx': 1} | 8.643 | 8.643 | 7.984 | +8.2% | 7.885 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950PR_957b | [2, 4096, 8, 2] · {'addend': True, 'materialize': True, 'write_idx': 2} | 9.618 | 9.618 | 9.007 | +6.8% | 8.771 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950PR_957b | [2, 7168, 8, 3] · {'addend': True, 'materialize': True, 'write_idx': 3} | 12.799 | 12.799 | 11.686 | +9.5% | 11.639 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950PR_957b | [3, 512, 8, 1] · {'addend': True, 'materialize': True, 'write_idx': 1} | 6.802 | 6.802 | 6.697 | +1.6% | 6.230 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950PR_957b | [3, 512, 8, 3] · {'addend': True, 'materialize': True, 'write_idx': 3} | 8.019 | 8.019 | 7.776 | +3.1% | 7.344 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950PR_957b | [3, 2048, 8, 4] · {'addend': True, 'materialize': True, 'write_idx': 4} | 9.888 | 9.888 | 9.579 | +3.2% | 9.033 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950PR_957b | [3, 7168, 8, 1] · {'addend': True, 'materialize': True, 'write_idx': 1} | 10.221 | 10.221 | 9.474 | +7.9% | 9.303 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950PR_957b | [4, 256, 8, 3] · {'addend': True, 'materialize': True, 'write_idx': 3} | 7.835 | 7.835 | 7.368 | +6.3% | 7.179 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950PR_957b | [4, 256, 8, 6] · {'addend': True, 'materialize': True, 'write_idx': 6} | 9.583 | 9.583 | 9.181 | +4.4% | 8.780 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950PR_957b | [4, 1024, 8, 2] · {'addend': True, 'materialize': True, 'write_idx': 2} | 7.726 | 7.726 | 7.494 | +3.1% | 7.070 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950PR_957b | [5, 512, 8, 4] · {'addend': True, 'materialize': True, 'write_idx': 4} | 8.628 | 8.628 | 8.473 | +1.8% | 7.901 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950PR_957b | [5, 2048, 8, 1] · {'addend': True, 'materialize': True, 'write_idx': 1} | 7.591 | 7.591 | 7.433 | +2.1% | 6.939 |
| attn_res_fwd · AttnResFwd_9dfad72fc51beb0819dd59ab91fb1abf_40020 | Ascend950PR_957b | [8, 512, 8, 2] · {'addend': True, 'materialize': True, 'write_idx': 2} | 7.410 | 7.410 | 7.223 | +2.6% | 6.787 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [1, 64] · {} | 1.957 | 1.957 | 2.172 | -9.9% | 1.794 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [1, 3584] · {} | 2.099 | 2.099 | 2.216 | -5.3% | 1.924 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [3, 7168] · {} | 2.203 | 2.203 | 2.379 | -7.4% | 2.019 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [9, 4096] · {} | 2.278 | 2.278 | 2.325 | -2.0% | 2.088 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [9, 5120] · {} | 2.326 | 2.326 | 2.412 | -3.6% | 2.132 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [16, 7168] · {} | 2.657 | 2.657 | 2.722 | -2.4% | 2.435 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [17, 32] · {} | 1.995 | 1.995 | 2.406 | -17.1% | 1.828 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [32, 4096] · {} | 2.736 | 2.736 | 2.771 | -1.3% | 2.508 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [48, 7168] · {} | 2.991 | 2.991 | 2.892 | +3.4% | 3.078 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [64, 4096] · {} | 2.893 | 2.893 | 2.893 | -0.0% | 3.030 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [65, 5120] · {} | 2.978 | 2.978 | 3.050 | -2.4% | 3.071 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [72, 4096] · {} | 2.932 | 2.932 | 2.851 | +2.8% | 3.049 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [96, 7168] · {} | 3.401 | 3.401 | 3.096 | +9.9% | 3.278 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [111, 7168] · {} | 3.531 | 3.531 | 3.172 | +11.3% | 3.341 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [128, 5120] · {} | 3.362 | 3.362 | 3.064 | +9.7% | 3.260 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [144, 6144] · {} | 3.636 | 3.636 | 3.522 | +3.2% | 3.393 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [255, 7168] · {} | 4.576 | 4.576 | 4.784 | -4.3% | 3.943 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [479, 5120] · {} | 5.078 | 5.078 | 4.902 | +3.6% | 4.308 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [511, 5120] · {} | 5.210 | 5.210 | 5.380 | -3.2% | 4.387 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [3072, 5120] · {} | 15.749 | 15.749 | 16.367 | -3.8% | 9.538 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [4096, 7168] · {} | 28.424 | 28.424 | 29.724 | -4.4% | 14.894 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [8192, 5120] · {} | 52.256 | 52.256 | 55.286 | -5.5% | 19.838 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [32769, 32] · {} | 3.834 | 3.834 | 3.676 | +4.3% | 3.489 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [98305, 64] · {} | 8.165 | 8.165 | 8.234 | -0.8% | 5.831 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [131073, 64] · {} | 9.850 | 9.850 | 10.518 | -6.3% | 6.656 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [262144, 32] · {} | 9.850 | 9.850 | 9.970 | -1.2% | 6.656 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [262144, 64] · {} | 16.592 | 16.592 | 17.041 | -2.6% | 9.951 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [393216, 32] · {} | 13.221 | 13.221 | 13.728 | -3.7% | 8.303 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [1048576, 32] · {} | 36.368 | 36.368 | 35.535 | +2.3% | 16.542 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [1048576, 64] · {} | 99.918 | 99.918 | 105.374 | -5.2% | 42.536 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [1572864, 32] · {} | 68.143 | 68.143 | 72.238 | -5.7% | 27.001 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950DT_9582 | [2097152, 32] · {} | 99.918 | 99.918 | 106.757 | -6.4% | 42.536 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [1, 64] · {} | 1.957 | 1.957 | 1.996 | -1.9% | 1.794 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [1, 3584] · {} | 2.100 | 2.100 | 2.104 | -0.2% | 1.924 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [3, 7168] · {} | 2.204 | 2.204 | 2.441 | -9.7% | 2.019 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [9, 4096] · {} | 2.279 | 2.279 | 2.220 | +2.7% | 2.088 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [9, 5120] · {} | 2.327 | 2.327 | 2.613 | -11.0% | 2.132 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [16, 7168] · {} | 2.658 | 2.658 | 2.830 | -6.1% | 2.435 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [17, 32] · {} | 1.995 | 1.995 | 1.953 | +2.2% | 1.828 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [32, 4096] · {} | 2.681 | 2.681 | 2.867 | -6.5% | 2.508 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [48, 7168] · {} | 2.972 | 2.972 | 2.914 | +2.0% | 3.078 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [65, 5120] · {} | 2.958 | 2.958 | 2.936 | +0.7% | 3.071 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [72, 4096] · {} | 2.907 | 2.907 | 2.975 | -2.3% | 3.049 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [96, 7168] · {} | 3.445 | 3.445 | 2.875 | +19.8% | 3.278 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [111, 7168] · {} | 3.592 | 3.592 | 3.744 | -4.1% | 3.341 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [128, 5120] · {} | 3.400 | 3.400 | 3.562 | -4.5% | 3.260 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [144, 6144] · {} | 3.715 | 3.715 | 4.111 | -9.6% | 3.393 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [255, 7168] · {} | 4.719 | 4.719 | 5.884 | -19.8% | 3.943 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [479, 5120] · {} | 5.296 | 5.296 | 6.532 | -18.9% | 4.308 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [511, 5120] · {} | 5.446 | 5.446 | 6.576 | -17.2% | 4.387 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [3072, 5120] · {} | 23.532 | 23.532 | 23.237 | +1.3% | 9.538 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [4096, 7168] · {} | 70.967 | 57.128 | 54.612 | +4.6% | 14.894 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [8192, 5120] · {} | 124.894 | 111.054 | 114.128 | -2.7% | 19.838 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [32769, 32] · {} | 3.941 | 3.941 | 4.479 | -12.0% | 3.489 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [98305, 64] · {} | 10.050 | 10.050 | 11.218 | -10.4% | 5.831 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [131073, 64] · {} | 13.046 | 13.046 | 14.119 | -7.6% | 6.656 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [262144, 32] · {} | 13.046 | 13.046 | 14.080 | -7.3% | 6.656 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [262144, 64] · {} | 25.030 | 25.030 | 23.762 | +5.3% | 9.951 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [393216, 32] · {} | 19.038 | 19.038 | 18.826 | +1.1% | 8.303 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [1048576, 32] · {} | 88.943 | 75.103 | 70.711 | +6.2% | 16.542 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [1048576, 64] · {} | 232.748 | 218.908 | 224.719 | -2.6% | 42.536 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [1572864, 32] · {} | 160.845 | 147.006 | 149.959 | -2.0% | 27.001 |
| cast_bf16_fp32 · Cast_b2717900ea7de2d41ea9c63d23aa269b_high_performance_0 | Ascend950PR_957b | [2097152, 32] · {} | 232.748 | 218.908 | 223.493 | -2.1% | 42.536 |
| fa · entry_kernel_2050000040100220 | Ascend950DT_9582 | [1, 64, 64, 1, 1] · {} | 8.616 | 8.616 | 8.937 | -3.6% | 7.898 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950DT_9582 | [1, 128, 64, 1, 1] · {} | 8.687 | 8.687 | 8.784 | -1.1% | 7.963 |
| fa · entry_kernel_2050000040100220 | Ascend950DT_9582 | [1, 224, 64, 1, 1] · {} | 8.946 | 8.946 | 8.826 | +1.4% | 8.200 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950DT_9582 | [3, 192, 64, 1, 2] · {} | 8.955 | 8.955 | 9.167 | -2.3% | 8.214 |
| fa · entry_kernel_2050000040100220 | Ascend950DT_9582 | [4, 160, 64, 1, 1] · {} | 8.956 | 8.956 | 9.461 | -5.3% | 8.220 |
| fa · entry_kernel_2050000070100220 | Ascend950DT_9582 | [4, 256, 128, 1, 1] · {} | 9.352 | 9.352 | 9.367 | -0.2% | 8.594 |
| fa · entry_kernel_2050000070100220 | Ascend950DT_9582 | [8, 128, 128, 1, 1] · {} | 8.889 | 8.889 | 9.136 | -2.7% | 8.190 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950DT_9582 | [8, 192, 64, 1, 1] · {} | 8.973 | 8.973 | 8.805 | +1.9% | 8.247 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic | Ascend950DT_9582 | [9, 256, 128, 4, 1] · {} | 9.399 | 9.399 | 9.690 | -3.0% | 8.658 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950DT_9582 | [9, 384, 64, 1, 1] · {} | 9.239 | 9.239 | 9.455 | -2.3% | 8.490 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950DT_9582 | [9, 1008, 64, 1, 1] · {} | 10.532 | 10.532 | 11.415 | -7.7% | 9.676 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950DT_9582 | [9, 1023, 64, 1, 1] · {} | 10.532 | 10.532 | 11.287 | -6.7% | 9.676 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic | Ascend950DT_9582 | [9, 3072, 64, 1, 1] · {} | 16.380 | 16.380 | 16.010 | +2.3% | 15.036 |
| fa · entry_kernel_2050000040140220 | Ascend950DT_9582 | [9, 4096, 64, 1, 1] · {} | 18.448 | 18.448 | 18.226 | +1.2% | 16.932 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic | Ascend950DT_9582 | [9, 8191, 64, 1, 1] · {} | 26.724 | 26.724 | 26.269 | +1.7% | 24.518 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic | Ascend950DT_9582 | [9, 16384, 64, 1, 1] · {} | 43.275 | 43.275 | 42.932 | +0.8% | 39.690 |
| fa · entry_kernel_2050000040140220 | Ascend950DT_9582 | [9, 16385, 64, 1, 1] · {} | 43.533 | 43.533 | 43.219 | +0.7% | 39.927 |
| fa · entry_kernel_2050000040140220 | Ascend950DT_9582 | [9, 32768, 64, 1, 1] · {} | 76.377 | 76.377 | 75.980 | +0.5% | 70.033 |
| fa · entry_kernel_2050000040140220 | Ascend950DT_9582 | [9, 131072, 64, 1, 1] · {} | 274.987 | 274.987 | 274.821 | +0.1% | 252.093 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950DT_9582 | [16, 64, 64, 1, 1] · {} | 8.678 | 8.678 | 8.856 | -2.0% | 7.997 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950DT_9582 | [16, 64, 64, 1, 1] · {} | 8.678 | 8.678 | 9.052 | -4.1% | 7.997 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950DT_9582 | [16, 128, 64, 1, 4] · {} | 8.749 | 8.749 | 8.758 | -0.1% | 8.063 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950DT_9582 | [16, 128, 64, 1, 8] · {} | 8.749 | 8.749 | 9.402 | -6.9% | 8.063 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic | Ascend950DT_9582 | [16, 128, 128, 1, 1] · {} | 8.955 | 8.955 | 8.715 | +2.8% | 8.293 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic | Ascend950DT_9582 | [16, 1025, 64, 1, 1] · {} | 12.528 | 12.528 | 12.384 | +1.2% | 12.046 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950DT_9582 | [17, 96, 64, 1, 1] · {} | 8.735 | 8.735 | 9.162 | -4.7% | 8.050 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic | Ascend950DT_9582 | [17, 192, 128, 1, 1] · {} | 9.266 | 9.266 | 9.050 | +2.4% | 8.670 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic | Ascend950DT_9582 | [17, 257, 128, 1, 1] · {} | 9.762 | 9.762 | 9.551 | +2.2% | 9.057 |
| fa · entry_kernel_2050000040100220 | Ascend950DT_9582 | [17, 384, 64, 1, 2] · {} | 9.293 | 9.293 | 9.757 | -4.8% | 8.865 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic | Ascend950DT_9582 | [17, 512, 128, 1, 1] · {} | 10.496 | 10.496 | 10.595 | -0.9% | 9.777 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950DT_9582 | [17, 895, 64, 1, 2] · {} | 10.328 | 10.328 | 11.455 | -9.8% | 10.417 |
| fa · entry_kernel_2050000040140220 | Ascend950DT_9582 | [17, 1153, 64, 1, 1] · {} | 12.833 | 12.833 | 12.024 | +6.7% | 12.447 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic | Ascend950DT_9582 | [17, 1153, 64, 1, 1] · {} | 12.833 | 12.833 | 12.150 | +5.6% | 12.447 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic | Ascend950DT_9582 | [17, 1153, 64, 1, 1] · {} | 12.833 | 12.833 | 12.487 | +2.8% | 12.447 |
| fa · entry_kernel_2050000040140220 | Ascend950DT_9582 | [17, 1153, 64, 1, 1] · {} | 12.833 | 12.833 | 12.553 | +2.2% | 12.447 |
| fa · entry_kernel_2050000040140220 | Ascend950DT_9582 | [17, 1153, 64, 1, 1] · {} | 12.833 | 12.833 | 12.556 | +2.2% | 12.447 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950DT_9582 | [32, 129, 64, 1, 4] · {} | 9.097 | 9.097 | 9.150 | -0.6% | 8.423 |
| fa · entry_kernel_2050000040100220 | Ascend950DT_9582 | [32, 512, 64, 1, 1] · {} | 10.037 | 10.037 | 9.722 | +3.2% | 10.382 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic | Ascend950DT_9582 | [33, 319, 128, 1, 1] · {} | 10.057 | 10.057 | 9.610 | +4.6% | 10.202 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950DT_9582 | [64, 256, 64, 1, 1] · {} | 9.821 | 9.821 | 9.409 | +4.4% | 10.464 |
| fa · entry_kernel_2050000070100220 | Ascend950DT_9582 | [64, 320, 128, 1, 1] · {} | 11.208 | 11.208 | 9.968 | +12.4% | 12.691 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950DT_9582 | [128, 128, 64, 1, 1] · {} | 9.022 | 9.022 | 9.498 | -5.0% | 9.001 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950DT_9582 | [129, 128, 64, 1, 1] · {} | 9.022 | 9.022 | 8.871 | +1.7% | 9.001 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950PR_957b | [1, 64, 64, 1, 1] · {} | 8.616 | 8.616 | 8.535 | +0.9% | 7.898 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950PR_957b | [1, 128, 64, 1, 1] · {} | 8.687 | 8.687 | 8.190 | +6.1% | 7.963 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950PR_957b | [1, 224, 64, 1, 1] · {} | 8.946 | 8.946 | 8.634 | +3.6% | 8.200 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950PR_957b | [3, 192, 64, 1, 2] · {} | 8.955 | 8.955 | 8.766 | +2.2% | 8.214 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950PR_957b | [4, 160, 64, 1, 1] · {} | 8.956 | 8.956 | 8.657 | +3.5% | 8.220 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic | Ascend950PR_957b | [4, 256, 128, 1, 1] · {} | 9.353 | 9.353 | 8.520 | +9.8% | 8.594 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic | Ascend950PR_957b | [8, 128, 128, 1, 1] · {} | 8.890 | 8.890 | 8.317 | +6.9% | 8.190 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950PR_957b | [8, 192, 64, 1, 1] · {} | 8.974 | 8.974 | 8.552 | +4.9% | 8.247 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950PR_957b | [9, 256, 64, 4, 2] · {} | 8.981 | 8.981 | 9.164 | -2.0% | 8.253 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic | Ascend950PR_957b | [9, 256, 128, 4, 1] · {} | 9.401 | 9.401 | 9.058 | +3.8% | 8.658 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950PR_957b | [9, 384, 64, 1, 1] · {} | 9.240 | 9.240 | 8.834 | +4.6% | 8.490 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic | Ascend950PR_957b | [9, 384, 128, 2, 2] · {} | 9.897 | 9.897 | 9.421 | +5.1% | 9.113 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950PR_957b | [9, 1008, 64, 1, 1] · {} | 10.533 | 10.533 | 10.915 | -3.5% | 9.676 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950PR_957b | [9, 1023, 64, 1, 1] · {} | 10.533 | 10.533 | 10.985 | -4.1% | 9.676 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic | Ascend950PR_957b | [9, 2048, 64, 1, 1] · {} | 14.312 | 14.312 | 13.273 | +7.8% | 13.139 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic | Ascend950PR_957b | [9, 2049, 64, 1, 1] · {} | 14.570 | 14.570 | 13.146 | +10.8% | 13.376 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic | Ascend950PR_957b | [9, 3072, 64, 1, 1] · {} | 16.380 | 16.380 | 15.156 | +8.1% | 15.036 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic | Ascend950PR_957b | [9, 4096, 64, 1, 1] · {} | 18.449 | 18.449 | 16.880 | +9.3% | 16.932 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic | Ascend950PR_957b | [9, 8191, 64, 1, 1] · {} | 26.725 | 26.725 | 25.197 | +6.1% | 24.518 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic | Ascend950PR_957b | [9, 8192, 64, 1, 1] · {} | 26.725 | 26.725 | 25.149 | +6.3% | 24.518 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic | Ascend950PR_957b | [9, 12289, 64, 1, 1] · {} | 35.259 | 35.259 | 33.039 | +6.7% | 32.341 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic | Ascend950PR_957b | [9, 16384, 64, 1, 1] · {} | 43.276 | 43.276 | 41.556 | +4.1% | 39.690 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic | Ascend950PR_957b | [9, 16385, 64, 1, 1] · {} | 43.534 | 43.534 | 41.293 | +5.4% | 39.927 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic | Ascend950PR_957b | [9, 32767, 64, 1, 1] · {} | 76.377 | 76.377 | 73.035 | +4.6% | 70.033 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic | Ascend950PR_957b | [9, 32768, 64, 1, 1] · {} | 76.377 | 76.377 | 72.856 | +4.8% | 70.033 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950PR_957b | [16, 64, 64, 1, 1] · {} | 8.679 | 8.679 | 8.472 | +2.4% | 7.997 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950PR_957b | [16, 128, 64, 1, 4] · {} | 8.751 | 8.751 | 8.539 | +2.5% | 8.063 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950PR_957b | [16, 128, 64, 1, 8] · {} | 8.751 | 8.751 | 8.731 | +0.2% | 8.063 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic | Ascend950PR_957b | [16, 128, 128, 1, 1] · {} | 8.957 | 8.957 | 8.224 | +8.9% | 8.293 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic | Ascend950PR_957b | [16, 1025, 64, 1, 1] · {} | 12.529 | 12.529 | 11.365 | +10.2% | 12.046 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950PR_957b | [17, 96, 64, 1, 1] · {} | 8.737 | 8.737 | 8.736 | +0.0% | 8.050 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic | Ascend950PR_957b | [17, 192, 128, 1, 1] · {} | 9.269 | 9.269 | 8.833 | +4.9% | 8.670 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic | Ascend950PR_957b | [17, 257, 128, 1, 1] · {} | 9.765 | 9.765 | 9.144 | +6.8% | 9.057 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950PR_957b | [17, 384, 64, 1, 2] · {} | 9.295 | 9.295 | 9.347 | -0.6% | 8.865 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic | Ascend950PR_957b | [17, 512, 128, 1, 1] · {} | 10.498 | 10.498 | 10.058 | +4.4% | 9.777 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950PR_957b | [17, 895, 64, 1, 2] · {} | 10.329 | 10.329 | 11.462 | -9.9% | 10.417 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic | Ascend950PR_957b | [17, 1153, 64, 1, 1] · {} | 12.834 | 12.834 | 11.489 | +11.7% | 12.447 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950PR_957b | [32, 129, 64, 1, 4] · {} | 9.099 | 9.099 | 8.850 | +2.8% | 8.423 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950PR_957b | [32, 512, 64, 1, 1] · {} | 10.072 | 10.072 | 9.345 | +7.8% | 10.382 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic | Ascend950PR_957b | [33, 319, 128, 1, 1] · {} | 10.086 | 10.086 | 9.252 | +9.0% | 10.202 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950PR_957b | [64, 256, 64, 1, 1] · {} | 9.865 | 9.865 | 8.935 | +10.4% | 10.464 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361009230643744_mix_aic | Ascend950PR_957b | [64, 320, 128, 1, 1] · {} | 11.280 | 11.280 | 9.582 | +17.7% | 12.691 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950PR_957b | [128, 128, 64, 1, 1] · {} | 9.043 | 9.043 | 8.793 | +2.8% | 9.001 |
| fa · FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425337376_mix_aic | Ascend950PR_957b | [129, 128, 64, 1, 1] · {} | 9.043 | 9.043 | 8.508 | +6.3% | 9.001 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950DT_9582 | [1, 2, 1, 1, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 21.746 | 21.746 | 23.241 | -6.4% | 19.932 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950DT_9582 | [1, 3, 1, 1, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 21.839 | 21.839 | 23.422 | -6.8% | 20.017 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950DT_9582 | [1, 5, 2, 2, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 22.024 | 22.024 | 22.822 | -3.5% | 20.187 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950DT_9582 | [1, 16, 1, 1, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 23.049 | 23.049 | 24.125 | -4.5% | 21.126 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950DT_9582 | [1, 17, 1, 1, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 23.395 | 23.395 | 24.547 | -4.7% | 21.443 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950DT_9582 | [1, 31, 2, 2, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 24.709 | 24.709 | 25.276 | -2.2% | 22.647 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950DT_9582 | [1, 32, 1, 1, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 24.803 | 24.803 | 24.936 | -0.5% | 22.733 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950DT_9582 | [1, 33, 2, 2, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 25.709 | 25.709 | 25.521 | +0.7% | 23.563 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950DT_9582 | [1, 63, 4, 4, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 29.045 | 29.045 | 29.621 | -1.9% | 26.619 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950DT_9582 | [1, 64, 2, 2, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 29.141 | 29.141 | 28.839 | +1.0% | 26.707 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950DT_9582 | [1, 64, 3, 3, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 29.141 | 29.141 | 28.744 | +1.4% | 26.707 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950DT_9582 | [1, 65, 1, 1, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 31.611 | 31.611 | 30.878 | +2.4% | 28.971 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950DT_9582 | [1, 65, 2, 2, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 31.611 | 31.611 | 30.795 | +2.7% | 28.971 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950DT_9582 | [1, 65, 4, 4, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 31.611 | 31.611 | 31.675 | -0.2% | 28.971 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950DT_9582 | [1, 80, 1, 1, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 31.751 | 31.751 | 30.882 | +2.8% | 29.099 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950DT_9582 | [1, 127, 1, 1, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 32.464 | 32.464 | 31.728 | +2.3% | 29.752 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950DT_9582 | [1, 129, 3, 3, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 34.944 | 34.944 | 34.128 | +2.4% | 32.025 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950DT_9582 | [1, 145, 2, 2, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 35.185 | 35.185 | 34.586 | +1.7% | 32.246 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950PR_957b | [1, 2, 1, 1, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 21.774 | 21.774 | 22.213 | -2.0% | 19.932 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950PR_957b | [1, 3, 1, 1, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 21.868 | 21.868 | 22.270 | -1.8% | 20.017 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950PR_957b | [1, 5, 2, 2, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 22.056 | 22.056 | 22.397 | -1.5% | 20.187 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950PR_957b | [1, 16, 1, 1, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 23.093 | 23.093 | 23.421 | -1.4% | 21.126 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950PR_957b | [1, 17, 1, 1, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 23.443 | 23.443 | 23.802 | -1.5% | 21.443 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950PR_957b | [1, 31, 2, 2, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 24.773 | 24.773 | 24.796 | -0.1% | 22.647 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950PR_957b | [1, 32, 1, 1, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 24.869 | 24.869 | 24.924 | -0.2% | 22.733 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950PR_957b | [1, 33, 2, 2, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 25.781 | 25.781 | 25.819 | -0.1% | 23.563 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950PR_957b | [1, 63, 4, 4, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 29.159 | 29.159 | 29.487 | -1.1% | 26.619 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950PR_957b | [1, 64, 2, 2, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 29.257 | 29.257 | 28.999 | +0.9% | 26.707 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950PR_957b | [1, 64, 3, 3, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 29.257 | 29.257 | 28.820 | +1.5% | 26.707 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950PR_957b | [1, 65, 1, 1, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 31.731 | 31.731 | 30.764 | +3.1% | 28.971 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950PR_957b | [1, 65, 2, 2, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 31.731 | 31.731 | 31.164 | +1.8% | 28.971 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950PR_957b | [1, 65, 4, 4, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 31.731 | 31.731 | 31.970 | -0.7% | 28.971 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950PR_957b | [1, 80, 1, 1, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 31.872 | 31.872 | 31.285 | +1.9% | 29.099 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950PR_957b | [1, 129, 3, 3, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 35.072 | 35.072 | 34.527 | +1.6% | 32.025 |
| fla_kda_v2 · ['ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_32513546_mix_aic', 'ChunkFwdH_72371232203476155baf45810c33a7f9_7710_mix_aic', 'ChunkKdaFwdFinalize_c0d447f32579acd296e1c16d6d9c7d38_0'] | Ascend950PR_957b | [1, 145, 2, 2, 128, 128] · {'implementation': 'fla_26_9_1_v2'} | 35.313 | 35.313 | 35.012 | +0.9% | 32.246 |
| gmm2 · entry_kernel_0 | Ascend950DT_9582 | [1, 64, 64, 1] · {'expert_counts': [1]} | 3.410 | 3.410 | 3.151 | +8.2% | 3.126 |
| gmm2 · GroupedMatmul_037cec40685a17f15b75759e999767c3_0 | Ascend950DT_9582 | [2, 64, 64, 1] · {'expert_counts': [2]} | 3.410 | 3.410 | 3.090 | +10.3% | 3.126 |
| gmm2 · entry_kernel_0 | Ascend950DT_9582 | [2, 64, 64, 1] · {'expert_counts': [2]} | 3.410 | 3.410 | 3.248 | +5.0% | 3.126 |
| gmm2 · entry_kernel_0 | Ascend950DT_9582 | [2, 64, 128, 1] · {'expert_counts': [2]} | 3.695 | 3.695 | 3.545 | +4.2% | 3.387 |
| gmm2 · entry_kernel_0 | Ascend950DT_9582 | [3, 64, 64, 2] · {'expert_counts': [1, 2]} | 3.410 | 3.410 | 3.771 | -9.6% | 3.125 |
| gmm2 · entry_kernel_0 | Ascend950DT_9582 | [4, 128, 64, 2] · {'expert_counts': [1, 3]} | 3.410 | 3.410 | 3.707 | -8.0% | 3.125 |
| gmm2 · entry_kernel_0 | Ascend950DT_9582 | [6, 128, 128, 2] · {'expert_counts': [1, 5]} | 3.695 | 3.695 | 3.697 | -0.0% | 3.386 |
| gmm2 · entry_kernel_0 | Ascend950DT_9582 | [8, 64, 128, 1] · {'expert_counts': [8]} | 3.695 | 3.695 | 3.228 | +14.5% | 3.387 |
| gmm2 · entry_kernel_0 | Ascend950DT_9582 | [8, 256, 128, 2] · {'expert_counts': [1, 7]} | 3.695 | 3.695 | 3.630 | +1.8% | 3.385 |
| gmm2 · entry_kernel_0 | Ascend950DT_9582 | [16, 128, 128, 2] · {'expert_counts': [1, 15]} | 3.695 | 3.695 | 3.838 | -3.7% | 3.386 |
| gmm2 · entry_kernel_0 | Ascend950DT_9582 | [16, 256, 256, 1] · {'expert_counts': [16]} | 4.516 | 4.516 | 3.560 | +26.9% | 4.136 |
| gmm2 · entry_kernel_0 | Ascend950DT_9582 | [17, 128, 64, 2] · {'expert_counts': [8, 9]} | 3.410 | 3.410 | 3.921 | -13.0% | 3.125 |
| gmm2 · entry_kernel_0 | Ascend950DT_9582 | [32, 128, 128, 4] · {'expert_counts': [2, 6, 8, 16]} | 3.695 | 3.695 | 3.885 | -4.9% | 3.385 |
| gmm2 · entry_kernel_0 | Ascend950DT_9582 | [32, 256, 128, 2] · {'expert_counts': [1, 31]} | 3.695 | 3.695 | 3.905 | -5.4% | 3.385 |
| gmm2 · entry_kernel_0 | Ascend950DT_9582 | [64, 128, 256, 4] · {'expert_counts': [8, 8, 16, 32]} | 4.516 | 4.516 | 4.142 | +9.0% | 4.132 |
| gmm2 · GroupedMatmul_037cec40685a17f15b75759e999767c3_0 | Ascend950PR_957b | [1, 64, 64, 1] · {'expert_counts': [1]} | 3.410 | 3.410 | 3.210 | +6.2% | 3.126 |
| gmm2 · GroupedMatmul_037cec40685a17f15b75759e999767c3_0 | Ascend950PR_957b | [2, 64, 64, 1] · {'expert_counts': [2]} | 3.410 | 3.410 | 3.107 | +9.8% | 3.126 |
| gmm2 · GroupedMatmul_037cec40685a17f15b75759e999767c3_0 | Ascend950PR_957b | [2, 64, 128, 1] · {'expert_counts': [2]} | 3.696 | 3.696 | 3.065 | +20.6% | 3.387 |
| gmm2 · GroupedMatmul_037cec40685a17f15b75759e999767c3_0 | Ascend950PR_957b | [3, 64, 64, 2] · {'expert_counts': [1, 2]} | 3.410 | 3.410 | 3.404 | +0.2% | 3.125 |
| gmm2 · GroupedMatmul_037cec40685a17f15b75759e999767c3_0 | Ascend950PR_957b | [4, 128, 64, 2] · {'expert_counts': [1, 3]} | 3.410 | 3.410 | 3.664 | -6.9% | 3.125 |
| gmm2 · GroupedMatmul_037cec40685a17f15b75759e999767c3_0 | Ascend950PR_957b | [6, 128, 128, 2] · {'expert_counts': [1, 5]} | 3.698 | 3.698 | 3.812 | -3.0% | 3.386 |
| gmm2 · GroupedMatmul_037cec40685a17f15b75759e999767c3_0 | Ascend950PR_957b | [8, 64, 128, 1] · {'expert_counts': [8]} | 3.696 | 3.696 | 3.164 | +16.8% | 3.387 |
| gmm2 · GroupedMatmul_037cec40685a17f15b75759e999767c3_0 | Ascend950PR_957b | [8, 256, 128, 2] · {'expert_counts': [1, 7]} | 3.701 | 3.701 | 3.663 | +1.0% | 3.385 |
| gmm2 · GroupedMatmul_037cec40685a17f15b75759e999767c3_0 | Ascend950PR_957b | [16, 128, 128, 2] · {'expert_counts': [1, 15]} | 3.699 | 3.699 | 3.979 | -7.1% | 3.386 |
| gmm2 · GroupedMatmul_037cec40685a17f15b75759e999767c3_0 | Ascend950PR_957b | [17, 128, 64, 2] · {'expert_counts': [8, 9]} | 3.410 | 3.410 | 3.812 | -10.5% | 3.125 |
| grouped_matmul_swiglu_quant_v2 · entry_kernel_0 | Ascend950DT_9582 | [1, 64, 64, 1] · {'expert_counts': [1]} | 7.272 | 7.272 | 7.044 | +3.2% | 6.666 |
| grouped_matmul_swiglu_quant_v2 · entry_kernel_0 | Ascend950DT_9582 | [2, 64, 64, 1] · {'expert_counts': [2]} | 7.272 | 7.272 | 7.005 | +3.8% | 6.666 |
| grouped_matmul_swiglu_quant_v2 · GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic | Ascend950DT_9582 | [2, 64, 64, 1] · {'expert_counts': [2]} | 7.272 | 7.272 | 7.105 | +2.4% | 6.666 |
| grouped_matmul_swiglu_quant_v2 · entry_kernel_0 | Ascend950DT_9582 | [2, 64, 128, 1] · {'expert_counts': [2]} | 7.272 | 7.272 | 7.280 | -0.1% | 6.666 |
| grouped_matmul_swiglu_quant_v2 · entry_kernel_0 | Ascend950DT_9582 | [3, 64, 64, 2] · {'expert_counts': [1, 2]} | 7.272 | 7.272 | 7.081 | +2.7% | 6.666 |
| grouped_matmul_swiglu_quant_v2 · entry_kernel_0 | Ascend950DT_9582 | [4, 128, 64, 2] · {'expert_counts': [1, 3]} | 7.858 | 7.858 | 7.364 | +6.7% | 7.202 |
| grouped_matmul_swiglu_quant_v2 · entry_kernel_0 | Ascend950DT_9582 | [6, 128, 128, 2] · {'expert_counts': [1, 5]} | 7.858 | 7.858 | 7.390 | +6.3% | 7.201 |
| grouped_matmul_swiglu_quant_v2 · entry_kernel_0 | Ascend950DT_9582 | [8, 64, 128, 1] · {'expert_counts': [8]} | 7.272 | 7.272 | 7.405 | -1.8% | 6.666 |
| grouped_matmul_swiglu_quant_v2 · entry_kernel_0 | Ascend950DT_9582 | [8, 256, 128, 2] · {'expert_counts': [1, 7]} | 9.941 | 9.941 | 7.489 | +32.7% | 9.104 |
| grouped_matmul_swiglu_quant_v2 · entry_kernel_0 | Ascend950DT_9582 | [16, 128, 128, 2] · {'expert_counts': [1, 15]} | 7.858 | 7.858 | 7.372 | +6.6% | 7.201 |
| grouped_matmul_swiglu_quant_v2 · entry_kernel_0 | Ascend950DT_9582 | [16, 256, 256, 1] · {'expert_counts': [16]} | 9.941 | 9.941 | 8.193 | +21.3% | 9.104 |
| grouped_matmul_swiglu_quant_v2 · entry_kernel_0 | Ascend950DT_9582 | [17, 128, 64, 2] · {'expert_counts': [8, 9]} | 7.858 | 7.858 | 7.414 | +6.0% | 7.202 |
| grouped_matmul_swiglu_quant_v2 · entry_kernel_0 | Ascend950DT_9582 | [32, 128, 128, 4] · {'expert_counts': [2, 6, 8, 16]} | 7.858 | 7.858 | 7.686 | +2.2% | 7.199 |
| grouped_matmul_swiglu_quant_v2 · entry_kernel_0 | Ascend950DT_9582 | [32, 256, 128, 2] · {'expert_counts': [1, 31]} | 9.941 | 9.941 | 8.214 | +21.0% | 9.104 |
| grouped_matmul_swiglu_quant_v2 · entry_kernel_0 | Ascend950DT_9582 | [64, 128, 256, 4] · {'expert_counts': [8, 8, 16, 32]} | 7.858 | 7.858 | 8.355 | -5.9% | 7.195 |
| grouped_matmul_swiglu_quant_v2 · GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic | Ascend950PR_957b | [1, 64, 64, 1] · {'expert_counts': [1]} | 7.272 | 7.272 | 6.785 | +7.2% | 6.666 |
| grouped_matmul_swiglu_quant_v2 · GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic | Ascend950PR_957b | [2, 64, 64, 1] · {'expert_counts': [2]} | 7.272 | 7.272 | 6.885 | +5.6% | 6.666 |
| grouped_matmul_swiglu_quant_v2 · GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic | Ascend950PR_957b | [2, 64, 128, 1] · {'expert_counts': [2]} | 7.272 | 7.272 | 6.755 | +7.7% | 6.666 |
| grouped_matmul_swiglu_quant_v2 · GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic | Ascend950PR_957b | [3, 64, 64, 2] · {'expert_counts': [1, 2]} | 7.272 | 7.272 | 6.862 | +6.0% | 6.666 |
| grouped_matmul_swiglu_quant_v2 · GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic | Ascend950PR_957b | [4, 128, 64, 2] · {'expert_counts': [1, 3]} | 7.862 | 7.862 | 7.064 | +11.3% | 7.202 |
| grouped_matmul_swiglu_quant_v2 · GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic | Ascend950PR_957b | [6, 128, 128, 2] · {'expert_counts': [1, 5]} | 7.865 | 7.865 | 7.030 | +11.9% | 7.201 |
| grouped_matmul_swiglu_quant_v2 · GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic | Ascend950PR_957b | [8, 64, 128, 1] · {'expert_counts': [8]} | 7.272 | 7.272 | 7.159 | +1.6% | 6.666 |
| grouped_matmul_swiglu_quant_v2 · GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic | Ascend950PR_957b | [8, 256, 128, 2] · {'expert_counts': [1, 7]} | 9.964 | 9.964 | 7.393 | +34.8% | 9.104 |
| grouped_matmul_swiglu_quant_v2 · GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic | Ascend950PR_957b | [16, 128, 128, 2] · {'expert_counts': [1, 15]} | 7.866 | 7.866 | 7.073 | +11.2% | 7.201 |
| grouped_matmul_swiglu_quant_v2 · GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic | Ascend950PR_957b | [17, 128, 64, 2] · {'expert_counts': [8, 9]} | 7.862 | 7.862 | 7.210 | +9.0% | 7.202 |
| kda_recurrent · RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0 | Ascend950DT_9582 | [1, 1, 1, 1, 128, 128] · {'layout': 'BSND', 'dtype': 'BF16', 'gate_dtype': 'FP32', 'state_dtype': 'FP32', 'state_v_first': False, 'use_gate_in_kernel': False, 'chunk_size': None, 'output_final_state': True, 'export_gk': False} | 15.072 | 15.072 | 14.834 | +1.6% | 13.804 |
| kda_recurrent · entry_kernel_0 | Ascend950DT_9582 | [1, 1, 1, 1, 128, 128] · {'layout': 'BSND', 'dtype': 'BF16', 'gate_dtype': 'FP32', 'state_dtype': 'FP32', 'state_v_first': False, 'use_gate_in_kernel': False, 'chunk_size': None, 'output_final_state': True, 'export_gk': False} | 15.072 | 15.072 | 15.219 | -1.0% | 13.804 |
| kda_recurrent · entry_kernel_0 | Ascend950DT_9582 | [1, 1, 1, 1, 128, 256] · {'layout': 'BSND', 'dtype': 'BF16', 'gate_dtype': 'FP32', 'state_dtype': 'FP32', 'state_v_first': False, 'use_gate_in_kernel': False, 'chunk_size': None, 'output_final_state': True, 'export_gk': False} | 18.547 | 18.547 | 17.434 | +6.4% | 16.984 |
| kda_recurrent · entry_kernel_0 | Ascend950DT_9582 | [1, 2, 1, 1, 128, 128] · {'layout': 'BSND', 'dtype': 'BF16', 'gate_dtype': 'FP32', 'state_dtype': 'FP32', 'state_v_first': False, 'use_gate_in_kernel': False, 'chunk_size': None, 'output_final_state': True, 'export_gk': False} | 18.547 | 18.547 | 18.467 | +0.4% | 16.987 |
| kda_recurrent · entry_kernel_0 | Ascend950DT_9582 | [1, 2, 1, 1, 128, 256] · {'layout': 'BSND', 'dtype': 'BF16', 'gate_dtype': 'FP32', 'state_dtype': 'FP32', 'state_v_first': False, 'use_gate_in_kernel': False, 'chunk_size': None, 'output_final_state': True, 'export_gk': False} | 25.498 | 25.498 | 22.684 | +12.4% | 23.349 |
| kda_recurrent · entry_kernel_0 | Ascend950DT_9582 | [1, 3, 1, 1, 128, 128] · {'layout': 'BSND', 'dtype': 'BF16', 'gate_dtype': 'FP32', 'state_dtype': 'FP32', 'state_v_first': False, 'use_gate_in_kernel': False, 'chunk_size': None, 'output_final_state': True, 'export_gk': False} | 22.022 | 22.022 | 21.932 | +0.4% | 20.169 |
| kda_recurrent · entry_kernel_0 | Ascend950DT_9582 | [1, 3, 1, 1, 128, 256] · {'layout': 'BSND', 'dtype': 'BF16', 'gate_dtype': 'FP32', 'state_dtype': 'FP32', 'state_v_first': False, 'use_gate_in_kernel': False, 'chunk_size': None, 'output_final_state': True, 'export_gk': False} | 32.449 | 32.449 | 31.071 | +4.4% | 29.714 |
| kda_recurrent · entry_kernel_0 | Ascend950DT_9582 | [1, 4, 1, 1, 128, 128] · {'layout': 'BSND', 'dtype': 'BF16', 'gate_dtype': 'FP32', 'state_dtype': 'FP32', 'state_v_first': False, 'use_gate_in_kernel': False, 'chunk_size': None, 'output_final_state': True, 'export_gk': False} | 25.498 | 25.498 | 25.430 | +0.3% | 23.352 |
| kda_recurrent · entry_kernel_0 | Ascend950DT_9582 | [1, 4, 1, 1, 128, 256] · {'layout': 'BSND', 'dtype': 'BF16', 'gate_dtype': 'FP32', 'state_dtype': 'FP32', 'state_v_first': False, 'use_gate_in_kernel': False, 'chunk_size': None, 'output_final_state': True, 'export_gk': False} | 39.399 | 39.399 | 38.342 | +2.8% | 36.079 |
| kda_recurrent · entry_kernel_0 | Ascend950DT_9582 | [1, 5, 1, 1, 128, 128] · {'layout': 'BSND', 'dtype': 'BF16', 'gate_dtype': 'FP32', 'state_dtype': 'FP32', 'state_v_first': False, 'use_gate_in_kernel': False, 'chunk_size': None, 'output_final_state': True, 'export_gk': False} | 28.973 | 28.973 | 24.648 | +17.5% | 26.534 |
| kda_recurrent · entry_kernel_0 | Ascend950DT_9582 | [1, 6, 1, 1, 128, 128] · {'layout': 'BSND', 'dtype': 'BF16', 'gate_dtype': 'FP32', 'state_dtype': 'FP32', 'state_v_first': False, 'use_gate_in_kernel': False, 'chunk_size': None, 'output_final_state': True, 'export_gk': False} | 32.449 | 32.449 | 28.441 | +14.1% | 29.716 |
| kda_recurrent · entry_kernel_0 | Ascend950DT_9582 | [1, 6, 1, 1, 128, 256] · {'layout': 'BSND', 'dtype': 'BF16', 'gate_dtype': 'FP32', 'state_dtype': 'FP32', 'state_v_first': False, 'use_gate_in_kernel': False, 'chunk_size': None, 'output_final_state': True, 'export_gk': False} | 53.301 | 53.301 | 52.691 | +1.2% | 48.808 |
| kda_recurrent · entry_kernel_0 | Ascend950DT_9582 | [1, 7, 1, 1, 128, 128] · {'layout': 'BSND', 'dtype': 'BF16', 'gate_dtype': 'FP32', 'state_dtype': 'FP32', 'state_v_first': False, 'use_gate_in_kernel': False, 'chunk_size': None, 'output_final_state': True, 'export_gk': False} | 35.924 | 35.924 | 31.195 | +15.2% | 32.899 |
| kda_recurrent · entry_kernel_0 | Ascend950DT_9582 | [1, 8, 1, 1, 128, 128] · {'layout': 'BSND', 'dtype': 'BF16', 'gate_dtype': 'FP32', 'state_dtype': 'FP32', 'state_v_first': False, 'use_gate_in_kernel': False, 'chunk_size': None, 'output_final_state': True, 'export_gk': False} | 39.399 | 39.399 | 35.396 | +11.3% | 36.081 |
| kda_recurrent · RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0 | Ascend950PR_957b | [1, 1, 1, 1, 128, 128] · {'layout': 'BSND', 'dtype': 'BF16', 'gate_dtype': 'FP32', 'state_dtype': 'FP32', 'state_v_first': False, 'use_gate_in_kernel': False, 'chunk_size': None, 'output_final_state': True, 'export_gk': False} | 15.277 | 15.277 | 14.622 | +4.5% | 13.804 |
| kda_recurrent · RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0 | Ascend950PR_957b | [1, 1, 1, 1, 128, 256] · {'layout': 'BSND', 'dtype': 'BF16', 'gate_dtype': 'FP32', 'state_dtype': 'FP32', 'state_v_first': False, 'use_gate_in_kernel': False, 'chunk_size': None, 'output_final_state': True, 'export_gk': False} | 18.841 | 18.841 | 21.030 | -10.4% | 16.984 |
| kda_recurrent · RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0 | Ascend950PR_957b | [1, 2, 1, 1, 128, 128] · {'layout': 'BSND', 'dtype': 'BF16', 'gate_dtype': 'FP32', 'state_dtype': 'FP32', 'state_v_first': False, 'use_gate_in_kernel': False, 'chunk_size': None, 'output_final_state': True, 'export_gk': False} | 18.832 | 18.832 | 18.048 | +4.3% | 16.987 |
| kda_recurrent · RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0 | Ascend950PR_957b | [1, 2, 1, 1, 128, 256] · {'layout': 'BSND', 'dtype': 'BF16', 'gate_dtype': 'FP32', 'state_dtype': 'FP32', 'state_v_first': False, 'use_gate_in_kernel': False, 'chunk_size': None, 'output_final_state': True, 'export_gk': False} | 25.951 | 25.951 | 28.402 | -8.6% | 23.349 |
| kda_recurrent · RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0 | Ascend950PR_957b | [1, 3, 1, 1, 128, 128] · {'layout': 'BSND', 'dtype': 'BF16', 'gate_dtype': 'FP32', 'state_dtype': 'FP32', 'state_v_first': False, 'use_gate_in_kernel': False, 'chunk_size': None, 'output_final_state': True, 'export_gk': False} | 22.388 | 22.388 | 21.221 | +5.5% | 20.169 |
| kda_recurrent · RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0 | Ascend950PR_957b | [1, 4, 1, 1, 128, 128] · {'layout': 'BSND', 'dtype': 'BF16', 'gate_dtype': 'FP32', 'state_dtype': 'FP32', 'state_v_first': False, 'use_gate_in_kernel': False, 'chunk_size': None, 'output_final_state': True, 'export_gk': False} | 25.943 | 25.943 | 25.084 | +3.4% | 23.352 |
| kda_recurrent · RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0 | Ascend950PR_957b | [1, 5, 1, 1, 128, 128] · {'layout': 'BSND', 'dtype': 'BF16', 'gate_dtype': 'FP32', 'state_dtype': 'FP32', 'state_v_first': False, 'use_gate_in_kernel': False, 'chunk_size': None, 'output_final_state': True, 'export_gk': False} | 29.498 | 29.498 | 28.819 | +2.4% | 26.534 |
| kda_recurrent · RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0 | Ascend950PR_957b | [1, 6, 1, 1, 128, 128] · {'layout': 'BSND', 'dtype': 'BF16', 'gate_dtype': 'FP32', 'state_dtype': 'FP32', 'state_v_first': False, 'use_gate_in_kernel': False, 'chunk_size': None, 'output_final_state': True, 'export_gk': False} | 33.054 | 33.054 | 31.955 | +3.4% | 29.716 |
| kda_recurrent · RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0 | Ascend950PR_957b | [1, 7, 1, 1, 128, 128] · {'layout': 'BSND', 'dtype': 'BF16', 'gate_dtype': 'FP32', 'state_dtype': 'FP32', 'state_v_first': False, 'use_gate_in_kernel': False, 'chunk_size': None, 'output_final_state': True, 'export_gk': False} | 36.609 | 36.609 | 35.718 | +2.5% | 32.899 |
| kda_recurrent · RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0 | Ascend950PR_957b | [1, 8, 1, 1, 128, 128] · {'layout': 'BSND', 'dtype': 'BF16', 'gate_dtype': 'FP32', 'state_dtype': 'FP32', 'state_v_first': False, 'use_gate_in_kernel': False, 'chunk_size': None, 'output_final_state': True, 'export_gk': False} | 40.164 | 40.164 | 39.450 | +1.8% | 36.081 |
| lightning_indexer · LightningIndexer_2517851d7f53b28f971ff94bfa4b7037_203547_mix_aic | Ascend950DT_9582 | [1, 1, 512, 8, 128] · {'dtype': 'bf16', 'layout': 'BSND', 'sparse_count': 512, 'sparse_mode': 3, 'return_values': False} | 5.771 | 5.771 | 5.862 | -1.6% | 5.288 |
| lightning_indexer · entry_kernel_31b1b | Ascend950DT_9582 | [1, 1, 512, 8, 128] · {'dtype': 'bf16', 'layout': 'BSND', 'sparse_count': 512, 'sparse_mode': 3, 'return_values': False} | 5.771 | 5.771 | 6.093 | -5.3% | 5.288 |
| lightning_indexer · entry_kernel_31b1b | Ascend950DT_9582 | [1, 1, 640, 8, 128] · {'dtype': 'bf16', 'layout': 'BSND', 'sparse_count': 512, 'sparse_mode': 3, 'return_values': False} | 6.652 | 6.652 | 6.200 | +7.3% | 6.093 |
| lightning_indexer · entry_kernel_31b1b | Ascend950DT_9582 | [1, 1, 896, 8, 128] · {'dtype': 'bf16', 'layout': 'BSND', 'sparse_count': 512, 'sparse_mode': 3, 'return_values': False} | 6.711 | 6.711 | 6.439 | +4.2% | 6.146 |
| lightning_indexer · entry_kernel_31b1b | Ascend950DT_9582 | [1, 1, 1024, 8, 128] · {'dtype': 'bf16', 'layout': 'BSND', 'sparse_count': 512, 'sparse_mode': 3, 'return_values': False} | 6.755 | 6.755 | 6.741 | +0.2% | 6.178 |
| lightning_indexer · entry_kernel_31b1b | Ascend950DT_9582 | [1, 1, 2048, 8, 128] · {'dtype': 'bf16', 'layout': 'BSND', 'sparse_count': 512, 'sparse_mode': 3, 'return_values': False} | 10.051 | 10.051 | 8.614 | +16.7% | 9.178 |
| lightning_indexer · entry_kernel_31b1b | Ascend950DT_9582 | [1, 1, 8192, 8, 128] · {'dtype': 'bf16', 'layout': 'BSND', 'sparse_count': 512, 'sparse_mode': 3, 'return_values': False} | 29.909 | 29.909 | 22.709 | +31.7% | 27.231 |
| lightning_indexer · entry_kernel_31b1b | Ascend950DT_9582 | [1, 2, 768, 8, 128] · {'dtype': 'bf16', 'layout': 'BSND', 'sparse_count': 512, 'sparse_mode': 3, 'return_values': False} | 6.682 | 6.682 | 6.365 | +5.0% | 6.120 |
| lightning_indexer · entry_kernel_31b1b | Ascend950DT_9582 | [1, 2, 1024, 8, 128] · {'dtype': 'bf16', 'layout': 'BSND', 'sparse_count': 512, 'sparse_mode': 3, 'return_values': False} | 6.755 | 6.755 | 6.516 | +3.7% | 6.178 |
| lightning_indexer · entry_kernel_31b1b | Ascend950DT_9582 | [1, 2, 3072, 8, 128] · {'dtype': 'bf16', 'layout': 'BSND', 'sparse_count': 512, 'sparse_mode': 3, 'return_values': False} | 13.361 | 13.361 | 11.101 | +20.4% | 12.188 |
| lightning_indexer · entry_kernel_31b1b | Ascend950DT_9582 | [1, 4, 1024, 8, 128] · {'dtype': 'bf16', 'layout': 'BSND', 'sparse_count': 512, 'sparse_mode': 3, 'return_values': False} | 6.755 | 6.755 | 8.043 | -16.0% | 6.178 |
| lightning_indexer · entry_kernel_31b1b | Ascend950DT_9582 | [1, 4, 4096, 8, 128] · {'dtype': 'bf16', 'layout': 'BSND', 'sparse_count': 512, 'sparse_mode': 3, 'return_values': False} | 16.670 | 16.670 | 15.674 | +6.4% | 15.196 |
| lightning_indexer · entry_kernel_31b1b | Ascend950DT_9582 | [1, 8, 1536, 8, 128] · {'dtype': 'bf16', 'layout': 'BSND', 'sparse_count': 512, 'sparse_mode': 3, 'return_values': False} | 8.396 | 8.396 | 8.862 | -5.3% | 7.671 |
| lightning_indexer · entry_kernel_31b1b | Ascend950DT_9582 | [1, 9, 4096, 8, 128] · {'dtype': 'bf16', 'layout': 'BSND', 'sparse_count': 512, 'sparse_mode': 3, 'return_values': False} | 16.670 | 16.670 | 15.334 | +8.7% | 15.195 |
| lightning_indexer · LightningIndexer_2517851d7f53b28f971ff94bfa4b7037_203547_mix_aic | Ascend950PR_957b | [1, 1, 512, 8, 128] · {'dtype': 'bf16', 'layout': 'BSND', 'sparse_count': 512, 'sparse_mode': 3, 'return_values': False} | 5.790 | 5.790 | 5.779 | +0.2% | 5.288 |
| lightning_indexer · LightningIndexer_2517851d7f53b28f971ff94bfa4b7037_203547_mix_aic | Ascend950PR_957b | [1, 1, 640, 8, 128] · {'dtype': 'bf16', 'layout': 'BSND', 'sparse_count': 512, 'sparse_mode': 3, 'return_values': False} | 6.700 | 6.700 | 5.935 | +12.9% | 6.093 |
| lightning_indexer · LightningIndexer_2517851d7f53b28f971ff94bfa4b7037_203547_mix_aic | Ascend950PR_957b | [1, 1, 896, 8, 128] · {'dtype': 'bf16', 'layout': 'BSND', 'sparse_count': 512, 'sparse_mode': 3, 'return_values': False} | 6.765 | 6.765 | 6.129 | +10.4% | 6.146 |
| lightning_indexer · LightningIndexer_2517851d7f53b28f971ff94bfa4b7037_203547_mix_aic | Ascend950PR_957b | [1, 1, 1024, 8, 128] · {'dtype': 'bf16', 'layout': 'BSND', 'sparse_count': 512, 'sparse_mode': 3, 'return_values': False} | 6.822 | 6.822 | 6.493 | +5.1% | 6.178 |
| lightning_indexer · LightningIndexer_2517851d7f53b28f971ff94bfa4b7037_203547_mix_aic | Ascend950PR_957b | [1, 1, 2048, 8, 128] · {'dtype': 'bf16', 'layout': 'BSND', 'sparse_count': 512, 'sparse_mode': 3, 'return_values': False} | 10.220 | 10.220 | 8.455 | +20.9% | 9.178 |
| lightning_indexer · LightningIndexer_2517851d7f53b28f971ff94bfa4b7037_203547_mix_aic | Ascend950PR_957b | [1, 2, 768, 8, 128] · {'dtype': 'bf16', 'layout': 'BSND', 'sparse_count': 512, 'sparse_mode': 3, 'return_values': False} | 6.733 | 6.733 | 6.171 | +9.1% | 6.120 |
| lightning_indexer · LightningIndexer_2517851d7f53b28f971ff94bfa4b7037_203547_mix_aic | Ascend950PR_957b | [1, 2, 1024, 8, 128] · {'dtype': 'bf16', 'layout': 'BSND', 'sparse_count': 512, 'sparse_mode': 3, 'return_values': False} | 6.823 | 6.823 | 6.570 | +3.8% | 6.178 |
| lightning_indexer · LightningIndexer_2517851d7f53b28f971ff94bfa4b7037_203547_mix_aic | Ascend950PR_957b | [1, 2, 3072, 8, 128] · {'dtype': 'bf16', 'layout': 'BSND', 'sparse_count': 512, 'sparse_mode': 3, 'return_values': False} | 13.633 | 13.633 | 10.776 | +26.5% | 12.188 |
| lightning_indexer · LightningIndexer_2517851d7f53b28f971ff94bfa4b7037_203547_mix_aic | Ascend950PR_957b | [1, 4, 1024, 8, 128] · {'dtype': 'bf16', 'layout': 'BSND', 'sparse_count': 512, 'sparse_mode': 3, 'return_values': False} | 6.824 | 6.824 | 8.030 | -15.0% | 6.178 |
| lightning_indexer · LightningIndexer_2517851d7f53b28f971ff94bfa4b7037_203547_mix_aic | Ascend950PR_957b | [1, 8, 1536, 8, 128] · {'dtype': 'bf16', 'layout': 'BSND', 'sparse_count': 512, 'sparse_mode': 3, 'return_values': False} | 8.521 | 8.521 | 8.829 | -3.5% | 7.671 |
| matmul · entry_kernel_2 | Ascend950DT_9582 | [1, 32, 16] · {} | 3.091 | 3.091 | 3.030 | +2.0% | 2.833 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [1, 128, 128] · {} | 3.414 | 3.414 | 3.356 | +1.7% | 3.129 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [1, 256, 1024] · {} | 5.127 | 5.127 | 4.624 | +10.9% | 4.700 |
| matmul · entry_kernel_2 | Ascend950DT_9582 | [2, 32, 32] · {} | 3.109 | 3.109 | 3.244 | -4.2% | 2.850 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [3, 64, 64] · {} | 3.217 | 3.217 | 3.065 | +4.9% | 2.949 |
| matmul · entry_kernel_2 | Ascend950DT_9582 | [4, 80, 48] · {} | 3.237 | 3.237 | 3.524 | -8.1% | 2.967 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [8, 96, 256] · {} | 3.702 | 3.702 | 3.755 | -1.4% | 3.393 |
| matmul · entry_kernel_2 | Ascend950DT_9582 | [9, 128, 256] · {} | 3.703 | 3.703 | 4.019 | -7.9% | 3.394 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [17, 96, 128] · {} | 3.476 | 3.476 | 3.494 | -0.5% | 3.187 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [17, 256, 1024] · {} | 5.190 | 5.190 | 4.742 | +9.5% | 4.758 |
| matmul · entry_kernel_2 | Ascend950DT_9582 | [31, 64, 257] · {} | 3.781 | 3.781 | 4.091 | -7.6% | 3.466 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [32, 64, 64] · {} | 3.278 | 3.278 | 3.281 | -0.1% | 3.005 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [32, 64, 64] · {} | 3.278 | 3.278 | 3.292 | -0.4% | 3.005 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [33, 128, 129] · {} | 3.513 | 3.513 | 3.671 | -4.3% | 3.220 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [48, 128, 192] · {} | 3.651 | 3.651 | 3.575 | +2.1% | 3.347 |
| matmul · entry_kernel_2 | Ascend950DT_9582 | [64, 128, 512] · {} | 4.348 | 4.348 | 4.437 | -2.0% | 3.985 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [65, 128, 129] · {} | 3.554 | 3.554 | 3.712 | -4.2% | 3.258 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [127, 384, 256] · {} | 3.954 | 3.954 | 3.819 | +3.5% | 3.630 |
| matmul · entry_kernel_2 | Ascend950DT_9582 | [128, 192, 256] · {} | 3.906 | 3.906 | 3.798 | +2.8% | 3.579 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [128, 256, 512] · {} | 4.463 | 4.463 | 4.393 | +1.6% | 4.087 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [128, 512, 512] · {} | 4.541 | 4.541 | 4.382 | +3.6% | 4.137 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [129, 256, 256] · {} | 3.936 | 3.936 | 3.664 | +7.4% | 3.608 |
| matmul · entry_kernel_2 | Ascend950DT_9582 | [192, 384, 256] · {} | 4.002 | 4.002 | 3.925 | +1.9% | 3.659 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [225, 512, 256] · {} | 4.079 | 4.079 | 4.113 | -0.8% | 3.697 |
| matmul · entry_kernel_2 | Ascend950DT_9582 | [256, 512, 256] · {} | 4.086 | 4.086 | 4.192 | -2.5% | 3.693 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [449, 512, 256] · {} | 4.759 | 4.759 | 4.361 | +9.1% | 3.740 |
| matmul · entry_kernel_2 | Ascend950DT_9582 | [512, 512, 256] · {} | 4.771 | 4.771 | 4.799 | -0.6% | 3.759 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [1024, 1792, 32] · {} | 4.610 | 4.610 | 4.260 | +8.2% | 3.477 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [1024, 2304, 16] · {} | 4.531 | 4.531 | 4.258 | +6.4% | 3.450 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [1280, 2048, 16] · {} | 4.630 | 4.630 | 4.397 | +5.3% | 3.481 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [1280, 2048, 16] · {} | 4.630 | 4.630 | 4.472 | +3.5% | 3.481 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [1280, 2048, 512] · {} | 12.449 | 12.449 | 12.219 | +1.9% | 6.350 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950DT_9582 | [1536, 1792, 16] · {} | 4.680 | 4.680 | 4.438 | +5.5% | 3.490 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_4098_mix_aic | Ascend950DT_9582 | [2048, 32, 5120] · {} | 12.123 | 12.123 | 13.023 | -6.9% | 7.287 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [1, 32, 16] · {} | 3.091 | 3.091 | 2.734 | +13.1% | 2.833 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [1, 128, 128] · {} | 3.414 | 3.414 | 3.138 | +8.8% | 3.129 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [1, 256, 1024] · {} | 5.127 | 5.127 | 4.612 | +11.2% | 4.700 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [2, 32, 32] · {} | 3.109 | 3.109 | 2.813 | +10.5% | 2.850 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [3, 64, 64] · {} | 3.217 | 3.217 | 2.858 | +12.6% | 2.949 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [4, 80, 48] · {} | 3.237 | 3.237 | 3.341 | -3.1% | 2.967 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [8, 96, 256] · {} | 3.702 | 3.702 | 3.620 | +2.3% | 3.393 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [9, 128, 256] · {} | 3.703 | 3.703 | 3.595 | +3.0% | 3.394 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [16, 192, 384] · {} | 4.039 | 4.039 | 3.724 | +8.5% | 3.703 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [17, 96, 128] · {} | 3.476 | 3.476 | 3.414 | +1.8% | 3.187 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [17, 256, 1024] · {} | 5.190 | 5.190 | 4.687 | +10.7% | 4.758 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [31, 64, 257] · {} | 3.781 | 3.781 | 3.801 | -0.5% | 3.466 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [32, 64, 64] · {} | 3.278 | 3.278 | 3.158 | +3.8% | 3.005 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [33, 128, 129] · {} | 3.513 | 3.513 | 3.679 | -4.5% | 3.220 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [48, 128, 192] · {} | 3.651 | 3.651 | 3.481 | +4.9% | 3.347 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [64, 128, 512] · {} | 4.348 | 4.348 | 4.275 | +1.7% | 3.985 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [65, 128, 129] · {} | 3.554 | 3.554 | 3.733 | -4.8% | 3.258 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [127, 384, 256] · {} | 3.956 | 3.956 | 3.793 | +4.3% | 3.630 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [128, 192, 256] · {} | 3.907 | 3.907 | 3.815 | +2.4% | 3.579 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [128, 256, 512] · {} | 4.477 | 4.477 | 4.368 | +2.5% | 4.087 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [128, 512, 512] · {} | 4.600 | 4.600 | 4.604 | -0.1% | 4.137 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [129, 256, 256] · {} | 3.936 | 3.936 | 3.609 | +9.1% | 3.608 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [192, 384, 256] · {} | 4.006 | 4.006 | 4.040 | -0.8% | 3.659 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [224, 512, 256] · {} | 4.225 | 4.225 | 4.115 | +2.7% | 3.692 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [225, 512, 256] · {} | 4.225 | 4.225 | 4.116 | +2.6% | 3.697 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [255, 512, 256] · {} | 4.390 | 4.390 | 4.183 | +4.9% | 3.693 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [256, 512, 256] · {} | 4.390 | 4.390 | 4.239 | +3.5% | 3.693 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [257, 512, 256] · {} | 4.390 | 4.390 | 4.197 | +4.6% | 3.702 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [384, 512, 256] · {} | 4.734 | 4.734 | 4.504 | +5.1% | 3.723 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [448, 512, 256] · {} | 4.760 | 4.760 | 4.515 | +5.4% | 3.735 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [449, 512, 256] · {} | 4.826 | 4.826 | 4.595 | +5.0% | 3.740 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [511, 512, 256] · {} | 4.852 | 4.852 | 4.612 | +5.2% | 3.759 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [512, 384, 256] · {} | 4.734 | 4.734 | 4.541 | +4.3% | 3.723 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [512, 512, 256] · {} | 4.852 | 4.852 | 4.656 | +4.2% | 3.759 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [513, 384, 256] · {} | 4.743 | 4.743 | 4.515 | +5.1% | 3.727 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [768, 512, 128] · {} | 4.312 | 4.312 | 4.068 | +6.0% | 3.737 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [1024, 1792, 32] · {} | 4.709 | 4.709 | 4.347 | +8.3% | 3.477 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [1024, 2048, 32] · {} | 4.823 | 4.823 | 4.464 | +8.0% | 3.547 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [1024, 2304, 16] · {} | 4.648 | 4.648 | 4.419 | +5.2% | 3.450 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [1280, 2048, 16] · {} | 4.762 | 4.762 | 4.633 | +2.8% | 3.481 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [1280, 2048, 512] · {} | 14.027 | 14.027 | 13.218 | +6.1% | 6.350 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [1536, 1792, 16] · {} | 4.818 | 4.818 | 4.540 | +6.1% | 3.490 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2 | Ascend950PR_957b | [1536, 2048, 16] · {} | 4.969 | 4.969 | 4.755 | +4.5% | 3.540 |
| matmul · MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_4098_mix_aic | Ascend950PR_957b | [2048, 32, 5120] · {} | 19.016 | 19.016 | 20.822 | -8.7% | 7.287 |
| mhc · entry_kernel_1+entry_kernel_0+entry_kernel_5 | Ascend950DT_9582 | [1, 4, 64] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 18.002 | 18.002 | 17.143 | +5.0% | 16.502 |
| mhc · entry_kernel_1+entry_kernel_0+entry_kernel_5 | Ascend950DT_9582 | [1, 4, 80] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 18.230 | 18.230 | 17.559 | +3.8% | 16.710 |
| mhc · MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic+MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0+MhcPost_59130ccee6717ff2c0f15b485e731162_5 | Ascend950DT_9582 | [1, 4, 128] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 16.903 | 16.903 | 15.768 | +7.2% | 15.494 |
| mhc · entry_kernel_2 + entry_kernel_0 + entry_kernel_5 | Ascend950DT_9582 | [1, 4, 128] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 16.903 | 16.903 | 15.970 | +5.8% | 15.494 |
| mhc · entry_kernel_2+entry_kernel_0+entry_kernel_5 | Ascend950DT_9582 | [1, 4, 512] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 17.457 | 17.457 | 17.343 | +0.7% | 16.000 |
| mhc · entry_kernel_1+entry_kernel_0+entry_kernel_5 | Ascend950DT_9582 | [2, 4, 64] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 18.028 | 18.028 | 17.935 | +0.5% | 16.525 |
| mhc · entry_kernel_2+entry_kernel_0+entry_kernel_5 | Ascend950DT_9582 | [2, 4, 128] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 16.923 | 16.923 | 16.416 | +3.1% | 15.512 |
| mhc · entry_kernel_2 + entry_kernel_0 + entry_kernel_5 | Ascend950DT_9582 | [3, 4, 128] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 16.942 | 16.942 | 16.835 | +0.6% | 15.530 |
| mhc · entry_kernel_1+entry_kernel_0+entry_kernel_5 | Ascend950DT_9582 | [4, 4, 96] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 18.452 | 18.452 | 17.570 | +5.0% | 16.913 |
| mhc · entry_kernel_2+entry_kernel_0+entry_kernel_5 | Ascend950DT_9582 | [4, 4, 256] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 17.300 | 17.300 | 17.344 | -0.3% | 15.857 |
| mhc · entry_kernel_2+entry_kernel_0+entry_kernel_5 | Ascend950DT_9582 | [4, 4, 1024] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 18.127 | 18.127 | 18.220 | -0.5% | 16.608 |
| mhc · entry_kernel_2+entry_kernel_0+entry_kernel_5 | Ascend950DT_9582 | [5, 4, 256] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 17.326 | 17.326 | 17.125 | +1.2% | 15.881 |
| mhc · entry_kernel_2+entry_kernel_0+entry_kernel_5 | Ascend950DT_9582 | [6, 4, 128] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 17.001 | 17.001 | 17.016 | -0.1% | 15.584 |
| mhc · entry_kernel_2+entry_kernel_0+entry_kernel_5 | Ascend950DT_9582 | [6, 4, 1024] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 18.259 | 18.259 | 18.361 | -0.6% | 16.728 |
| mhc · entry_kernel_2 + entry_kernel_0 + entry_kernel_5 | Ascend950DT_9582 | [8, 4, 128] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 17.040 | 17.040 | 16.791 | +1.5% | 15.620 |
| mhc · entry_kernel_2+entry_kernel_0+entry_kernel_5 | Ascend950DT_9582 | [8, 4, 512] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 17.734 | 17.734 | 18.652 | -4.9% | 16.253 |
| mhc · entry_kernel_2+entry_kernel_0+entry_kernel_5 | Ascend950DT_9582 | [9, 4, 1024] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 18.458 | 18.458 | 18.482 | -0.1% | 16.909 |
| mhc · entry_kernel_2+entry_kernel_0+entry_kernel_5 | Ascend950DT_9582 | [12, 4, 384] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 17.601 | 17.601 | 18.486 | -4.8% | 16.132 |
| mhc · entry_kernel_2+entry_kernel_0+entry_kernel_5 | Ascend950DT_9582 | [12, 4, 768] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 18.175 | 18.175 | 18.787 | -3.3% | 16.652 |
| mhc · entry_kernel_2+entry_kernel_0+entry_kernel_5 | Ascend950DT_9582 | [16, 4, 128] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 17.196 | 17.196 | 18.202 | -5.5% | 15.762 |
| mhc · entry_kernel_2+entry_kernel_0+entry_kernel_5 | Ascend950DT_9582 | [16, 4, 512] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 18.051 | 18.051 | 18.691 | -3.4% | 16.541 |
| mhc · entry_kernel_2+entry_kernel_0+entry_kernel_5 | Ascend950DT_9582 | [17, 4, 512] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 19.343 | 19.343 | 19.328 | +0.1% | 17.726 |
| mhc · entry_kernel_2+entry_kernel_0+entry_kernel_5 | Ascend950DT_9582 | [24, 4, 1536] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 22.595 | 22.595 | 21.986 | +2.8% | 19.762 |
| mhc · entry_kernel_2+entry_kernel_0+entry_kernel_6 | Ascend950DT_9582 | [32, 4, 1024] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 19.558 | 19.558 | 20.070 | -2.5% | 17.913 |
| mhc · MhcPre_52e042c374f575edbe357a32523dba2c_1_mix_aic+MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0+MhcPost_59130ccee6717ff2c0f15b485e731162_5 | Ascend950PR_957b | [1, 4, 64] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 18.006 | 18.006 | 16.464 | +9.4% | 16.502 |
| mhc · MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic+MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0+MhcPost_59130ccee6717ff2c0f15b485e731162_5 | Ascend950PR_957b | [1, 4, 128] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 16.909 | 16.909 | 15.496 | +9.1% | 15.494 |
| mhc · MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic+MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0+MhcPost_59130ccee6717ff2c0f15b485e731162_5 | Ascend950PR_957b | [1, 4, 512] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 17.473 | 17.473 | 16.794 | +4.0% | 16.000 |
| mhc · MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic+MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0+MhcPost_59130ccee6717ff2c0f15b485e731162_5 | Ascend950PR_957b | [2, 4, 128] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 16.929 | 16.929 | 16.054 | +5.4% | 15.512 |
| mhc · MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic+MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0+MhcPost_59130ccee6717ff2c0f15b485e731162_5 | Ascend950PR_957b | [3, 4, 128] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 16.948 | 16.948 | 16.406 | +3.3% | 15.530 |
| mhc · MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic+MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0+MhcPost_59130ccee6717ff2c0f15b485e731162_5 | Ascend950PR_957b | [4, 4, 256] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 17.312 | 17.312 | 17.061 | +1.5% | 15.857 |
| mhc · MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic+MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0+MhcPost_59130ccee6717ff2c0f15b485e731162_5 | Ascend950PR_957b | [4, 4, 1024] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 18.160 | 18.160 | 18.525 | -2.0% | 16.608 |
| mhc · MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic+MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0+MhcPost_59130ccee6717ff2c0f15b485e731162_5 | Ascend950PR_957b | [8, 4, 128] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 17.046 | 17.046 | 16.738 | +1.8% | 15.620 |
| mhc · MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic+MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0+MhcPost_59130ccee6717ff2c0f15b485e731162_5 | Ascend950PR_957b | [9, 4, 1024] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 18.502 | 18.502 | 19.199 | -3.6% | 16.909 |
| mhc · MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic+MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0+MhcPost_59130ccee6717ff2c0f15b485e731162_5 | Ascend950PR_957b | [16, 4, 128] · {'dtype': 'bf16', 'implementation': 'split', 'num_iters': 20} | 17.202 | 17.202 | 17.677 | -2.7% | 15.762 |
| mhc_pre · entry_kernel_1 | Ascend950DT_9582 | [1, 4, 64] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 9.579 | 9.579 | 9.482 | +1.0% | 8.780 |
| mhc_pre · entry_kernel_1 | Ascend950DT_9582 | [1, 4, 80] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 9.705 | 9.705 | 9.802 | -1.0% | 8.896 |
| mhc_pre · MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic | Ascend950DT_9582 | [1, 4, 128] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 8.378 | 8.378 | 7.879 | +6.3% | 7.679 |
| mhc_pre · entry_kernel_2 | Ascend950DT_9582 | [1, 4, 128] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 8.378 | 8.378 | 8.271 | +1.3% | 7.679 |
| mhc_pre · entry_kernel_2 | Ascend950DT_9582 | [1, 4, 512] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 8.717 | 8.717 | 8.544 | +2.0% | 7.989 |
| mhc_pre · entry_kernel_1 | Ascend950DT_9582 | [2, 4, 64] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 9.589 | 9.589 | 9.962 | -3.7% | 8.790 |
| mhc_pre · entry_kernel_2 | Ascend950DT_9582 | [2, 4, 128] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 8.380 | 8.380 | 8.005 | +4.7% | 7.681 |
| mhc_pre · entry_kernel_2 | Ascend950DT_9582 | [3, 4, 128] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 8.382 | 8.382 | 8.312 | +0.8% | 7.684 |
| mhc_pre · entry_kernel_1 | Ascend950DT_9582 | [4, 4, 96] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 9.879 | 9.879 | 9.312 | +6.1% | 9.055 |
| mhc_pre · entry_kernel_2 | Ascend950DT_9582 | [4, 4, 256] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 8.505 | 8.505 | 8.421 | +1.0% | 7.796 |
| mhc_pre · entry_kernel_2 | Ascend950DT_9582 | [4, 4, 1024] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 9.226 | 9.226 | 8.730 | +5.7% | 8.450 |
| mhc_pre · entry_kernel_2 | Ascend950DT_9582 | [5, 4, 256] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 8.509 | 8.509 | 8.251 | +3.1% | 7.800 |
| mhc_pre · entry_kernel_2 | Ascend950DT_9582 | [6, 4, 128] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 8.389 | 8.389 | 8.410 | -0.2% | 7.690 |
| mhc_pre · entry_kernel_2 | Ascend950DT_9582 | [6, 4, 1024] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 9.263 | 9.263 | 8.973 | +3.2% | 8.483 |
| mhc_pre · entry_kernel_2 | Ascend950DT_9582 | [8, 4, 128] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 8.394 | 8.394 | 8.542 | -1.7% | 7.694 |
| mhc_pre · entry_kernel_2 | Ascend950DT_9582 | [8, 4, 512] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 8.782 | 8.782 | 8.866 | -0.9% | 8.048 |
| mhc_pre · entry_kernel_2 | Ascend950DT_9582 | [9, 4, 1024] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 9.318 | 9.318 | 9.024 | +3.3% | 8.533 |
| mhc_pre · entry_kernel_2 | Ascend950DT_9582 | [12, 4, 384] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 8.680 | 8.680 | 9.137 | -5.0% | 7.956 |
| mhc_pre · entry_kernel_2 | Ascend950DT_9582 | [12, 4, 768] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 9.097 | 9.097 | 9.058 | +0.4% | 8.333 |
| mhc_pre · entry_kernel_2 | Ascend950DT_9582 | [16, 4, 128] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 8.412 | 8.412 | 9.308 | -9.6% | 7.711 |
| mhc_pre · entry_kernel_2 | Ascend950DT_9582 | [16, 4, 512] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 8.856 | 8.856 | 9.385 | -5.6% | 8.115 |
| mhc_pre · entry_kernel_2 | Ascend950DT_9582 | [17, 4, 512] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 10.118 | 10.118 | 10.095 | +0.2% | 9.272 |
| mhc_pre · entry_kernel_2 | Ascend950DT_9582 | [24, 4, 1536] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 11.514 | 11.514 | 12.201 | -5.6% | 9.632 |
| mhc_pre · entry_kernel_2 | Ascend950DT_9582 | [32, 4, 1024] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 10.996 | 10.996 | 11.368 | -3.3% | 10.064 |
| mhc_pre · MhcPre_52e042c374f575edbe357a32523dba2c_1_mix_aic | Ascend950PR_957b | [1, 4, 64] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 9.579 | 9.579 | 8.952 | +7.0% | 8.780 |
| mhc_pre · MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic | Ascend950PR_957b | [1, 4, 128] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 8.378 | 8.378 | 7.638 | +9.7% | 7.679 |
| mhc_pre · MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic | Ascend950PR_957b | [1, 4, 512] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 8.722 | 8.722 | 7.876 | +10.7% | 7.989 |
| mhc_pre · MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic | Ascend950PR_957b | [2, 4, 128] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 8.380 | 8.380 | 7.644 | +9.6% | 7.681 |
| mhc_pre · MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic | Ascend950PR_957b | [3, 4, 128] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 8.383 | 8.383 | 7.921 | +5.8% | 7.684 |
| mhc_pre · MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic | Ascend950PR_957b | [4, 4, 256] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 8.506 | 8.506 | 8.085 | +5.2% | 7.796 |
| mhc_pre · MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic | Ascend950PR_957b | [4, 4, 1024] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 9.247 | 9.247 | 9.420 | -1.8% | 8.450 |
| mhc_pre · MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic | Ascend950PR_957b | [8, 4, 128] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 8.394 | 8.394 | 8.072 | +4.0% | 7.694 |
| mhc_pre · MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic | Ascend950PR_957b | [9, 4, 1024] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 9.344 | 9.344 | 9.865 | -5.3% | 8.533 |
| mhc_pre · MhcPre_52e042c374f575edbe357a32523dba2c_2_mix_aic | Ascend950PR_957b | [16, 4, 128] · {'dtype': 'bf16', 'alpha_count': 3, 'has_gamma': False, 'return_intermediates': True} | 8.413 | 8.413 | 8.696 | -3.3% | 7.711 |
| mhc_pre_sinkhorn · entry_kernel_3e8 | Ascend950DT_9582 | [1, 4, 4096] · {'dtype': 'bf16', 'num_iters': 20, 'need_backward': False} | 9.939 | 9.939 | 9.701 | +2.5% | 9.109 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950DT_9582 | [1, 4, 4096] · {'dtype': 'bf16', 'num_iters': 20, 'need_backward': False} | 9.939 | 9.939 | 10.108 | -1.7% | 9.109 |
| mhc_pre_sinkhorn · entry_kernel_3e8 | Ascend950DT_9582 | [2, 4, 4096] · {'dtype': 'bf16', 'num_iters': 20, 'need_backward': False} | 9.939 | 9.939 | 10.648 | -6.7% | 9.109 |
| mhc_pre_sinkhorn · entry_kernel_3e8 | Ascend950DT_9582 | [3, 4, 4096] · {'dtype': 'bf16', 'num_iters': 20, 'need_backward': False} | 9.971 | 9.971 | 10.373 | -3.9% | 9.109 |
| mhc_pre_sinkhorn · entry_kernel_3e8 | Ascend950DT_9582 | [4, 4, 4096] · {'dtype': 'bf16', 'num_iters': 20, 'need_backward': False} | 9.971 | 9.971 | 9.939 | +0.3% | 9.109 |
| mhc_pre_sinkhorn · entry_kernel_3e8 | Ascend950DT_9582 | [5, 4, 4096] · {'dtype': 'bf16', 'num_iters': 20, 'need_backward': False} | 10.002 | 10.002 | 9.761 | +2.5% | 9.109 |
| mhc_pre_sinkhorn · entry_kernel_3e8 | Ascend950DT_9582 | [6, 4, 4096] · {'dtype': 'bf16', 'num_iters': 20, 'need_backward': False} | 10.002 | 10.002 | 9.785 | +2.2% | 9.109 |
| mhc_pre_sinkhorn · entry_kernel_3e8 | Ascend950DT_9582 | [7, 4, 4096] · {'dtype': 'bf16', 'num_iters': 20, 'need_backward': False} | 10.033 | 10.033 | 11.518 | -12.9% | 9.109 |
| mhc_pre_sinkhorn · entry_kernel_3e8 | Ascend950DT_9582 | [8, 4, 4096] · {'dtype': 'bf16', 'num_iters': 20, 'need_backward': False} | 10.033 | 10.033 | 10.529 | -4.7% | 9.109 |
| mhc_pre_sinkhorn · entry_kernel_3e8 | Ascend950DT_9582 | [9, 4, 4096] · {'dtype': 'bf16', 'num_iters': 20, 'need_backward': False} | 10.064 | 10.064 | 10.202 | -1.3% | 9.109 |
| mhc_pre_sinkhorn · entry_kernel_3e8 | Ascend950DT_9582 | [12, 4, 4096] · {'dtype': 'bf16', 'num_iters': 20, 'need_backward': False} | 10.096 | 10.096 | 10.952 | -7.8% | 9.109 |
| mhc_pre_sinkhorn · entry_kernel_3e8 | Ascend950DT_9582 | [16, 4, 4096] · {'dtype': 'bf16', 'num_iters': 20, 'need_backward': False} | 10.158 | 10.158 | 11.560 | -12.1% | 9.109 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950DT_9582 | [20, 4, 4096] · {'num_iters': 20, 'need_backward': False} | 10.298 | 10.298 | 11.519 | -10.6% | 9.109 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950DT_9582 | [22, 4, 4096] · {'num_iters': 20, 'need_backward': False} | 10.329 | 10.329 | 11.058 | -6.6% | 9.109 |
| mhc_pre_sinkhorn · entry_kernel_3e8 | Ascend950DT_9582 | [24, 4, 4096] · {'dtype': 'bf16', 'num_iters': 20, 'need_backward': False} | 10.360 | 10.360 | 11.583 | -10.6% | 9.109 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950DT_9582 | [26, 4, 4096] · {'num_iters': 20, 'need_backward': False} | 10.391 | 10.391 | 11.182 | -7.1% | 9.109 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950DT_9582 | [30, 4, 4096] · {'num_iters': 20, 'need_backward': False} | 10.454 | 10.454 | 11.050 | -5.4% | 9.109 |
| mhc_pre_sinkhorn · entry_kernel_3e8 | Ascend950DT_9582 | [32, 4, 4096] · {'dtype': 'bf16', 'num_iters': 20, 'need_backward': False} | 10.485 | 10.485 | 11.114 | -5.7% | 9.109 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950DT_9582 | [33, 4, 4096] · {'num_iters': 20, 'need_backward': False} | 10.686 | 10.686 | 11.301 | -5.4% | 10.754 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950DT_9582 | [36, 4, 4096] · {'num_iters': 20, 'need_backward': False} | 10.718 | 10.718 | 11.522 | -7.0% | 10.754 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950DT_9582 | [48, 4, 4096] · {'num_iters': 20, 'need_backward': False} | 10.905 | 10.905 | 11.890 | -8.3% | 10.754 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950DT_9582 | [56, 4, 4096] · {'num_iters': 20, 'need_backward': False} | 11.107 | 11.107 | 11.595 | -4.2% | 12.566 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950DT_9582 | [64, 4, 4096] · {'num_iters': 20, 'need_backward': False} | 11.232 | 11.232 | 12.468 | -9.9% | 12.566 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950DT_9582 | [80, 4, 4096] · {'num_iters': 20, 'need_backward': False} | 13.651 | 13.651 | 13.782 | -0.9% | 14.378 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950PR_957b | [1, 4, 4096] · {'dtype': 'bf16', 'num_iters': 20, 'need_backward': False} | 9.990 | 9.990 | 8.763 | +14.0% | 9.109 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950PR_957b | [2, 4, 4096] · {'dtype': 'bf16', 'num_iters': 20, 'need_backward': False} | 9.990 | 9.990 | 10.198 | -2.0% | 9.109 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950PR_957b | [3, 4, 4096] · {'dtype': 'bf16', 'num_iters': 20, 'need_backward': False} | 10.038 | 10.038 | 9.250 | +8.5% | 9.109 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950PR_957b | [4, 4, 4096] · {'dtype': 'bf16', 'num_iters': 20, 'need_backward': False} | 10.038 | 10.038 | 9.470 | +6.0% | 9.109 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950PR_957b | [5, 4, 4096] · {'dtype': 'bf16', 'num_iters': 20, 'need_backward': False} | 10.086 | 10.086 | 9.495 | +6.2% | 9.109 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950PR_957b | [6, 4, 4096] · {'dtype': 'bf16', 'num_iters': 20, 'need_backward': False} | 10.086 | 10.086 | 9.405 | +7.2% | 9.109 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950PR_957b | [7, 4, 4096] · {'dtype': 'bf16', 'num_iters': 20, 'need_backward': False} | 10.134 | 10.134 | 9.388 | +7.9% | 9.109 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950PR_957b | [8, 4, 4096] · {'dtype': 'bf16', 'num_iters': 20, 'need_backward': False} | 10.134 | 10.134 | 9.713 | +4.3% | 9.109 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950PR_957b | [9, 4, 4096] · {'dtype': 'bf16', 'num_iters': 20, 'need_backward': False} | 10.182 | 10.182 | 9.390 | +8.4% | 9.109 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950PR_957b | [12, 4, 4096] · {'dtype': 'bf16', 'num_iters': 20, 'need_backward': False} | 10.230 | 10.230 | 9.686 | +5.6% | 9.109 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950PR_957b | [20, 4, 4096] · {'num_iters': 20, 'need_backward': False} | 10.539 | 10.539 | 11.231 | -6.2% | 9.109 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950PR_957b | [22, 4, 4096] · {'num_iters': 20, 'need_backward': False} | 10.587 | 10.587 | 11.103 | -4.6% | 9.109 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950PR_957b | [26, 4, 4096] · {'num_iters': 20, 'need_backward': False} | 10.683 | 10.683 | 10.893 | -1.9% | 9.109 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950PR_957b | [30, 4, 4096] · {'num_iters': 20, 'need_backward': False} | 10.779 | 10.779 | 10.511 | +2.6% | 9.109 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950PR_957b | [33, 4, 4096] · {'num_iters': 20, 'need_backward': False} | 11.084 | 11.084 | 10.855 | +2.1% | 10.754 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950PR_957b | [36, 4, 4096] · {'num_iters': 20, 'need_backward': False} | 11.132 | 11.132 | 10.701 | +4.0% | 10.754 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950PR_957b | [48, 4, 4096] · {'num_iters': 20, 'need_backward': False} | 11.421 | 11.421 | 11.271 | +1.3% | 10.754 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950PR_957b | [56, 4, 4096] · {'num_iters': 20, 'need_backward': False} | 11.776 | 11.776 | 12.215 | -3.6% | 12.566 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950PR_957b | [64, 4, 4096] · {'num_iters': 20, 'need_backward': False} | 14.022 | 14.022 | 13.152 | +6.6% | 12.566 |
| mhc_pre_sinkhorn · MhcPreSinkhorn_bd8e37c67f2294e1437f03bb2bace96c_1000_mix_aic | Ascend950PR_957b | [80, 4, 4096] · {'num_iters': 20, 'need_backward': False} | 14.616 | 14.616 | 13.653 | +7.0% | 14.378 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950DT_9582 | [1, 4] · {'dtype': 'fp32', 'num_iters': 20, 'return_intermediates': False} | 5.786 | 5.786 | 6.149 | -5.9% | 5.300 |
| mhc_sinkhorn · entry_kernel_0 | Ascend950DT_9582 | [1, 4] · {'dtype': 'fp32', 'num_iters': 20, 'return_intermediates': False} | 5.786 | 5.786 | 6.296 | -8.1% | 5.300 |
| mhc_sinkhorn · entry_kernel_0 | Ascend950DT_9582 | [2, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 2} | 3.077 | 3.077 | 3.593 | -14.4% | 2.820 |
| mhc_sinkhorn · entry_kernel_0 | Ascend950DT_9582 | [3, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 2} | 3.102 | 3.102 | 3.730 | -16.8% | 2.843 |
| mhc_sinkhorn · entry_kernel_0 | Ascend950DT_9582 | [4, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 5} | 3.578 | 3.578 | 2.945 | +21.5% | 3.279 |
| mhc_sinkhorn · entry_kernel_0 | Ascend950DT_9582 | [5, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 6} | 3.744 | 3.744 | 4.066 | -7.9% | 3.431 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950DT_9582 | [6, 4] · {'num_iters': 2, 'return_intermediates': False} | 3.144 | 3.144 | 3.580 | -12.2% | 2.882 |
| mhc_sinkhorn · entry_kernel_0 | Ascend950DT_9582 | [7, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 12} | 4.682 | 4.682 | 5.142 | -8.9% | 4.290 |
| mhc_sinkhorn · entry_kernel_0 | Ascend950DT_9582 | [8, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 10} | 4.385 | 4.385 | 4.995 | -12.2% | 4.017 |
| mhc_sinkhorn · entry_kernel_0 | Ascend950DT_9582 | [9, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 20} | 5.920 | 5.920 | 6.212 | -4.7% | 5.423 |
| mhc_sinkhorn · entry_kernel_0 | Ascend950DT_9582 | [11, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 20} | 5.932 | 5.932 | 6.406 | -7.4% | 5.435 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950DT_9582 | [14, 4] · {'num_iters': 7, 'return_intermediates': False} | 3.960 | 3.960 | 4.636 | -14.6% | 3.629 |
| mhc_sinkhorn · entry_kernel_0 | Ascend950DT_9582 | [15, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 24} | 6.563 | 6.563 | 6.967 | -5.8% | 6.012 |
| mhc_sinkhorn · entry_kernel_0 | Ascend950DT_9582 | [16, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 32} | 7.789 | 7.789 | 8.156 | -4.5% | 7.135 |
| mhc_sinkhorn · entry_kernel_0 | Ascend950DT_9582 | [17, 4] · {'dtype': 'fp32', 'num_iters': 20, 'return_intermediates': False} | 5.959 | 5.959 | 6.532 | -8.8% | 5.459 |
| mhc_sinkhorn · entry_kernel_0 | Ascend950DT_9582 | [21, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 16} | 5.361 | 5.361 | 5.597 | -4.2% | 4.911 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950DT_9582 | [28, 4] · {'num_iters': 15, 'return_intermediates': False} | 5.225 | 5.225 | 5.886 | -11.2% | 4.787 |
| mhc_sinkhorn · entry_kernel_0 | Ascend950DT_9582 | [31, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 20} | 5.996 | 5.996 | 6.333 | -5.3% | 5.493 |
| mhc_sinkhorn · entry_kernel_0 | Ascend950DT_9582 | [32, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 20} | 5.998 | 5.998 | 5.336 | +12.4% | 5.494 |
| mhc_sinkhorn · entry_kernel_0 | Ascend950DT_9582 | [33, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 20} | 6.000 | 6.000 | 6.377 | -5.9% | 5.496 |
| mhc_sinkhorn · entry_kernel_0 | Ascend950DT_9582 | [47, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 4} | 3.576 | 3.576 | 3.996 | -10.5% | 3.277 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950DT_9582 | [54, 4] · {'num_iters': 20, 'return_intermediates': False} | 6.030 | 6.030 | 7.551 | -20.1% | 5.524 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950DT_9582 | [55, 4] · {'num_iters': 32, 'return_intermediates': False} | 7.865 | 7.865 | 9.274 | -15.2% | 7.204 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950DT_9582 | [56, 4] · {'num_iters': 4, 'return_intermediates': False} | 3.587 | 3.587 | 5.068 | -29.2% | 3.287 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950DT_9582 | [57, 4] · {'num_iters': 20, 'return_intermediates': False} | 6.033 | 6.033 | 7.321 | -17.6% | 5.527 |
| mhc_sinkhorn · entry_kernel_0 | Ascend950DT_9582 | [63, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 5} | 3.747 | 3.747 | 4.101 | -8.6% | 3.434 |
| mhc_sinkhorn · entry_kernel_0 | Ascend950DT_9582 | [64, 4] · {'dtype': 'fp32', 'num_iters': 20, 'return_intermediates': False} | 6.040 | 6.040 | 7.598 | -20.5% | 5.533 |
| mhc_sinkhorn · entry_kernel_0 | Ascend950DT_9582 | [65, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 10} | 4.471 | 4.471 | 3.258 | +37.2% | 4.135 |
| mhc_sinkhorn · entry_kernel_0 | Ascend950DT_9582 | [66, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 10} | 4.471 | 4.471 | 3.212 | +39.2% | 4.136 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950DT_9582 | [68, 4] · {'num_iters': 10, 'return_intermediates': False} | 4.473 | 4.473 | 4.922 | -9.1% | 4.138 |
| mhc_sinkhorn · entry_kernel_0 | Ascend950DT_9582 | [96, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 20} | 6.023 | 6.023 | 5.334 | +12.9% | 5.556 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950DT_9582 | [113, 4] · {'num_iters': 20, 'return_intermediates': False} | 6.033 | 6.033 | 6.996 | -13.8% | 5.565 |
| mhc_sinkhorn · entry_kernel_0 | Ascend950DT_9582 | [128, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 20} | 6.040 | 6.040 | 5.475 | +10.3% | 5.533 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950DT_9582 | [193, 4] · {'num_iters': 12, 'return_intermediates': False} | 4.801 | 4.801 | 6.014 | -20.2% | 4.437 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950PR_957b | [1, 4] · {'dtype': 'fp32', 'num_iters': 20, 'return_intermediates': False} | 5.869 | 5.869 | 5.948 | -1.3% | 5.300 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950PR_957b | [2, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 2} | 3.085 | 3.085 | 3.697 | -16.5% | 2.820 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950PR_957b | [4, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 5} | 3.599 | 3.599 | 3.938 | -8.6% | 3.279 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950PR_957b | [6, 4] · {'num_iters': 2, 'return_intermediates': False} | 3.153 | 3.153 | 3.438 | -8.3% | 2.882 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950PR_957b | [8, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 10} | 4.426 | 4.426 | 4.719 | -6.2% | 4.017 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950PR_957b | [9, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 20} | 6.003 | 6.003 | 6.268 | -4.2% | 5.423 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950PR_957b | [14, 4] · {'num_iters': 7, 'return_intermediates': False} | 3.989 | 3.989 | 4.395 | -9.2% | 3.629 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950PR_957b | [16, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 32} | 7.922 | 7.922 | 8.328 | -4.9% | 7.135 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950PR_957b | [17, 4] · {'dtype': 'fp32', 'num_iters': 20, 'return_intermediates': False} | 6.042 | 6.042 | 6.280 | -3.8% | 5.459 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950PR_957b | [28, 4] · {'num_iters': 15, 'return_intermediates': False} | 5.288 | 5.288 | 5.691 | -7.1% | 4.787 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950PR_957b | [31, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 20} | 6.079 | 6.079 | 6.634 | -8.4% | 5.493 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950PR_957b | [32, 4] · {'dtype': 'fp32', 'return_intermediates': False, 'num_iters': 20} | 6.081 | 6.081 | 6.616 | -8.1% | 5.494 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950PR_957b | [54, 4] · {'num_iters': 20, 'return_intermediates': False} | 6.113 | 6.113 | 7.287 | -16.1% | 5.524 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950PR_957b | [55, 4] · {'num_iters': 32, 'return_intermediates': False} | 7.998 | 7.998 | 9.111 | -12.2% | 7.204 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950PR_957b | [56, 4] · {'num_iters': 4, 'return_intermediates': False} | 3.603 | 3.603 | 4.464 | -19.3% | 3.287 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950PR_957b | [57, 4] · {'num_iters': 20, 'return_intermediates': False} | 6.075 | 6.075 | 6.410 | -5.2% | 5.527 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950PR_957b | [64, 4] · {'dtype': 'fp32', 'num_iters': 20, 'return_intermediates': False} | 6.081 | 6.081 | 6.533 | -6.9% | 5.533 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950PR_957b | [68, 4] · {'num_iters': 10, 'return_intermediates': False} | 4.515 | 4.515 | 4.937 | -8.6% | 4.138 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950PR_957b | [113, 4] · {'num_iters': 20, 'return_intermediates': False} | 6.091 | 6.091 | 6.827 | -10.8% | 5.565 |
| mhc_sinkhorn · MhcSinkhorn_8997d9c5c22e82688b98bc25d8daa9cd_0 | Ascend950PR_957b | [193, 4] · {'num_iters': 12, 'return_intermediates': False} | 4.851 | 4.851 | 5.953 | -18.5% | 4.437 |
| moe_ffn · entry_kernel_0+entry_kernel_0 | Ascend950DT_9582 | [1, 64, 64, 1] · {'expert_counts': [1]} | 10.682 | 10.682 | 10.195 | +4.8% | 9.792 |
| moe_ffn · GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic+GroupedMatmul_037cec40685a17f15b75759e999767c3_0 | Ascend950DT_9582 | [2, 64, 64, 1] · {'expert_counts': [2]} | 10.682 | 10.682 | 10.195 | +4.8% | 9.792 |
| moe_ffn · entry_kernel_0 + entry_kernel_0 | Ascend950DT_9582 | [2, 64, 64, 1] · {'expert_counts': [2]} | 10.682 | 10.682 | 10.253 | +4.2% | 9.792 |
| moe_ffn · entry_kernel_0 + entry_kernel_0 | Ascend950DT_9582 | [2, 64, 128, 1] · {'expert_counts': [2]} | 10.967 | 10.967 | 10.825 | +1.3% | 10.053 |
| moe_ffn · entry_kernel_0+entry_kernel_0 | Ascend950DT_9582 | [3, 64, 64, 2] · {'expert_counts': [1, 2]} | 10.682 | 10.682 | 10.852 | -1.6% | 9.792 |
| moe_ffn · entry_kernel_0 + entry_kernel_0 | Ascend950DT_9582 | [4, 128, 64, 2] · {'expert_counts': [1, 3]} | 11.268 | 11.268 | 11.070 | +1.8% | 10.328 |
| moe_ffn · entry_kernel_0 + entry_kernel_0 | Ascend950DT_9582 | [6, 128, 128, 2] · {'expert_counts': [1, 5]} | 11.553 | 11.553 | 11.087 | +4.2% | 10.588 |
| moe_ffn · entry_kernel_0+entry_kernel_0 | Ascend950DT_9582 | [8, 64, 128, 1] · {'expert_counts': [8]} | 10.967 | 10.967 | 10.633 | +3.1% | 10.053 |
| moe_ffn · entry_kernel_0+entry_kernel_0 | Ascend950DT_9582 | [8, 256, 128, 2] · {'expert_counts': [1, 7]} | 13.636 | 13.636 | 11.119 | +22.6% | 12.490 |
| moe_ffn · entry_kernel_0+entry_kernel_0 | Ascend950DT_9582 | [16, 128, 128, 2] · {'expert_counts': [1, 15]} | 11.553 | 11.553 | 11.209 | +3.1% | 10.587 |
| moe_ffn · entry_kernel_0+entry_kernel_0 | Ascend950DT_9582 | [16, 256, 256, 1] · {'expert_counts': [16]} | 14.457 | 14.457 | 11.753 | +23.0% | 13.240 |
| moe_ffn · entry_kernel_0+entry_kernel_0 | Ascend950DT_9582 | [17, 128, 64, 2] · {'expert_counts': [8, 9]} | 11.268 | 11.268 | 11.335 | -0.6% | 10.327 |
| moe_ffn · entry_kernel_0+entry_kernel_0 | Ascend950DT_9582 | [32, 128, 128, 4] · {'expert_counts': [2, 6, 8, 16]} | 11.553 | 11.553 | 11.571 | -0.2% | 10.585 |
| moe_ffn · entry_kernel_0+entry_kernel_0 | Ascend950DT_9582 | [32, 256, 128, 2] · {'expert_counts': [1, 31]} | 13.636 | 13.636 | 12.119 | +12.5% | 12.489 |
| moe_ffn · entry_kernel_0+entry_kernel_0 | Ascend950DT_9582 | [64, 128, 256, 4] · {'expert_counts': [8, 8, 16, 32]} | 12.374 | 12.374 | 12.496 | -1.0% | 11.327 |
| moe_ffn · GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic+GroupedMatmul_037cec40685a17f15b75759e999767c3_0 | Ascend950PR_957b | [1, 64, 64, 1] · {'expert_counts': [1]} | 10.682 | 10.682 | 9.995 | +6.9% | 9.792 |
| moe_ffn · GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic+GroupedMatmul_037cec40685a17f15b75759e999767c3_0 | Ascend950PR_957b | [2, 64, 64, 1] · {'expert_counts': [2]} | 10.682 | 10.682 | 9.992 | +6.9% | 9.792 |
| moe_ffn · GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic+GroupedMatmul_037cec40685a17f15b75759e999767c3_0 | Ascend950PR_957b | [2, 64, 128, 1] · {'expert_counts': [2]} | 10.968 | 10.968 | 9.819 | +11.7% | 10.053 |
| moe_ffn · GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic+GroupedMatmul_037cec40685a17f15b75759e999767c3_0 | Ascend950PR_957b | [3, 64, 64, 2] · {'expert_counts': [1, 2]} | 10.682 | 10.682 | 10.267 | +4.0% | 9.792 |
| moe_ffn · GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic+GroupedMatmul_037cec40685a17f15b75759e999767c3_0 | Ascend950PR_957b | [4, 128, 64, 2] · {'expert_counts': [1, 3]} | 11.272 | 11.272 | 10.727 | +5.1% | 10.328 |
| moe_ffn · GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic+GroupedMatmul_037cec40685a17f15b75759e999767c3_0 | Ascend950PR_957b | [6, 128, 128, 2] · {'expert_counts': [1, 5]} | 11.564 | 11.564 | 10.842 | +6.7% | 10.588 |
| moe_ffn · GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic+GroupedMatmul_037cec40685a17f15b75759e999767c3_0 | Ascend950PR_957b | [8, 64, 128, 1] · {'expert_counts': [8]} | 10.968 | 10.968 | 10.324 | +6.2% | 10.053 |
| moe_ffn · GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic+GroupedMatmul_037cec40685a17f15b75759e999767c3_0 | Ascend950PR_957b | [8, 256, 128, 2] · {'expert_counts': [1, 7]} | 13.666 | 13.666 | 11.056 | +23.6% | 12.490 |
| moe_ffn · GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic+GroupedMatmul_037cec40685a17f15b75759e999767c3_0 | Ascend950PR_957b | [16, 128, 128, 2] · {'expert_counts': [1, 15]} | 11.564 | 11.564 | 11.053 | +4.6% | 10.587 |
| moe_ffn · GroupedMatmulSwigluQuantV2_201ac99772cea936e22117f5ac241d38_0_mix_aic+GroupedMatmul_037cec40685a17f15b75759e999767c3_0 | Ascend950PR_957b | [17, 128, 64, 2] · {'expert_counts': [8, 9]} | 11.272 | 11.272 | 11.022 | +2.3% | 10.327 |
| moe_gating_top_k · MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005 | Ascend950DT_9582 | [1, 128, 8] · {} | 3.825 | 3.825 | 3.822 | +0.1% | 3.506 |
| moe_gating_top_k · MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005 | Ascend950DT_9582 | [1, 384, 6] · {} | 4.330 | 4.330 | 4.239 | +2.1% | 3.968 |
| moe_gating_top_k · MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005 | Ascend950DT_9582 | [1, 896, 16] · {} | 5.340 | 5.340 | 5.344 | -0.1% | 4.893 |
| moe_gating_top_k · MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005 | Ascend950DT_9582 | [2, 896, 16] · {} | 5.340 | 5.340 | 5.472 | -2.4% | 4.893 |
| moe_gating_top_k · MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005 | Ascend950DT_9582 | [4, 256, 8] · {} | 4.078 | 4.078 | 4.865 | -16.2% | 3.737 |
| moe_gating_top_k · MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005 | Ascend950DT_9582 | [7, 896, 16] · {} | 5.340 | 5.340 | 5.576 | -4.2% | 4.893 |
| moe_gating_top_k · MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005 | Ascend950DT_9582 | [9, 384, 8] · {} | 4.330 | 4.330 | 4.717 | -8.2% | 3.968 |
| moe_gating_top_k · MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005 | Ascend950DT_9582 | [12, 896, 16] · {} | 5.340 | 5.340 | 6.121 | -12.8% | 4.893 |
| moe_gating_top_k · MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005 | Ascend950DT_9582 | [16, 512, 8] · {} | 4.583 | 4.583 | 5.278 | -13.2% | 4.200 |
| moe_gating_top_k · MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005 | Ascend950DT_9582 | [24, 896, 16] · {} | 5.340 | 5.340 | 5.915 | -9.7% | 4.893 |
| moe_gating_top_k · MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005 | Ascend950DT_9582 | [33, 384, 8] · {} | 4.330 | 4.330 | 4.903 | -11.7% | 3.968 |
| moe_gating_top_k · MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005 | Ascend950DT_9582 | [65, 128, 8] · {} | 4.162 | 4.162 | 4.598 | -9.5% | 3.506 |
| moe_gating_top_k · MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005 | Ascend950PR_957b | [1, 384, 6] · {} | 4.353 | 4.353 | 4.094 | +6.3% | 3.968 |
| moe_gating_top_k · MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005 | Ascend950PR_957b | [2, 896, 16] · {} | 5.391 | 5.391 | 5.446 | -1.0% | 4.893 |
| moe_gating_top_k · MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005 | Ascend950PR_957b | [4, 256, 8] · {} | 4.094 | 4.094 | 4.410 | -7.2% | 3.737 |
| moe_gating_top_k · MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005 | Ascend950PR_957b | [7, 896, 16] · {} | 5.391 | 5.391 | 5.567 | -3.2% | 4.893 |
| moe_gating_top_k · MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005 | Ascend950PR_957b | [9, 384, 8] · {} | 4.353 | 4.353 | 4.535 | -4.0% | 3.968 |
| moe_gating_top_k · MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005 | Ascend950PR_957b | [12, 896, 16] · {} | 5.391 | 5.391 | 5.720 | -5.8% | 4.893 |
| moe_gating_top_k · MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005 | Ascend950PR_957b | [16, 512, 8] · {} | 4.612 | 4.612 | 4.861 | -5.1% | 4.200 |
| moe_gating_top_k · MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005 | Ascend950PR_957b | [24, 896, 16] · {} | 5.391 | 5.391 | 5.682 | -5.1% | 4.893 |
| moe_gating_top_k · MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005 | Ascend950PR_957b | [33, 384, 8] · {} | 4.353 | 4.353 | 4.831 | -9.9% | 3.968 |
| moe_gating_top_k · MoeGatingTopK_4ca391a7fc5a94c6f1e7d38c9bb0fa20_10005 | Ascend950PR_957b | [65, 128, 8] · {} | 4.180 | 4.180 | 4.698 | -11.0% | 3.506 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [1, 64] · {} | 4.231 | 4.231 | 4.326 | -2.2% | 3.692 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [1, 128] · {} | 4.231 | 4.231 | 4.325 | -2.2% | 3.693 |
| quant · entry_kernel_21 | Ascend950DT_9582 | [1, 192] · {} | 4.231 | 4.231 | 4.129 | +2.5% | 3.693 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [1, 256] · {} | 4.231 | 4.231 | 4.493 | -5.8% | 3.693 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [1, 704] · {} | 4.630 | 4.630 | 4.332 | +6.9% | 4.076 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [1, 4096] · {} | 5.430 | 5.430 | 5.686 | -4.5% | 4.532 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [2, 832] · {} | 5.098 | 5.098 | 4.819 | +5.8% | 4.459 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [3, 1920] · {} | 5.430 | 5.430 | 5.788 | -6.2% | 4.657 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [4, 256] · {} | 4.759 | 4.759 | 4.588 | +3.7% | 4.268 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [4, 256] · {} | 4.759 | 4.759 | 4.620 | +3.0% | 4.268 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [4, 1472] · {} | 5.430 | 5.430 | 5.685 | -4.5% | 4.722 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [5, 320] · {} | 5.214 | 5.214 | 4.662 | +11.8% | 4.842 |
| quant · entry_kernel_21 | Ascend950DT_9582 | [7, 320] · {} | 5.393 | 5.393 | 5.146 | +4.8% | 4.987 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [7, 4480] · {} | 5.430 | 5.430 | 5.748 | -5.5% | 5.840 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [8, 384] · {} | 5.430 | 5.430 | 5.760 | -5.7% | 4.871 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [9, 4096] · {} | 5.430 | 5.430 | 5.465 | -0.6% | 6.127 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [9, 5120] · {} | 4.980 | 4.980 | 4.867 | +2.3% | 7.084 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [11, 3136] · {} | 5.430 | 5.430 | 5.507 | -1.4% | 6.353 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [17, 512] · {} | 5.430 | 5.430 | 5.790 | -6.2% | 4.876 |
| quant · entry_kernel_21 | Ascend950DT_9582 | [17, 1024] · {} | 5.430 | 5.430 | 5.772 | -5.9% | 5.982 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [19, 1024] · {} | 5.430 | 5.430 | 6.230 | -12.8% | 6.256 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [26, 704] · {} | 5.430 | 5.430 | 5.640 | -3.7% | 6.430 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [27, 576] · {} | 5.430 | 5.430 | 6.130 | -11.4% | 6.561 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [30, 768] · {} | 5.430 | 5.430 | 5.472 | -0.8% | 6.734 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [33, 768] · {} | 5.430 | 5.430 | 5.210 | +4.2% | 7.042 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [36, 512] · {} | 5.430 | 5.430 | 5.646 | -3.8% | 6.119 |
| quant · entry_kernel_21 | Ascend950DT_9582 | [36, 3072] · {} | 5.156 | 5.156 | 4.971 | +3.7% | 7.373 |
| quant · entry_kernel_21 | Ascend950DT_9582 | [36, 5120] · {} | 5.292 | 5.292 | 5.584 | -5.2% | 4.769 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [53, 1984] · {} | 5.156 | 5.156 | 5.616 | -8.2% | 7.506 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [55, 1152] · {} | 5.180 | 5.180 | 5.900 | -12.2% | 6.542 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [56, 4096] · {} | 5.430 | 5.430 | 6.285 | -13.6% | 4.581 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [57, 3200] · {} | 5.331 | 5.331 | 4.945 | +7.8% | 4.970 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [59, 2880] · {} | 5.292 | 5.292 | 5.651 | -6.4% | 5.152 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [64, 1024] · {} | 5.214 | 5.214 | 5.532 | -5.7% | 5.870 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [65, 256] · {} | 5.430 | 5.430 | 4.898 | +10.9% | 5.846 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [65, 2624] · {} | 5.194 | 5.194 | 5.306 | -2.1% | 4.942 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [97, 2752] · {} | 5.316 | 5.316 | 5.279 | +0.7% | 4.781 |
| quant · entry_kernel_21 | Ascend950DT_9582 | [128, 5120] · {} | 5.794 | 5.794 | 5.885 | -1.5% | 4.904 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [144, 5120] · {} | 5.886 | 5.886 | 5.905 | -0.3% | 4.728 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [144, 7168] · {} | 6.207 | 6.207 | 6.332 | -2.0% | 5.022 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [255, 5440] · {} | 6.522 | 6.522 | 5.747 | +13.5% | 5.612 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [257, 4416] · {} | 6.528 | 6.528 | 6.467 | +0.9% | 5.083 |
| quant · entry_kernel_21 | Ascend950DT_9582 | [288, 7168] · {} | 7.181 | 7.181 | 6.876 | +4.4% | 5.526 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [384, 4096] · {} | 6.695 | 6.695 | 6.673 | +0.3% | 5.091 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [448, 6592] · {} | 6.726 | 6.726 | 6.300 | +6.8% | 6.381 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [512, 256] · {} | 5.430 | 5.430 | 5.208 | +4.3% | 4.734 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [512, 5120] · {} | 6.364 | 6.364 | 6.299 | +1.0% | 5.727 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [513, 5504] · {} | 6.845 | 6.845 | 6.751 | +1.4% | 6.009 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [576, 4096] · {} | 7.470 | 7.470 | 6.948 | +7.5% | 5.440 |
| quant · entry_kernel_21 | Ascend950DT_9582 | [576, 7168] · {} | 7.577 | 7.577 | 7.758 | -2.3% | 6.553 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [640, 4096] · {} | 6.364 | 6.364 | 6.218 | +2.4% | 5.727 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [641, 5120] · {} | 7.001 | 7.001 | 7.244 | -3.3% | 6.040 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [768, 4096] · {} | 6.743 | 6.743 | 7.202 | -6.4% | 5.984 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [1023, 7168] · {} | 9.811 | 9.811 | 9.311 | +5.4% | 6.713 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [1024, 4096] · {} | 7.502 | 7.502 | 8.047 | -6.8% | 6.352 |
| quant · entry_kernel_21 | Ascend950DT_9582 | [1024, 6144] · {} | 8.935 | 8.935 | 8.704 | +2.7% | 6.245 |
| quant · entry_kernel_21 | Ascend950DT_9582 | [1024, 7168] · {} | 9.586 | 9.586 | 9.332 | +2.7% | 6.713 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [1152, 5120] · {} | 8.942 | 8.942 | 8.684 | +3.0% | 5.893 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950DT_9582 | [2048, 5120] · {} | 11.531 | 11.531 | 11.364 | +1.5% | 7.733 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [1, 64] · {} | 3.591 | 3.591 | 3.946 | -9.0% | 3.692 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [1, 128] · {} | 3.592 | 3.592 | 3.828 | -6.2% | 3.693 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [1, 192] · {} | 3.592 | 3.592 | 3.833 | -6.3% | 3.693 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [1, 256] · {} | 3.592 | 3.592 | 3.859 | -6.9% | 3.693 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [1, 704] · {} | 4.010 | 4.010 | 3.964 | +1.2% | 4.076 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [1, 4096] · {} | 4.507 | 4.507 | 4.945 | -8.9% | 4.532 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [2, 832] · {} | 4.428 | 4.428 | 4.421 | +0.2% | 4.459 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [3, 1920] · {} | 4.643 | 4.643 | 5.315 | -12.6% | 4.657 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [4, 256] · {} | 4.219 | 4.219 | 4.157 | +1.5% | 4.268 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [4, 1472] · {} | 4.715 | 4.715 | 5.398 | -12.7% | 4.722 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [5, 320] · {} | 4.846 | 4.846 | 4.592 | +5.5% | 4.842 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [7, 320] · {} | 5.004 | 5.004 | 5.042 | -0.8% | 4.987 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [7, 4480] · {} | 4.865 | 4.865 | 4.559 | +6.7% | 5.840 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [8, 384] · {} | 4.877 | 4.877 | 5.136 | -5.0% | 4.871 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [9, 4096] · {} | 5.072 | 5.072 | 4.825 | +5.1% | 6.127 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [9, 5120] · {} | 4.377 | 4.377 | 5.025 | -12.9% | 7.084 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [11, 3136] · {} | 5.319 | 5.319 | 4.986 | +6.7% | 6.353 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [17, 512] · {} | 4.882 | 4.882 | 5.432 | -10.1% | 4.876 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [17, 1024] · {} | 4.966 | 4.966 | 5.311 | -6.5% | 5.982 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [19, 1024] · {} | 5.102 | 5.102 | 5.983 | -14.7% | 6.256 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [26, 704] · {} | 5.636 | 5.636 | 5.452 | +3.4% | 6.430 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [27, 576] · {} | 5.711 | 5.711 | 5.547 | +3.0% | 6.561 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [30, 768] · {} | 4.770 | 4.770 | 4.917 | -3.0% | 6.734 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [33, 768] · {} | 4.845 | 4.845 | 4.915 | -1.4% | 7.042 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [36, 512] · {} | 4.906 | 4.906 | 5.795 | -15.3% | 6.119 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [36, 3072] · {} | 4.448 | 4.448 | 5.170 | -14.0% | 7.373 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [36, 5120] · {} | 4.547 | 4.547 | 5.052 | -10.0% | 4.769 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [53, 1984] · {} | 4.657 | 4.657 | 5.105 | -8.8% | 7.506 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [55, 1152] · {} | 4.873 | 4.873 | 5.336 | -8.7% | 6.542 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [56, 4096] · {} | 4.554 | 4.554 | 4.513 | +0.9% | 4.581 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [57, 3200] · {} | 4.776 | 4.776 | 4.430 | +7.8% | 4.970 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [59, 2880] · {} | 4.734 | 4.734 | 4.695 | +0.8% | 5.152 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [64, 1024] · {} | 4.599 | 4.599 | 5.373 | -14.4% | 5.870 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [65, 256] · {} | 4.757 | 4.757 | 4.609 | +3.2% | 5.846 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [65, 2624] · {} | 4.753 | 4.753 | 4.736 | +0.4% | 4.942 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [97, 2752] · {} | 4.831 | 4.831 | 5.107 | -5.4% | 4.781 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [128, 5120] · {} | 5.192 | 5.192 | 5.730 | -9.4% | 4.904 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [144, 5120] · {} | 5.180 | 5.180 | 5.984 | -13.4% | 4.728 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [144, 7168] · {} | 5.434 | 5.434 | 6.415 | -15.3% | 5.022 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [255, 5440] · {} | 6.694 | 6.694 | 6.868 | -2.5% | 5.612 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [257, 4416] · {} | 6.069 | 6.069 | 6.130 | -1.0% | 5.083 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [288, 7168] · {} | 7.203 | 7.203 | 8.009 | -10.1% | 5.526 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [384, 4096] · {} | 6.397 | 6.397 | 7.332 | -12.8% | 5.091 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [448, 6592] · {} | 7.560 | 7.560 | 8.222 | -8.0% | 6.381 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [512, 256] · {} | 4.667 | 4.667 | 5.090 | -8.3% | 4.734 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [512, 5120] · {} | 6.870 | 6.870 | 7.690 | -10.7% | 5.727 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [513, 5504] · {} | 6.845 | 6.845 | 8.230 | -16.8% | 6.009 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [576, 4096] · {} | 6.425 | 6.425 | 7.692 | -16.5% | 5.440 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [576, 7168] · {} | 9.614 | 9.614 | 10.007 | -3.9% | 6.553 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [640, 4096] · {} | 6.912 | 6.912 | 8.108 | -14.8% | 5.727 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [641, 5120] · {} | 7.813 | 7.813 | 9.092 | -14.1% | 6.040 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [1023, 7168] · {} | 14.026 | 14.026 | 14.080 | -0.4% | 6.713 |
| quant · DynamicMxQuant_bf16toe4m3_high_performance_33_mix_aiv | Ascend950PR_957b | [2048, 5120] · {} | 17.834 | 17.834 | 18.504 | -3.6% | 7.733 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950DT_9582 | [1, 12] · {} | 2.689 | 2.689 | 2.724 | -1.3% | 2.465 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950DT_9582 | [3, 96] · {} | 2.722 | 2.722 | 3.035 | -10.3% | 2.495 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950DT_9582 | [9, 96] · {} | 2.790 | 2.790 | 3.032 | -8.0% | 2.558 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950DT_9582 | [72, 12] · {} | 2.790 | 2.790 | 3.114 | -10.4% | 2.558 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950DT_9582 | [129, 12] · {} | 2.925 | 2.925 | 2.926 | -0.0% | 2.681 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950DT_9582 | [300, 12] · {} | 3.042 | 3.042 | 3.085 | -1.4% | 2.789 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950DT_9582 | [2048, 12] · {} | 3.345 | 3.345 | 3.176 | +5.3% | 3.066 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950DT_9582 | [3072, 96] · {} | 3.938 | 3.938 | 3.621 | +8.7% | 3.477 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950DT_9582 | [3073, 12] · {} | 3.418 | 3.418 | 3.066 | +11.5% | 3.133 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950DT_9582 | [4096, 24] · {} | 3.571 | 3.571 | 3.425 | +4.3% | 3.280 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950DT_9582 | [4097, 96] · {} | 4.171 | 4.171 | 3.696 | +12.9% | 3.581 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950DT_9582 | [4545, 24] · {} | 3.604 | 3.604 | 3.267 | +10.3% | 3.297 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950DT_9582 | [16384, 48] · {} | 4.854 | 4.854 | 4.855 | -0.0% | 3.929 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950DT_9582 | [16385, 48] · {} | 4.910 | 4.910 | 4.769 | +3.0% | 3.929 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950DT_9582 | [32768, 12] · {} | 4.121 | 4.121 | 3.698 | +11.4% | 3.581 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950DT_9582 | [65537, 24] · {} | 5.768 | 5.768 | 5.867 | -1.7% | 4.657 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950DT_9582 | [98304, 12] · {} | 5.476 | 5.476 | 5.523 | -0.9% | 4.317 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950DT_9582 | [131072, 12] · {} | 5.709 | 5.709 | 5.987 | -4.6% | 4.657 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950DT_9582 | [262144, 12] · {} | 8.002 | 8.002 | 7.759 | +3.1% | 5.553 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950DT_9582 | [524288, 12] · {} | 11.908 | 11.908 | 11.230 | +6.0% | 7.556 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950DT_9582 | [1048576, 12] · {} | 19.720 | 19.720 | 18.553 | +6.3% | 11.321 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950DT_9582 | [1572864, 12] · {} | 30.140 | 30.140 | 28.561 | +5.5% | 15.408 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950DT_9582 | [2097152, 12] · {} | 45.897 | 45.897 | 42.686 | +7.5% | 19.353 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950PR_957b | [3, 96] · {} | 2.723 | 2.723 | 2.628 | +3.6% | 2.495 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950PR_957b | [9, 96] · {} | 2.793 | 2.793 | 2.658 | +5.1% | 2.558 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950PR_957b | [72, 12] · {} | 2.793 | 2.793 | 2.672 | +4.5% | 2.558 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950PR_957b | [129, 12] · {} | 2.928 | 2.928 | 2.716 | +7.8% | 2.681 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950PR_957b | [2048, 12] · {} | 3.347 | 3.347 | 2.793 | +19.8% | 3.066 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950PR_957b | [3073, 12] · {} | 3.420 | 3.420 | 2.932 | +16.7% | 3.133 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950PR_957b | [4096, 24] · {} | 3.587 | 3.587 | 3.187 | +12.5% | 3.280 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950PR_957b | [4097, 96] · {} | 4.338 | 4.338 | 4.248 | +2.1% | 3.581 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950PR_957b | [4545, 24] · {} | 3.607 | 3.607 | 3.178 | +13.5% | 3.297 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950PR_957b | [16384, 48] · {} | 6.015 | 6.015 | 4.823 | +24.7% | 3.929 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950PR_957b | [16385, 48] · {} | 6.015 | 6.015 | 5.723 | +5.1% | 3.929 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950PR_957b | [32768, 12] · {} | 4.338 | 4.338 | 4.128 | +5.1% | 3.581 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950PR_957b | [65537, 24] · {} | 8.114 | 8.114 | 7.585 | +7.0% | 4.657 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950PR_957b | [98304, 12] · {} | 6.685 | 6.685 | 6.867 | -2.7% | 4.317 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950PR_957b | [131072, 12] · {} | 8.114 | 8.114 | 7.882 | +2.9% | 4.657 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950PR_957b | [262144, 12] · {} | 12.344 | 12.344 | 11.428 | +8.0% | 5.553 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950PR_957b | [524288, 12] · {} | 21.815 | 21.815 | 19.424 | +12.3% | 7.556 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950PR_957b | [1048576, 12] · {} | 39.630 | 39.630 | 34.285 | +15.6% | 11.321 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950PR_957b | [1572864, 12] · {} | 69.041 | 69.041 | 54.379 | +27.0% | 15.408 |
| sigmoid_fp32 · Sigmoid_5c5e58e043224cccf51a06108010ae13_high_precision_300000000000100 | Ascend950PR_957b | [2097152, 12] · {} | 104.717 | 104.717 | 88.593 | +18.2% | 19.353 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950DT_9582 | [1, 1, 4, 512] · {'implementation': 'vllm_hc_post'} | 3.184 | 3.184 | 3.364 | -5.3% | 2.918 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950DT_9582 | [1, 1, 4, 1024] · {'implementation': 'vllm_hc_post'} | 3.404 | 3.404 | 3.479 | -2.2% | 3.120 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950DT_9582 | [1, 1, 4, 16384] · {'implementation': 'vllm_hc_post'} | 8.661 | 8.661 | 8.379 | +3.4% | 7.933 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950DT_9582 | [1, 2, 4, 1280] · {'implementation': 'vllm_hc_post'} | 3.531 | 3.531 | 3.570 | -1.1% | 3.236 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950DT_9582 | [1, 2, 4, 8192] · {'implementation': 'vllm_hc_post'} | 6.058 | 6.058 | 5.795 | +4.5% | 5.550 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950DT_9582 | [1, 2, 4, 16448] · {'implementation': 'vllm_hc_post'} | 8.869 | 8.869 | 8.636 | +2.7% | 8.124 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950DT_9582 | [1, 3, 4, 5120] · {'implementation': 'vllm_hc_post'} | 5.084 | 5.084 | 5.122 | -0.7% | 4.658 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950DT_9582 | [1, 4, 4, 4080] · {'implementation': 'vllm_hc_post'} | 4.771 | 4.771 | 4.743 | +0.6% | 4.372 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950DT_9582 | [1, 5, 4, 7168] · {'implementation': 'vllm_hc_post'} | 5.779 | 5.779 | 5.729 | +0.9% | 5.295 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950DT_9582 | [1, 8, 4, 4096] · {'implementation': 'vllm_hc_post'} | 4.844 | 4.844 | 4.813 | +0.6% | 4.439 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950DT_9582 | [1, 9, 4, 4160] · {'implementation': 'vllm_hc_post'} | 5.060 | 5.060 | 4.956 | +2.1% | 4.637 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950DT_9582 | [1, 33, 4, 7168] · {'implementation': 'vllm_hc_post'} | 6.265 | 6.265 | 6.101 | +2.7% | 5.740 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950DT_9582 | [1, 57, 4, 192] · {'implementation': 'vllm_hc_post'} | 4.019 | 4.019 | 3.941 | +2.0% | 3.683 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950DT_9582 | [1, 65, 4, 256] · {'implementation': 'vllm_hc_post'} | 4.168 | 4.168 | 4.202 | -0.8% | 3.836 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950DT_9582 | [1, 129, 4, 128] · {'implementation': 'vllm_hc_post'} | 4.412 | 4.412 | 4.612 | -4.3% | 4.660 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950PR_957b | [1, 1, 4, 512] · {'implementation': 'vllm_hc_post'} | 3.196 | 3.196 | 3.154 | +1.3% | 2.918 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950PR_957b | [1, 1, 4, 1024] · {'implementation': 'vllm_hc_post'} | 3.419 | 3.419 | 3.451 | -0.9% | 3.120 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950PR_957b | [1, 1, 4, 16384] · {'implementation': 'vllm_hc_post'} | 8.807 | 8.807 | 8.341 | +5.6% | 7.933 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950PR_957b | [1, 2, 4, 1280] · {'implementation': 'vllm_hc_post'} | 3.548 | 3.548 | 3.499 | +1.4% | 3.236 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950PR_957b | [1, 2, 4, 8192] · {'implementation': 'vllm_hc_post'} | 6.131 | 6.131 | 5.878 | +4.3% | 5.550 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950PR_957b | [1, 3, 4, 5120] · {'implementation': 'vllm_hc_post'} | 5.136 | 5.136 | 5.024 | +2.2% | 4.658 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950PR_957b | [1, 4, 4, 4080] · {'implementation': 'vllm_hc_post'} | 4.807 | 4.807 | 4.901 | -1.9% | 4.372 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950PR_957b | [1, 5, 4, 7168] · {'implementation': 'vllm_hc_post'} | 5.845 | 5.845 | 5.798 | +0.8% | 5.295 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950PR_957b | [1, 8, 4, 4096] · {'implementation': 'vllm_hc_post'} | 4.881 | 4.881 | 4.923 | -0.9% | 4.439 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950PR_957b | [1, 9, 4, 4160] · {'implementation': 'vllm_hc_post'} | 5.097 | 5.097 | 4.843 | +5.2% | 4.637 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950PR_957b | [1, 33, 4, 7168] · {'implementation': 'vllm_hc_post'} | 6.331 | 6.331 | 6.508 | -2.7% | 5.740 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950PR_957b | [1, 57, 4, 192] · {'implementation': 'vllm_hc_post'} | 4.011 | 4.011 | 4.024 | -0.3% | 3.683 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950PR_957b | [1, 65, 4, 256] · {'implementation': 'vllm_hc_post'} | 4.039 | 4.039 | 4.054 | -0.4% | 3.836 |
| vllm_hc_post · HcPost_d2fe6614722b27afa6f8d964c215447e_1 | Ascend950PR_957b | [1, 129, 4, 128] · {'implementation': 'vllm_hc_post'} | 4.292 | 4.292 | 4.364 | -1.7% | 4.660 |
