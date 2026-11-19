# 本版新增双端实测与外推

RMSNorm 默认列为只给 shape，H 列按目标芯片自己的流历史。KDA 计三个 kernel 的设备窗之和；HcPost 用显式 vLLM 实现的各片资源排程。970DT 全列为外推。单位均为微秒。开发回放与冻结独立数据按行标明。[既有全量逐形](../v0.22/shape-errors.md) · [本版机器可读数据](evaluation.json) · [HcPost 的数值失败和未覆盖形状](hc-timing/comparison.md)

| 算子 / 状态 | shape | 芯片 | 默认或显式模型 | 给 H 的模型 | CA | 默认误差 | 给 H 的误差 | 970DT 模型 / 给 H |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cast_bf16_fp32 archived full prefill replay | [1048576, 32] | 950DT | 36.368 | N/A | 35.535 | +2.35% | N/A | 16.542 / N/A |
| cast_bf16_fp32 archived full prefill replay | [1048576, 32] | 950PR | 88.943 | N/A | 70.711 | +25.78% | N/A | 16.542 / N/A |
| cast_bf16_fp32 archived full prefill replay | [2097152, 32] | 950DT | 99.918 | N/A | 106.757 | -6.41% | N/A | 42.536 / N/A |
| cast_bf16_fp32 archived full prefill replay | [2097152, 32] | 950PR | 232.748 | N/A | 223.493 | +4.14% | N/A | 42.536 / N/A |
| cast_bf16_fp32 archived full prefill replay | [262144, 32] | 950DT | 9.850 | N/A | 9.970 | -1.20% | N/A | 6.656 / N/A |
| cast_bf16_fp32 archived full prefill replay | [262144, 32] | 950PR | 13.046 | N/A | 14.080 | -7.34% | N/A | 6.656 / N/A |
| sigmoid_fp32 archived full prefill replay | [1048576, 12] | 950DT | 19.720 | N/A | 18.553 | +6.29% | N/A | 11.321 / N/A |
| sigmoid_fp32 archived full prefill replay | [1048576, 12] | 950PR | 39.630 | N/A | 34.285 | +15.59% | N/A | 11.321 / N/A |
| sigmoid_fp32 archived full prefill replay | [2097152, 12] | 950DT | 45.897 | N/A | 42.686 | +7.52% | N/A | 19.353 / N/A |
| sigmoid_fp32 archived full prefill replay | [2097152, 12] | 950PR | 104.717 | N/A | 88.593 | +18.20% | N/A | 19.353 / N/A |
| sigmoid_fp32 archived full prefill replay | [262144, 12] | 950DT | 8.002 | N/A | 7.759 | +3.14% | N/A | 5.553 / N/A |
| sigmoid_fp32 archived full prefill replay | [262144, 12] | 950PR | 12.344 | N/A | 11.428 | +8.01% | N/A | 5.553 / N/A |
| cast_bf16_fp32 archived full prefill replay | [262144, 64] | 950DT | 16.592 | N/A | 17.041 | -2.64% | N/A | 9.951 / N/A |
| cast_bf16_fp32 archived full prefill replay | [262144, 64] | 950PR | 25.030 | N/A | 23.762 | +5.34% | N/A | 9.951 / N/A |
| fla_26_9_1_v2 frozen independent | [1, 3, 1, 1, 128, 128] | 950DT | 22.076 | N/A | 23.422 | -5.75% | N/A | 20.235 / N/A |
| fla_26_9_1_v2 frozen independent | [1, 3, 1, 1, 128, 128] | 950PR | 22.105 | N/A | 22.270 | -0.74% | N/A | 20.235 / N/A |
| fla_26_9_1_v2 frozen independent | [1, 5, 2, 2, 128, 128] | 950DT | 22.275 | N/A | 22.822 | -2.40% | N/A | 20.417 / N/A |
| fla_26_9_1_v2 frozen independent | [1, 5, 2, 2, 128, 128] | 950PR | 22.306 | N/A | 22.397 | -0.41% | N/A | 20.417 / N/A |
| fla_26_9_1_v2 frozen independent | [1, 16, 1, 1, 128, 128] | 950DT | 23.372 | N/A | 24.125 | -3.12% | N/A | 21.422 / N/A |
| fla_26_9_1_v2 frozen independent | [1, 16, 1, 1, 128, 128] | 950PR | 23.416 | N/A | 23.421 | -0.02% | N/A | 21.422 / N/A |
| fla_26_9_1_v2 frozen independent | [1, 17, 1, 1, 128, 128] | 950DT | 23.724 | N/A | 24.547 | -3.35% | N/A | 21.745 / N/A |
| fla_26_9_1_v2 frozen independent | [1, 17, 1, 1, 128, 128] | 950PR | 23.772 | N/A | 23.802 | -0.12% | N/A | 21.745 / N/A |
| fla_26_9_1_v2 frozen independent | [1, 31, 2, 2, 128, 128] | 950DT | 25.130 | N/A | 25.276 | -0.57% | N/A | 23.033 / N/A |
| fla_26_9_1_v2 frozen independent | [1, 31, 2, 2, 128, 128] | 950PR | 25.195 | N/A | 24.796 | +1.61% | N/A | 23.033 / N/A |
| fla_26_9_1_v2 frozen independent | [1, 32, 1, 1, 128, 128] | 950DT | 25.231 | N/A | 24.936 | +1.18% | N/A | 23.126 / N/A |
| fla_26_9_1_v2 frozen independent | [1, 32, 1, 1, 128, 128] | 950PR | 25.297 | N/A | 24.924 | +1.50% | N/A | 23.126 / N/A |
| fla_26_9_1_v2 frozen independent | [1, 33, 2, 2, 128, 128] | 950DT | 26.144 | N/A | 25.521 | +2.44% | N/A | 23.962 / N/A |
| fla_26_9_1_v2 frozen independent | [1, 33, 2, 2, 128, 128] | 950PR | 26.216 | N/A | 25.819 | +1.54% | N/A | 23.962 / N/A |
| fla_26_9_1_v2 frozen independent | [1, 64, 2, 2, 128, 128] | 950DT | 29.931 | N/A | 28.839 | +3.79% | N/A | 27.431 / N/A |
| fla_26_9_1_v2 frozen independent | [1, 64, 2, 2, 128, 128] | 950PR | 30.047 | N/A | 28.999 | +3.61% | N/A | 27.431 / N/A |
| fla_26_9_1_v2 frozen independent | [1, 65, 2, 2, 128, 128] | 950DT | 32.630 | N/A | 30.795 | +5.96% | N/A | 29.905 / N/A |
| fla_26_9_1_v2 frozen independent | [1, 65, 2, 2, 128, 128] | 950PR | 32.750 | N/A | 31.164 | +5.09% | N/A | 29.905 / N/A |
| fla_26_9_1_v2 frozen independent | [1, 129, 3, 3, 128, 128] | 950DT | 36.814 | N/A | 34.128 | +7.87% | N/A | 33.739 / N/A |
| fla_26_9_1_v2 frozen independent | [1, 129, 3, 3, 128, 128] | 950PR | 36.941 | N/A | 34.527 | +6.99% | N/A | 33.739 / N/A |
| vllm_hc_post development | [1, 1, 4, 1024] | 950DT | 3.404 | N/A | 3.479 | -2.15% | N/A | 3.120 / N/A |
| vllm_hc_post development | [1, 3, 4, 5120] | 950DT | 5.084 | N/A | 5.122 | -0.73% | N/A | 4.658 / N/A |
| vllm_hc_post development | [1, 5, 4, 7168] | 950DT | 5.779 | N/A | 5.729 | +0.88% | N/A | 5.295 / N/A |
| vllm_hc_post development | [1, 1, 4, 1024] | 950PR | 3.419 | N/A | 3.451 | -0.92% | N/A | 3.120 / N/A |
| vllm_hc_post development | [1, 3, 4, 5120] | 950PR | 5.136 | N/A | 5.024 | +2.22% | N/A | 4.658 / N/A |
| vllm_hc_post development | [1, 5, 4, 7168] | 950PR | 5.845 | N/A | 5.798 | +0.82% | N/A | 5.295 / N/A |
| vllm_hc_post development | [1, 2, 4, 16448] | 950DT | 8.869 | N/A | 8.636 | +2.70% | N/A | 8.124 / N/A |
| vllm_hc_post development | [1, 65, 4, 256] | 950DT | 4.168 | N/A | 4.202 | -0.83% | N/A | 3.836 / N/A |
| vllm_hc_post development | [1, 65, 4, 256] | 950PR | 4.039 | N/A | 4.054 | -0.37% | N/A | 3.836 / N/A |
| vllm_hc_post independent | [1, 1, 4, 512] | 950DT | 3.184 | N/A | 3.364 | -5.33% | N/A | 2.918 / N/A |
| vllm_hc_post independent | [1, 1, 4, 512] | 950PR | 3.196 | N/A | 3.154 | +1.33% | N/A | 2.918 / N/A |
| vllm_hc_post independent | [1, 2, 4, 1280] | 950DT | 3.531 | N/A | 3.570 | -1.10% | N/A | 3.236 / N/A |
| vllm_hc_post independent | [1, 2, 4, 1280] | 950PR | 3.548 | N/A | 3.499 | +1.41% | N/A | 3.236 / N/A |
| vllm_hc_post independent | [1, 4, 4, 4080] | 950DT | 4.771 | N/A | 4.743 | +0.59% | N/A | 4.372 / N/A |
| vllm_hc_post independent | [1, 4, 4, 4080] | 950PR | 4.807 | N/A | 4.901 | -1.90% | N/A | 4.372 / N/A |
| vllm_hc_post independent | [1, 8, 4, 4096] | 950DT | 4.844 | N/A | 4.813 | +0.64% | N/A | 4.439 / N/A |
| vllm_hc_post independent | [1, 8, 4, 4096] | 950PR | 4.881 | N/A | 4.923 | -0.86% | N/A | 4.439 / N/A |
| vllm_hc_post independent | [1, 33, 4, 7168] | 950DT | 6.265 | N/A | 6.101 | +2.69% | N/A | 5.740 / N/A |
| vllm_hc_post independent | [1, 33, 4, 7168] | 950PR | 6.331 | N/A | 6.508 | -2.72% | N/A | 5.740 / N/A |
| vllm_hc_post independent | [1, 2, 4, 8192] | 950DT | 6.058 | N/A | 5.795 | +4.53% | N/A | 5.550 / N/A |
| vllm_hc_post independent | [1, 2, 4, 8192] | 950PR | 6.131 | N/A | 5.878 | +4.29% | N/A | 5.550 / N/A |
| vllm_hc_post independent | [1, 1, 4, 16384] | 950DT | 8.661 | N/A | 8.379 | +3.36% | N/A | 7.933 / N/A |
| vllm_hc_post independent | [1, 1, 4, 16384] | 950PR | 8.807 | N/A | 8.341 | +5.59% | N/A | 7.933 / N/A |
| vllm_hc_post independent | [1, 57, 4, 192] | 950DT | 4.019 | N/A | 3.941 | +1.97% | N/A | 3.683 / N/A |
| vllm_hc_post independent | [1, 57, 4, 192] | 950PR | 4.011 | N/A | 4.024 | -0.33% | N/A | 3.683 / N/A |
| vllm_hc_post independent | [1, 129, 4, 128] | 950DT | 4.412 | N/A | 4.612 | -4.32% | N/A | 4.660 / N/A |
| vllm_hc_post independent | [1, 129, 4, 128] | 950PR | 4.292 | N/A | 4.364 | -1.67% | N/A | 4.660 / N/A |
| vllm_hc_post supplement | [1, 9, 4, 4160] | 950DT | 5.060 | N/A | 4.956 | +2.09% | N/A | 4.637 / N/A |
| vllm_hc_post supplement | [1, 9, 4, 4160] | 950PR | 5.097 | N/A | 4.843 | +5.23% | N/A | 4.637 / N/A |
| rmsnorm entry H=0 | [563, 4048] | 950DT | 6.223 | 6.409 | 6.658 | -6.53% | -3.74% | 5.101 / 5.527 |
| rmsnorm immediate_repeat H=63 | [563, 4048] | 950DT | 6.223 | 4.455 | 4.777 | +30.27% | -6.73% | 5.101 / 3.320 |
| rmsnorm entry H=0 | [563, 4048] | 950PR | 6.681 | 6.755 | 8.210 | -18.63% | -17.73% | 5.101 / 5.527 |
| rmsnorm immediate_repeat H=52 | [563, 4048] | 950PR | 6.681 | 5.112 | 5.197 | +28.55% | -1.63% | 5.101 / 3.320 |
| rmsnorm entry H=0 | [1, 672] | 950DT | 2.287 | 3.187 | 3.354 | -31.82% | -4.97% | 2.096 / 2.922 |
| rmsnorm immediate_repeat H=1 | [1, 672] | 950DT | 2.287 | 1.619 | 1.474 | +55.15% | +9.82% | 2.096 / 1.484 |
| rmsnorm entry H=1 | [4, 608] | 950DT | 2.443 | 4.165 | 4.451 | -45.12% | -6.42% | 2.213 / 3.818 |
| rmsnorm immediate_repeat H=4 | [4, 608] | 950DT | 2.443 | 1.734 | 1.745 | +39.98% | -0.64% | 2.213 / 1.589 |
| rmsnorm entry H=4 | [17, 2400] | 950DT | 4.010 | 3.987 | 4.224 | -5.08% | -5.62% | 3.559 / 3.655 |
| rmsnorm immediate_repeat H=17 | [17, 2400] | 950DT | 4.010 | 3.087 | 3.160 | +26.89% | -2.31% | 3.559 / 2.830 |
| rmsnorm entry H=17 | [34, 1664] | 950DT | 5.657 | 4.589 | 4.819 | +17.38% | -4.78% | 4.969 / 4.206 |
| rmsnorm immediate_repeat H=34 | [34, 1664] | 950DT | 5.657 | 4.581 | 4.434 | +27.58% | +3.32% | 4.969 / 4.199 |
| rmsnorm entry H=34 | [55, 2272] | 950DT | 6.070 | 4.744 | 5.116 | +18.65% | -7.28% | 5.225 / 4.349 |
| rmsnorm immediate_repeat H=55 | [55, 2272] | 950DT | 6.070 | 4.739 | 4.695 | +29.31% | +0.96% | 5.225 / 4.344 |
| rmsnorm entry H=0 | [1, 672] | 950PR | 2.230 | 2.975 | 3.188 | -30.03% | -6.69% | 2.096 / 2.922 |
| rmsnorm immediate_repeat H=1 | [1, 672] | 950PR | 2.230 | 1.619 | 1.468 | +51.89% | +10.27% | 2.096 / 1.484 |
| rmsnorm entry H=1 | [4, 608] | 950PR | 2.390 | 4.166 | 4.578 | -47.80% | -9.01% | 2.213 / 3.818 |
| rmsnorm immediate_repeat H=4 | [4, 608] | 950PR | 2.390 | 1.734 | 1.762 | +35.61% | -1.64% | 2.213 / 1.589 |
| rmsnorm entry H=4 | [17, 2400] | 950PR | 3.966 | 3.987 | 4.348 | -8.79% | -8.32% | 3.559 / 3.655 |
| rmsnorm immediate_repeat H=17 | [17, 2400] | 950PR | 3.966 | 3.087 | 3.202 | +23.87% | -3.59% | 3.559 / 2.830 |
| rmsnorm entry H=17 | [34, 1664] | 950PR | 5.632 | 4.589 | 4.776 | +17.92% | -3.91% | 4.969 / 4.206 |
| rmsnorm immediate_repeat H=34 | [34, 1664] | 950PR | 5.632 | 4.581 | 4.612 | +22.10% | -0.67% | 4.969 / 4.199 |
| rmsnorm entry H=34 | [55, 2272] | 950PR | 6.074 | 4.744 | 5.241 | +15.89% | -9.49% | 5.225 / 4.349 |
| rmsnorm immediate_repeat H=55 | [55, 2272] | 950PR | 6.074 | 4.739 | 4.846 | +25.33% | -2.20% | 5.225 / 4.344 |
| rmsnorm entry H=0 | [128, 1184] | 950DT | 4.807 | 4.986 | 5.315 | -9.55% | -6.19% | 4.003 / 4.570 |
| rmsnorm immediate_repeat H=64 | [128, 1184] | 950DT | 4.807 | 3.260 | 3.044 | +57.94% | +7.12% | 4.003 / 2.989 |
| rmsnorm entry H=0 | [128, 1184] | 950PR | 3.997 | 4.169 | 4.358 | -8.28% | -4.33% | 4.003 / 4.570 |
| rmsnorm immediate_repeat H=43 | [128, 1184] | 950PR | 3.997 | 2.779 | 2.633 | +51.78% | +5.54% | 4.003 / 2.989 |
| rmsnorm entry H=1 | [2, 2944] | 950DT | 2.534 | 4.317 | 4.628 | -45.25% | -6.72% | 2.314 / 3.957 |
| rmsnorm immediate_repeat H=2 | [2, 2944] | 950DT | 2.534 | 1.850 | 1.604 | +57.94% | +15.33% | 2.314 / 1.696 |
| rmsnorm entry H=2 | [3, 3264] | 950DT | 2.592 | 2.728 | 2.740 | -5.40% | -0.44% | 2.359 / 3.056 |
| rmsnorm immediate_repeat H=3 | [3, 3264] | 950DT | 2.592 | 1.908 | 1.827 | +41.90% | +4.47% | 2.359 / 1.749 |
| rmsnorm entry H=3 | [4, 6080] | 950DT | 2.894 | 2.949 | 3.253 | -11.02% | -9.35% | 2.628 / 3.258 |
| rmsnorm immediate_repeat H=4 | [4, 6080] | 950DT | 2.894 | 2.150 | 2.015 | +43.62% | +6.70% | 2.628 / 1.971 |
| rmsnorm entry H=1 | [2, 2944] | 950PR | 2.479 | 4.319 | 4.361 | -43.15% | -0.96% | 2.314 / 3.957 |
| rmsnorm immediate_repeat H=2 | [2, 2944] | 950PR | 2.479 | 1.852 | 1.610 | +53.97% | +15.04% | 2.314 / 1.696 |
| rmsnorm entry H=2 | [3, 3264] | 950PR | 2.540 | 2.731 | 2.739 | -7.27% | -0.30% | 2.359 / 3.056 |
| rmsnorm immediate_repeat H=3 | [3, 3264] | 950PR | 2.540 | 1.911 | 1.835 | +38.44% | +4.16% | 2.359 / 1.749 |
| rmsnorm entry H=3 | [4, 6080] | 950PR | 2.844 | 2.953 | 2.916 | -2.46% | +1.29% | 2.628 / 3.258 |
| rmsnorm immediate_repeat H=4 | [4, 6080] | 950PR | 2.844 | 2.155 | 2.032 | +39.96% | +6.04% | 2.628 / 1.971 |
