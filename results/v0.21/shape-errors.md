# 逐算子、逐芯片的校准与外推

设备时间均为 µs。DT 和 PR 各行对应自己的完整 CA 窗口；970DT 列是同一输入在假定芯片上的外推。v0.21 新独立组使用采集前封存的预测。源回执、库 kernel、数值结果见同目录 evaluation.json 及其 source。

## DynamicMXQuant

| 数据组 | 芯片 | shape | v0.20 | 当前 | CA | 有符号误差 | 970DT 外推 |
|---|---|---|---:|---:|---:|---:|---:|
| DT v0.12 development replay | 950DT | [1, 128] | — | 4.231 | 4.325 | -2.2% | 3.693 |
| DT v0.12 development replay | 950DT | [36, 512] | — | 5.430 | 5.646 | -3.8% | 6.119 |
| DT v0.12 development replay | 950DT | [9, 5120] | — | 4.980 | 4.867 | +2.3% | 7.084 |
| DT v0.12 development replay | 950DT | [144, 7168] | — | 6.207 | 6.332 | -2.0% | 5.022 |
| DT v0.12 development replay | 950DT | [512, 5120] | — | 6.364 | 6.299 | +1.0% | 5.727 |
| DT v0.12 development replay | 950DT | [1152, 5120] | — | 8.942 | 8.684 | +3.0% | 5.893 |
| DT v0.12 development replay | 950DT | [512, 256] | — | 5.430 | 5.208 | +4.3% | 4.734 |
| DT v0.12 development replay | 950DT | [576, 4096] | — | 7.470 | 6.948 | +7.5% | 5.440 |
| DT v0.12 development replay | 950DT | [768, 4096] | — | 6.743 | 7.202 | -6.4% | 5.984 |
| DT v0.12 development replay | 950DT | [1, 64] | — | 4.231 | 4.326 | -2.2% | 3.692 |
| DT v0.12 development replay | 950DT | [65, 256] | — | 5.430 | 4.898 | +10.9% | 5.846 |
| DT v0.12 development replay | 950DT | [1024, 4096] | — | 7.502 | 8.047 | -6.8% | 6.352 |
| DT v0.12 development replay | 950DT | [1, 4096] | — | 5.430 | 5.686 | -4.5% | 4.532 |
| DT v0.12 development replay | 950DT | [640, 4096] | — | 6.364 | 6.218 | +2.4% | 5.727 |
| DT v0.12 development replay | 950DT | [33, 768] | — | 5.430 | 5.210 | +4.2% | 7.042 |
| DT v0.12 development replay | 950DT | [4, 256] | — | 4.759 | 4.588 | +3.7% | 4.268 |
| DT v0.12 development replay | 950DT | [17, 512] | — | 5.430 | 5.790 | -6.2% | 4.876 |
| DT v0.12 development replay | 950DT | [8, 384] | — | 5.430 | 5.760 | -5.7% | 4.871 |
| DT v0.12 development replay | 950DT | [64, 1024] | — | 5.214 | 5.532 | -5.7% | 5.870 |
| DT v0.12 holdout replay | 950DT | [36, 5120] | — | 5.292 | 5.584 | -5.2% | 4.769 |
| DT v0.12 holdout replay | 950DT | [288, 7168] | — | 7.181 | 6.876 | +4.4% | 5.526 |
| DT v0.12 holdout replay | 950DT | [1024, 7168] | — | 9.586 | 9.332 | +2.7% | 6.713 |
| DT v0.12 holdout replay | 950DT | [1024, 6144] | — | 8.935 | 8.704 | +2.7% | 6.245 |
| DT v0.12 holdout replay | 950DT | [128, 5120] | — | 5.794 | 5.885 | -1.5% | 4.904 |
| DT v0.12 holdout replay | 950DT | [17, 1024] | — | 5.430 | 5.772 | -5.9% | 5.982 |
| DT v0.12 holdout replay | 950DT | [1, 192] | — | 4.231 | 4.129 | +2.5% | 3.693 |
| DT v0.12 holdout replay | 950DT | [36, 3072] | — | 5.156 | 4.971 | +3.7% | 7.373 |
| DT v0.12 holdout replay | 950DT | [576, 7168] | — | 7.577 | 7.758 | -2.3% | 6.553 |
| DT v0.12 holdout replay | 950DT | [7, 320] | — | 5.393 | 5.146 | +4.8% | 4.987 |
| primary | 950DT | [2, 832] | — | 5.098 | 4.819 | +5.8% | 4.459 |
| primary | 950PR | [2, 832] | — | 4.428 | 4.421 | +0.2% | 4.459 |
| primary | 950DT | [4, 1472] | — | 5.430 | 5.685 | -4.5% | 4.722 |
| primary | 950PR | [4, 1472] | — | 4.715 | 5.398 | -12.7% | 4.722 |
| primary | 950DT | [11, 3136] | — | 5.430 | 5.507 | -1.4% | 6.353 |
| primary | 950PR | [11, 3136] | — | 5.319 | 4.986 | +6.7% | 6.353 |
| primary | 950DT | [26, 704] | — | 5.430 | 5.640 | -3.7% | 6.430 |
| primary | 950PR | [26, 704] | — | 5.636 | 5.452 | +3.4% | 6.430 |
| primary | 950DT | [53, 1984] | — | 5.156 | 5.616 | -8.2% | 7.506 |
| primary | 950PR | [53, 1984] | — | 4.657 | 5.105 | -8.8% | 7.506 |
| primary | 950DT | [59, 2880] | — | 5.292 | 5.651 | -6.4% | 5.152 |
| primary | 950PR | [59, 2880] | — | 4.734 | 4.695 | +0.8% | 5.152 |
| primary | 950DT | [97, 2752] | — | 5.316 | 5.279 | +0.7% | 4.781 |
| primary | 950PR | [97, 2752] | — | 4.831 | 5.107 | -5.4% | 4.781 |
| primary | 950DT | [257, 4416] | — | 6.528 | 6.467 | +0.9% | 5.083 |
| primary | 950PR | [257, 4416] | — | 6.069 | 6.130 | -1.0% | 5.083 |
| primary | 950DT | [641, 5120] | — | 7.001 | 7.244 | -3.3% | 6.040 |
| primary | 950PR | [641, 5120] | — | 7.813 | 9.092 | -14.1% | 6.040 |
| primary | 950DT | [1023, 7168] | — | 9.811 | 9.311 | +5.4% | 6.713 |
| primary | 950PR | [1023, 7168] | — | 14.026 | 14.080 | -0.4% | 6.713 |
| historical coverage replay | 950DT | [1, 64] | — | 4.231 | 4.326 | -2.2% | 3.692 |
| historical coverage replay | 950DT | [1, 128] | — | 4.231 | 4.325 | -2.2% | 3.693 |
| historical coverage replay | 950DT | [1, 256] | — | 4.231 | 4.493 | -5.8% | 3.693 |
| historical coverage replay | 950DT | [1, 704] | — | 4.630 | 4.332 | +6.9% | 4.076 |
| historical coverage replay | 950DT | [1, 4096] | — | 5.430 | 5.686 | -4.5% | 4.532 |
| historical coverage replay | 950DT | [3, 1920] | — | 5.430 | 5.788 | -6.2% | 4.657 |
| historical coverage replay | 950DT | [4, 256] | — | 4.759 | 4.588 | +3.7% | 4.268 |
| historical coverage replay | 950DT | [4, 256] | — | 4.759 | 4.620 | +3.0% | 4.268 |
| historical coverage replay | 950DT | [5, 320] | — | 5.214 | 4.662 | +11.8% | 4.842 |
| historical coverage replay | 950DT | [7, 4480] | — | 5.430 | 5.748 | -5.5% | 5.840 |
| historical coverage replay | 950DT | [8, 384] | — | 5.430 | 5.760 | -5.7% | 4.871 |
| historical coverage replay | 950DT | [9, 4096] | — | 5.430 | 5.465 | -0.6% | 6.127 |
| historical coverage replay | 950DT | [9, 5120] | — | 4.980 | 4.867 | +2.3% | 7.084 |
| historical coverage replay | 950DT | [17, 512] | — | 5.430 | 5.790 | -6.2% | 4.876 |
| historical coverage replay | 950DT | [19, 1024] | — | 5.430 | 6.230 | -12.8% | 6.256 |
| historical coverage replay | 950DT | [27, 576] | — | 5.430 | 6.130 | -11.4% | 6.561 |
| historical coverage replay | 950DT | [30, 768] | — | 5.430 | 5.472 | -0.8% | 6.734 |
| historical coverage replay | 950DT | [33, 768] | — | 5.430 | 5.210 | +4.2% | 7.042 |
| historical coverage replay | 950DT | [36, 512] | — | 5.430 | 5.646 | -3.8% | 6.119 |
| historical coverage replay | 950DT | [55, 1152] | — | 5.180 | 5.900 | -12.2% | 6.542 |
| historical coverage replay | 950DT | [56, 4096] | — | 5.430 | 6.285 | -13.6% | 4.581 |
| historical coverage replay | 950DT | [57, 3200] | — | 5.331 | 4.945 | +7.8% | 4.970 |
| historical coverage replay | 950DT | [64, 1024] | — | 5.214 | 5.532 | -5.7% | 5.870 |
| historical coverage replay | 950DT | [65, 256] | — | 5.430 | 4.898 | +10.9% | 5.846 |
| historical coverage replay | 950DT | [65, 2624] | — | 5.194 | 5.306 | -2.1% | 4.942 |
| historical coverage replay | 950DT | [144, 5120] | — | 5.886 | 5.905 | -0.3% | 4.728 |
| historical coverage replay | 950DT | [144, 7168] | — | 6.207 | 6.332 | -2.0% | 5.022 |
| historical coverage replay | 950DT | [255, 5440] | — | 6.522 | 5.747 | +13.5% | 5.612 |
| historical coverage replay | 950DT | [384, 4096] | — | 6.695 | 6.673 | +0.3% | 5.091 |
| historical coverage replay | 950DT | [448, 6592] | — | 6.726 | 6.300 | +6.8% | 6.381 |
| historical coverage replay | 950DT | [512, 256] | — | 5.430 | 5.208 | +4.3% | 4.734 |
| historical coverage replay | 950DT | [512, 5120] | — | 6.364 | 6.299 | +1.0% | 5.727 |
| historical coverage replay | 950DT | [513, 5504] | — | 6.845 | 6.751 | +1.4% | 6.009 |
| historical coverage replay | 950DT | [576, 4096] | — | 7.470 | 6.948 | +7.5% | 5.440 |
| historical coverage replay | 950DT | [640, 4096] | — | 6.364 | 6.218 | +2.4% | 5.727 |
| historical coverage replay | 950DT | [768, 4096] | — | 6.743 | 7.202 | -6.4% | 5.984 |
| historical coverage replay | 950DT | [1024, 4096] | — | 7.502 | 8.047 | -6.8% | 6.352 |
| historical coverage replay | 950DT | [1152, 5120] | — | 8.942 | 8.684 | +3.0% | 5.893 |
| historical coverage replay | 950DT | [2048, 5120] | — | 11.531 | 11.364 | +1.5% | 7.733 |
| historical coverage replay | 950PR | [1, 64] | — | 3.591 | 3.946 | -9.0% | 3.692 |
| historical coverage replay | 950PR | [1, 128] | — | 3.592 | 3.828 | -6.2% | 3.693 |
| historical coverage replay | 950PR | [1, 192] | — | 3.592 | 3.833 | -6.3% | 3.693 |
| historical coverage replay | 950PR | [1, 256] | — | 3.592 | 3.859 | -6.9% | 3.693 |
| historical coverage replay | 950PR | [1, 704] | — | 4.010 | 3.964 | +1.2% | 4.076 |
| historical coverage replay | 950PR | [1, 4096] | — | 4.507 | 4.945 | -8.9% | 4.532 |
| historical coverage replay | 950PR | [3, 1920] | — | 4.643 | 5.315 | -12.6% | 4.657 |
| historical coverage replay | 950PR | [4, 256] | — | 4.219 | 4.157 | +1.5% | 4.268 |
| historical coverage replay | 950PR | [5, 320] | — | 4.846 | 4.592 | +5.5% | 4.842 |
| historical coverage replay | 950PR | [7, 320] | — | 5.004 | 5.042 | -0.8% | 4.987 |
| historical coverage replay | 950PR | [7, 4480] | — | 4.865 | 4.559 | +6.7% | 5.840 |
| historical coverage replay | 950PR | [8, 384] | — | 4.877 | 5.136 | -5.0% | 4.871 |
| historical coverage replay | 950PR | [9, 4096] | — | 5.072 | 4.825 | +5.1% | 6.127 |
| historical coverage replay | 950PR | [9, 5120] | — | 4.377 | 5.025 | -12.9% | 7.084 |
| historical coverage replay | 950PR | [17, 512] | — | 4.882 | 5.432 | -10.1% | 4.876 |
| historical coverage replay | 950PR | [17, 1024] | — | 4.966 | 5.311 | -6.5% | 5.982 |
| historical coverage replay | 950PR | [19, 1024] | — | 5.102 | 5.983 | -14.7% | 6.256 |
| historical coverage replay | 950PR | [27, 576] | — | 5.711 | 5.547 | +3.0% | 6.561 |
| historical coverage replay | 950PR | [30, 768] | — | 4.770 | 4.917 | -3.0% | 6.734 |
| historical coverage replay | 950PR | [33, 768] | — | 4.845 | 4.915 | -1.4% | 7.042 |
| historical coverage replay | 950PR | [36, 512] | — | 4.906 | 5.795 | -15.3% | 6.119 |
| historical coverage replay | 950PR | [36, 3072] | — | 4.448 | 5.170 | -14.0% | 7.373 |
| historical coverage replay | 950PR | [36, 5120] | — | 4.547 | 5.052 | -10.0% | 4.769 |
| historical coverage replay | 950PR | [55, 1152] | — | 4.873 | 5.336 | -8.7% | 6.542 |
| historical coverage replay | 950PR | [56, 4096] | — | 4.554 | 4.513 | +0.9% | 4.581 |
| historical coverage replay | 950PR | [57, 3200] | — | 4.776 | 4.430 | +7.8% | 4.970 |
| historical coverage replay | 950PR | [64, 1024] | — | 4.599 | 5.373 | -14.4% | 5.870 |
| historical coverage replay | 950PR | [65, 256] | — | 4.757 | 4.609 | +3.2% | 5.846 |
| historical coverage replay | 950PR | [65, 2624] | — | 4.753 | 4.736 | +0.4% | 4.942 |
| historical coverage replay | 950PR | [128, 5120] | — | 5.192 | 5.730 | -9.4% | 4.904 |
| historical coverage replay | 950PR | [144, 5120] | — | 5.180 | 5.984 | -13.4% | 4.728 |
| historical coverage replay | 950PR | [144, 7168] | — | 5.434 | 6.415 | -15.3% | 5.022 |
| historical coverage replay | 950PR | [255, 5440] | — | 6.694 | 6.868 | -2.5% | 5.612 |
| historical coverage replay | 950PR | [288, 7168] | — | 7.203 | 8.009 | -10.1% | 5.526 |
| historical coverage replay | 950PR | [384, 4096] | — | 6.397 | 7.332 | -12.8% | 5.091 |
| historical coverage replay | 950PR | [448, 6592] | — | 7.560 | 8.222 | -8.0% | 6.381 |
| historical coverage replay | 950PR | [512, 256] | — | 4.667 | 5.090 | -8.3% | 4.734 |
| historical coverage replay | 950PR | [512, 5120] | — | 6.870 | 7.690 | -10.7% | 5.727 |
| historical coverage replay | 950PR | [513, 5504] | — | 6.845 | 8.230 | -16.8% | 6.009 |
| historical coverage replay | 950PR | [576, 4096] | — | 6.425 | 7.692 | -16.5% | 5.440 |
| historical coverage replay | 950PR | [576, 7168] | — | 9.614 | 10.007 | -3.9% | 6.553 |
| historical coverage replay | 950PR | [640, 4096] | — | 6.912 | 8.108 | -14.8% | 5.727 |
| historical coverage replay | 950PR | [2048, 5120] | — | 17.834 | 18.504 | -3.6% | 7.733 |

## Matmul

| 数据组 | 芯片 | shape | v0.20 | 当前 | CA | 有符号误差 | 970DT 外推 |
|---|---|---|---:|---:|---:|---:|---:|
| DT v0.12 development replay | 950DT | [1, 256, 1024] | — | 5.127 | 4.624 | +10.9% | 4.700 |
| DT v0.12 development replay | 950DT | [65, 128, 129] | — | 3.554 | 3.712 | -4.2% | 3.258 |
| DT v0.12 development replay | 950DT | [128, 512, 512] | — | 4.541 | 4.382 | +3.6% | 4.137 |
| DT v0.12 development replay | 950DT | [129, 256, 256] | — | 3.936 | 3.664 | +7.4% | 3.608 |
| DT v0.12 development replay | 950DT | [17, 256, 1024] | — | 5.190 | 4.742 | +9.5% | 4.758 |
| DT v0.12 development replay | 950DT | [128, 256, 512] | — | 4.463 | 4.393 | +1.6% | 4.087 |
| DT v0.12 development replay | 950DT | [1, 128, 128] | — | 3.414 | 3.356 | +1.7% | 3.129 |
| DT v0.12 development replay | 950DT | [32, 64, 64] | — | 3.278 | 3.292 | -0.4% | 3.005 |
| DT v0.12 development replay | 950DT | [48, 128, 192] | — | 3.651 | 3.575 | +2.1% | 3.347 |
| DT v0.12 development replay | 950DT | [17, 96, 128] | — | 3.476 | 3.494 | -0.5% | 3.187 |
| DT v0.12 holdout replay | 950DT | [128, 192, 256] | — | 3.906 | 3.798 | +2.8% | 3.579 |
| DT v0.12 holdout replay | 950DT | [1, 32, 16] | — | 3.091 | 3.030 | +2.0% | 2.833 |
| DT v0.12 holdout replay | 950DT | [256, 512, 256] | — | 4.086 | 4.192 | -2.5% | 3.693 |
| DT v0.12 holdout replay | 950DT | [31, 64, 257] | — | 3.781 | 4.091 | -7.6% | 3.466 |
| DT v0.12 holdout replay | 950DT | [4, 80, 48] | — | 3.237 | 3.524 | -8.1% | 2.967 |
| DT v0.12 holdout replay | 950DT | [512, 512, 256] | — | 4.771 | 4.799 | -0.6% | 3.759 |
| DT v0.12 holdout replay | 950DT | [64, 128, 512] | — | 4.348 | 4.437 | -2.0% | 3.985 |
| DT v0.12 holdout replay | 950DT | [192, 384, 256] | — | 4.002 | 3.925 | +1.9% | 3.659 |
| DT v0.12 holdout replay | 950DT | [2, 32, 32] | — | 3.109 | 3.244 | -4.2% | 2.850 |
| DT v0.12 holdout replay | 950DT | [9, 128, 256] | — | 3.703 | 4.019 | -7.9% | 3.394 |
| primary | 950DT | [3, 64, 64] | — | 3.217 | 3.065 | +4.9% | 2.949 |
| primary | 950PR | [3, 64, 64] | — | 3.217 | 2.858 | +12.6% | 2.949 |
| primary | 950DT | [8, 96, 256] | — | 3.702 | 3.755 | -1.4% | 3.393 |
| primary | 950PR | [8, 96, 256] | — | 3.702 | 3.620 | +2.3% | 3.393 |
| primary | 950DT | [33, 128, 129] | — | 3.513 | 3.671 | -4.3% | 3.220 |
| primary | 950PR | [33, 128, 129] | — | 3.513 | 3.679 | -4.5% | 3.220 |
| primary | 950DT | [127, 384, 256] | — | 3.954 | 3.819 | +3.5% | 3.630 |
| primary | 950PR | [127, 384, 256] | — | 3.956 | 3.793 | +4.3% | 3.630 |
| primary | 950DT | [225, 512, 256] | — | 4.079 | 4.113 | -0.8% | 3.697 |
| primary | 950PR | [225, 512, 256] | — | 4.225 | 4.116 | +2.6% | 3.697 |
| primary | 950DT | [449, 512, 256] | — | 4.759 | 4.361 | +9.1% | 3.740 |
| primary | 950PR | [449, 512, 256] | — | 4.826 | 4.595 | +5.0% | 3.740 |
| primary | 950DT | [1024, 1792, 32] | — | 4.610 | 4.260 | +8.2% | 3.477 |
| primary | 950PR | [1024, 1792, 32] | — | 4.709 | 4.347 | +8.3% | 3.477 |
| primary | 950DT | [1536, 1792, 16] | — | 4.680 | 4.438 | +5.5% | 3.490 |
| primary | 950PR | [1536, 1792, 16] | — | 4.818 | 4.540 | +6.1% | 3.490 |
| primary | 950DT | [1024, 2304, 16] | — | 4.531 | 4.258 | +6.4% | 3.450 |
| primary | 950PR | [1024, 2304, 16] | — | 4.648 | 4.419 | +5.2% | 3.450 |
| primary | 950DT | [1280, 2048, 16] | — | 4.630 | 4.397 | +5.3% | 3.481 |
| primary | 950PR | [1280, 2048, 16] | — | 4.762 | 4.633 | +2.8% | 3.481 |
| primary | 950DT | [2048, 32, 5120] | — | 12.123 | 13.023 | -6.9% | 7.287 |
| primary | 950PR | [2048, 32, 5120] | — | 19.016 | 20.822 | -8.7% | 7.287 |
| historical coverage replay | 950DT | [1, 128, 128] | — | 3.414 | 3.356 | +1.7% | 3.129 |
| historical coverage replay | 950DT | [1, 256, 1024] | — | 5.127 | 4.624 | +10.9% | 4.700 |
| historical coverage replay | 950DT | [17, 96, 128] | — | 3.476 | 3.494 | -0.5% | 3.187 |
| historical coverage replay | 950DT | [17, 256, 1024] | — | 5.190 | 4.742 | +9.5% | 4.758 |
| historical coverage replay | 950DT | [32, 64, 64] | — | 3.278 | 3.292 | -0.4% | 3.005 |
| historical coverage replay | 950DT | [32, 64, 64] | — | 3.278 | 3.281 | -0.1% | 3.005 |
| historical coverage replay | 950DT | [48, 128, 192] | — | 3.651 | 3.575 | +2.1% | 3.347 |
| historical coverage replay | 950DT | [65, 128, 129] | — | 3.554 | 3.712 | -4.2% | 3.258 |
| historical coverage replay | 950DT | [128, 256, 512] | — | 4.463 | 4.393 | +1.6% | 4.087 |
| historical coverage replay | 950DT | [128, 512, 512] | — | 4.541 | 4.382 | +3.6% | 4.137 |
| historical coverage replay | 950DT | [129, 256, 256] | — | 3.936 | 3.664 | +7.4% | 3.608 |
| historical coverage replay | 950DT | [1280, 2048, 16] | — | 4.630 | 4.472 | +3.5% | 3.481 |
| historical coverage replay | 950DT | [1280, 2048, 512] | — | 12.449 | 12.219 | +1.9% | 6.350 |
| historical coverage replay | 950PR | [1, 32, 16] | — | 3.091 | 2.734 | +13.1% | 2.833 |
| historical coverage replay | 950PR | [1, 128, 128] | — | 3.414 | 3.138 | +8.8% | 3.129 |
| historical coverage replay | 950PR | [1, 256, 1024] | — | 5.127 | 4.612 | +11.2% | 4.700 |
| historical coverage replay | 950PR | [2, 32, 32] | — | 3.109 | 2.813 | +10.5% | 2.850 |
| historical coverage replay | 950PR | [4, 80, 48] | — | 3.237 | 3.341 | -3.1% | 2.967 |
| historical coverage replay | 950PR | [9, 128, 256] | — | 3.703 | 3.595 | +3.0% | 3.394 |
| historical coverage replay | 950PR | [16, 192, 384] | — | 4.039 | 3.724 | +8.5% | 3.703 |
| historical coverage replay | 950PR | [17, 96, 128] | — | 3.476 | 3.414 | +1.8% | 3.187 |
| historical coverage replay | 950PR | [17, 256, 1024] | — | 5.190 | 4.687 | +10.7% | 4.758 |
| historical coverage replay | 950PR | [31, 64, 257] | — | 3.781 | 3.801 | -0.5% | 3.466 |
| historical coverage replay | 950PR | [32, 64, 64] | — | 3.278 | 3.158 | +3.8% | 3.005 |
| historical coverage replay | 950PR | [48, 128, 192] | — | 3.651 | 3.481 | +4.9% | 3.347 |
| historical coverage replay | 950PR | [64, 128, 512] | — | 4.348 | 4.275 | +1.7% | 3.985 |
| historical coverage replay | 950PR | [65, 128, 129] | — | 3.554 | 3.733 | -4.8% | 3.258 |
| historical coverage replay | 950PR | [128, 192, 256] | — | 3.907 | 3.815 | +2.4% | 3.579 |
| historical coverage replay | 950PR | [128, 256, 512] | — | 4.477 | 4.368 | +2.5% | 4.087 |
| historical coverage replay | 950PR | [128, 512, 512] | — | 4.600 | 4.604 | -0.1% | 4.137 |
| historical coverage replay | 950PR | [129, 256, 256] | — | 3.936 | 3.609 | +9.1% | 3.608 |
| historical coverage replay | 950PR | [192, 384, 256] | — | 4.006 | 4.040 | -0.8% | 3.659 |
| historical coverage replay | 950PR | [224, 512, 256] | — | 4.225 | 4.115 | +2.7% | 3.692 |
| historical coverage replay | 950PR | [255, 512, 256] | — | 4.390 | 4.183 | +4.9% | 3.693 |
| historical coverage replay | 950PR | [256, 512, 256] | — | 4.390 | 4.239 | +3.5% | 3.693 |
| historical coverage replay | 950PR | [257, 512, 256] | — | 4.390 | 4.197 | +4.6% | 3.702 |
| historical coverage replay | 950PR | [384, 512, 256] | — | 4.734 | 4.504 | +5.1% | 3.723 |
| historical coverage replay | 950PR | [448, 512, 256] | — | 4.760 | 4.515 | +5.4% | 3.735 |
| historical coverage replay | 950PR | [511, 512, 256] | — | 4.852 | 4.612 | +5.2% | 3.759 |
| historical coverage replay | 950PR | [512, 384, 256] | — | 4.734 | 4.541 | +4.3% | 3.723 |
| historical coverage replay | 950PR | [512, 512, 256] | — | 4.852 | 4.656 | +4.2% | 3.759 |
| historical coverage replay | 950PR | [513, 384, 256] | — | 4.743 | 4.515 | +5.1% | 3.727 |
| historical coverage replay | 950PR | [768, 512, 128] | — | 4.312 | 4.068 | +6.0% | 3.737 |
| historical coverage replay | 950PR | [1024, 2048, 32] | — | 4.823 | 4.464 | +8.0% | 3.547 |
| historical coverage replay | 950PR | [1536, 2048, 16] | — | 4.969 | 4.755 | +4.5% | 3.540 |

## FA

| 数据组 | 芯片 | shape | v0.20 | 当前 | CA | 有符号误差 | 970DT 外推 |
|---|---|---|---:|---:|---:|---:|---:|
| DT v0.12 holdout replay | 950DT | [9, 256, 128, 4, 1] | — | 9.767 | 9.690 | +0.8% | 8.658 |
| DT v0.12 development replay | 950DT | [32, 129, 64, 1, 4] | — | 9.720 | 9.150 | +6.2% | 8.423 |
| DT v0.12 development replay | 950DT | [17, 257, 128, 1, 1] | — | 10.471 | 9.551 | +9.6% | 9.057 |
| DT v0.12 development replay | 950DT | [17, 1153, 64, 1, 1] | — | 12.322 | 12.487 | -1.3% | 12.447 |
| DT v0.12 development replay | 950DT | [128, 128, 64, 1, 1] | — | 9.248 | 9.498 | -2.6% | 9.001 |
| DT v0.12 development replay | 950DT | [16, 1025, 64, 1, 1] | — | 12.313 | 12.384 | -0.6% | 12.046 |
| DT v0.12 development replay | 950DT | [16, 128, 64, 1, 4] | — | 8.894 | 8.758 | +1.6% | 8.063 |
| DT v0.12 development replay | 950DT | [16, 128, 128, 1, 1] | — | 8.829 | 8.715 | +1.3% | 8.293 |
| DT v0.12 development replay | 950DT | [129, 128, 64, 1, 1] | — | 9.081 | 8.871 | +2.4% | 9.001 |
| DT v0.12 development replay | 950DT | [16, 128, 64, 1, 8] | — | 8.992 | 9.402 | -4.4% | 8.063 |
| DT v0.12 development replay | 950DT | [17, 192, 128, 1, 1] | — | 9.653 | 9.050 | +6.7% | 8.670 |
| DT v0.12 development replay | 950DT | [17, 96, 64, 1, 1] | — | 8.692 | 9.162 | -5.1% | 8.050 |
| DT v0.12 development replay | 950DT | [16, 64, 64, 1, 1] | — | 8.657 | 9.052 | -4.4% | 7.997 |
| DT v0.12 development replay | 950DT | [8, 192, 64, 1, 1] | — | 9.425 | 8.805 | +7.0% | 8.247 |
| DT v0.12 development replay | 950DT | [64, 256, 64, 1, 1] | — | 9.703 | 9.409 | +3.1% | 10.464 |
| DT v0.12 development replay | 950DT | [1, 128, 64, 1, 1] | — | 8.718 | 8.784 | -0.7% | 7.963 |
| DT v0.12 development replay | 950DT | [9, 32768, 64, 1, 1] | — | 76.184 | 75.980 | +0.3% | 70.033 |
| DT v0.12 holdout replay | 950DT | [9, 131072, 64, 1, 1] | — | 274.794 | 274.821 | -0.0% | 252.093 |
| DT v0.12 holdout replay | 950DT | [9, 16385, 64, 1, 1] | — | 43.599 | 43.219 | +0.9% | 39.927 |
| DT v0.12 holdout replay | 950DT | [17, 384, 64, 1, 2] | — | 10.301 | 9.757 | +5.6% | 8.865 |
| DT v0.12 holdout replay | 950DT | [1, 64, 64, 1, 1] | — | 8.657 | 8.937 | -3.1% | 7.898 |
| DT v0.12 holdout replay | 950DT | [32, 512, 64, 1, 1] | — | 11.046 | 9.722 | +13.6% | 10.382 |
| DT v0.12 holdout replay | 950DT | [4, 256, 128, 1, 1] | — | 9.551 | 9.367 | +2.0% | 8.594 |
| DT v0.12 holdout replay | 950DT | [64, 320, 128, 1, 1] | — | 10.863 | 9.968 | +9.0% | 12.691 |
| DT v0.12 holdout replay | 950DT | [9, 4096, 64, 1, 1] | — | 18.256 | 18.226 | +0.2% | 16.932 |
| DT v0.12 holdout replay | 950DT | [1, 224, 64, 1, 1] | — | 9.465 | 8.826 | +7.2% | 8.200 |
| DT v0.12 holdout replay | 950DT | [4, 160, 64, 1, 1] | — | 9.406 | 9.461 | -0.6% | 8.220 |
| DT v0.12 holdout replay | 950DT | [8, 128, 128, 1, 1] | — | 8.770 | 9.136 | -4.0% | 8.190 |
| primary | 950DT | [17, 192, 128, 1, 1] | — | 9.653 | 9.050 | +6.7% | 8.670 |
| primary | 950PR | [17, 192, 128, 1, 1] | — | 9.269 | 8.833 | +4.9% | 8.670 |
| primary | 950DT | [4, 160, 64, 1, 1] | — | 9.406 | 9.461 | -0.6% | 8.220 |
| primary | 950PR | [4, 160, 64, 1, 1] | — | 8.956 | 8.657 | +3.5% | 8.220 |
| primary | 950DT | [8, 128, 128, 1, 1] | — | 8.770 | 9.136 | -4.0% | 8.190 |
| primary | 950PR | [8, 128, 128, 1, 1] | — | 8.890 | 8.317 | +6.9% | 8.190 |
| primary | 950DT | [3, 192, 64, 1, 2] | — | 9.484 | 9.167 | +3.5% | 8.214 |
| primary | 950PR | [3, 192, 64, 1, 2] | — | 8.955 | 8.766 | +2.2% | 8.214 |
| primary | 950DT | [9, 384, 64, 1, 1] | — | 10.175 | 9.455 | +7.6% | 8.490 |
| primary | 950PR | [9, 384, 64, 1, 1] | — | 9.240 | 8.834 | +4.6% | 8.490 |
| primary | 950DT | [17, 512, 128, 1, 1] | — | 11.288 | 10.595 | +6.5% | 9.777 |
| primary | 950PR | [17, 512, 128, 1, 1] | — | 10.498 | 10.058 | +4.4% | 9.777 |
| primary | 950DT | [9, 1023, 64, 1, 1] | — | 13.899 | 11.287 | +23.1% | 9.676 |
| primary | 950PR | [9, 1023, 64, 1, 1] | — | 10.533 | 10.985 | -4.1% | 9.676 |
| primary | 950DT | [9, 3072, 64, 1, 1] | — | 17.924 | 16.010 | +12.0% | 15.036 |
| primary | 950PR | [9, 3072, 64, 1, 1] | — | 16.380 | 15.156 | +8.1% | 15.036 |
| primary | 950DT | [9, 8191, 64, 1, 1] | — | 26.531 | 26.269 | +1.0% | 24.518 |
| primary | 950PR | [9, 8191, 64, 1, 1] | — | 26.725 | 25.197 | +6.1% | 24.518 |
| primary | 950DT | [9, 16384, 64, 1, 1] | — | 43.082 | 42.932 | +0.4% | 39.690 |
| primary | 950PR | [9, 16384, 64, 1, 1] | — | 43.276 | 41.556 | +4.1% | 39.690 |
| historical coverage replay | 950DT | [1, 128, 64, 1, 1] | — | 8.718 | 8.784 | -0.7% | 7.963 |
| historical coverage replay | 950DT | [8, 192, 64, 1, 1] | — | 9.425 | 8.805 | +7.0% | 8.247 |
| historical coverage replay | 950DT | [9, 32768, 64, 1, 1] | — | 76.184 | 75.980 | +0.3% | 70.033 |
| historical coverage replay | 950DT | [16, 64, 64, 1, 1] | — | 8.657 | 9.052 | -4.4% | 7.997 |
| historical coverage replay | 950DT | [16, 64, 64, 1, 1] | — | 8.657 | 8.856 | -2.2% | 7.997 |
| historical coverage replay | 950DT | [16, 128, 64, 1, 4] | — | 8.894 | 8.758 | +1.6% | 8.063 |
| historical coverage replay | 950DT | [16, 128, 64, 1, 8] | — | 8.992 | 9.402 | -4.4% | 8.063 |
| historical coverage replay | 950DT | [16, 128, 128, 1, 1] | — | 8.829 | 8.715 | +1.3% | 8.293 |
| historical coverage replay | 950DT | [16, 1025, 64, 1, 1] | — | 12.313 | 12.384 | -0.6% | 12.046 |
| historical coverage replay | 950DT | [17, 96, 64, 1, 1] | — | 8.692 | 9.162 | -5.1% | 8.050 |
| historical coverage replay | 950DT | [17, 257, 128, 1, 1] | — | 10.471 | 9.551 | +9.6% | 9.057 |
| historical coverage replay | 950DT | [17, 1153, 64, 1, 1] | — | 12.322 | 12.150 | +1.4% | 12.447 |
| historical coverage replay | 950DT | [17, 1153, 64, 1, 1] | — | 12.322 | 12.024 | +2.5% | 12.447 |
| historical coverage replay | 950DT | [17, 1153, 64, 1, 1] | — | 12.322 | 12.556 | -1.9% | 12.447 |
| historical coverage replay | 950DT | [17, 1153, 64, 1, 1] | — | 12.322 | 12.553 | -1.8% | 12.447 |
| historical coverage replay | 950DT | [17, 1153, 64, 1, 1] | — | 12.322 | 12.487 | -1.3% | 12.447 |
| historical coverage replay | 950DT | [32, 129, 64, 1, 4] | — | 9.720 | 9.150 | +6.2% | 8.423 |
| historical coverage replay | 950DT | [64, 256, 64, 1, 1] | — | 9.703 | 9.409 | +3.1% | 10.464 |
| historical coverage replay | 950DT | [128, 128, 64, 1, 1] | — | 9.248 | 9.498 | -2.6% | 9.001 |
| historical coverage replay | 950DT | [129, 128, 64, 1, 1] | — | 9.081 | 8.871 | +2.4% | 9.001 |
| historical coverage replay | 950PR | [1, 64, 64, 1, 1] | — | 8.616 | 8.535 | +0.9% | 7.898 |
| historical coverage replay | 950PR | [1, 128, 64, 1, 1] | — | 8.687 | 8.190 | +6.1% | 7.963 |
| historical coverage replay | 950PR | [1, 224, 64, 1, 1] | — | 8.946 | 8.634 | +3.6% | 8.200 |
| historical coverage replay | 950PR | [4, 256, 128, 1, 1] | — | 9.353 | 8.520 | +9.8% | 8.594 |
| historical coverage replay | 950PR | [8, 192, 64, 1, 1] | — | 8.974 | 8.552 | +4.9% | 8.247 |
| historical coverage replay | 950PR | [9, 256, 64, 4, 2] | — | 8.981 | 9.164 | -2.0% | 8.253 |
| historical coverage replay | 950PR | [9, 256, 128, 4, 1] | — | 9.401 | 9.058 | +3.8% | 8.658 |
| historical coverage replay | 950PR | [9, 384, 128, 2, 2] | — | 9.897 | 9.421 | +5.1% | 9.113 |
| historical coverage replay | 950PR | [9, 2048, 64, 1, 1] | — | 14.312 | 13.273 | +7.8% | 13.139 |
| historical coverage replay | 950PR | [9, 2049, 64, 1, 1] | — | 14.570 | 13.146 | +10.8% | 13.376 |
| historical coverage replay | 950PR | [9, 4096, 64, 1, 1] | — | 18.449 | 16.880 | +9.3% | 16.932 |
| historical coverage replay | 950PR | [9, 8192, 64, 1, 1] | — | 26.725 | 25.149 | +6.3% | 24.518 |
| historical coverage replay | 950PR | [9, 12289, 64, 1, 1] | — | 35.259 | 33.039 | +6.7% | 32.341 |
| historical coverage replay | 950PR | [9, 16385, 64, 1, 1] | — | 43.534 | 41.293 | +5.4% | 39.927 |
| historical coverage replay | 950PR | [9, 32767, 64, 1, 1] | — | 76.377 | 73.035 | +4.6% | 70.033 |
| historical coverage replay | 950PR | [9, 32768, 64, 1, 1] | — | 76.377 | 72.856 | +4.8% | 70.033 |
| historical coverage replay | 950PR | [16, 64, 64, 1, 1] | — | 8.679 | 8.472 | +2.4% | 7.997 |
| historical coverage replay | 950PR | [16, 128, 64, 1, 4] | — | 8.751 | 8.539 | +2.5% | 8.063 |
| historical coverage replay | 950PR | [16, 128, 64, 1, 8] | — | 8.751 | 8.731 | +0.2% | 8.063 |
| historical coverage replay | 950PR | [16, 128, 128, 1, 1] | — | 8.957 | 8.224 | +8.9% | 8.293 |
| historical coverage replay | 950PR | [16, 1025, 64, 1, 1] | — | 12.529 | 11.365 | +10.2% | 12.046 |
| historical coverage replay | 950PR | [17, 96, 64, 1, 1] | — | 8.737 | 8.736 | +0.0% | 8.050 |
| historical coverage replay | 950PR | [17, 257, 128, 1, 1] | — | 9.765 | 9.144 | +6.8% | 9.057 |
| historical coverage replay | 950PR | [17, 384, 64, 1, 2] | — | 9.295 | 9.347 | -0.6% | 8.865 |
| historical coverage replay | 950PR | [17, 1153, 64, 1, 1] | — | 12.834 | 11.489 | +11.7% | 12.447 |
| historical coverage replay | 950PR | [32, 129, 64, 1, 4] | — | 9.099 | 8.850 | +2.8% | 8.423 |
| historical coverage replay | 950PR | [32, 512, 64, 1, 1] | — | 10.072 | 9.345 | +7.8% | 10.382 |
| historical coverage replay | 950PR | [64, 256, 64, 1, 1] | — | 9.865 | 8.935 | +10.4% | 10.464 |
| historical coverage replay | 950PR | [64, 320, 128, 1, 1] | — | 11.280 | 9.582 | +17.7% | 12.691 |
| historical coverage replay | 950PR | [128, 128, 64, 1, 1] | — | 9.043 | 8.793 | +2.8% | 9.001 |
| historical coverage replay | 950PR | [129, 128, 64, 1, 1] | — | 9.043 | 8.508 | +6.3% | 9.001 |

## RMSNorm

| 数据组 | 芯片 | shape | v0.20 | 当前 | CA | 有符号误差 | 970DT 外推 |
|---|---|---|---:|---:|---:|---:|---:|
| DT v0.12 development replay | 950DT | [9, 5120] | — | 3.241 | 4.188 | -22.6% | 2.907 |
| DT v0.12 development replay | 950DT | [36, 7168] | — | 5.952 | 6.614 | -10.0% | 5.228 |
| DT v0.12 development replay | 950DT | [1, 128] | — | 2.226 | 3.454 | -35.6% | 2.040 |
| DT v0.12 development replay | 950DT | [1, 256] | — | 2.239 | 3.637 | -38.4% | 2.052 |
| DT v0.12 development replay | 950DT | [4, 512] | — | 2.437 | 4.292 | -43.2% | 2.208 |
| DT v0.12 development replay | 950DT | [16, 1024] | — | 3.832 | 3.815 | +0.5% | 3.402 |
| DT v0.12 development replay | 950DT | [9, 7168] | — | 3.339 | 2.790 | +19.7% | 2.998 |
| DT v0.12 development replay | 950DT | [36, 5120] | — | 5.856 | 5.289 | +10.7% | 5.140 |
| DT v0.12 development replay | 950DT | [32, 4096] | — | 5.734 | 4.581 | +25.2% | 5.052 |
| DT v0.12 development replay | 950DT | [64, 2048] | — | 6.220 | 4.352 | +42.9% | 5.312 |
| DT v0.12 development replay | 950DT | [65, 128] | — | 4.130 | 3.084 | +33.9% | 5.241 |
| DT v0.12 development replay | 950DT | [128, 1024] | — | 4.787 | 3.090 | +54.9% | 3.985 |
| DT v0.12 development replay | 950DT | [1, 8192] | — | 3.034 | 2.055 | +47.6% | 2.780 |
| DT v0.12 development replay | 950DT | [2, 128] | — | 2.249 | 3.407 | -34.0% | 2.053 |
| DT v0.12 development replay | 950DT | [3, 384] | — | 2.312 | 2.870 | -19.4% | 2.101 |
| DT v0.12 development replay | 950DT | [8, 1536] | — | 2.953 | 3.772 | -21.7% | 2.650 |
| DT v0.12 development replay | 950DT | [15, 3072] | — | 3.818 | 3.682 | +3.7% | 3.396 |
| DT v0.12 development replay | 950DT | [17, 7168] | — | 4.235 | 3.867 | +9.5% | 3.767 |
| DT v0.12 development replay | 950DT | [24, 4096] | — | 4.861 | 4.310 | +12.8% | 4.297 |
| DT v0.12 development replay | 950DT | [63, 512] | — | 6.131 | 4.432 | +38.3% | 5.235 |
| DT v0.12 development replay | 950DT | [66, 256] | — | 4.142 | 2.831 | +46.3% | 5.257 |
| DT v0.12 development replay | 950DT | [96, 2048] | — | 4.592 | 3.448 | +33.2% | 5.658 |
| DT v0.12 development replay | 950DT | [1, 6144] | — | 2.831 | 1.937 | +46.1% | 2.595 |
| DT v0.12 development replay | 950DT | [2, 4608] | — | 2.700 | 3.819 | -29.3% | 2.466 |
| DT v0.12 development replay | 950DT | [7, 640] | — | 2.794 | 3.388 | -17.5% | 2.511 |
| DT v0.12 development replay | 950DT | [1, 5120] | — | 2.729 | 1.927 | +41.6% | 2.501 |
| DT v0.12 development replay | 950DT | [34, 1536] | — | 5.651 | 5.576 | +1.3% | 4.963 |
| DT v0.12 development replay | 950DT | [4, 8192] | — | 3.099 | 2.395 | +29.4% | 2.816 |
| DT v0.12 development replay | 950DT | [64, 768] | — | 6.160 | 4.410 | +39.7% | 5.257 |
| DT v0.12 development replay | 950DT | [65, 384] | — | 4.154 | 3.073 | +35.2% | 5.252 |
| DT v0.12 development replay | 950DT | [2, 7168] | — | 2.954 | 2.127 | +38.9% | 2.699 |
| DT v0.12 development replay | 950DT | [96, 1280] | — | 4.520 | 3.601 | +25.5% | 5.625 |
| DT v0.12 development replay | 950DT | [3, 3072] | — | 2.569 | 1.840 | +39.6% | 2.338 |
| DT v0.12 development replay | 950DT | [3, 5120] | — | 2.773 | 3.808 | -27.2% | 2.525 |
| DT v0.12 development replay | 950DT | [2, 5376] | — | 2.777 | 1.979 | +40.3% | 2.537 |
| DT v0.12 development replay | 950DT | [4, 4608] | — | 2.745 | 2.633 | +4.3% | 2.490 |
| DT v0.12 development replay | 950DT | [1, 6912] | — | 2.907 | 2.008 | +44.7% | 2.664 |
| DT v0.12 development replay | 950DT | [33, 1536] | — | 5.632 | 5.548 | +1.5% | 4.952 |
| DT v0.12 development replay | 950DT | [38, 1536] | — | 5.725 | 4.921 | +16.3% | 5.007 |
| DT v0.12 development replay | 950DT | [64, 4096] | — | 6.315 | 6.457 | -2.2% | 5.401 |
| DT v0.12 development replay | 950DT | [128, 4096] | — | 5.074 | 3.838 | +32.2% | 4.251 |
| DT v0.12 development replay | 950DT | [256, 4096] | — | 5.042 | 3.589 | +40.5% | 4.227 |
| DT v0.12 development replay | 950DT | [512, 4096] | — | 6.005 | 4.851 | +23.8% | 5.002 |
| DT v0.12 holdout replay | 950DT | [1, 4352] | — | 2.652 | 3.837 | -30.9% | 2.431 |
| DT v0.12 holdout replay | 950DT | [2, 5888] | — | 2.828 | 4.542 | -37.8% | 2.583 |
| DT v0.12 holdout replay | 950DT | [4, 5632] | — | 2.846 | 2.841 | +0.2% | 2.584 |
| DT v0.12 holdout replay | 950DT | [3, 6656] | — | 2.926 | 2.119 | +38.0% | 2.665 |
| DT v0.12 holdout replay | 950DT | [37, 1536] | — | 5.707 | 5.454 | +4.6% | 4.996 |
| DT v0.12 holdout replay | 950DT | [192, 3072] | — | 4.644 | 4.850 | -4.2% | 4.512 |
| DT v0.12 holdout replay | 950DT | [448, 4096] | — | 6.200 | 4.968 | +24.8% | 4.743 |
| DT v0.12 holdout replay | 950DT | [480, 4096] | — | 5.934 | 5.175 | +14.7% | 4.830 |
| DT v0.12 holdout replay | 950DT | [384, 6144] | — | 6.238 | 6.838 | -8.8% | 5.283 |
| DT v0.12 holdout replay | 950DT | [512, 6144] | — | 7.129 | 6.374 | +11.8% | 5.377 |
| prior published pair replay | 950DT | [3, 6080] | — | 2.872 | 3.868 | -25.8% | 2.616 |
| prior published pair replay | 950PR | [3, 6080] | — | 2.820 | 3.604 | -21.7% | 2.616 |
| prior published pair replay | 950DT | [7, 7360] | — | 3.122 | 4.050 | -22.9% | 2.814 |
| prior published pair replay | 950PR | [7, 7360] | — | 3.072 | 3.766 | -18.4% | 2.814 |
| prior published pair replay | 950DT | [18, 4992] | — | 4.243 | 4.995 | -15.1% | 3.767 |
| prior published pair replay | 950PR | [18, 4992] | — | 4.201 | 5.028 | -16.4% | 3.767 |
| prior published pair replay | 950DT | [38, 2624] | — | 5.778 | 6.330 | -8.7% | 5.055 |
| prior published pair replay | 950PR | [38, 2624] | — | 5.759 | 6.238 | -7.7% | 5.055 |
| prior published pair replay | 950DT | [57, 2304] | — | 6.107 | 6.394 | -4.5% | 5.247 |
| prior published pair replay | 950PR | [57, 2304] | — | 4.233 | 4.592 | -7.8% | 5.247 |
| prior published pair replay | 950DT | [69, 3328] | — | 4.469 | 4.867 | -8.2% | 5.422 |
| prior published pair replay | 950PR | [69, 3328] | — | 4.454 | 4.788 | -7.0% | 5.422 |
| prior published pair replay | 950DT | [103, 2944] | — | 4.749 | 5.104 | -7.0% | 5.771 |
| prior published pair replay | 950PR | [103, 2944] | — | 4.755 | 4.839 | -1.7% | 5.771 |
| prior published pair replay | 950DT | [229, 3968] | — | 4.886 | 5.587 | -12.5% | 4.795 |
| prior published pair replay | 950PR | [229, 3968] | — | 5.063 | 5.427 | -6.7% | 4.795 |
| prior published pair replay | 950DT | [367, 5312] | — | 5.867 | 6.665 | -12.0% | 4.963 |
| prior published pair replay | 950PR | [367, 5312] | — | 6.148 | 7.618 | -19.3% | 4.963 |
| prior published pair replay | 950DT | [495, 6912] | — | 7.547 | 7.846 | -3.8% | 5.564 |
| prior published pair replay | 950PR | [495, 6912] | — | 7.489 | 10.059 | -25.5% | 5.564 |
| v0.17 historical RMS replay | 950DT | [9, 5120] | — | 3.241 | 4.161 | -22.1% | 2.907 |
| v0.17 historical RMS replay | 950PR | [9, 5120] | — | 3.191 | 3.748 | -14.9% | 2.907 |
| v0.17 historical RMS replay | 950DT | [36, 7168] | — | 5.952 | 6.614 | -10.0% | 5.228 |
| v0.17 historical RMS replay | 950PR | [36, 7168] | — | 5.932 | 6.468 | -8.3% | 5.228 |
| v0.17 historical RMS replay | 950DT | [1, 128] | — | 2.226 | 3.454 | -35.6% | 2.040 |
| v0.17 historical RMS replay | 950PR | [1, 128] | — | 2.169 | 3.129 | -30.7% | 2.040 |
| v0.17 historical RMS replay | 950DT | [1, 256] | — | 2.239 | 3.637 | -38.4% | 2.052 |
| v0.17 historical RMS replay | 950PR | [1, 256] | — | 2.183 | 3.155 | -30.8% | 2.052 |
| v0.17 historical RMS replay | 950DT | [4, 512] | — | 2.437 | 4.292 | -43.2% | 2.208 |
| v0.17 historical RMS replay | 950PR | [4, 512] | — | 2.384 | 3.473 | -31.3% | 2.208 |
| v0.17 historical RMS replay | 950DT | [16, 1024] | — | 3.832 | 3.815 | +0.5% | 3.402 |
| v0.17 historical RMS replay | 950PR | [16, 1024] | — | 3.787 | 4.242 | -10.7% | 3.402 |
| v0.17 historical RMS replay | 950DT | [9, 7168] | — | 3.339 | 2.790 | +19.7% | 2.998 |
| v0.17 historical RMS replay | 950PR | [9, 7168] | — | 3.290 | 3.776 | -12.9% | 2.998 |
| v0.17 historical RMS replay | 950DT | [36, 5120] | — | 5.856 | 5.289 | +10.7% | 5.140 |
| v0.17 historical RMS replay | 950PR | [36, 5120] | — | 5.835 | 6.183 | -5.6% | 5.140 |
| v0.17 historical RMS replay | 950DT | [32, 4096] | — | 5.734 | 4.581 | +25.2% | 5.052 |
| v0.17 historical RMS replay | 950PR | [32, 4096] | — | 5.707 | 6.096 | -6.4% | 5.052 |
| v0.17 historical RMS replay | 950DT | [64, 2048] | — | 6.220 | 4.352 | +42.9% | 5.312 |
| v0.17 historical RMS replay | 950PR | [64, 2048] | — | 4.271 | 4.568 | -6.5% | 5.312 |
| v0.18 historical RMS replay | 950DT | [1, 5504] | — | 2.767 | 3.763 | -26.5% | 2.536 |
| v0.18 historical RMS replay | 950PR | [1, 5504] | — | 2.711 | 3.573 | -24.1% | 2.536 |
| v0.18 historical RMS replay | 950DT | [3, 7552] | — | 3.014 | 3.728 | -19.2% | 2.746 |
| v0.18 historical RMS replay | 950PR | [3, 7552] | — | 2.963 | 3.841 | -22.9% | 2.746 |
| v0.18 historical RMS replay | 950DT | [8, 6016] | — | 3.170 | 3.844 | -17.5% | 2.850 |
| v0.18 historical RMS replay | 950PR | [8, 6016] | — | 3.120 | 3.685 | -15.3% | 2.850 |
| v0.18 historical RMS replay | 950DT | [18, 6912] | — | 4.333 | 4.897 | -11.5% | 3.851 |
| v0.18 historical RMS replay | 950PR | [18, 6912] | — | 4.292 | 4.553 | -5.7% | 3.851 |
| v0.18 historical RMS replay | 950DT | [55, 1728] | — | 6.046 | 6.044 | +0.0% | 5.202 |
| v0.18 historical RMS replay | 950PR | [55, 1728] | — | 6.049 | 5.736 | +5.5% | 5.202 |
| v0.18 historical RMS replay | 950DT | [57, 1536] | — | 6.071 | 6.085 | -0.2% | 5.214 |
| v0.18 historical RMS replay | 950PR | [57, 1536] | — | 4.160 | 4.451 | -6.5% | 5.214 |
| v0.18 historical RMS replay | 950DT | [65, 2816] | — | 4.383 | 4.912 | -10.8% | 5.356 |
| v0.18 historical RMS replay | 950PR | [65, 2816] | — | 4.364 | 4.593 | -5.0% | 5.356 |
| v0.18 historical RMS replay | 950DT | [113, 2112] | — | 4.766 | 5.076 | -6.1% | 5.844 |
| v0.18 historical RMS replay | 950PR | [113, 2112] | — | 4.026 | 4.367 | -7.8% | 5.844 |
| v0.18 historical RMS replay | 950DT | [393, 3840] | — | 5.907 | 6.088 | -3.0% | 4.514 |
| v0.18 historical RMS replay | 950PR | [393, 3840] | — | 5.619 | 6.853 | -18.0% | 4.514 |
| v0.18 historical RMS replay | 950DT | [511, 3968] | — | 5.920 | 6.613 | -10.5% | 4.947 |
| v0.18 historical RMS replay | 950PR | [511, 3968] | — | 6.292 | 7.773 | -19.1% | 4.947 |
| v0.20 frozen RMS replay | 950DT | [32, 1280] | — | 5.602 | 6.392 | -12.4% | 4.930 |
| v0.20 frozen RMS replay | 950PR | [32, 1280] | — | 5.573 | 6.309 | -11.7% | 4.930 |
| v0.20 frozen RMS replay | 950DT | [96, 1536] | — | 4.544 | 4.819 | -5.7% | 5.636 |
| v0.20 frozen RMS replay | 950PR | [96, 1536] | — | 4.543 | 4.670 | -2.7% | 5.636 |
| v0.20 frozen RMS replay | 950DT | [512, 1280] | — | 4.463 | 4.827 | -7.5% | 3.804 |
| v0.20 frozen RMS replay | 950PR | [512, 1280] | — | 4.507 | 5.047 | -10.7% | 3.804 |
| v0.20 frozen RMS replay | 950DT | [768, 1536] | — | 5.235 | 5.522 | -5.2% | 4.232 |
| v0.20 frozen RMS replay | 950PR | [768, 1536] | — | 5.398 | 6.008 | -10.2% | 4.232 |
| v0.20 frozen RMS replay | 950DT | [1024, 5120] | — | 8.534 | 10.000 | -14.7% | 6.482 |
| v0.20 frozen RMS replay | 950PR | [1024, 5120] | — | 9.621 | 12.051 | -20.2% | 6.482 |
| v0.20 frozen RMS replay | 950DT | [1536, 7168] | — | 13.865 | 14.834 | -6.5% | 8.675 |
| v0.20 frozen RMS replay | 950PR | [1536, 7168] | — | 17.428 | 19.896 | -12.4% | 8.675 |
| v0.20 frozen RMS replay | 950DT | [2048, 1280] | — | 6.553 | 6.981 | -6.1% | 5.343 |
| v0.20 frozen RMS replay | 950PR | [2048, 1280] | — | 6.945 | 9.008 | -22.9% | 5.343 |
| v0.20 frozen RMS replay | 950DT | [4096, 1536] | — | 9.541 | 10.825 | -11.9% | 7.048 |
| v0.20 frozen RMS replay | 950PR | [4096, 1536] | — | 10.704 | 13.522 | -20.8% | 7.048 |
| v0.20 frozen RMS replay | 950DT | [8192, 5120] | — | 39.379 | 44.868 | -12.2% | 21.582 |
| v0.20 frozen RMS replay | 950PR | [8192, 5120] | — | 61.622 | 70.485 | -12.6% | 21.582 |

## Addmm

| 数据组 | 芯片 | shape | v0.20 | 当前 | CA | 有符号误差 | 970DT 外推 |
|---|---|---|---:|---:|---:|---:|---:|
| DT v0.12 development replay | 950DT | [16, 64, 256] | — | 3.653 | 3.447 | +6.0% | 3.348 |
| DT v0.12 development replay | 950DT | [9, 64, 128] | — | 3.364 | 3.342 | +0.7% | 3.084 |
| DT v0.12 development replay | 950DT | [2, 16, 16] | — | 3.083 | 2.992 | +3.0% | 2.826 |
| DT v0.12 development replay | 950DT | [128, 192, 256] | — | 3.908 | 3.763 | +3.8% | 3.581 |
| DT v0.12 development replay | 950DT | [16, 192, 384] | — | 4.042 | 3.877 | +4.2% | 3.705 |
| DT v0.12 development replay | 950DT | [256, 512, 256] | — | 4.088 | 4.327 | -5.5% | 3.695 |
| DT v0.12 development replay | 950DT | [4, 80, 48] | — | 3.239 | 3.625 | -10.7% | 2.969 |
| DT v0.12 development replay | 950DT | [512, 512, 256] | — | 4.773 | 4.479 | +6.6% | 3.761 |
| DT v0.12 development replay | 950DT | [64, 128, 512] | — | 4.350 | 4.519 | -3.8% | 3.987 |
| DT v0.12 development replay | 950DT | [1, 32, 16] | — | 3.092 | 3.288 | -6.0% | 2.834 |
| DT v0.12 development replay | 950DT | [2, 32, 32] | — | 3.110 | 3.408 | -8.8% | 2.851 |
| DT v0.12 development replay | 950DT | [31, 64, 257] | — | 3.783 | 4.253 | -11.0% | 3.468 |
| DT v0.12 development replay | 950DT | [9, 128, 256] | — | 3.705 | 3.756 | -1.4% | 3.396 |
| DT v0.12 holdout replay | 950DT | [128, 256, 384] | — | 4.235 | 4.275 | -0.9% | 3.878 |
| DT v0.12 holdout replay | 950DT | [17, 128, 128] | — | 3.478 | 3.393 | +2.5% | 3.188 |
| DT v0.12 holdout replay | 950DT | [1, 64, 32] | — | 3.147 | 3.145 | +0.0% | 2.884 |
| DT v0.12 holdout replay | 950DT | [256, 384, 256] | — | 4.052 | 4.381 | -7.5% | 3.682 |
| DT v0.12 holdout replay | 950DT | [2, 64, 32] | — | 3.147 | 3.423 | -8.1% | 2.885 |
| DT v0.12 holdout replay | 950DT | [32, 80, 272] | — | 3.863 | 4.228 | -8.6% | 3.541 |
| DT v0.12 holdout replay | 950DT | [3, 48, 96] | — | 3.236 | 3.352 | -3.5% | 2.966 |
| DT v0.12 holdout replay | 950DT | [48, 256, 384] | — | 4.152 | 4.027 | +3.1% | 3.806 |
| DT v0.12 holdout replay | 950DT | [8, 96, 192] | — | 3.562 | 3.687 | -3.4% | 3.265 |
| DT v0.12 holdout replay | 950DT | [96, 192, 256] | — | 3.883 | 3.925 | -1.1% | 3.560 |
| prior published pair replay | 950DT | [16, 64, 256] | — | 3.653 | 3.588 | +1.8% | 3.348 |
| prior published pair replay | 950PR | [16, 64, 256] | — | 3.653 | 3.215 | +13.6% | 3.348 |
| prior published pair replay | 950DT | [9, 64, 128] | — | 3.364 | 3.342 | +0.7% | 3.084 |
| prior published pair replay | 950PR | [9, 64, 128] | — | 3.364 | 3.059 | +10.0% | 3.084 |
| prior published pair replay | 950DT | [2, 16, 16] | — | 3.083 | 2.992 | +3.0% | 2.826 |
| prior published pair replay | 950PR | [2, 16, 16] | — | 3.083 | 2.728 | +13.0% | 2.826 |
| prior published pair replay | 950DT | [128, 192, 256] | — | 3.908 | 3.763 | +3.8% | 3.581 |
| prior published pair replay | 950PR | [128, 192, 256] | — | 3.909 | 3.753 | +4.2% | 3.581 |
| prior published pair replay | 950DT | [16, 192, 384] | — | 4.042 | 3.877 | +4.2% | 3.705 |
| prior published pair replay | 950PR | [16, 192, 384] | — | 4.042 | 3.741 | +8.0% | 3.705 |
| prior published pair replay | 950DT | [256, 512, 256] | — | 4.088 | 4.327 | -5.5% | 3.695 |
| prior published pair replay | 950PR | [256, 512, 256] | — | 4.392 | 4.230 | +3.8% | 3.695 |
| prior published pair replay | 950DT | [4, 80, 48] | — | 3.239 | 3.625 | -10.7% | 2.969 |
| prior published pair replay | 950PR | [4, 80, 48] | — | 3.239 | 3.307 | -2.0% | 2.969 |
| prior published pair replay | 950DT | [512, 512, 256] | — | 4.773 | 4.479 | +6.6% | 3.761 |
| prior published pair replay | 950PR | [512, 512, 256] | — | 4.856 | 4.687 | +3.6% | 3.761 |
| prior published pair replay | 950DT | [64, 128, 512] | — | 4.350 | 4.519 | -3.8% | 3.987 |
| prior published pair replay | 950PR | [64, 128, 512] | — | 4.350 | 4.293 | +1.3% | 3.987 |
| prior published pair replay | 950DT | [1, 32, 16] | — | 3.092 | 3.288 | -6.0% | 2.834 |
| prior published pair replay | 950PR | [1, 32, 16] | — | 3.092 | 3.002 | +3.0% | 2.834 |
| primary | 950PR | [17, 80, 96] | 3.714 | 3.407 | 3.410 | -0.1% | 3.123 |
| primary | 950DT | [272, 480, 192] | 4.679 | 4.044 | 4.538 | -10.9% | 3.564 |
| primary | 950PR | [272, 480, 192] | 4.688 | 4.175 | 4.410 | -5.3% | 3.564 |
| primary | 950DT | [511, 384, 192] | 4.661 | 4.333 | 4.158 | +4.2% | 3.585 |
| primary | 950PR | [511, 384, 192] | 4.671 | 4.448 | 4.253 | +4.6% | 3.585 |
| primary | 950DT | [1, 48, 80] | 3.511 | 3.208 | 3.078 | +4.2% | 2.941 |
| primary | 950PR | [1, 48, 80] | 3.511 | 3.208 | 2.828 | +13.5% | 2.941 |
| primary | 950DT | [31, 144, 160] | 3.752 | 3.582 | 4.082 | -12.2% | 3.283 |
| primary | 950PR | [31, 144, 160] | 3.752 | 3.582 | 3.818 | -6.2% | 3.283 |
| primary | 950DT | [96, 176, 320] | 3.853 | 4.069 | 4.580 | -11.1% | 3.730 |
| primary | 950PR | [96, 176, 320] | 3.856 | 4.069 | 4.499 | -9.6% | 3.730 |
| primary | 950DT | [192, 384, 192] | 4.178 | 3.842 | 3.840 | +0.1% | 3.530 |
| primary | 950PR | [192, 384, 192] | 4.183 | 3.846 | 3.794 | +1.4% | 3.530 |
| primary | 950DT | [257, 256, 192] | 4.190 | 3.839 | 3.645 | +5.3% | 3.526 |
| primary | 950PR | [257, 256, 192] | 4.195 | 3.843 | 3.750 | +2.5% | 3.526 |
| primary | 950DT | [384, 480, 256] | 4.694 | 4.428 | 5.079 | -12.8% | 3.725 |
| primary | 950PR | [384, 480, 256] | 4.707 | 4.731 | 5.075 | -6.8% | 3.725 |
| primary | 950DT | [1024, 192, 256] | 4.713 | 4.585 | 4.252 | +7.9% | 3.725 |
| primary | 950PR | [1024, 192, 256] | 7.101 | 4.736 | 4.449 | +6.4% | 3.725 |

## Lightning Indexer

| 数据组 | 芯片 | shape | v0.20 | 当前 | CA | 有符号误差 | 970DT 外推 |
|---|---|---|---:|---:|---:|---:|---:|
| DT v0.12 development replay | 950DT | [1, 1, 512, 8, 128] | — | 5.771 | 6.093 | -5.3% | 5.288 |
| DT v0.12 development replay | 950DT | [1, 1, 1024, 8, 128] | — | 6.755 | 6.741 | +0.2% | 6.178 |
| DT v0.12 development replay | 950DT | [1, 2, 768, 8, 128] | — | 6.682 | 6.365 | +5.0% | 6.120 |
| primary | 950DT | [1, 1, 640, 8, 128] | — | 6.652 | 6.200 | +7.3% | 6.093 |
| primary | 950DT | [1, 1, 896, 8, 128] | — | 6.711 | 6.439 | +4.2% | 6.146 |
| primary | 950DT | [1, 2, 1024, 8, 128] | — | 6.755 | 6.516 | +3.7% | 6.178 |
| primary | 950DT | [1, 4, 1024, 8, 128] | — | 6.755 | 8.043 | -16.0% | 6.178 |
| primary | 950DT | [1, 8, 1536, 8, 128] | — | 8.396 | 8.862 | -5.3% | 7.671 |
| primary | 950DT | [1, 1, 2048, 8, 128] | — | 10.051 | 8.614 | +16.7% | 9.178 |
| primary | 950DT | [1, 2, 3072, 8, 128] | — | 13.361 | 11.101 | +20.4% | 12.188 |
| primary | 950DT | [1, 4, 4096, 8, 128] | — | 16.670 | 15.674 | +6.4% | 15.196 |
| primary | 950DT | [1, 9, 4096, 8, 128] | — | 16.670 | 15.334 | +8.7% | 15.195 |
| primary | 950DT | [1, 1, 8192, 8, 128] | — | 29.909 | 22.709 | +31.7% | 27.231 |
| prior published pair replay | 950DT | [1, 1, 512, 8, 128] | — | 5.771 | 5.862 | -1.6% | 5.288 |
| primary | 950PR | [1, 1, 512, 8, 128] | — | 5.790 | 5.779 | +0.2% | 5.288 |
| prior published pair replay | 950DT | [1, 1, 1024, 8, 128] | — | 6.755 | 6.741 | +0.2% | 6.178 |
| primary | 950PR | [1, 1, 1024, 8, 128] | — | 6.822 | 6.493 | +5.1% | 6.178 |
| prior published pair replay | 950DT | [1, 2, 768, 8, 128] | — | 6.682 | 6.365 | +5.0% | 6.120 |
| primary | 950PR | [1, 2, 768, 8, 128] | — | 6.733 | 6.171 | +9.1% | 6.120 |
| prior published pair replay | 950DT | [1, 1, 640, 8, 128] | — | 6.652 | 6.200 | +7.3% | 6.093 |
| primary | 950PR | [1, 1, 640, 8, 128] | — | 6.700 | 5.935 | +12.9% | 6.093 |
| prior published pair replay | 950DT | [1, 1, 896, 8, 128] | — | 6.711 | 6.439 | +4.2% | 6.146 |
| primary | 950PR | [1, 1, 896, 8, 128] | — | 6.765 | 6.129 | +10.4% | 6.146 |
| prior published pair replay | 950DT | [1, 2, 1024, 8, 128] | — | 6.755 | 6.516 | +3.7% | 6.178 |
| primary | 950PR | [1, 2, 1024, 8, 128] | — | 6.823 | 6.570 | +3.8% | 6.178 |
| prior published pair replay | 950DT | [1, 4, 1024, 8, 128] | — | 6.755 | 8.043 | -16.0% | 6.178 |
| primary | 950PR | [1, 4, 1024, 8, 128] | — | 6.824 | 8.030 | -15.0% | 6.178 |
| prior published pair replay | 950DT | [1, 8, 1536, 8, 128] | — | 8.396 | 8.862 | -5.3% | 7.671 |
| primary | 950PR | [1, 8, 1536, 8, 128] | — | 8.521 | 8.829 | -3.5% | 7.671 |
| prior published pair replay | 950DT | [1, 1, 2048, 8, 128] | — | 10.051 | 8.614 | +16.7% | 9.178 |
| primary | 950PR | [1, 1, 2048, 8, 128] | — | 10.220 | 8.455 | +20.9% | 9.178 |
| prior published pair replay | 950DT | [1, 2, 3072, 8, 128] | — | 13.361 | 11.101 | +20.4% | 12.188 |
| primary | 950PR | [1, 2, 3072, 8, 128] | — | 13.633 | 10.776 | +26.5% | 12.188 |

## mHC Pre

| 数据组 | 芯片 | shape | v0.20 | 当前 | CA | 有符号误差 | 970DT 外推 |
|---|---|---|---:|---:|---:|---:|---:|
| DT v0.12 development replay | 950DT | [1, 4, 128] | — | 8.378 | 8.271 | +1.3% | 7.679 |
| DT v0.12 development replay | 950DT | [8, 4, 128] | — | 8.394 | 8.542 | -1.7% | 7.694 |
| DT v0.12 development replay | 950DT | [3, 4, 128] | — | 8.382 | 8.312 | +0.8% | 7.684 |
| DT v0.12 development replay | 950DT | [1, 4, 64] | — | 9.579 | 9.482 | +1.0% | 8.780 |
| DT v0.12 development replay | 950DT | [2, 4, 128] | — | 8.380 | 8.005 | +4.7% | 7.681 |
| DT v0.12 development replay | 950DT | [4, 4, 256] | — | 8.505 | 8.421 | +1.0% | 7.796 |
| DT v0.12 development replay | 950DT | [16, 4, 128] | — | 8.412 | 9.308 | -9.6% | 7.711 |
| DT v0.12 development replay | 950DT | [1, 4, 512] | — | 8.717 | 8.544 | +2.0% | 7.989 |
| DT v0.12 development replay | 950DT | [4, 4, 1024] | — | 9.226 | 8.730 | +5.7% | 8.450 |
| DT v0.12 development replay | 950DT | [9, 4, 1024] | — | 9.318 | 9.024 | +3.3% | 8.533 |
| DT v0.12 development replay | 950DT | [16, 4, 512] | — | 8.856 | 9.385 | -5.6% | 8.115 |
| DT v0.12 development replay | 950DT | [32, 4, 1024] | — | 10.996 | 11.368 | -3.3% | 10.064 |
| DT v0.12 development replay | 950DT | [1, 4, 80] | — | 9.705 | 9.802 | -1.0% | 8.896 |
| primary | 950DT | [2, 4, 64] | — | 9.589 | 9.962 | -3.7% | 8.790 |
| primary | 950DT | [4, 4, 96] | — | 9.879 | 9.312 | +6.1% | 9.055 |
| primary | 950DT | [6, 4, 128] | — | 8.389 | 8.410 | -0.2% | 7.690 |
| primary | 950DT | [5, 4, 256] | — | 8.509 | 8.251 | +3.1% | 7.800 |
| primary | 950DT | [12, 4, 384] | — | 8.680 | 9.137 | -5.0% | 7.956 |
| primary | 950DT | [8, 4, 512] | — | 8.782 | 8.866 | -0.9% | 8.048 |
| primary | 950DT | [12, 4, 768] | — | 9.097 | 9.058 | +0.4% | 8.333 |
| primary | 950DT | [6, 4, 1024] | — | 9.263 | 8.973 | +3.2% | 8.483 |
| primary | 950DT | [17, 4, 512] | — | 10.118 | 10.095 | +0.2% | 9.272 |
| primary | 950DT | [24, 4, 1536] | — | 11.514 | 12.201 | -5.6% | 9.632 |
| prior published pair replay | 950DT | [1, 4, 128] | — | 8.378 | 7.879 | +6.3% | 7.679 |
| primary | 950PR | [1, 4, 128] | — | 8.378 | 7.638 | +9.7% | 7.679 |
| prior published pair replay | 950DT | [8, 4, 128] | — | 8.394 | 8.542 | -1.7% | 7.694 |
| primary | 950PR | [8, 4, 128] | — | 8.394 | 8.072 | +4.0% | 7.694 |
| prior published pair replay | 950DT | [3, 4, 128] | — | 8.382 | 8.312 | +0.8% | 7.684 |
| primary | 950PR | [3, 4, 128] | — | 8.383 | 7.921 | +5.8% | 7.684 |
| prior published pair replay | 950DT | [1, 4, 64] | — | 9.579 | 9.482 | +1.0% | 8.780 |
| primary | 950PR | [1, 4, 64] | — | 9.579 | 8.952 | +7.0% | 8.780 |
| prior published pair replay | 950DT | [2, 4, 128] | — | 8.380 | 8.005 | +4.7% | 7.681 |
| primary | 950PR | [2, 4, 128] | — | 8.380 | 7.644 | +9.6% | 7.681 |
| prior published pair replay | 950DT | [4, 4, 256] | — | 8.505 | 8.421 | +1.0% | 7.796 |
| primary | 950PR | [4, 4, 256] | — | 8.506 | 8.085 | +5.2% | 7.796 |
| prior published pair replay | 950DT | [16, 4, 128] | — | 8.412 | 9.308 | -9.6% | 7.711 |
| primary | 950PR | [16, 4, 128] | — | 8.413 | 8.696 | -3.3% | 7.711 |
| prior published pair replay | 950DT | [1, 4, 512] | — | 8.717 | 8.544 | +2.0% | 7.989 |
| primary | 950PR | [1, 4, 512] | — | 8.722 | 7.876 | +10.7% | 7.989 |
| prior published pair replay | 950DT | [4, 4, 1024] | — | 9.226 | 8.730 | +5.7% | 8.450 |
| primary | 950PR | [4, 4, 1024] | — | 9.247 | 9.420 | -1.8% | 8.450 |
| prior published pair replay | 950DT | [9, 4, 1024] | — | 9.318 | 9.024 | +3.3% | 8.533 |
| primary | 950PR | [9, 4, 1024] | — | 9.344 | 9.865 | -5.3% | 8.533 |

## mHC Sinkhorn

| 数据组 | 芯片 | shape | v0.20 | 当前 | CA | 有符号误差 | 970DT 外推 |
|---|---|---|---:|---:|---:|---:|---:|
| DT v0.12 development replay | 950DT | [1, 4] | — | 5.786 | 6.296 | -8.1% | 5.300 |
| DT v0.12 development replay | 950DT | [64, 4] | — | 6.040 | 7.598 | -20.5% | 5.533 |
| DT v0.12 development replay | 950DT | [17, 4] | — | 5.959 | 6.532 | -8.8% | 5.459 |
| DT v0.12 development replay | 950DT | [2, 4] | — | 3.077 | 3.593 | -14.4% | 2.820 |
| DT v0.12 development replay | 950DT | [4, 4] | — | 3.578 | 2.945 | +21.5% | 3.279 |
| DT v0.12 development replay | 950DT | [8, 4] | — | 4.385 | 4.995 | -12.2% | 4.017 |
| DT v0.12 development replay | 950DT | [9, 4] | — | 5.920 | 6.212 | -4.7% | 5.423 |
| DT v0.12 development replay | 950DT | [16, 4] | — | 7.789 | 8.156 | -4.5% | 7.135 |
| DT v0.12 development replay | 950DT | [31, 4] | — | 5.996 | 6.333 | -5.3% | 5.493 |
| DT v0.12 development replay | 950DT | [32, 4] | — | 5.998 | 5.336 | +12.4% | 5.494 |
| DT v0.12 development replay | 950DT | [63, 4] | — | 3.747 | 4.101 | -8.6% | 3.434 |
| DT v0.12 development replay | 950DT | [65, 4] | — | 4.471 | 3.258 | +37.2% | 4.135 |
| DT v0.12 development replay | 950DT | [128, 4] | — | 6.040 | 5.475 | +10.3% | 5.533 |
| DT v0.12 holdout replay | 950DT | [3, 4] | — | 3.102 | 3.730 | -16.8% | 2.843 |
| DT v0.12 holdout replay | 950DT | [5, 4] | — | 3.744 | 4.066 | -7.9% | 3.431 |
| DT v0.12 holdout replay | 950DT | [7, 4] | — | 4.682 | 5.142 | -8.9% | 4.290 |
| DT v0.12 holdout replay | 950DT | [11, 4] | — | 5.932 | 6.406 | -7.4% | 5.435 |
| DT v0.12 holdout replay | 950DT | [15, 4] | — | 6.563 | 6.967 | -5.8% | 6.012 |
| DT v0.12 holdout replay | 950DT | [21, 4] | — | 5.361 | 5.597 | -4.2% | 4.911 |
| DT v0.12 holdout replay | 950DT | [33, 4] | — | 6.000 | 6.377 | -5.9% | 5.496 |
| DT v0.12 holdout replay | 950DT | [47, 4] | — | 3.576 | 3.996 | -10.5% | 3.277 |
| DT v0.12 holdout replay | 950DT | [66, 4] | — | 4.471 | 3.212 | +39.2% | 4.136 |
| DT v0.12 holdout replay | 950DT | [96, 4] | — | 6.023 | 5.334 | +12.9% | 5.556 |
| prior published pair replay | 950DT | [1, 4] | — | 5.786 | 6.149 | -5.9% | 5.300 |
| prior published pair replay | 950PR | [1, 4] | — | 5.869 | 5.948 | -1.3% | 5.300 |
| prior published pair replay | 950DT | [64, 4] | — | 6.040 | 7.598 | -20.5% | 5.533 |
| prior published pair replay | 950PR | [64, 4] | — | 6.081 | 6.533 | -6.9% | 5.533 |
| prior published pair replay | 950DT | [17, 4] | — | 5.959 | 6.532 | -8.8% | 5.459 |
| prior published pair replay | 950PR | [17, 4] | — | 6.042 | 6.280 | -3.8% | 5.459 |
| prior published pair replay | 950DT | [2, 4] | — | 3.077 | 3.593 | -14.4% | 2.820 |
| prior published pair replay | 950PR | [2, 4] | — | 3.085 | 3.697 | -16.5% | 2.820 |
| prior published pair replay | 950DT | [4, 4] | — | 3.578 | 2.945 | +21.5% | 3.279 |
| prior published pair replay | 950PR | [4, 4] | — | 3.599 | 3.938 | -8.6% | 3.279 |
| prior published pair replay | 950DT | [8, 4] | — | 4.385 | 4.995 | -12.2% | 4.017 |
| prior published pair replay | 950PR | [8, 4] | — | 4.426 | 4.719 | -6.2% | 4.017 |
| prior published pair replay | 950DT | [9, 4] | — | 5.920 | 6.212 | -4.7% | 5.423 |
| prior published pair replay | 950PR | [9, 4] | — | 6.003 | 6.268 | -4.2% | 5.423 |
| prior published pair replay | 950DT | [16, 4] | — | 7.789 | 8.156 | -4.5% | 7.135 |
| prior published pair replay | 950PR | [16, 4] | — | 7.922 | 8.328 | -4.9% | 7.135 |
| prior published pair replay | 950DT | [31, 4] | — | 5.996 | 6.333 | -5.3% | 5.493 |
| prior published pair replay | 950PR | [31, 4] | — | 6.079 | 6.634 | -8.4% | 5.493 |
| prior published pair replay | 950DT | [32, 4] | — | 5.998 | 5.336 | +12.4% | 5.494 |
| prior published pair replay | 950PR | [32, 4] | — | 6.081 | 6.616 | -8.1% | 5.494 |
| primary | 950DT | [57, 4] | 6.033 | 6.033 | 7.321 | -17.6% | 5.527 |
| primary | 950PR | [57, 4] | 10.536 | 6.075 | 6.410 | -5.2% | 5.527 |
| primary | 950DT | [6, 4] | 3.144 | 3.144 | 3.580 | -12.2% | 2.882 |
| primary | 950PR | [6, 4] | 3.169 | 3.153 | 3.438 | -8.3% | 2.882 |
| primary | 950DT | [14, 4] | 3.960 | 3.960 | 4.636 | -14.6% | 3.629 |
| primary | 950PR | [14, 4] | 4.006 | 3.989 | 4.395 | -9.2% | 3.629 |
| primary | 950DT | [28, 4] | 5.225 | 5.225 | 5.886 | -11.2% | 4.787 |
| primary | 950PR | [28, 4] | 5.304 | 5.288 | 5.691 | -7.1% | 4.787 |
| primary | 950DT | [54, 4] | 6.030 | 6.030 | 7.551 | -20.1% | 5.524 |
| primary | 950PR | [54, 4] | 6.129 | 6.113 | 7.287 | -16.1% | 5.524 |
| primary | 950DT | [55, 4] | 7.865 | 7.865 | 9.274 | -15.2% | 7.204 |
| primary | 950PR | [55, 4] | 8.014 | 7.998 | 9.111 | -12.2% | 7.204 |
| primary | 950DT | [56, 4] | 3.587 | 3.587 | 5.068 | -29.2% | 3.287 |
| primary | 950PR | [56, 4] | 3.620 | 3.603 | 4.464 | -19.3% | 3.287 |
| fusion distinct historical windows | 950DT | [1, 4] | — | 5.786 | 6.296 | -8.1% | 5.300 |
| fusion distinct historical windows | 950DT | [64, 4] | — | 6.040 | 7.598 | -20.5% | 5.533 |
| fusion distinct historical windows | 950DT | [17, 4] | — | 5.959 | 6.532 | -8.8% | 5.459 |
| fusion distinct historical windows | 950DT | [2, 4] | — | 3.077 | 3.593 | -14.4% | 2.820 |
| fusion distinct historical windows | 950DT | [4, 4] | — | 3.578 | 2.945 | +21.5% | 3.279 |
| fusion distinct historical windows | 950DT | [8, 4] | — | 4.385 | 4.995 | -12.2% | 4.017 |
| fusion distinct historical windows | 950DT | [9, 4] | — | 5.920 | 6.212 | -4.7% | 5.423 |
| fusion distinct historical windows | 950DT | [16, 4] | — | 7.789 | 8.156 | -4.5% | 7.135 |
| fusion distinct historical windows | 950DT | [31, 4] | — | 5.996 | 6.333 | -5.3% | 5.493 |
| fusion distinct historical windows | 950DT | [32, 4] | — | 5.998 | 5.336 | +12.4% | 5.494 |
| fusion distinct historical windows | 950DT | [63, 4] | — | 3.747 | 4.101 | -8.6% | 3.434 |
| fusion distinct historical windows | 950DT | [65, 4] | — | 4.471 | 3.258 | +37.2% | 4.135 |
| fusion distinct historical windows | 950DT | [128, 4] | — | 6.040 | 5.475 | +10.3% | 5.533 |
| fusion distinct historical windows | 950DT | [3, 4] | — | 3.102 | 3.730 | -16.8% | 2.843 |
| fusion distinct historical windows | 950DT | [5, 4] | — | 3.744 | 4.066 | -7.9% | 3.431 |
| fusion distinct historical windows | 950DT | [7, 4] | — | 4.682 | 5.142 | -8.9% | 4.290 |
| fusion distinct historical windows | 950DT | [11, 4] | — | 5.932 | 6.406 | -7.4% | 5.435 |
| fusion distinct historical windows | 950DT | [15, 4] | — | 6.563 | 6.967 | -5.8% | 6.012 |
| fusion distinct historical windows | 950DT | [21, 4] | — | 5.361 | 5.597 | -4.2% | 4.911 |
| fusion distinct historical windows | 950DT | [33, 4] | — | 6.000 | 6.377 | -5.9% | 5.496 |
| fusion distinct historical windows | 950DT | [47, 4] | — | 3.576 | 3.996 | -10.5% | 3.277 |
| fusion distinct historical windows | 950DT | [66, 4] | — | 4.471 | 3.212 | +39.2% | 4.136 |
| fusion distinct historical windows | 950DT | [96, 4] | — | 6.023 | 5.334 | +12.9% | 5.556 |
| fusion distinct historical windows | 950DT | [1, 4] | — | 5.786 | 6.149 | -5.9% | 5.300 |
| fusion distinct historical windows | 950PR | [1, 4] | — | 5.869 | 5.948 | -1.3% | 5.300 |
| fusion distinct historical windows | 950PR | [64, 4] | — | 6.081 | 6.533 | -6.9% | 5.533 |
| fusion distinct historical windows | 950PR | [17, 4] | — | 6.042 | 6.280 | -3.8% | 5.459 |
| fusion distinct historical windows | 950PR | [2, 4] | — | 3.085 | 3.697 | -16.5% | 2.820 |
| fusion distinct historical windows | 950PR | [4, 4] | — | 3.599 | 3.938 | -8.6% | 3.279 |
| fusion distinct historical windows | 950PR | [8, 4] | — | 4.426 | 4.719 | -6.2% | 4.017 |
| fusion distinct historical windows | 950PR | [9, 4] | — | 6.003 | 6.268 | -4.2% | 5.423 |
| fusion distinct historical windows | 950PR | [16, 4] | — | 7.922 | 8.328 | -4.9% | 7.135 |
| fusion distinct historical windows | 950PR | [31, 4] | — | 6.079 | 6.634 | -8.4% | 5.493 |
| fusion distinct historical windows | 950PR | [32, 4] | — | 6.081 | 6.616 | -8.1% | 5.494 |

## mHC PreSinkhorn

| 数据组 | 芯片 | shape | v0.20 | 当前 | CA | 有符号误差 | 970DT 外推 |
|---|---|---|---:|---:|---:|---:|---:|
| DT v0.12 development replay | 950DT | [1, 4, 4096] | — | 9.939 | 9.701 | +2.5% | 9.109 |
| DT v0.12 development replay | 950DT | [4, 4, 4096] | — | 9.939 | 9.939 | +0.0% | 9.109 |
| DT v0.12 development replay | 950DT | [2, 4, 4096] | — | 9.939 | 10.648 | -6.7% | 9.109 |
| DT v0.12 holdout replay | 950DT | [3, 4, 4096] | — | 9.939 | 10.373 | -4.2% | 9.109 |
| DT v0.12 holdout replay | 950DT | [5, 4, 4096] | — | 9.939 | 9.761 | +1.8% | 9.109 |
| DT v0.12 holdout replay | 950DT | [6, 4, 4096] | — | 9.939 | 9.785 | +1.6% | 9.109 |
| DT v0.12 holdout replay | 950DT | [7, 4, 4096] | — | 9.939 | 11.518 | -13.7% | 9.109 |
| DT v0.12 holdout replay | 950DT | [8, 4, 4096] | — | 9.939 | 10.529 | -5.6% | 9.109 |
| DT v0.12 holdout replay | 950DT | [9, 4, 4096] | — | 9.939 | 10.202 | -2.6% | 9.109 |
| DT v0.12 holdout replay | 950DT | [12, 4, 4096] | — | 9.939 | 10.952 | -9.2% | 9.109 |
| DT v0.12 holdout replay | 950DT | [16, 4, 4096] | — | 9.939 | 11.560 | -14.0% | 9.109 |
| DT v0.12 holdout replay | 950DT | [24, 4, 4096] | — | 9.939 | 11.583 | -14.2% | 9.109 |
| DT v0.12 holdout replay | 950DT | [32, 4, 4096] | — | 9.939 | 11.114 | -10.6% | 9.109 |
| prior published pair replay | 950DT | [1, 4, 4096] | — | 9.939 | 10.108 | -1.7% | 9.109 |
| prior published pair replay | 950PR | [1, 4, 4096] | — | 9.990 | 8.763 | +14.0% | 9.109 |
| prior published pair replay | 950DT | [4, 4, 4096] | — | 9.939 | 9.939 | +0.0% | 9.109 |
| prior published pair replay | 950PR | [4, 4, 4096] | — | 9.990 | 9.470 | +5.5% | 9.109 |
| prior published pair replay | 950DT | [2, 4, 4096] | — | 9.939 | 10.648 | -6.7% | 9.109 |
| prior published pair replay | 950PR | [2, 4, 4096] | — | 9.990 | 10.198 | -2.0% | 9.109 |
| prior published pair replay | 950DT | [3, 4, 4096] | — | 9.939 | 10.373 | -4.2% | 9.109 |
| prior published pair replay | 950PR | [3, 4, 4096] | — | 9.990 | 9.250 | +8.0% | 9.109 |
| prior published pair replay | 950DT | [5, 4, 4096] | — | 9.939 | 9.761 | +1.8% | 9.109 |
| prior published pair replay | 950PR | [5, 4, 4096] | — | 9.990 | 9.495 | +5.2% | 9.109 |
| prior published pair replay | 950DT | [6, 4, 4096] | — | 9.939 | 9.785 | +1.6% | 9.109 |
| prior published pair replay | 950PR | [6, 4, 4096] | — | 9.990 | 9.405 | +6.2% | 9.109 |
| prior published pair replay | 950DT | [7, 4, 4096] | — | 9.939 | 11.518 | -13.7% | 9.109 |
| prior published pair replay | 950PR | [7, 4, 4096] | — | 9.990 | 9.388 | +6.4% | 9.109 |
| prior published pair replay | 950DT | [8, 4, 4096] | — | 9.939 | 10.529 | -5.6% | 9.109 |
| prior published pair replay | 950PR | [8, 4, 4096] | — | 9.990 | 9.713 | +2.8% | 9.109 |
| prior published pair replay | 950DT | [9, 4, 4096] | — | 9.939 | 10.202 | -2.6% | 9.109 |
| prior published pair replay | 950PR | [9, 4, 4096] | — | 9.990 | 9.390 | +6.4% | 9.109 |
| prior published pair replay | 950DT | [12, 4, 4096] | — | 9.939 | 10.952 | -9.2% | 9.109 |
| prior published pair replay | 950PR | [12, 4, 4096] | — | 9.990 | 9.686 | +3.1% | 9.109 |
| primary | 950DT | [20, 4, 4096] | 9.939 | 9.939 | 11.519 | -13.7% | 9.109 |
| primary | 950PR | [20, 4, 4096] | 11.161 | 10.115 | 11.231 | -9.9% | 9.109 |
| primary | 950DT | [33, 4, 4096] | 19.879 | 11.739 | 11.301 | +3.9% | 10.754 |
| primary | 950PR | [22, 4, 4096] | 11.173 | 10.115 | 11.103 | -8.9% | 9.109 |
| primary | 950DT | [26, 4, 4096] | 9.939 | 9.939 | 11.182 | -11.1% | 9.109 |
| primary | 950PR | [26, 4, 4096] | 11.195 | 10.115 | 10.893 | -7.1% | 9.109 |
| primary | 950DT | [30, 4, 4096] | 9.939 | 9.939 | 11.050 | -10.0% | 9.109 |
| primary | 950PR | [30, 4, 4096] | 11.218 | 10.115 | 10.511 | -3.8% | 9.109 |
| primary | 950DT | [36, 4, 4096] | 19.879 | 11.739 | 11.522 | +1.9% | 10.754 |
| primary | 950PR | [36, 4, 4096] | 21.537 | 12.272 | 10.701 | +14.7% | 10.754 |
| primary | 950DT | [48, 4, 4096] | 19.879 | 11.739 | 11.890 | -1.3% | 10.754 |
| primary | 950PR | [48, 4, 4096] | 21.608 | 12.272 | 11.271 | +8.9% | 10.754 |
| fusion distinct historical windows | 950DT | [1, 4, 4096] | — | 9.939 | 9.701 | +2.5% | 9.109 |
| fusion distinct historical windows | 950DT | [4, 4, 4096] | — | 9.939 | 9.939 | +0.0% | 9.109 |
| fusion distinct historical windows | 950DT | [2, 4, 4096] | — | 9.939 | 10.648 | -6.7% | 9.109 |
| fusion distinct historical windows | 950DT | [3, 4, 4096] | — | 9.939 | 10.373 | -4.2% | 9.109 |
| fusion distinct historical windows | 950DT | [5, 4, 4096] | — | 9.939 | 9.761 | +1.8% | 9.109 |
| fusion distinct historical windows | 950DT | [6, 4, 4096] | — | 9.939 | 9.785 | +1.6% | 9.109 |
| fusion distinct historical windows | 950DT | [7, 4, 4096] | — | 9.939 | 11.518 | -13.7% | 9.109 |
| fusion distinct historical windows | 950DT | [8, 4, 4096] | — | 9.939 | 10.529 | -5.6% | 9.109 |
| fusion distinct historical windows | 950DT | [9, 4, 4096] | — | 9.939 | 10.202 | -2.6% | 9.109 |
| fusion distinct historical windows | 950DT | [12, 4, 4096] | — | 9.939 | 10.952 | -9.2% | 9.109 |
| fusion distinct historical windows | 950DT | [16, 4, 4096] | — | 9.939 | 11.560 | -14.0% | 9.109 |
| fusion distinct historical windows | 950DT | [24, 4, 4096] | — | 9.939 | 11.583 | -14.2% | 9.109 |
| fusion distinct historical windows | 950DT | [32, 4, 4096] | — | 9.939 | 11.114 | -10.6% | 9.109 |
| fusion distinct historical windows | 950DT | [1, 4, 4096] | — | 9.939 | 10.108 | -1.7% | 9.109 |
| fusion distinct historical windows | 950PR | [1, 4, 4096] | — | 9.990 | 8.763 | +14.0% | 9.109 |
| fusion distinct historical windows | 950PR | [4, 4, 4096] | — | 9.990 | 9.470 | +5.5% | 9.109 |
| fusion distinct historical windows | 950PR | [2, 4, 4096] | — | 9.990 | 10.198 | -2.0% | 9.109 |
| fusion distinct historical windows | 950PR | [3, 4, 4096] | — | 9.990 | 9.250 | +8.0% | 9.109 |
| fusion distinct historical windows | 950PR | [5, 4, 4096] | — | 9.990 | 9.495 | +5.2% | 9.109 |
| fusion distinct historical windows | 950PR | [6, 4, 4096] | — | 9.990 | 9.405 | +6.2% | 9.109 |
| fusion distinct historical windows | 950PR | [7, 4, 4096] | — | 9.990 | 9.388 | +6.4% | 9.109 |
| fusion distinct historical windows | 950PR | [8, 4, 4096] | — | 9.990 | 9.713 | +2.8% | 9.109 |
| fusion distinct historical windows | 950PR | [9, 4, 4096] | — | 9.990 | 9.390 | +6.4% | 9.109 |
| fusion distinct historical windows | 950PR | [12, 4, 4096] | — | 9.990 | 9.686 | +3.1% | 9.109 |

## mHC Post

| 数据组 | 芯片 | shape | v0.20 | 当前 | CA | 有符号误差 | 970DT 外推 |
|---|---|---|---:|---:|---:|---:|---:|
| DT v0.12 development replay | 950DT | [1, 4, 128] | — | 3.044 | 4.646 | -34.5% | 2.791 |
| DT v0.12 development replay | 950DT | [32, 4, 128] | — | 2.550 | 3.408 | -25.2% | 2.791 |
| DT v0.12 development replay | 950DT | [9, 4, 128] | — | 3.044 | 4.613 | -34.0% | 2.791 |
| DT v0.12 development replay | 950DT | [1, 4, 64] | — | 3.007 | 4.588 | -34.5% | 2.756 |
| DT v0.12 development replay | 950DT | [2, 4, 256] | — | 3.120 | 5.276 | -40.9% | 2.859 |
| DT v0.12 development replay | 950DT | [4, 4, 512] | — | 3.120 | 4.108 | -24.1% | 2.859 |
| DT v0.12 development replay | 950DT | [9, 4, 1024] | — | 3.120 | 3.966 | -21.3% | 2.859 |
| DT v0.12 development replay | 950DT | [16, 4, 256] | — | 3.120 | 2.214 | +40.9% | 2.859 |
| DT v0.12 development replay | 950DT | [32, 4, 512] | — | 2.811 | 2.388 | +17.7% | 2.859 |
| DT v0.12 development replay | 950DT | [36, 4, 128] | — | 2.550 | 1.772 | +44.0% | 2.791 |
| DT v0.12 development replay | 950DT | [64, 4, 256] | — | 2.638 | 3.007 | -12.3% | 2.418 |
| DT v0.12 development replay | 950DT | [65, 4, 128] | — | 2.751 | 2.110 | +30.4% | 2.338 |
| DT v0.12 development replay | 950DT | [1, 4, 7168] | — | 3.120 | 2.430 | +28.4% | 2.859 |
| DT v0.12 development replay | 950DT | [1, 4, 96] | — | 3.044 | 4.745 | -35.8% | 2.791 |
| DT v0.12 development replay | 950DT | [3, 4, 256] | — | 3.120 | 5.496 | -43.2% | 2.859 |
| DT v0.12 development replay | 950DT | [6, 4, 512] | — | 3.120 | 3.996 | -21.9% | 2.859 |
| DT v0.12 development replay | 950DT | [11, 4, 1024] | — | 3.120 | 4.087 | -23.7% | 2.859 |
| DT v0.12 development replay | 950DT | [17, 4, 256] | — | 3.120 | 2.276 | +37.0% | 2.859 |
| DT v0.12 development replay | 950DT | [31, 4, 512] | — | 3.120 | 3.996 | -21.9% | 2.859 |
| DT v0.12 development replay | 950DT | [33, 4, 256] | — | 2.638 | 2.483 | +6.2% | 2.859 |
| DT v0.12 development replay | 950DT | [63, 4, 512] | — | 2.811 | 2.502 | +12.3% | 2.577 |
| DT v0.12 development replay | 950DT | [66, 4, 128] | — | 2.751 | 2.018 | +36.3% | 2.338 |
| DT v0.12 development replay | 950DT | [2, 4, 4096] | — | 3.120 | 2.130 | +46.4% | 2.859 |
| DT v0.12 development replay | 950DT | [2, 4, 6144] | — | 3.082 | 4.838 | -36.3% | 2.825 |
| DT v0.12 development replay | 950DT | [7, 4, 384] | — | 3.082 | 2.144 | +43.8% | 2.825 |
| DT v0.12 development replay | 950DT | [1, 4, 4096] | — | 3.120 | 2.256 | +38.3% | 2.859 |
| DT v0.12 development replay | 950DT | [30, 4, 768] | — | 3.195 | 2.719 | +17.5% | 2.825 |
| DT v0.12 development replay | 950DT | [4, 4, 8192] | — | 3.269 | 2.794 | +17.0% | 2.996 |
| DT v0.12 development replay | 950DT | [32, 4, 384] | — | 2.725 | 2.292 | +18.9% | 2.825 |
| DT v0.12 development replay | 950DT | [34, 4, 512] | — | 2.811 | 1.912 | +47.1% | 2.859 |
| DT v0.12 development replay | 950DT | [3, 4, 3072] | — | 3.082 | 2.238 | +37.7% | 2.825 |
| DT v0.12 development replay | 950DT | [72, 4, 256] | — | 2.924 | 2.221 | +31.7% | 2.418 |
| DT v0.12 development replay | 950DT | [1, 4, 5120] | — | 3.082 | 2.167 | +42.2% | 2.825 |
| DT v0.12 development replay | 950DT | [3, 4, 4352] | — | 3.157 | 5.288 | -40.3% | 2.825 |
| DT v0.12 development replay | 950DT | [8, 4, 640] | — | 3.082 | 2.332 | +32.2% | 2.825 |
| DT v0.12 development replay | 950DT | [2, 4, 3328] | — | 3.120 | 2.573 | +21.3% | 2.859 |
| DT v0.12 development replay | 950DT | [4, 4, 6144] | — | 3.195 | 4.095 | -22.0% | 2.928 |
| DT v0.12 development replay | 950DT | [1, 4, 2304] | — | 3.082 | 2.349 | +31.2% | 2.825 |
| DT v0.12 development replay | 950DT | [1, 4, 7680] | — | 3.120 | 2.298 | +35.7% | 2.859 |
| DT v0.12 development replay | 950DT | [64, 4, 1024] | — | 3.155 | 3.019 | +4.5% | 2.892 |
| DT v0.12 development replay | 950DT | [128, 4, 1024] | — | 3.946 | 3.715 | +6.2% | 3.616 |
| DT v0.12 development replay | 950DT | [192, 4, 1024] | — | 4.728 | 4.177 | +13.2% | 3.616 |
| DT v0.12 development replay | 950DT | [256, 4, 1024] | — | 5.505 | 4.646 | +18.5% | 4.333 |
| DT v0.12 holdout replay | 950DT | [1, 4, 3456] | — | 3.120 | 4.797 | -35.0% | 2.859 |
| DT v0.12 holdout replay | 950DT | [2, 4, 4480] | — | 3.082 | 4.292 | -28.2% | 2.825 |
| DT v0.12 holdout replay | 950DT | [3, 4, 3840] | — | 3.120 | 2.892 | +7.9% | 2.859 |
| DT v0.12 holdout replay | 950DT | [1, 4, 2816] | — | 3.082 | 4.877 | -36.8% | 2.825 |
| DT v0.12 holdout replay | 950DT | [4, 4, 5888] | — | 3.195 | 4.520 | -29.3% | 2.928 |
| DT v0.12 holdout replay | 950DT | [12, 4, 1024] | — | 3.120 | 2.346 | +33.0% | 2.859 |
| DT v0.12 holdout replay | 950DT | [31, 4, 256] | — | 3.120 | 2.278 | +37.0% | 2.859 |
| DT v0.12 holdout replay | 950DT | [64, 4, 1536] | — | 3.496 | 3.459 | +1.1% | 3.204 |
| DT v0.12 holdout replay | 950DT | [128, 4, 1536] | — | 4.618 | 3.959 | +16.6% | 4.232 |
| DT v0.12 holdout replay | 950DT | [256, 4, 1536] | — | 6.833 | 5.201 | +31.4% | 5.249 |
| prior published pair replay | 950DT | [1, 4, 128] | — | 3.044 | 4.752 | -35.9% | 2.791 |
| prior published pair replay | 950PR | [1, 4, 128] | — | 3.046 | 4.255 | -28.4% | 2.791 |
| prior published pair replay | 950DT | [32, 4, 128] | — | 2.550 | 3.408 | -25.2% | 2.791 |
| prior published pair replay | 950PR | [32, 4, 128] | — | 2.552 | 3.245 | -21.4% | 2.791 |
| prior published pair replay | 950DT | [9, 4, 128] | — | 3.044 | 4.613 | -34.0% | 2.791 |
| prior published pair replay | 950PR | [9, 4, 128] | — | 3.046 | 4.313 | -29.4% | 2.791 |
| prior published pair replay | 950DT | [1, 4, 64] | — | 3.007 | 4.588 | -34.5% | 2.756 |
| prior published pair replay | 950PR | [1, 4, 64] | — | 3.008 | 4.132 | -27.2% | 2.756 |
| prior published pair replay | 950DT | [2, 4, 256] | — | 3.120 | 5.276 | -40.9% | 2.859 |
| prior published pair replay | 950PR | [2, 4, 256] | — | 3.124 | 4.415 | -29.2% | 2.859 |
| prior published pair replay | 950DT | [4, 4, 512] | — | 3.120 | 4.108 | -24.1% | 2.859 |
| prior published pair replay | 950PR | [4, 4, 512] | — | 3.124 | 4.501 | -30.6% | 2.859 |
| prior published pair replay | 950DT | [9, 4, 1024] | — | 3.120 | 3.966 | -21.3% | 2.859 |
| prior published pair replay | 950PR | [9, 4, 1024] | — | 3.124 | 4.700 | -33.5% | 2.859 |
| prior published pair replay | 950DT | [16, 4, 256] | — | 3.120 | 2.214 | +40.9% | 2.859 |
| prior published pair replay | 950PR | [16, 4, 256] | — | 3.124 | 4.255 | -26.6% | 2.859 |
| prior published pair replay | 950DT | [32, 4, 512] | — | 2.811 | 2.388 | +17.7% | 2.859 |
| prior published pair replay | 950PR | [32, 4, 512] | — | 2.816 | 3.323 | -15.3% | 2.859 |
| prior published pair replay | 950DT | [36, 4, 128] | — | 2.550 | 1.772 | +44.0% | 2.791 |
| prior published pair replay | 950PR | [36, 4, 128] | — | 2.552 | 3.210 | -20.5% | 2.791 |
| primary | 950DT | [5, 4, 288] | 3.082 | 3.082 | 4.962 | -37.9% | 2.825 |
| primary | 950DT | [37, 4, 144] | 2.057 | 2.579 | 3.489 | -26.1% | 2.825 |
| primary | 950PR | [37, 4, 144] | 2.079 | 2.580 | 3.407 | -24.3% | 2.825 |
| primary | 950DT | [7, 4, 1728] | 3.120 | 3.120 | 5.380 | -42.0% | 2.859 |
| primary | 950PR | [7, 4, 1728] | 3.173 | 3.124 | 4.816 | -35.1% | 2.859 |
| primary | 950DT | [15, 4, 240] | 4.211 | 3.120 | 4.658 | -33.0% | 2.859 |
| primary | 950PR | [15, 4, 240] | 4.243 | 3.124 | 4.758 | -34.3% | 2.859 |
| primary | 950DT | [29, 4, 640] | 3.157 | 3.157 | 5.098 | -38.1% | 2.825 |
| primary | 950PR | [29, 4, 640] | 5.043 | 2.904 | 3.558 | -18.4% | 2.825 |
| primary | 950DT | [55, 4, 352] | 2.185 | 2.714 | 3.848 | -29.5% | 2.825 |
| primary | 950PR | [55, 4, 352] | 2.252 | 2.718 | 3.789 | -28.3% | 2.825 |
| primary | 950DT | [71, 4, 1280] | 3.695 | 4.283 | 4.550 | -5.9% | 3.048 |
| primary | 950PR | [71, 4, 1280] | 4.046 | 4.307 | 4.333 | -0.6% | 3.048 |
| primary | 950DT | [129, 4, 1664] | 5.360 | 5.978 | 5.202 | +14.9% | 4.386 |
| primary | 950PR | [129, 4, 1664] | 6.218 | 6.024 | 5.305 | +13.5% | 4.386 |
| primary | 950DT | [191, 4, 2112] | 6.220 | 6.848 | 6.072 | +12.8% | 4.921 |

## mHC 拆分链

| 数据组 | 芯片 | shape | v0.20 | 当前 | CA | 有符号误差 | 970DT 外推 |
|---|---|---|---:|---:|---:|---:|---:|
| DT v0.12 development replay | 950DT | [1, 4, 128] | — | 16.903 | 15.970 | +5.8% | 15.494 |
| DT v0.12 development replay | 950DT | [8, 4, 128] | — | 17.040 | 16.791 | +1.5% | 15.620 |
| DT v0.12 development replay | 950DT | [3, 4, 128] | — | 16.942 | 16.835 | +0.6% | 15.530 |
| DT v0.12 development replay | 950DT | [1, 4, 64] | — | 18.002 | 17.143 | +5.0% | 16.502 |
| DT v0.12 development replay | 950DT | [2, 4, 128] | — | 16.923 | 16.416 | +3.1% | 15.512 |
| DT v0.12 development replay | 950DT | [4, 4, 256] | — | 17.300 | 17.344 | -0.3% | 15.857 |
| DT v0.12 development replay | 950DT | [16, 4, 128] | — | 17.196 | 18.202 | -5.5% | 15.762 |
| DT v0.12 development replay | 950DT | [1, 4, 512] | — | 17.457 | 17.343 | +0.7% | 16.000 |
| DT v0.12 development replay | 950DT | [4, 4, 1024] | — | 18.127 | 18.220 | -0.5% | 16.608 |
| DT v0.12 development replay | 950DT | [9, 4, 1024] | — | 18.458 | 18.482 | -0.1% | 16.909 |
| DT v0.12 development replay | 950DT | [16, 4, 512] | — | 18.051 | 18.691 | -3.4% | 16.541 |
| DT v0.12 development replay | 950DT | [32, 4, 1024] | — | 19.558 | 20.070 | -2.5% | 17.913 |
| DT v0.12 development replay | 950DT | [1, 4, 80] | — | 18.230 | 17.559 | +3.8% | 16.710 |
| primary | 950DT | [2, 4, 64] | — | 18.028 | 17.935 | +0.5% | 16.525 |
| primary | 950DT | [4, 4, 96] | — | 18.452 | 17.570 | +5.0% | 16.913 |
| primary | 950DT | [6, 4, 128] | — | 17.001 | 17.016 | -0.1% | 15.584 |
| primary | 950DT | [5, 4, 256] | — | 17.326 | 17.125 | +1.2% | 15.881 |
| primary | 950DT | [12, 4, 384] | — | 17.601 | 18.486 | -4.8% | 16.132 |
| primary | 950DT | [8, 4, 512] | — | 17.734 | 18.652 | -4.9% | 16.253 |
| primary | 950DT | [12, 4, 768] | — | 18.175 | 18.787 | -3.3% | 16.652 |
| primary | 950DT | [6, 4, 1024] | — | 18.259 | 18.361 | -0.6% | 16.728 |
| primary | 950DT | [17, 4, 512] | — | 19.343 | 19.328 | +0.1% | 17.726 |
| primary | 950DT | [24, 4, 1536] | — | 22.595 | 21.986 | +2.8% | 19.762 |
| prior published pair replay | 950DT | [1, 4, 128] | — | 16.903 | 15.768 | +7.2% | 15.494 |
| primary | 950PR | [1, 4, 128] | — | 16.909 | 15.496 | +9.1% | 15.494 |
| prior published pair replay | 950DT | [8, 4, 128] | — | 17.040 | 16.791 | +1.5% | 15.620 |
| primary | 950PR | [8, 4, 128] | — | 17.046 | 16.738 | +1.8% | 15.620 |
| prior published pair replay | 950DT | [3, 4, 128] | — | 16.942 | 16.835 | +0.6% | 15.530 |
| primary | 950PR | [3, 4, 128] | — | 16.948 | 16.406 | +3.3% | 15.530 |
| prior published pair replay | 950DT | [1, 4, 64] | — | 18.002 | 17.143 | +5.0% | 16.502 |
| primary | 950PR | [1, 4, 64] | — | 18.006 | 16.464 | +9.4% | 16.502 |
| prior published pair replay | 950DT | [2, 4, 128] | — | 16.923 | 16.416 | +3.1% | 15.512 |
| primary | 950PR | [2, 4, 128] | — | 16.929 | 16.054 | +5.4% | 15.512 |
| prior published pair replay | 950DT | [4, 4, 256] | — | 17.300 | 17.344 | -0.3% | 15.857 |
| primary | 950PR | [4, 4, 256] | — | 17.312 | 17.061 | +1.5% | 15.857 |
| prior published pair replay | 950DT | [16, 4, 128] | — | 17.196 | 18.202 | -5.5% | 15.762 |
| primary | 950PR | [16, 4, 128] | — | 17.202 | 17.677 | -2.7% | 15.762 |
| prior published pair replay | 950DT | [1, 4, 512] | — | 17.457 | 17.343 | +0.7% | 16.000 |
| primary | 950PR | [1, 4, 512] | — | 17.473 | 16.794 | +4.0% | 16.000 |
| prior published pair replay | 950DT | [4, 4, 1024] | — | 18.127 | 18.220 | -0.5% | 16.608 |
| primary | 950PR | [4, 4, 1024] | — | 18.160 | 18.525 | -2.0% | 16.608 |
| prior published pair replay | 950DT | [9, 4, 1024] | — | 18.458 | 18.482 | -0.1% | 16.909 |
| primary | 950PR | [9, 4, 1024] | — | 18.502 | 19.199 | -3.6% | 16.909 |

## KDA recurrent

| 数据组 | 芯片 | shape | v0.20 | 当前 | CA | 有符号误差 | 970DT 外推 |
|---|---|---|---:|---:|---:|---:|---:|
| DT v0.12 development replay | 950DT | [1, 1, 1, 1, 128, 128] | — | 15.072 | 15.219 | -1.0% | 13.804 |
| DT v0.12 development replay | 950DT | [1, 2, 1, 1, 128, 128] | — | 18.547 | 18.467 | +0.4% | 16.987 |
| DT v0.12 development replay | 950DT | [1, 4, 1, 1, 128, 128] | — | 25.498 | 25.430 | +0.3% | 23.352 |
| primary | 950DT | [1, 3, 1, 1, 128, 128] | — | 22.022 | 21.932 | +0.4% | 20.169 |
| primary | 950DT | [1, 5, 1, 1, 128, 128] | — | 28.973 | 24.648 | +17.5% | 26.534 |
| primary | 950DT | [1, 6, 1, 1, 128, 128] | — | 32.449 | 28.441 | +14.1% | 29.716 |
| primary | 950DT | [1, 7, 1, 1, 128, 128] | — | 35.924 | 31.195 | +15.2% | 32.899 |
| primary | 950DT | [1, 8, 1, 1, 128, 128] | — | 39.399 | 35.396 | +11.3% | 36.081 |
| primary | 950DT | [1, 1, 1, 1, 128, 256] | — | 18.547 | 17.434 | +6.4% | 16.984 |
| primary | 950DT | [1, 2, 1, 1, 128, 256] | — | 25.498 | 22.684 | +12.4% | 23.349 |
| primary | 950DT | [1, 3, 1, 1, 128, 256] | — | 32.449 | 31.071 | +4.4% | 29.714 |
| primary | 950DT | [1, 4, 1, 1, 128, 256] | — | 39.399 | 38.342 | +2.8% | 36.079 |
| primary | 950DT | [1, 6, 1, 1, 128, 256] | — | 53.301 | 52.691 | +1.2% | 48.808 |
| prior published pair replay | 950DT | [1, 1, 1, 1, 128, 128] | — | 15.072 | 14.834 | +1.6% | 13.804 |
| primary | 950PR | [1, 1, 1, 1, 128, 128] | — | 15.277 | 14.622 | +4.5% | 13.804 |
| prior published pair replay | 950DT | [1, 2, 1, 1, 128, 128] | — | 18.547 | 18.467 | +0.4% | 16.987 |
| primary | 950PR | [1, 2, 1, 1, 128, 128] | — | 18.832 | 18.048 | +4.3% | 16.987 |
| prior published pair replay | 950DT | [1, 4, 1, 1, 128, 128] | — | 25.498 | 25.430 | +0.3% | 23.352 |
| primary | 950PR | [1, 4, 1, 1, 128, 128] | — | 25.943 | 25.084 | +3.4% | 23.352 |
| prior published pair replay | 950DT | [1, 3, 1, 1, 128, 128] | — | 22.022 | 21.932 | +0.4% | 20.169 |
| primary | 950PR | [1, 3, 1, 1, 128, 128] | — | 22.388 | 21.221 | +5.5% | 20.169 |
| prior published pair replay | 950DT | [1, 5, 1, 1, 128, 128] | — | 28.973 | 24.648 | +17.5% | 26.534 |
| primary | 950PR | [1, 5, 1, 1, 128, 128] | — | 29.498 | 28.819 | +2.4% | 26.534 |
| prior published pair replay | 950DT | [1, 6, 1, 1, 128, 128] | — | 32.449 | 28.441 | +14.1% | 29.716 |
| primary | 950PR | [1, 6, 1, 1, 128, 128] | — | 33.054 | 31.955 | +3.4% | 29.716 |
| prior published pair replay | 950DT | [1, 7, 1, 1, 128, 128] | — | 35.924 | 31.195 | +15.2% | 32.899 |
| primary | 950PR | [1, 7, 1, 1, 128, 128] | — | 36.609 | 35.718 | +2.5% | 32.899 |
| prior published pair replay | 950DT | [1, 8, 1, 1, 128, 128] | — | 39.399 | 35.396 | +11.3% | 36.081 |
| primary | 950PR | [1, 8, 1, 1, 128, 128] | — | 40.164 | 39.450 | +1.8% | 36.081 |
| prior published pair replay | 950DT | [1, 1, 1, 1, 128, 256] | — | 18.547 | 17.434 | +6.4% | 16.984 |
| primary | 950PR | [1, 1, 1, 1, 128, 256] | — | 18.841 | 21.030 | -10.4% | 16.984 |
| prior published pair replay | 950DT | [1, 2, 1, 1, 128, 256] | — | 25.498 | 22.684 | +12.4% | 23.349 |
| primary | 950PR | [1, 2, 1, 1, 128, 256] | — | 25.951 | 28.402 | -8.6% | 23.349 |

## KDA chunk

| 数据组 | 芯片 | shape | v0.20 | 当前 | CA | 有符号误差 | 970DT 外推 |
|---|---|---|---:|---:|---:|---:|---:|
| 无合格窗口 | DT、PR | — | — | — | N/A | N/A | N/A |

## GMM1–SwiGLU–MXQuant

| 数据组 | 芯片 | shape | v0.20 | 当前 | CA | 有符号误差 | 970DT 外推 |
|---|---|---|---:|---:|---:|---:|---:|
| DT v0.12 development replay | 950DT | [2, 64, 64, 1] | — | 7.272 | 7.005 | +3.8% | 6.666 |
| DT v0.12 development replay | 950DT | [2, 64, 128, 1] | — | 7.272 | 7.280 | -0.1% | 6.666 |
| DT v0.12 development replay | 950DT | [4, 128, 64, 2] | — | 7.858 | 7.364 | +6.7% | 7.202 |
| DT v0.12 development replay | 950DT | [6, 128, 128, 2] | — | 7.858 | 7.390 | +6.3% | 7.201 |
| primary | 950DT | [1, 64, 64, 1] | — | 7.272 | 7.044 | +3.2% | 6.666 |
| primary | 950DT | [3, 64, 64, 2] | — | 7.272 | 7.081 | +2.7% | 6.666 |
| primary | 950DT | [8, 64, 128, 1] | — | 7.272 | 7.405 | -1.8% | 6.666 |
| primary | 950DT | [16, 128, 128, 2] | — | 7.858 | 7.372 | +6.6% | 7.201 |
| primary | 950DT | [17, 128, 64, 2] | — | 7.858 | 7.414 | +6.0% | 7.202 |
| primary | 950DT | [8, 256, 128, 2] | — | 9.941 | 7.489 | +32.7% | 9.104 |
| primary | 950DT | [32, 128, 128, 4] | — | 7.858 | 7.686 | +2.2% | 7.199 |
| primary | 950DT | [16, 256, 256, 1] | — | 9.941 | 8.193 | +21.3% | 9.104 |
| primary | 950DT | [32, 256, 128, 2] | — | 9.941 | 8.214 | +21.0% | 9.104 |
| primary | 950DT | [64, 128, 256, 4] | — | 7.858 | 8.355 | -5.9% | 7.195 |
| prior published pair replay | 950DT | [2, 64, 64, 1] | — | 7.272 | 7.105 | +2.4% | 6.666 |
| primary | 950PR | [2, 64, 64, 1] | — | 7.272 | 6.885 | +5.6% | 6.666 |
| prior published pair replay | 950DT | [2, 64, 128, 1] | — | 7.272 | 7.280 | -0.1% | 6.666 |
| primary | 950PR | [2, 64, 128, 1] | — | 7.272 | 6.755 | +7.7% | 6.666 |
| prior published pair replay | 950DT | [4, 128, 64, 2] | — | 7.858 | 7.364 | +6.7% | 7.202 |
| primary | 950PR | [4, 128, 64, 2] | — | 7.862 | 7.064 | +11.3% | 7.202 |
| prior published pair replay | 950DT | [6, 128, 128, 2] | — | 7.858 | 7.390 | +6.3% | 7.201 |
| primary | 950PR | [6, 128, 128, 2] | — | 7.865 | 7.030 | +11.9% | 7.201 |
| prior published pair replay | 950DT | [1, 64, 64, 1] | — | 7.272 | 7.044 | +3.2% | 6.666 |
| primary | 950PR | [1, 64, 64, 1] | — | 7.272 | 6.785 | +7.2% | 6.666 |
| prior published pair replay | 950DT | [3, 64, 64, 2] | — | 7.272 | 7.081 | +2.7% | 6.666 |
| primary | 950PR | [3, 64, 64, 2] | — | 7.272 | 6.862 | +6.0% | 6.666 |
| prior published pair replay | 950DT | [8, 64, 128, 1] | — | 7.272 | 7.405 | -1.8% | 6.666 |
| primary | 950PR | [8, 64, 128, 1] | — | 7.272 | 7.159 | +1.6% | 6.666 |
| prior published pair replay | 950DT | [16, 128, 128, 2] | — | 7.858 | 7.372 | +6.6% | 7.201 |
| primary | 950PR | [16, 128, 128, 2] | — | 7.866 | 7.073 | +11.2% | 7.201 |
| prior published pair replay | 950DT | [17, 128, 64, 2] | — | 7.858 | 7.414 | +6.0% | 7.202 |
| primary | 950PR | [17, 128, 64, 2] | — | 7.862 | 7.210 | +9.0% | 7.202 |
| prior published pair replay | 950DT | [8, 256, 128, 2] | — | 9.941 | 7.489 | +32.7% | 9.104 |
| primary | 950PR | [8, 256, 128, 2] | — | 9.964 | 7.393 | +34.8% | 9.104 |

## MoE GMM2

| 数据组 | 芯片 | shape | v0.20 | 当前 | CA | 有符号误差 | 970DT 外推 |
|---|---|---|---:|---:|---:|---:|---:|
| DT v0.12 development replay | 950DT | [2, 64, 64, 1] | — | 3.410 | 3.248 | +5.0% | 3.126 |
| DT v0.12 development replay | 950DT | [2, 64, 128, 1] | — | 3.695 | 3.545 | +4.2% | 3.387 |
| DT v0.12 development replay | 950DT | [4, 128, 64, 2] | — | 3.410 | 3.707 | -8.0% | 3.125 |
| DT v0.12 development replay | 950DT | [6, 128, 128, 2] | — | 3.695 | 3.697 | -0.0% | 3.386 |
| primary | 950DT | [1, 64, 64, 1] | — | 3.410 | 3.151 | +8.2% | 3.126 |
| primary | 950DT | [3, 64, 64, 2] | — | 3.410 | 3.771 | -9.6% | 3.125 |
| primary | 950DT | [8, 64, 128, 1] | — | 3.695 | 3.228 | +14.5% | 3.387 |
| primary | 950DT | [16, 128, 128, 2] | — | 3.695 | 3.838 | -3.7% | 3.386 |
| primary | 950DT | [17, 128, 64, 2] | — | 3.410 | 3.921 | -13.0% | 3.125 |
| primary | 950DT | [8, 256, 128, 2] | — | 3.695 | 3.630 | +1.8% | 3.385 |
| primary | 950DT | [32, 128, 128, 4] | — | 3.695 | 3.885 | -4.9% | 3.385 |
| primary | 950DT | [16, 256, 256, 1] | — | 4.516 | 3.560 | +26.9% | 4.136 |
| primary | 950DT | [32, 256, 128, 2] | — | 3.695 | 3.905 | -5.4% | 3.385 |
| primary | 950DT | [64, 128, 256, 4] | — | 4.516 | 4.142 | +9.0% | 4.132 |
| prior published pair replay | 950DT | [2, 64, 64, 1] | — | 3.410 | 3.090 | +10.3% | 3.126 |
| primary | 950PR | [2, 64, 64, 1] | — | 3.410 | 3.107 | +9.8% | 3.126 |
| prior published pair replay | 950DT | [2, 64, 128, 1] | — | 3.695 | 3.545 | +4.2% | 3.387 |
| primary | 950PR | [2, 64, 128, 1] | — | 3.696 | 3.065 | +20.6% | 3.387 |
| prior published pair replay | 950DT | [4, 128, 64, 2] | — | 3.410 | 3.707 | -8.0% | 3.125 |
| primary | 950PR | [4, 128, 64, 2] | — | 3.410 | 3.664 | -6.9% | 3.125 |
| prior published pair replay | 950DT | [6, 128, 128, 2] | — | 3.695 | 3.697 | -0.0% | 3.386 |
| primary | 950PR | [6, 128, 128, 2] | — | 3.698 | 3.812 | -3.0% | 3.386 |
| prior published pair replay | 950DT | [1, 64, 64, 1] | — | 3.410 | 3.151 | +8.2% | 3.126 |
| primary | 950PR | [1, 64, 64, 1] | — | 3.410 | 3.210 | +6.2% | 3.126 |
| prior published pair replay | 950DT | [3, 64, 64, 2] | — | 3.410 | 3.771 | -9.6% | 3.125 |
| primary | 950PR | [3, 64, 64, 2] | — | 3.410 | 3.404 | +0.2% | 3.125 |
| prior published pair replay | 950DT | [8, 64, 128, 1] | — | 3.695 | 3.228 | +14.5% | 3.387 |
| primary | 950PR | [8, 64, 128, 1] | — | 3.696 | 3.164 | +16.8% | 3.387 |
| prior published pair replay | 950DT | [16, 128, 128, 2] | — | 3.695 | 3.838 | -3.7% | 3.386 |
| primary | 950PR | [16, 128, 128, 2] | — | 3.699 | 3.979 | -7.1% | 3.386 |
| prior published pair replay | 950DT | [17, 128, 64, 2] | — | 3.410 | 3.921 | -13.0% | 3.125 |
| primary | 950PR | [17, 128, 64, 2] | — | 3.410 | 3.812 | -10.5% | 3.125 |
| prior published pair replay | 950DT | [8, 256, 128, 2] | — | 3.695 | 3.630 | +1.8% | 3.385 |
| primary | 950PR | [8, 256, 128, 2] | — | 3.701 | 3.663 | +1.0% | 3.385 |

## MoE FFN 组合

| 数据组 | 芯片 | shape | v0.20 | 当前 | CA | 有符号误差 | 970DT 外推 |
|---|---|---|---:|---:|---:|---:|---:|
| DT v0.12 development replay | 950DT | [2, 64, 64, 1] | — | 10.682 | 10.253 | +4.2% | 9.792 |
| DT v0.12 development replay | 950DT | [2, 64, 128, 1] | — | 10.967 | 10.825 | +1.3% | 10.053 |
| DT v0.12 development replay | 950DT | [4, 128, 64, 2] | — | 11.268 | 11.070 | +1.8% | 10.328 |
| DT v0.12 development replay | 950DT | [6, 128, 128, 2] | — | 11.553 | 11.087 | +4.2% | 10.588 |
| primary | 950DT | [1, 64, 64, 1] | — | 10.682 | 10.195 | +4.8% | 9.792 |
| primary | 950DT | [3, 64, 64, 2] | — | 10.682 | 10.852 | -1.6% | 9.792 |
| primary | 950DT | [8, 64, 128, 1] | — | 10.967 | 10.633 | +3.1% | 10.053 |
| primary | 950DT | [16, 128, 128, 2] | — | 11.553 | 11.209 | +3.1% | 10.587 |
| primary | 950DT | [17, 128, 64, 2] | — | 11.268 | 11.335 | -0.6% | 10.327 |
| primary | 950DT | [8, 256, 128, 2] | — | 13.636 | 11.119 | +22.6% | 12.490 |
| primary | 950DT | [32, 128, 128, 4] | — | 11.553 | 11.571 | -0.2% | 10.585 |
| primary | 950DT | [16, 256, 256, 1] | — | 14.457 | 11.753 | +23.0% | 13.240 |
| primary | 950DT | [32, 256, 128, 2] | — | 13.636 | 12.119 | +12.5% | 12.489 |
| primary | 950DT | [64, 128, 256, 4] | — | 12.374 | 12.496 | -1.0% | 11.327 |
| prior published pair replay | 950DT | [2, 64, 64, 1] | — | 10.682 | 10.195 | +4.8% | 9.792 |
| primary | 950PR | [2, 64, 64, 1] | — | 10.682 | 9.992 | +6.9% | 9.792 |
| prior published pair replay | 950DT | [2, 64, 128, 1] | — | 10.967 | 10.825 | +1.3% | 10.053 |
| primary | 950PR | [2, 64, 128, 1] | — | 10.968 | 9.819 | +11.7% | 10.053 |
| prior published pair replay | 950DT | [4, 128, 64, 2] | — | 11.268 | 11.070 | +1.8% | 10.328 |
| primary | 950PR | [4, 128, 64, 2] | — | 11.272 | 10.727 | +5.1% | 10.328 |
| prior published pair replay | 950DT | [6, 128, 128, 2] | — | 11.553 | 11.087 | +4.2% | 10.588 |
| primary | 950PR | [6, 128, 128, 2] | — | 11.564 | 10.842 | +6.7% | 10.588 |
| prior published pair replay | 950DT | [1, 64, 64, 1] | — | 10.682 | 10.195 | +4.8% | 9.792 |
| primary | 950PR | [1, 64, 64, 1] | — | 10.682 | 9.995 | +6.9% | 9.792 |
| prior published pair replay | 950DT | [3, 64, 64, 2] | — | 10.682 | 10.852 | -1.6% | 9.792 |
| primary | 950PR | [3, 64, 64, 2] | — | 10.682 | 10.267 | +4.0% | 9.792 |
| prior published pair replay | 950DT | [8, 64, 128, 1] | — | 10.967 | 10.633 | +3.1% | 10.053 |
| primary | 950PR | [8, 64, 128, 1] | — | 10.968 | 10.324 | +6.2% | 10.053 |
| prior published pair replay | 950DT | [16, 128, 128, 2] | — | 11.553 | 11.209 | +3.1% | 10.587 |
| primary | 950PR | [16, 128, 128, 2] | — | 11.564 | 11.053 | +4.6% | 10.587 |
| prior published pair replay | 950DT | [17, 128, 64, 2] | — | 11.268 | 11.335 | -0.6% | 10.327 |
| primary | 950PR | [17, 128, 64, 2] | — | 11.272 | 11.022 | +2.3% | 10.327 |
| prior published pair replay | 950DT | [8, 256, 128, 2] | — | 13.636 | 11.119 | +22.6% | 12.490 |
| primary | 950PR | [8, 256, 128, 2] | — | 13.666 | 11.056 | +23.6% | 12.490 |

## MoE sigmoid Top-K

| 数据组 | 芯片 | shape | v0.20 | 当前 | CA | 有符号误差 | 970DT 外推 |
|---|---|---|---:|---:|---:|---:|---:|
| primary | 950DT | [1, 384, 6] | — | 4.330 | 4.239 | +2.1% | 3.968 |
| primary | 950PR | [1, 384, 6] | — | 4.353 | 4.094 | +6.3% | 3.968 |
| primary | 950DT | [2, 896, 16] | — | 5.340 | 5.472 | -2.4% | 4.893 |
| primary | 950PR | [2, 896, 16] | — | 5.391 | 5.446 | -1.0% | 4.893 |
| primary | 950DT | [4, 256, 8] | — | 4.078 | 4.865 | -16.2% | 3.737 |
| primary | 950PR | [4, 256, 8] | — | 4.094 | 4.410 | -7.2% | 3.737 |
| primary | 950DT | [7, 896, 16] | — | 5.340 | 5.576 | -4.2% | 4.893 |
| primary | 950PR | [7, 896, 16] | — | 5.391 | 5.567 | -3.2% | 4.893 |
| primary | 950DT | [9, 384, 8] | — | 4.330 | 4.717 | -8.2% | 3.968 |
| primary | 950PR | [9, 384, 8] | — | 4.353 | 4.535 | -4.0% | 3.968 |
| primary | 950DT | [12, 896, 16] | — | 5.340 | 6.121 | -12.8% | 4.893 |
| primary | 950PR | [12, 896, 16] | — | 5.391 | 5.720 | -5.8% | 4.893 |
| primary | 950DT | [16, 512, 8] | — | 4.583 | 5.278 | -13.2% | 4.200 |
| primary | 950PR | [16, 512, 8] | — | 4.612 | 4.861 | -5.1% | 4.200 |
| primary | 950DT | [24, 896, 16] | — | 5.340 | 5.915 | -9.7% | 4.893 |
| primary | 950PR | [24, 896, 16] | — | 5.391 | 5.682 | -5.1% | 4.893 |
| primary | 950DT | [33, 384, 8] | — | 4.330 | 4.903 | -11.7% | 3.968 |
| primary | 950PR | [33, 384, 8] | — | 4.353 | 4.831 | -9.9% | 3.968 |
| primary | 950DT | [65, 128, 8] | — | 4.162 | 4.598 | -9.5% | 3.506 |
| primary | 950PR | [65, 128, 8] | — | 4.180 | 4.698 | -11.0% | 3.506 |

## Cast BF16→FP32

| 数据组 | 芯片 | shape | v0.20 | 当前 | CA | 有符号误差 | 970DT 外推 |
|---|---|---|---:|---:|---:|---:|---:|
| vector development replay | 950DT | [262144, 32] | 8.363 | 8.363 | 9.970 | -16.1% | 6.051 |
| vector development replay | 950PR | [262144, 32] | 13.046 | 13.046 | 14.080 | -7.3% | 6.051 |
| vector development replay | 950DT | [1048576, 32] | 23.275 | 36.368 | 35.535 | +2.3% | 13.341 |
| vector development replay | 950PR | [1048576, 32] | 64.360 | 88.943 | 70.711 | +25.8% | 13.341 |
| vector development replay | 950DT | [2097152, 32] | 69.523 | 99.918 | 106.757 | -6.4% | 42.536 |
| vector development replay | 950PR | [2097152, 32] | 208.165 | 232.748 | 223.493 | +4.1% | 42.536 |
| vector development replay | 950DT | [8192, 5120] | 31.774 | 52.256 | 55.286 | -5.5% | 19.234 |
| vector development replay | 950PR | [8192, 5120] | 100.311 | 124.894 | 114.128 | +9.4% | 19.234 |
| vector development replay | 950DT | [4096, 7168] | 20.790 | 28.424 | 29.724 | -4.4% | 12.126 |
| vector development replay | 950PR | [4096, 7168] | 46.385 | 70.967 | 54.612 | +29.9% | 12.126 |
| vector development replay | 950DT | [262144, 64] | 13.334 | 13.334 | 17.041 | -21.8% | 8.481 |
| vector development replay | 950PR | [262144, 64] | 25.030 | 25.030 | 23.762 | +5.3% | 8.481 |
| vector development replay | 950DT | [32, 4096] | 2.736 | 2.736 | 2.771 | -1.3% | 2.508 |
| vector development replay | 950PR | [32, 4096] | 2.681 | 2.681 | 2.867 | -6.5% | 2.508 |
| vector development replay | 950DT | [1, 64] | 1.957 | 1.957 | 2.172 | -9.9% | 1.794 |
| vector development replay | 950PR | [1, 64] | 1.957 | 1.957 | 1.996 | -1.9% | 1.794 |
| vector development replay | 950DT | [72, 4096] | 2.932 | 2.932 | 2.851 | +2.8% | 3.049 |
| vector development replay | 950PR | [72, 4096] | 2.907 | 2.907 | 2.975 | -2.3% | 3.049 |
| vector development replay | 950DT | [1, 3584] | 2.099 | 2.099 | 2.216 | -5.3% | 1.924 |
| vector development replay | 950PR | [1, 3584] | 2.100 | 2.100 | 2.104 | -0.2% | 1.924 |
| vector development replay | 950DT | [96, 7168] | 3.401 | 3.401 | 3.096 | +9.9% | 3.278 |
| vector development replay | 950PR | [96, 7168] | 3.445 | 3.445 | 2.875 | +19.8% | 3.278 |
| vector development replay | 950DT | [9, 4096] | 2.278 | 2.278 | 2.325 | -2.0% | 2.088 |
| vector development replay | 950PR | [9, 4096] | 2.279 | 2.279 | 2.220 | +2.7% | 2.088 |
| vector development replay | 950DT | [48, 7168] | 2.991 | 2.991 | 2.892 | +3.4% | 3.078 |
| vector development replay | 950PR | [48, 7168] | 2.972 | 2.972 | 2.914 | +2.0% | 3.078 |
| vector development replay | 950DT | [16, 7168] | 2.657 | 2.657 | 2.722 | -2.4% | 2.435 |
| vector development replay | 950PR | [16, 7168] | 2.658 | 2.658 | 2.830 | -6.1% | 2.435 |
| vector development replay | 950DT | [128, 5120] | 3.362 | 3.362 | 3.064 | +9.7% | 3.260 |
| vector development replay | 950PR | [128, 5120] | 3.400 | 3.400 | 3.562 | -4.5% | 3.260 |
| vector development replay | 950DT | [144, 6144] | 3.636 | 3.636 | 3.522 | +3.2% | 3.393 |
| vector development replay | 950PR | [144, 6144] | 3.715 | 3.715 | 4.111 | -9.6% | 3.393 |
| primary | 950DT | [65, 5120] | 2.978 | 2.978 | 3.050 | -2.4% | 3.071 |
| primary | 950PR | [65, 5120] | 2.958 | 2.958 | 2.936 | +0.7% | 3.071 |
| primary | 950DT | [393216, 32] | 10.849 | 10.849 | 13.728 | -21.0% | 7.266 |
| primary | 950PR | [393216, 32] | 19.038 | 19.038 | 18.826 | +1.1% | 7.266 |
| primary | 950DT | [3072, 5120] | 12.712 | 12.712 | 16.367 | -22.3% | 8.177 |
| primary | 950PR | [3072, 5120] | 23.532 | 23.532 | 23.237 | +1.3% | 8.177 |

## Kimi KDA beta Sigmoid

| 数据组 | 芯片 | shape | v0.20 | 当前 | CA | 有符号误差 | 970DT 外推 |
|---|---|---|---:|---:|---:|---:|---:|
| vector development replay | 950DT | [262144, 12] | 10.279 | 8.002 | 7.759 | +3.1% | 5.553 |
| vector development replay | 950PR | [262144, 12] | 21.728 | 12.344 | 11.428 | +8.0% | 5.553 |
| vector development replay | 950DT | [72, 12] | 2.790 | 2.790 | 3.114 | -10.4% | 2.558 |
| vector development replay | 950PR | [72, 12] | 2.793 | 2.793 | 2.672 | +4.5% | 2.558 |
| vector development replay | 950DT | [1048576, 12] | 28.858 | 19.720 | 18.553 | +6.3% | 11.321 |
| vector development replay | 950PR | [1048576, 12] | 75.597 | 39.630 | 34.285 | +15.6% | 11.321 |
| vector development replay | 950DT | [2048, 12] | 3.345 | 3.345 | 3.176 | +5.3% | 3.066 |
| vector development replay | 950PR | [2048, 12] | 3.404 | 3.347 | 2.793 | +19.8% | 3.066 |
| vector development replay | 950DT | [2097152, 12] | 54.185 | 45.897 | 42.686 | +7.5% | 19.353 |
| vector development replay | 950PR | [2097152, 12] | 147.417 | 104.717 | 88.593 | +18.2% | 19.353 |
| vector development replay | 950DT | [32768, 12] | 4.407 | 4.121 | 3.698 | +11.4% | 3.581 |
| vector development replay | 950PR | [32768, 12] | 5.895 | 4.338 | 4.128 | +5.1% | 3.581 |
| vector development replay | 950DT | [9, 96] | 2.790 | 2.790 | 3.032 | -8.0% | 2.558 |
| vector development replay | 950PR | [9, 96] | 2.793 | 2.793 | 2.658 | +5.1% | 2.558 |
| vector development replay | 950DT | [4096, 24] | 3.643 | 3.571 | 3.425 | +4.3% | 3.280 |
| vector development replay | 950PR | [4096, 24] | 3.982 | 3.587 | 3.187 | +12.5% | 3.280 |
| vector development replay | 950DT | [16384, 48] | 5.426 | 4.854 | 4.855 | -0.0% | 3.929 |
| vector development replay | 950PR | [16384, 48] | 8.427 | 6.015 | 4.823 | +24.7% | 3.929 |
| vector development replay | 950DT | [131072, 12] | 6.863 | 5.709 | 5.987 | -4.6% | 4.657 |
| vector development replay | 950PR | [131072, 12] | 12.856 | 8.114 | 7.882 | +2.9% | 4.657 |
| vector development replay | 950DT | [1, 12] | 2.689 | 2.689 | 2.724 | -1.3% | 2.465 |
| vector development replay | 950DT | [300, 12] | 3.042 | 3.042 | 3.085 | -1.4% | 2.789 |
| vector development replay | 950DT | [3072, 96] | 4.152 | 3.938 | 3.621 | +8.7% | 3.477 |
| primary | 950PR | [3073, 12] | 3.553 | 3.420 | 2.932 | +16.7% | 3.133 |
| primary | 950DT | [3073, 12] | 3.418 | 3.418 | 3.066 | +11.5% | 3.133 |
| primary | 950DT | [524288, 12] | 16.472 | 11.908 | 11.230 | +6.0% | 7.556 |
| primary | 950DT | [65537, 24] | 6.895 | 5.768 | 5.867 | -1.7% | 4.657 |
| primary | 950PR | [65537, 24] | 12.856 | 8.114 | 7.585 | +7.0% | 4.657 |
| primary | 950PR | [524288, 12] | 39.880 | 21.815 | 19.424 | +12.3% | 7.556 |
| primary | 950DT | [1572864, 12] | 41.486 | 30.140 | 28.561 | +5.5% | 15.408 |
| primary | 950PR | [1572864, 12] | 111.497 | 69.041 | 54.379 | +27.0% | 15.408 |

## Kimi AttnResFwd

| 数据组 | 芯片 | shape | v0.20 | 当前 | CA | 有符号误差 | 970DT 外推 |
|---|---|---|---:|---:|---:|---:|---:|
| prior published pair replay | 950DT | [1, 7168, 8, 1] | — | 10.152 | 9.505 | +6.8% | 9.303 |
| prior published pair replay | 950PR | [1, 7168, 8, 1] | — | 10.221 | 9.499 | +7.6% | 9.303 |
| prior published pair replay | 950DT | [2, 512, 8, 1] | — | 6.797 | 7.122 | -4.6% | 6.230 |
| prior published pair replay | 950PR | [2, 512, 8, 1] | — | 6.802 | 6.598 | +3.1% | 6.230 |
| prior published pair replay | 950DT | [1, 7168, 8, 3] | — | 12.701 | 11.572 | +9.8% | 11.639 |
| prior published pair replay | 950PR | [1, 7168, 8, 3] | — | 12.799 | 11.488 | +11.4% | 11.639 |
| prior published pair replay | 950DT | [2, 4096, 8, 2] | — | 9.570 | 9.398 | +1.8% | 8.771 |
| prior published pair replay | 950PR | [2, 4096, 8, 2] | — | 9.618 | 9.007 | +6.8% | 8.771 |
| prior published pair replay | 950DT | [5, 2048, 8, 1] | — | 7.571 | 7.574 | -0.0% | 6.939 |
| prior published pair replay | 950PR | [5, 2048, 8, 1] | — | 7.591 | 7.433 | +2.1% | 6.939 |
| prior published pair replay | 950DT | [2, 1024, 8, 4] | — | 9.032 | 9.178 | -1.6% | 8.278 |
| prior published pair replay | 950PR | [2, 1024, 8, 4] | — | 9.048 | 8.973 | +0.8% | 8.278 |
| prior published pair replay | 950DT | [8, 512, 8, 2] | — | 7.404 | 7.652 | -3.2% | 6.787 |
| prior published pair replay | 950PR | [8, 512, 8, 2] | — | 7.410 | 7.223 | +2.6% | 6.787 |
| prior published pair replay | 950DT | [3, 512, 8, 1] | — | 6.797 | 7.010 | -3.0% | 6.230 |
| prior published pair replay | 950PR | [3, 512, 8, 1] | — | 6.802 | 6.697 | +1.6% | 6.230 |
| prior published pair replay | 950DT | [4, 256, 8, 3] | — | 7.832 | 7.585 | +3.2% | 7.179 |
| prior published pair replay | 950PR | [4, 256, 8, 3] | — | 7.835 | 7.368 | +6.3% | 7.179 |
| prior published pair replay | 950DT | [1, 256, 8, 2] | — | 7.250 | 7.004 | +3.5% | 6.645 |
| prior published pair replay | 950PR | [1, 256, 8, 2] | — | 7.253 | 6.583 | +10.2% | 6.645 |
| primary | 950DT | [2, 4096, 8, 1] | 10.949 | 8.603 | 8.113 | +6.0% | 7.885 |
| primary | 950PR | [2, 4096, 8, 1] | 11.066 | 8.643 | 7.984 | +8.2% | 7.885 |
| primary | 950PR | [4, 256, 8, 6] | 9.832 | 9.583 | 9.181 | +4.4% | 8.780 |
| primary | 950DT | [1, 1024, 8, 7] | 13.945 | 11.009 | 11.573 | -4.9% | 10.091 |
| primary | 950PR | [1, 1024, 8, 7] | 14.061 | 11.031 | 10.764 | +2.5% | 10.091 |
| primary | 950DT | [2, 2048, 8, 3] | 11.833 | 9.094 | 9.490 | -4.2% | 8.335 |
| primary | 950PR | [2, 2048, 8, 3] | 11.950 | 9.122 | 9.085 | +0.4% | 8.335 |
| primary | 950DT | [5, 512, 8, 4] | 9.142 | 8.620 | 9.237 | -6.7% | 7.901 |
| primary | 950PR | [5, 512, 8, 4] | 9.178 | 8.628 | 8.473 | +1.8% | 7.901 |
| primary | 950DT | [3, 7168, 8, 1] | 14.648 | 10.152 | 9.712 | +4.5% | 9.303 |
| primary | 950PR | [3, 7168, 8, 1] | 14.853 | 10.221 | 9.474 | +7.9% | 9.303 |
| fusion distinct historical windows | 950DT | [1, 7168, 8, 1] | — | 10.152 | 9.505 | +6.8% | 9.303 |
| fusion distinct historical windows | 950PR | [1, 7168, 8, 1] | — | 10.221 | 9.499 | +7.6% | 9.303 |
| fusion distinct historical windows | 950DT | [2, 512, 8, 1] | — | 6.797 | 7.122 | -4.6% | 6.230 |
| fusion distinct historical windows | 950PR | [2, 512, 8, 1] | — | 6.802 | 6.598 | +3.1% | 6.230 |
| fusion distinct historical windows | 950DT | [1, 256, 8, 1] | — | 6.668 | 6.325 | +5.4% | 6.112 |
| fusion distinct historical windows | 950PR | [1, 512, 8, 2] | — | 7.410 | 7.039 | +5.3% | 6.787 |
| fusion distinct historical windows | 950DT | [1, 7168, 8, 3] | — | 12.701 | 11.572 | +9.8% | 11.639 |
| fusion distinct historical windows | 950PR | [1, 7168, 8, 3] | — | 12.799 | 11.488 | +11.4% | 11.639 |
| fusion distinct historical windows | 950DT | [2, 4096, 8, 2] | — | 9.570 | 9.398 | +1.8% | 8.771 |
| fusion distinct historical windows | 950PR | [2, 4096, 8, 2] | — | 9.618 | 9.007 | +6.8% | 8.771 |
| fusion distinct historical windows | 950DT | [5, 2048, 8, 1] | — | 7.571 | 7.574 | -0.0% | 6.939 |
| fusion distinct historical windows | 950PR | [5, 2048, 8, 1] | — | 7.591 | 7.433 | +2.1% | 6.939 |
| fusion distinct historical windows | 950DT | [2, 1024, 8, 4] | — | 9.032 | 9.178 | -1.6% | 8.278 |
| fusion distinct historical windows | 950PR | [2, 1024, 8, 4] | — | 9.048 | 8.973 | +0.8% | 8.278 |
| fusion distinct historical windows | 950DT | [8, 512, 8, 2] | — | 7.404 | 7.652 | -3.2% | 6.787 |
| fusion distinct historical windows | 950PR | [8, 512, 8, 2] | — | 7.410 | 7.223 | +2.6% | 6.787 |
| fusion distinct historical windows | 950DT | [3, 512, 8, 1] | — | 6.797 | 7.010 | -3.0% | 6.230 |
| fusion distinct historical windows | 950PR | [3, 512, 8, 1] | — | 6.802 | 6.697 | +1.6% | 6.230 |
| fusion distinct historical windows | 950DT | [4, 256, 8, 3] | — | 7.832 | 7.585 | +3.2% | 7.179 |
| fusion distinct historical windows | 950PR | [4, 256, 8, 3] | — | 7.835 | 7.368 | +6.3% | 7.179 |
| fusion distinct historical windows | 950DT | [1, 256, 8, 2] | — | 7.250 | 7.004 | +3.5% | 6.645 |
| fusion distinct historical windows | 950PR | [1, 256, 8, 2] | — | 7.253 | 6.583 | +10.2% | 6.645 |
