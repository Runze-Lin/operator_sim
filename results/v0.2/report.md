# v0.2 分支与边界扩展结果

本轮复用 v0.1 的 15 个不同输入作为校准；本轮留出点全部是新 shape，预测在采集前冻结。不能将 v0.1 旧留出点在本轮继续算作独立测试。

| 算子 | 新留出数 | WAPE | P90 APE | 固定耗时 WAPE | 经验 Roofline WAPE | 达标 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| quant | 2 | 13.85% | 18.67% | 7.27% | 57.68% | False |
| matmul | 2 | 6.91% | 7.57% | 12.82% | 44.92% | True |
| fa | 3 | 4.47% | 6.20% | 3.21% | 32.91% | True |
| all | 7 | 7.25% | 12.61% | 5.92% | 41.22% | True |

达标目标为 WAPE≤10%、P90 APE≤20%。均匀权重；经验 Roofline 非理论峰值、非用户原始实现。

| 新留出 shape | CA μs | 预测 μs | APE | 预测/实际 AIC,AIV | 校准同机制点数 | 覆盖 |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| quant [1, 4096] | 5.6861 | 5.2775 | 7.18% | [0, 16] / [0, 16] | 4 | within_calibration_box |
| matmul [129, 256, 256] | 3.6642 | 3.9469 | 7.71% | [20, 0] / [20, 0] | 1 | provisional_regime |
| matmul [17, 256, 1024] | 4.7418 | 5.0402 | 6.29% | [8, 0] / [8, 0] | 4 | within_calibration_box |
| fa [129, 128, 64, 1, 1] | 8.8709 | 9.0807 | 2.37% | [3, 6] / [3, 6] | 6 | extrapolation |
| fa [16, 128, 64, 1, 8] | 9.4018 | 8.9921 | 4.36% | [8, 16] / [8, 16] | 6 | extrapolation |
| fa [17, 192, 128, 1, 1] | 9.0503 | 9.6533 | 6.66% | [1, 2] / [1, 2] | 1 | provisional_regime |
| quant [640, 4096] | 6.2176 | 7.4579 | 19.95% | [0, 64] / [0, 64] | 1 | provisional_regime |

## 分支、分核与指令证据

kernel 标识相同不表示 tile 相同；保留实际 MMAD 的 m/n/k 和次数。自定义的 `branch` 是模型中的机制分组，不冒充已解码的 CANN tilingKey。

| Case | 分组 | 核数匹配 | 最大核选定指令数：预测/实际 | Cube busy | Vector family busy |
| --- | --- | --- | --- | ---: | ---: |
| q2_cal_tiny | slice256 | True | 136.0/136 | 0.00% | 1.82% |
| q2_cal_saturation | slice512 | True | 180.0/180 | 0.00% | 1.74% |
| q2_cal_ub_loop | ub_loop | True | 11896.0/11896 | 0.00% | 25.73% |
| m2_cal_skinny_k | gemv | True | 4.0/4 | 8.56% | 0.00% |
| m2_cal_k_tail | k_tail | True | 1.0/1 | 2.32% | 0.00% |
| m2_cal_full | coarse_tile | True | 2.0/2 | 5.03% | 0.00% |
| f2_cal_q128 | d64_kv256 | True | 3.0/2 | 2.49% | 3.73% |
| f2_cal_longkv | d64_kv256 | True | 15.1/18 | 7.04% | 7.73% |
| f2_cal_heads4 | d64_kv128 | True | 3.0/2 | 1.37% | 1.99% |
| f2_cal_d128 | d128_kv128 | True | 3.0/2 | 1.82% | 2.31% |
| q2_test_wide | slice256 | True | 136.0/136 | 0.00% | 1.39% |
| m2_test_m_tail | coarse_tile | True | 1.0/1 | 3.89% | 0.00% |
| m2_test_k_work | dense | True | 4.0/4 | 8.37% | 0.00% |
| f2_test_q129 | d64_kv128 | True | 3.0/2 | 2.66% | 4.00% |
| f2_test_heads8 | d64_kv128 | True | 3.0/2 | 1.28% | 1.85% |
| f2_test_d128 | d128_kv128 | True | 6.1/4 | 3.94% | 3.68% |
| q2_test_ub_loop | ub_loop | True | 7480.0/7480 | 0.00% | 21.60% |

busy 为非同步指令区间并集/跨核指令时间窗，Vector family 包含 RegBase load/store/control；这是占用观察，不是峰值算力利用率。稀疏训练分组保持 provisional；不把低时延误差当作准确分核/指令模型的证明。

新留出核数匹配 7/7；选定指令次数 WAPE：quant=0.00%, matmul=0.00%, fa=51.28%。

部分 MMAD trace 事件没有 m/n/k 字段；指令总数使用 CSV call_count，维度列表仅代表带完整 detail 的事件，不能用它推导完整执行 FLOPs。存在该情况的 case：f2_cal_longkv。

## 开销与超时

新增采集尝试 18 次，成功 17 次，复用旧 shape 15 个；本轮墙钟 23.02 分钟、CPU 7.41 小时、生成原始数据 4.82 GiB。

两轮合计墙钟 40.81 分钟、CPU 11.53 小时。CPU 是累计计算资源，不是用户等待时间。

在线未缓存 P50/P95 33.56/44.02 μs；缓存 P50/P95 0.89/0.98 μs；批量 1,330,831 次/秒。

失败尝试计入开销，不作为设备耗时样本；慢例重试保留独立目录。缺失项：[]

- q2_cal_ub_loop attempt=None：failed，墙钟 152.9s；Incomplete/incorrect CA run: inspect run.log。

详细归因与未完成项见 [误差分析](../../docs/error-analysis-v2.md)。

## 结论边界

- 本轮覆盖 BF16 普通 MatMul、尾轴 block32 FP8 quant、BF16 标准 FA；没有覆盖所有 dtype、layout、mask、量化矩阵、稀疏/MLA/KDA 和所有算法分支。
- 分组只有 1–2 个校准点时不能识别全部成本参数；使用共享阶段成本与收缩残差，并显式标记 provisional。
- CA 配置、时钟与版本仍固定为 950DT/CANN9.2.1。没有将系数直接迁移到 A7/A8。
- 本轮新留出结果不用于修改已冻结模型；若未达标，误差保留，下一轮需要新留出点。
