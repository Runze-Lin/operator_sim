# 真实大形 CA 覆盖

本轮保留 21 条实现，其中 6 条已有同一大形、相同合同的双端完整设备计时。历史大形登记中仍在范围内的有 20 条，其中 6 条有双端完整窗。第 21 条是单状态链 recurrent KDA，其已测 S=8 边界样本与大形验收分别统计。已豁免多链 recurrent 与 chunk fused V1，V2 仍在表中。

RMSNorm 新增的是对过往真实采集的严格复核。原 large29 清单已经登记了 Flash 每次调度 8192 个 token、hidden=5120；此处沿用它的大形口径。已测 8192 行，B8 不切 chunk 时的 262144、1048576、2097152 行仍待测。

| 路线 | 已有双端合格大形 | 核对依据 |
|---|---|---|
| addmm | N/A | [上一版逐形资格与端点](../../v0.32/large/report.md) |
| attn_res_fwd | N/A | [上一版逐形资格与端点](../../v0.32/large/report.md) |
| cast_bf16_fp32 | [262144,32]、[262144,64]、[393216,32]、[1048576,32]、[1048576,64]、[1572864,32]、[2097152,32]、[2097152,64] | [上一版逐形资格与端点](../../v0.32/large/report.md) |
| fa | [9,32768,64,1,1]、[9,131072,64,1,1] | [上一版逐形资格与端点](../../v0.32/large/report.md) |
| fla_kda_v2 | N/A | [上一版逐形资格与端点](../../v0.32/large/report.md) |
| gmm2 | N/A | [上一版逐形资格与端点](../../v0.32/large/report.md) |
| grouped_matmul_swiglu_quant_v2 | N/A | [上一版逐形资格与端点](../../v0.32/large/report.md) |
| kda_recurrent (single chain) | N/A | [S≤8 的单链控制，旧登记无单链大形](#单状态链-kda) |
| lightning_indexer | N/A | [上一版逐形资格与端点](../../v0.32/large/report.md) |
| matmul | [262144,128,512]、[1048576,128,512] | [上一版逐形资格与端点](../../v0.32/large/report.md) |
| mhc | N/A | [上一版逐形资格与端点](../../v0.32/large/report.md) |
| mhc_post | N/A | [上一版逐形资格与端点](../../v0.32/large/report.md) |
| mhc_pre | N/A | [上一版逐形资格与端点](../../v0.32/large/report.md) |
| mhc_pre_sinkhorn | N/A | [上一版逐形资格与端点](../../v0.32/large/report.md) |
| mhc_sinkhorn | [513,4]、[4097,4] | [上一版逐形资格与端点](../../v0.32/large/report.md) |
| moe_ffn | N/A | [上一版逐形资格与端点](../../v0.32/large/report.md) |
| moe_gating_top_k | N/A | [上一版逐形资格与端点](../../v0.32/large/report.md) |
| quant | N/A | [上一版逐形资格与端点](../../v0.32/large/report.md) |
| rmsnorm | [8192,5120] | [本轮逐核复核](#rmsnorm-大形) |
| sigmoid_fp32 | [262144,12]、[524288,12]、[1048576,12]、[1572864,12]、[2097152,12] | [上一版逐形资格与端点](../../v0.32/large/report.md) |
| vllm_hc_post | N/A | [上一版逐形资格与端点](../../v0.32/large/report.md) |

### 其它优先排查的 Vector 路线

| 算子 | 已查到的对照 | 尚缺的双端大形 |
|---|---|---|
| DynamicMxQuant | [2048,5120] 双端指令窗，Kimi [262144,7168] 的 DT 已有数值和 wrapper 整窗，独立于指令窗口径 | [16384,7168] 双端指令、[262144,7168] 同合同双端完整计时 |
| MoE gating top-k | 两端真实 896 专家、top-16 的既有控制为 24 token | 既有登记 [4096,896,16] |
| AttnResFwd | 既有 7168 通道小形使用 materialize 或不同写回属性；不能替代 Kimi 的 in-place prefill 合同 | 已登记 [512,7168,8,1]、write_idx=-1、materialize=false |

本轮优先找回已登记且已有真实测量的调用。[原 Vector 合同、真实参数和既有采集队列](../../v0.29/large/vector/report.md)。

## 单状态链 KDA

单链解码合同 B=Hq=Hv=1、K=128、V=128 或 256、S≤8。Kimi 的 V=128 路径已测到 [1,8,1,1,128,128]，DT/PR 完整窗 35.396/39.450 µs；这是单核八步序列边界，状态张量为 64 KiB。旧大形登记采用的是 B6、16、128 且每请求含 12 条状态链。单链在此表保留为大形 N/A；保留的单链准确性与适用域见[逐形回放](../../v0.32/replay.json)。

## RMSNorm 大形

表内 950DT/PR 的 CA 是从全核最早指令发起到最后指令退休的微秒数。耗时预测使用隔离安装的 0.32.0，按调用方只传 shape、dtype、芯片的入口定价。所有实测属于开发回放，970DT 一列只作外推。

| 形状 | 芯片 | 模型 / CA µs | 有符号误差 | 970DT 外推 µs | 参与 AIV | 单核实测行数 / 块数 × 核数 | 原 CA 墙钟 |
|---|---|---:|---:|---:|---:|---|---:|
| [8192, 5120] | Ascend950DT_9582 | 39.379 / 44.868 | -12.23% | 21.582 | 64 | 128 / 26 × 64 | 68.9 min |
| [8192, 5120] | Ascend950PR_957b | 61.622 / 70.485 | -12.57% | 21.582 | 56 | 107 / 22 × 1，147 / 30 × 55 | 69.3 min |
| [32768, 1280] | Ascend950DT_9582 | 39.393 / 44.585 | -11.64% | 21.436 | 64 | 512 / 22 × 64 | 84.9 min |
| [32768, 1280] | Ascend950PR_957b | 61.616 / 69.729 | -11.63% | 21.436 | 56 | 538 / 23 × 1，586 / 25 × 55 | 96.3 min |

| 几何 | 950DT → PR 实测加速比 | 模型加速比 | 加速比误差 |
|---|---:|---:|---:|
| [8192, 5120] | 0.6366× | 0.6390× | +0.39% |
| [32768, 1280] | 0.6394× | 0.6393× | -0.01% |

第一个 shape 对应 Flash 归一化的实际 hidden 和 B4 × 2048 已调度 token，也可组成 B8 × 1024 的单步调度。后者是将同一输入形状映射到另一个请求调度组合。[32768,1280] 具有相同元素总数，是宽度/行数控制样本，不能代表对 32Ki 整上下文的全 hidden 归一化验证。

每个端点实查安装包 ops-nn 的 RmsNorm_normal_all_bf16_high_performance_5000、采集对象 .text、物理 SKU 与内存模型、完整 BF16 Y 与逐行 FP32 rstd 结果。原始合并 trace 的 SHA-256、每个 AIV 唯一物理 END、供应商各核 duration、实际 MTE3 描述符写出的 Y 和 rstd 总量均通过。[32768,1280] 的 PR 有核缺 profiler 的 END_LABEL 附注，对该核另完整扫描所有 X 事件，以最早发起和最晚退休交叉核对。逐核指令汇总 CSV 会把尾块与满块计到同一 PC，只能从原逐条描述符取实际 burst_number 和 processed_bytes。

计数遗漏来自原 large29 聚合只导入 vector/archive 下的 wrapper 和特定 Cast/Sigmoid 的大形回放，而后续分级表没有联入 v0.28 的 RMSNorm msopprof 大形库。这里复用同一次 CA 的已验证物理身份；数据库的历史误差分母与独立样本数不增加。两条旧采集登记已写跳过回执，明确本次未发起 CA、没有本次耗时，原预测与外部合格 trace 分别保存。[DT 去重回执](../../v0.29/large/vector/instruction-rmsnorm-8192-5120-Ascend950DT_9582.json) · [PR 去重回执](../../v0.29/large/vector/instruction-rmsnorm-8192-5120-Ascend950PR_957b.json)

[冻结的大形判据与原采集来源](freeze.json) · [计算过程与具体案例](../../../docs/examples/rmsnorm-large.md) · [重建脚本](../../../development/validation/coverage33/rms_archive.py)
