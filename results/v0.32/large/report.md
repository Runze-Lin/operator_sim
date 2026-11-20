# 大几何覆盖：22 条算子路线

按同形、同合同、双端完整计时统计，**5/22** 条路线至少有一个合格大形；不是 22 条都已验证。两种较长的 Sinkhorn [513,4] 与 [4097,4] 都是历史开发回放，迭代数分别 6、4。没有把它们算成生产配置 20 次迭代的独立校验。

| 算子 | 登记形状示例（小端 → 大端） | 双端完整的大形代表：DT / PR µs | 数据 |
|---|---|---|---|
| addmm | 262144×128×512 | N/A | [明细](../../v0.29/large/cube/freeze.json) |
| attn_res_fwd | 512×7168×8×1 | N/A | [明细](../../v0.29/large/vector/freeze.json) |
| cast_bf16_fp32 | 262144×32 → 262144×5120 | 2097152×64: 247.788 / 529.287 | [明细](../../v0.29/large/vector/freeze.json) |
| fa | 9×32768×64×1×1 → 9×524288×64×1×1 | 9×131072×64×1×1: 274.821 / 265.893 | [明细](../../v0.29/large/fa/freeze.json) |
| fla_kda_v2 | 1×1024×12×12×128×128 | N/A | [明细](../../v0.29/large/cube/freeze.json) |
| gmm2 | 4096×7168×384×1 | N/A | [明细](../../v0.29/large/cube/freeze.json) |
| grouped_matmul_swiglu_quant_v2 | 4096×7168×384×1 | N/A | [明细](../../v0.29/large/cube/freeze.json) |
| kda_chunk_fused_v1 | 1×16384×12×12×128×128 | N/A | [明细](../../v0.29/kda/report.md) |
| kda_recurrent | 6×1×12×12×128×128 → 128×1×12×12×128×128 | N/A | [明细](recurrent-freeze.json) |
| lightning_indexer | 1×9×32768×8×128 | N/A | [明细](../../v0.29/large/fusion/freeze.json) |
| matmul | 262144×128×512 → 2097152×128×512 | 1048576×128×512: 405.617 / 804.867 | [明细](../../v0.29/matmul/full-load/freeze.json) |
| mhc | 512×4×5120 → 4096×4×5120 | N/A | [明细](../../v0.29/large/fusion/freeze.json) |
| mhc_post | 512×4×5120 → 4096×4×5120 | N/A | [明细](../../v0.29/large/fusion/freeze.json) |
| mhc_pre | 512×4×5120 → 4096×4×5120 | N/A | [明细](../../v0.29/large/fusion/freeze.json) |
| mhc_pre_sinkhorn | 512×4×4096 | N/A | [明细](../../v0.29/large/fusion/freeze.json) |
| mhc_sinkhorn | 512×4 → 262144×4 | 4097×4: 8.964 / 9.112 | [明细](sink-score.json) |
| moe_ffn | 4096×7168×384×1 | N/A | [明细](../../v0.29/large/cube/freeze.json) |
| moe_gating_top_k | 4096×896×16 | N/A | [明细](../../v0.29/large/vector/freeze.json) |
| quant | 262144×7168 | N/A | [明细](../../v0.29/large/vector/freeze.json) |
| rmsnorm | 8192×5120 → 262144×5120 | N/A | [明细](../../v0.29/large/vector/freeze.json) |
| sigmoid_fp32 | 262144×12 → 2097152×12 | 2097152×12: 42.686 / 88.593 | [明细](../../v0.23/prefill/kimi-128ki-dt.json) |
| vllm_hc_post | 1×512×4×5120 → 1×1024×4×5120 | N/A | [明细](../../v0.29/large/fusion/freeze.json) |

### 每条路线的 DT / PR 状态

表中为已登记大形的原始状态汇总。完整数按具有合格指令窗的不同 shape 去重，wrapper 另列；非终态仍需查看正在执行该次采集的日志，不能把“已登记”当作已经运行。一个已有完整窗的算子仍可能有其他未验证的大形。

| 算子 | DT | PR |
|---|---|---|
| addmm | 完整 N/A；有登记待回执 | 完整 N/A；有登记待回执 |
| attn_res_fwd | 完整 N/A；有登记待回执 | 完整 N/A；有登记待回执 |
| cast_bf16_fp32 | 完整 8 形；wrapper 1 形 | 完整 8 形；有登记待回执 |
| fa | 完整 2 形；有登记待回执 | 完整 2 形；有登记待回执 |
| fla_kda_v2 | 完整 N/A；有登记待回执 | 完整 N/A；有登记待回执 |
| gmm2 | 完整 N/A；有登记待回执 | 完整 N/A；有登记待回执 |
| grouped_matmul_swiglu_quant_v2 | 完整 N/A；有登记待回执 | 完整 N/A；有登记待回执 |
| kda_chunk_fused_v1 | 完整 N/A；有未合格路线 | 完整 N/A；有未合格路线 |
| kda_recurrent | 完整 N/A；有登记待回执 | 完整 N/A；有登记待回执 |
| lightning_indexer | 完整 N/A；有登记待回执 | 完整 N/A；有登记待回执 |
| matmul | 完整 2 形；有采集中 | 完整 2 形；有登记待回执 |
| mhc | 完整 N/A；有登记待回执 | 完整 N/A；有登记待回执 |
| mhc_post | 完整 N/A；有登记待回执 | 完整 N/A；有登记待回执 |
| mhc_pre | 完整 N/A；有登记待回执 | 完整 N/A；有登记待回执 |
| mhc_pre_sinkhorn | 完整 N/A；有登记待回执 | 完整 N/A；有登记待回执 |
| mhc_sinkhorn | 完整 2 形；有登记待回执 | 完整 2 形；有登记待回执 |
| moe_ffn | 完整 N/A；有登记待回执 | 完整 N/A；有登记待回执 |
| moe_gating_top_k | 完整 N/A；有登记待回执 | 完整 N/A；有登记待回执 |
| quant | 完整 N/A；wrapper 1 形 | 完整 N/A；有登记待回执 |
| rmsnorm | 完整 N/A；有登记待回执 | 完整 N/A；有登记待回执 |
| sigmoid_fp32 | 完整 5 形 | 完整 5 形 |
| vllm_hc_post | 完整 N/A；有未合格路线；有登记待回执 | 完整 N/A；有采集中；有登记待回执 |

[原有 22 条路线的逐形口径](../../v0.29/large/report.md)。DT、PR 必须分别有同几何、同合同的合格完整窗口；个别 wrapper 包络单列，不列为完整指令窗。量化和 RMS 已有的 [2048,5120] 双端窗是 2048-token chunk 的控制，不等于 8192/262144 行整批验证。TopK [4096,896,16]、RMS [8192,5120] 的原任务仍由共享槽调度。未合格、等待与 970DT 外推均无 CA 数值。

## FP32 standalone MhcSinkhorn

| 形状 / 迭代数 / 角色 | DT 已知 H / 未知 H / CA µs | PR 已知 H / 未知 H / CA µs | 970DT 已知 H / 未知 H 外推 µs |
|---|---:|---:|---:|
| [513,4] / 6 / 旧形回放 | 3.388 / 4.640 / 3.622 | 3.443 / 4.656 / 3.525 | 2.400 / 3.938 |
| [4097,4] / 4 / 旧形回放 | 8.355 / 8.105 / 8.964 | 9.096 / 8.815 / 9.112 | 5.558 / 5.546 |
| [1024,4] / 20 / 冻结新形 | 9.360 / 8.174 / N/A | 11.177 / 10.313 / N/A | 9.535 / 7.814 |

[1024,4] 对应每卡 B8、每请求本次调度 128 token 的 chunk，不等于在设备上以一次调用处理整条 32Ki 输入。每端首发 H=0、20 次交替归一化，检查全部 16,384 个 FP32 输出。历史 H=57/52 的有限迭代控制分别报告。已知 H 的值使用相应分支；无 H 按可达状态均值，不把均值叫作实际运行历史。真实接口 aclnnMhcSinkhorn，CANN 9.2.1 安装版 ops-transformer 二进制。保留[模型参数和旧版基线](freeze.json)、[原始/汇总核验](archive.json)、[同集误差和墙钟](sink-score.json)。

## Kimi decode：独立状态链

| 单卡 B / 单卡形状 | DT 预测 / CA µs | PR 预测 / CA µs | 970DT 外推 µs |
|---|---:|---:|---:|
| 6 / [6, 1, 12, 12, 128, 128] | 26.713 / N/A | 27.124 / N/A | 13.804 |
| 16 / [16, 1, 12, 12, 128, 128] | 38.354 / N/A | 50.817 / N/A | 24.464 |
| 128 / [128, 1, 12, 12, 128, 128] | 282.821 / N/A | 335.138 / N/A | 141.723 |

B16 TP8 使用 Kimi 96/8=12 个本地头、K=V=128。192 条链使 DT 每核 3 轮、PR 最忙核 4 轮。B6/B128 不重复；B16 在新 Sinkhorn 两端结束后复用同一把设备锁，每端最多 1 小时、8 GiB。完整时间要通过全输出金标、发布库 .text、全核官方时长与 END、所有链的 beta 计数及状态写体积。970DT 始终无 CA。

[逐形完整 JSON](coverage.json) · [Sinkhorn 当前状态](state.json) · [B16 当前状态](recurrent-state.json) · [B16 冻结与预算](recurrent-freeze.json)

### 验收口径与开销

已重验历史循环的四个 950 窗，T=513/I6 每核都是 14 次 VF、总写 32,832 B；T=4097/I4 每核都是 10 次 VF、总写 262,208 B。这符合每核运行 2I+2 个 helper，寄存器的多次遍历发生在 helper 内。T=4097 同已知 H 时，旧模型 DT/PR 高估 26.37% / 37.53%，新模型偏差 −6.79% / −0.17%；H 不传时新模型偏差 −9.58% / −3.26%。T=513 H 不传的偏差仍有 +28.12% / +32.08%，不能据此宣称全部达标。四个窗来自同两次有序四连发；每个芯片整批用 248/252 秒，不能把该墙钟归到某一个 shape。

新增 DT/PR 用官方 msopprof simulator --dump=false，只收集判定整窗所需的合并/分核指令和 kernel 对象，保留全部输出金标；不重复抓取逐指令文本 dump。历史结果逐端复核输出字节、真实内存控制器、每核 END 与官方表、全输出、trace 哈希与安装 .text。当前 CPU 利用率与部分日志不能换算成可计分的 CA 时间。

[主机进程/锁的实际工作与等候情况](host-audit.json) · [3 秒 CPU 样本](host-sample.json)。主机有 256 个可用 CPU；采样时平均约 25.5 个 CPU 等效空闲，Sinkhorn CA 实际用了 14.5 个 CPU；高负载下旧实验耗时不是这轮的 ETA。

[每条线路最大的旧双端参照与大形目标](scale-ladder.md) · [资格、同集误差与主机开销](costs-and-evidence.md)
