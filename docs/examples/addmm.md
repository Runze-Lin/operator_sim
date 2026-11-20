# Addmm · 列偏置与目标芯片分波

![950PR 的模型关键路径与最后一个 AIC 的真实流水](figures/addmm-v21-pr-page.png)

对同一 BF16 输入 `[M,N,K]=[512,512,256]`，950PR 实测用了 28 个 AIC，一波内完成 56 条 MMAD。旧版按 DT 的分块推成额外一波，预测 7.101 µs。当前预测 4.856 µs，CA 4.687 µs，高估 3.62%。[每条图元的原始时间与 trace 哈希](../../results/v0.21/dispatch/addmm-pr-page.json)

```python
from operator_model import estimate
for chip in ('Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT'):
    r = estimate('addmm', [512,512,256], {'alpha':1, 'beta':1, 'self_shape':[512]}, chip=chip)
    print(r.latency_us, r.active_aic, r.work, r.projection['target_plan'])
```

输入为 BF16 的 A[512,256]、B[256,512] 和 bias[512]。有效工作为 134,217,728 FLOP，逻辑读为 525,312 B，写为 524,288 B。调用 CANN 9.2.1 ops-nn 的 `aclnnAddmm`，日志给出实际 kernel `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance_2`；已校验所有输出。列偏置进入 BT 参与计算，设备窗口包含此次 kernel 的所有核。

PR 的 28 个 AIC 用目标 L0/L1 容量和输出网格重选 [80,128,128] 主块。M 方向 7 块、N 方向 4 块，24 个 80×128 主输出和 4 个 32×128 尾输出，每个块 K 读一页 256，页内计算两次。bias 随 N 分区在同一个 tile 内复用，冷读一次共 1,024 B，跨各 AIC 的 L1→BT 搬运合计 28×128×2=7,168 B。数学字节量、核侧搬运量与片外事务分别计量。

| 关键路径部分 | 设备 µs | 来自哪些输入 |
|---|---:|---|
| 启动及核到达 | 2.315 | 启动周期和 log₂(28) 的到达偏斜，来自 950 的指令校准 |
| GM→L1 首次填充 | 1.557 | 操作数与列偏置，分别受本核端口、共享 L2 和片外读限制 |
| L1→L0/BT 先导 | 0.131 | 首个 K=128 段的数据服务与 L0 响应周期 |
| MMAD 两段 | 0.508 | 对齐后的矩阵运算数、目标 BF16 每核算力、发起间隔与末条响应 |
| FIXP 写出与排空 | 0.345 | 主块 20,480 B 的本核写口及共享写口上界，含末写退休 |
| 合计 | **4.856** | 28 核同一波，最慢输出块的关键路径 |

下图栏的偏置读和 B 读几乎同时发起。首条实际 MMAD 在 3.886 µs，模型在约 4.003 µs 才完成先导；收尾误差并非来自额外一波。FIXP 的长橙条从描述符发起开始，包含设备等待依赖的时间。本模型上栏的写尾是关键路径的暴露成本。[队列周期参数](../../results/v0.16/matmul.json)

| 平台 | 当前预测 µs | CA µs | AIC |
|---|---:|---:|---:|
| 950DT | 4.773 | 4.479 | 32 |
| 950PR | 4.856 | 4.687 | 28 |
| 970DT | 3.761 | N/A · 外推 | 88 |

DT 原始归档与 PR 单独发起的实验沿用各自的采集上下文。本例是开发回放，新验证的形状、预测、基线和资格记录另外保存。[开发形状原始回执](../../results/v0.17/evaluation.json) · [独立预注册](../../results/v0.21/dispatch/freeze.json)

## 65,536 行的多波 prefill 算例

设每卡有 8 个 32Ki 请求，本次每个请求处理 8192 个 token，取 Flash C1 来源层 `indexer.wk` 的输入宽度 512、输出宽度 128，得到 `A[65536,512]`、`B[512,128]`。源码的 `wk` 使用 `bias=False`。本例显式传入独立的非零 BF16 `self[128]`，用于验证系统仿真后端 `addmm` 的列偏置合同。实际调度的 `max_num_batched_tokens` 决定单次能否容纳这 65,536 行。[锁定的 vLLM 源码](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/models/deepseek_v41/indexer.py#L74-L80)

$2MNK=8,589,934,592$ FLOP，逻辑输入 A、B、self 分别占 64 MiB、128 KiB、256 B，输出 Y 占 16 MiB。950DT 当前模型为 32 个 AIC 各排 4 波、共 2,048 条 MMAD，给出 32.646 µs。950PR 为 28 个 AIC、6 波、2,688 条 MMAD，预测 53.193 µs。970DT 为 12.404 µs，属于外推。一个可验证的问题是 B 能否在每核 L1 常驻。以 DT 为例，如果设备采用 256×128×32 而非 512×128×32 的块，MMAD 将翻到 4,096 条，但 B 可能只需每核从 GM 读一次，核内波数上升仍可缩短整窗。实际分核、L1→BT 偏置指令、MMAD 尺寸和总时间随独立端点回执更新。[大形实测、候选和偏差](../../results/v0.34/addmm/report.md)
