# FlashAttentionScoreV2 算例

输入轴顺序为 **Q,KV,D,B,H**。本例比较同一输入契约在 950DT、950PR 与假设的 970DT 上的设备指令窗口。

```python
from operator_model import estimate

shape = [9, 8191, 64, 1, 1]
attrs = {}
for chip in ('Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT'):
    r = estimate('fa', shape, attrs, chip=chip, dtype='bf16', timing_mode='resource')
    print(r.latency_us, r.status, r.active_aic, r.active_aiv, r.work)
```

## 输入到工作量

该路线调用 CANN 9.2.1 的 ops-transformer 算子族。实际调用接口为 `aclnnFlashAttentionScoreV2`。
工作量接口给出的 Cube 操作为 18,872,064，Vector 计数为 442,296。逻辑读/写分别为 2098048 / 1728 B。此处 None 表示缺少计数；片外实际流量须另行观测，逻辑字节不含中间层重复搬运。

## 分块、资源和关键路径

Q、K、V 为 BF16 稠密 SBH。QK 与 PV 各需 2BHQ·KV·D，合计 18,872,064 个有用 Cube ops。模型按实际尾块与 16 对齐计算硬件工作。

#### Ascend950DT_9582 的首块、循环和排空

主 tile [Q,KV]=[64, 256]；并行任务 B·H·ceil(Q/tq)=1，1 波，每波遍历 ceil(KV/tk)=32 块。AIC/AIV=1/2。Cube 8181.818 ops/(cycle·AIC)，Vector 的 softmax 指令混合折合 20.185 等效操作/(cycle·AIV)，由 DT 指令流校准后按目标 FP32 每核峰值缩放。

这一类波重复 1 次；每块 9 个 Q 行，每个活跃 AIC 配 2.00 AIV，K 读 32,768 B、V 同量。每核可用读带宽 min(本地 MTE2, 共享带宽/占用核)=80.068 B/cycle。首读基准延迟 1106.18 cycles，Q 和 K 就绪后再搬 L0。K/V 首就绪 1607.43/3599.93 cycles；首次 QK/PV 开始 1781.43/6252.43 cycles。首块结束为 PV 开始加 145.00 cycles。

每个完整 KV 块各资源的发起服务（cycles）为 cube=356.000，vector=364.625，k_v_read=818.500，mte1=463.672，fix=178.031，issue_control=853.405。II=max(这些服务)=853.40 cycles，受 issue_control 限制；尾块 II=853.40。首块之后累计 max(0,32−2)×完整块 II 加一个尾块 II（仅当有第二块）。写侧按输出 1728 B / 80.068 B/cycle 加末次 Vector 和控制排空。

launch=6.479，first_tile=3.877，steady=16.034，drain=0.334 µs，相加为 **26.72393 µs**。

#### Ascend970DT 的首块、循环和排空

主 tile [Q,KV]=[64, 256]；并行任务 B·H·ceil(Q/tq)=1，1 波，每波遍历 ceil(KV/tk)=32 块。AIC/AIV=1/1。Cube 8194.444 ops/(cycle·AIC)，Vector 的 softmax 指令混合折合 20.210 等效操作/(cycle·AIV)，由 DT 指令流校准后按目标 FP32 每核峰值缩放。

这一类波重复 1 次；每块 9 个 Q 行，每个活跃 AIC 配 1.00 AIV，K 读 32,768 B、V 同量。每核可用读带宽 min(本地 MTE2, 共享带宽/占用核)=80.068 B/cycle。首读基准延迟 1106.18 cycles，Q 和 K 就绪后再搬 L0。K/V 首就绪 1607.43/3599.93 cycles；首次 QK/PV 开始 1781.43/6252.43 cycles。首块结束为 PV 开始加 144.90 cycles。

每个完整 KV 块各资源的发起服务（cycles）为 cube=355.803，vector=655.509，k_v_read=818.500，mte1=463.672，fix=178.031，issue_control=853.405。II=max(这些服务)=853.40 cycles，受 issue_control 限制；尾块 II=853.40。首块之后累计 max(0,32−2)×完整块 II 加一个尾块 II（仅当有第二块）。写侧按输出 1728 B / 80.068 B/cycle 加末次 Vector 和控制排空。

launch=5.939，first_tile=3.554，steady=14.698，drain=0.327 µs，相加为 **24.51796 µs**。

[DT 指令发起、读延迟和队列吞吐参数](../../results/v0.16/fa.json) · [改变绑定方式及 Q、KV 后的瓶颈](../../results/v0.17/fa-scenarios.md)

## 设备时间及误差

| 平台 | 模型 (µs) | CA (µs) | 有符号误差 | 预测 AIC / AIV | CA AIC / AIV |
|---|---:|---:|---:|---|---|
| 950DT | 26.72393 | 26.26909 | +1.73% | 1 / 2 | 1 / 2 |
| 950PR | 26.72471 | 25.19697 | +6.06% | 1 / 2 | 1 / 2 |
| 970DT 外推 | 24.51796 | N/A | N/A | 1 / 1 | N/A |

DT/PR 加速比取同形耗时之比，模型 1.0000×，CA 1.0425×。正时延误差为高估。

**950DT CA**　msopprof official full trace，角色 new paired DT CA；核名 `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic`。该宿主的数值检查共 1 组；子算子切片共享宿主检查。各设备窗以最早到最晚核指令计，多 kernel 取串行窗之和。

**950PR CA**　msopprof official full trace，角色 frozen PR holdout；核名 `FlashAttentionScore_50348df09d0b3ee451d3693a6b2ab3f5_2328361008425599520_mix_aic`。该宿主的数值检查共 1 组；子算子切片共享宿主检查。各设备窗以最早到最晚核指令计，多 kernel 取串行窗之和。

## CA 流水与模型差距

![FlashAttentionScoreV2 实测队列](figures/fa.png)

横轴为每次 kernel 自身的设备时间，两端各自从零开始。每种核取最晚结束的一个核。每行按该队列有效指令区间的并集着色，100 个等宽时间格，深蓝表示格内占满。不同队列可重叠，行长不能相加。DMA/FIX 行表示描述符的在途窗口，可能包含等待依赖的时间；颜色深度不能直接换算 HBM 总线利用率。

950DT 第 1 段 26.2691 µs；所选核上占用最长的队列为 AIC/Scalar 21.6212 µs、AIC/MTE2 18.3370 µs、AIV/Scalar 17.8661 µs。
950PR 第 1 段 25.1970 µs；所选核上占用最长的队列为 AIC/Scalar 20.9818 µs、AIC/MTE2 17.8224 µs、AIV/Scalar 17.3721 µs。

**950DT 发起与完成**　取 core0.cubecore0。全核 MMAD 计数为 128。

| 队列 | 指令及程序计数器 | 临界核次数 | 首次发起距全核起点 (µs) | 描述符中位持续 (µs) | 相邻发起中位间隔 |
|---|---|---:|---:|---:|---|
| CUBE | `MMAD` @ `0x9156267f94` | 64 | 7.4909 | 0.0600 | 0.0970 µs |
| CUBE | `MMAD` @ `0x91562687ac` | 64 | 9.7994 | 0.0600 | 0.1067 µs |
| MTE2 | `MOV_OUT_TO_L1_MULTI_ND2NZ` @ `0x9156267a20` | 32 | 6.5436 | 0.5761 | 0.4976 µs |
| MTE2 | `MOV_OUT_TO_L1_MULTI_ND2NZ` @ `0x9156268624` | 32 | 7.6133 | 0.5685 | 0.5036 µs |
| MTE2 | `MOV_OUT_TO_L1_MULTI_ND2NZ` @ `0x9156269034` | 1 | 6.4879 | 0.7400 | N/A（只有一次） |

模型对本次调用的全核 MMAD 预计 128 次。在有多核重叠时，全核次数用于核对分块，时间取设备窗。

到首次 QK 的模型时间为启动 6.4788 µs 加本波 QK 就绪 1.0797 µs。首个满 KV tile 的稳态间隔为 0.5172 µs；它应与同一 K 或 V 读程序计数器的连续发起间隔比较。

**950PR 发起与完成**　取 core0.cubecore0。全核 MMAD 计数为 128。

| 队列 | 指令及程序计数器 | 临界核次数 | 首次发起距全核起点 (µs) | 描述符中位持续 (µs) | 相邻发起中位间隔 |
|---|---|---:|---:|---:|---|
| CUBE | `MMAD` @ `0x9156267f94` | 64 | 6.8188 | 0.0600 | 0.0988 µs |
| CUBE | `MMAD` @ `0x91562687ac` | 64 | 9.0582 | 0.0600 | 0.0988 µs |
| MTE2 | `MOV_OUT_TO_L1_MULTI_ND2NZ` @ `0x9156267a20` | 32 | 5.9103 | 0.5664 | 0.4976 µs |
| MTE2 | `MOV_OUT_TO_L1_MULTI_ND2NZ` @ `0x9156268624` | 32 | 6.9291 | 0.5688 | 0.5000 µs |
| MTE2 | `MOV_OUT_TO_L1_MULTI_ND2NZ` @ `0x9156269034` | 1 | 5.8545 | 0.7012 | N/A（只有一次） |

模型对本次调用的全核 MMAD 预计 128 次。在有多核重叠时，全核次数用于核对分块，时间取设备窗。

到首次 QK 的模型时间为启动 6.4788 µs 加本波 QK 就绪 1.0797 µs。首个满 KV tile 的稳态间隔为 0.5172 µs；它应与同一 K 或 V 读程序计数器的连续发起间隔比较。

PR 的模型与 CA 相差 +1.5277 µs。只有工作量大到出现重复 KV 块时，发起间隔才会主导。首 MMAD 就绪、Vector 交接、K/V 的真实 cache 重用与读队列延迟仍可改变总时长。上图给出可观测的队列活动；一次端到端差值无法唯一分摊到这些因素。

[总建模方法](../modeling-strategy.md) · [全部 shape 的误差](../error-analysis.md) · [970DT 配置与来源](../../configs/chips/Ascend970DT.json)
