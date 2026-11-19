# Post 扩核与隔形回访

按同一进程各个已经派发过的真实 AIV 名单重建历史。若某个新 AIV 的同物理核兄弟此前已执行，将其标为“新兄弟”，整个物理核均未参与过则标为“新物理核”。该分类来自设备核标识，不预设工作集留在 L2、UB 或指令缓存中。每次设备时间独立从最早物理指令开始。

| SKU | shape / 状态 | 上次家族内已用核数 → 本次核数 | 最近一次/当前 ELF 函数后缀 | 最后完成的核与类别 | 已用 / 新兄弟 / 新物理核的最晚加入 µs | 预测 / 物理窗 µs |
|---|---|---:|---|---|---|---:|
| 950DT | [16, 4, 768] / first | 0 → 64 | — / 5 | core15.veccore0 (new physical core) | — / — / 0.679 | 4.755 / 4.990 |
| 950PR | [16, 4, 768] / first | 0 → 32 | — / 5 | core23.veccore1 (new physical core) | — / — / 0.402 | 4.861 / 4.582 |
| 950PR | [29, 4, 320] / reuse | 32 → 29 | 5 / 6 | core17.veccore1 (issued) | 0.456 / — / — | 2.157 / 2.290 |
| 950DT | [1, 4, 192] / first | 0 → 1 | — / 5 | core0.veccore0 (new physical core) | — / — / 0.000 | 4.755 / 4.527 |
| 950DT | [2, 4, 448] / expand | 1 → 4 | 5 / 5 | core16.veccore1 (new physical core) | 0.000 / 0.119 / 1.458 | 4.211 / 5.347 |
| 950DT | [8, 4, 896] / expand | 4 → 32 | 5 / 5 | core5.veccore0 (new physical core) | 0.049 / — / 0.785 | 4.211 / 4.228 |
| 950DT | [1, 4, 192] / revisit | 32 → 1 | 5 / 5 | core0.veccore0 (issued) | 0.000 / — / — | 2.280 / 1.742 |
| 950PR | [1, 4, 192] / first | 0 → 1 | — / 5 | core0.veccore0 (new physical core) | — / — / 0.000 | 4.758 / 4.281 |
| 950PR | [2, 4, 448] / expand | 1 → 4 | 5 / 5 | core16.veccore1 (new physical core) | 0.000 / 0.119 / 1.534 | 4.215 / 5.338 |
| 950PR | [8, 4, 896] / expand | 4 → 32 | 5 / 5 | core23.veccore1 (new physical core) | 0.048 / — / 0.800 | 4.215 / 4.135 |
| 950PR | [1, 4, 192] / revisit | 32 → 1 | 5 / 5 | core0.veccore0 (issued) | 0.000 / — / — | 2.283 / 1.730 |
| 950DT | [69, 4, 640] / first | 0 → 35 | — / 6 | core17.veccore1 (new physical core) | — / — / 0.551 | 3.858 / 4.195 |
| 950DT | [257, 4, 832] / expand | 35 → 52 | 6 / 6 | core12.veccore0 (new physical core) | 0.053 / 0.144 / 0.692 | 5.669 / 5.356 |
| 950PR | [69, 4, 640] / first | 0 → 35 | — / 6 | core19.veccore0 (new physical core) | — / — / 0.463 | 3.867 / 3.884 |
| 950PR | [257, 4, 832] / expand | 35 → 52 | 6 / 6 | core25.veccore1 (new physical core) | 0.053 / 0.145 / 0.696 | 5.707 / 5.251 |
| 950DT | [3, 4, 7168] / first | 0 → 48 | — / 5 | core2.veccore0 (new physical core) | — / — / 0.568 | 4.889 / 4.753 |
| 950DT | [3, 4, 7168] / revisit | 48 → 48 | 5 / 5 | core25.veccore1 (issued) | 0.053 / — / — | 2.414 / 2.745 |
| 950PR | [3, 4, 7168] / first | 0 → 48 | — / 5 | core27.veccore1 (new physical core) | — / — / 0.461 | 4.895 / 4.772 |
| 950PR | [3, 4, 7168] / revisit | 48 → 48 | 5 / 5 | core27.veccore1 (issued) | 0.053 / — / — | 2.420 / 2.722 |

旧训练集中扩核前只用过 1 个 AIV 时，2 个末核的加入偏移中位值是 1.413 µs；其余 10 个扩核窗中位值为 0.672 µs。旧扩核证据全部来自 DT。新样本可直接检验先前有一个 AIV 与已经覆盖一对 AIV 这两种历史，但这两个时延分组还不能唯一确定延迟提升来自哪一级硬件。

[每次调用的实际物理核分组及队列起止](topology.json)
