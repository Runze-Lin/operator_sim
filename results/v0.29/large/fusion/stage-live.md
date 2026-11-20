# 预登记大形阶段模型的实测进度

| 实际调用 | 路线 | 芯片 | 公开 / 已选候选 / CA (µs) | 候选误差 | 状态 |
|---|---|---|---:|---:|---|
| flash-chunk512 | mhc_pre | Ascend950DT_9582 | 2512.514 / 56.427 / — | — | queued |
| flash-chunk512 | mhc_pre | Ascend950PR_957b | 2982.251 / 67.586 / — | — | queued |
| flash-chunk512 | mhc_sinkhorn | Ascend950DT_9582 | — / — / — | — | queued |
| flash-chunk512 | mhc_sinkhorn | Ascend950PR_957b | — / — / — | — | queued |
| sink-chunk4096 | mhc_sinkhorn | Ascend950DT_9582 | — / 30.750 / — | — | queued |
| sink-chunk4096 | mhc_sinkhorn | Ascend950PR_957b | — / 38.401 / — | — | queued |
| flash-chunk4096 | mhc_pre | Ascend950DT_9582 | 20100.112 / 412.078 / — | — | queued |
| flash-chunk4096 | mhc_pre | Ascend950PR_957b | 23219.433 / 517.134 / — | — | queued |
| flash-chunk4096 | mhc_sinkhorn | Ascend950DT_9582 | — / — / — | — | queued |
| flash-chunk4096 | mhc_sinkhorn | Ascend950PR_957b | — / — / — | — | queued |
| sink-b8-32ki | mhc_sinkhorn | Ascend950DT_9582 | — / — / — | — | queued |
| sink-b8-32ki | mhc_sinkhorn | Ascend950PR_957b | — / — / — | — | queued |

独立单发且单 UB 的 Sinkhorn 可以按冷启动定价。三段链的 Sinkhorn 无可验证的同族前序；跨 UB 的阶段价格保留两种块容量假设。表内只给有明确比较资格的单点，所有场景价格及各自有符号偏差完整列在 JSON。970DT 始终作为架构外推。

[预注册方法](stage-report.md) · [逐假设数值与资格](stage-score.json)
