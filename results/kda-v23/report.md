# FLA Chunk KDA V2 开发采集

这组样本直接调用 FLA 26.9.1 的公开三阶段入口。BF16，BSND，B=1，K=V=128，C=64，Q/K/V 与 raw gate 的数值和布局沿用已通过的 Kimi 小例。T 是此次调用的 token 数，H 是本地 Q 与 V 的头数。它们都是开发数据，生产 Python 在这里的工作量会选 V1。表中设备耗时是同一条 ACL 流中各实际设备 launch 的独立全核窗口之和，排除 launch 间隔。

| 芯片 | T | H | 角色与状态 | Prepare / FwdH / Finalize µs | 全调用 µs | CA 墙钟 s | 逐核证据 |
|---|---:|---:|---|---|---:|---:|---|
| 950DT | 2 | 1 | reused development · complete | 15.2806 / 4.7394 / 3.2206 | 23.2406 | 110.5154 | [reused-ascend950dt-t2-h1](reused-ascend950dt-t2-h1.json) |
| 950PR | 2 | 1 | new development · complete | 14.5927 / 4.7170 / 2.9036 | 22.2133 | 92.1536 | [ascend950pr-t2-h1](ascend950pr-t2-h1.json) |
| 950DT | 64 | 1 | new development · ineligible | 20.3558 / 5.2430 / 3.2067 | N/A | 106.4863 | [ascend950dt-t64-h1](ascend950dt-t64-h1.json) |
| 950DT | 65 | 1 | new development · ineligible | 19.9739 / 7.2770 / 3.4315 | N/A | 116.5285 | [ascend950dt-t65-h1](ascend950dt-t65-h1.json) |
| 950PR | 65 | 1 | new development · ineligible | 19.9982 / 7.6206 / 3.2236 | N/A | 114.2112 | [ascend950pr-t65-h1](ascend950pr-t65-h1.json) |
| 950DT | 15 | 1 | new development · ineligible | 15.8376 / 4.4473 / 2.9818 | N/A | 102.4714 | [ascend950dt-t15-h1](ascend950dt-t15-h1.json) |
| 950DT | 15 | 2 | new development · ineligible | 15.8818 / 4.6442 / 3.0145 | N/A | 112.5289 | [ascend950dt-t15-h2](ascend950dt-t15-h2.json) |
| 950PR | 15 | 2 | new development · ineligible | 15.5921 / 4.9394 / 3.1527 | N/A | 124.2662 | [ascend950pr-t15-h2](ascend950pr-t15-h2.json) |
| 950DT | 63 | 1 | new development · ineligible | 20.3170 / 5.3121 / 3.2067 | N/A | 108.4823 | [ascend950dt-t63-h1](ascend950dt-t63-h1.json) |

## 同形双端全窗

| T | H | DT µs | PR µs | DT / PR |
|---:|---:|---:|---:|---:|
| 2 | 1 | 23.2406 | 22.2133 | 1.0462 |

硬件模型分别通过 runtime SoC 与物理 DRAM 配置核对。两端分别使用 HBM3 与 LPDDR5。每条合格记录核查四组逐元素参考、初态保持、发布库与设备 ELF、所有活动核真实 END、分核与合并 trace 的物理指令/同步完成点及官方时长。旧 DT 对照的本轮新增 CA 墙钟为零；表中保留其历史采集开销。

[原数学协议的开发登记](development-register.json) · [原协议同集结果](summary.json) · [单核资源证据](resources/) · [发布精度对照、独立留出及加速比](mechanisms.md)

资源表保留逐核 MMAD 的 M/N/K 及 PC、首发与后续发起间隔，并按 DMA 源/目的地址空间拆开搬运量、描述符时延和相邻发起间隔。OUT 是 GM 指令地址空间，实际到达芯片外存的流量还受缓存影响。单个双端比值覆盖一次仿真实现的同形复放；样本间差异不能直接解释成芯片理论峰值比。
