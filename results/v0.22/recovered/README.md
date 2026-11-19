# 八个 mHC 导出中断端点的核验

这些 shape 属于 v0.21 事前冻结的独立清单。原运行已经执行了库 kernel、逐元素金标和设备关闭，随后离线报告导出触及磁盘上限。每个 shape/SKU 在误差统计中只占一次。根目录同名 JSON 是当前回执，原始失败记录按 SHA-256 保留。恢复测量没有修改原始冻结预测。

| 实际 ACLNN | `[T,N,D]` | SKU | AIC / AIV | 完整 CA 窗口（µs） | 证据 |
|---|---|---|---:|---:|---|
| `aclnnMhcPreSinkhorn` | `[56,4,4096]` | 950PR | 28 / 56 | 12.214545 | 原导出逐核与原始时钟相同 |
| `aclnnMhcPreSinkhorn` | `[64,4,4096]` | 950DT | 32 / 64 | 12.468485 | 原导出逐核与原始时钟相同 |
| `aclnnMhcPreSinkhorn` | `[64,4,4096]` | 950PR | 22 / 44 | 13.152121 | 66 核原始指令恢复，同 SKU、同 ELF 的 84 核完整对照 |
| `aclnnMhcPreSinkhorn` | `[80,4,4096]` | 950DT | 32 / 64 | 13.781818 | 原导出逐核与原始时钟相同 |
| `aclnnMhcPreSinkhorn` | `[80,4,4096]` | 950PR | 22 / 44 | 13.653333 | 原导出逐核与原始时钟相同 |
| `aclnnMhcPost` | `[191,4,2112]` | 950PR | 0 / 48 | 7.398182 | 原导出；48 核最终 `END_LABEL` 与各核末时钟一致 |
| `aclnnMhcPost` | `[253,4,2048]` | 950DT | 0 / 64 | 6.766667 | 相同合同重采；64 核均验证 ELF 入口、物理 END、最终标签 |
| `aclnnMhcPost` | `[253,4,2048]` | 950PR | 0 / 51 | 7.583030 | 同合同重采；51 核均验证 ELF 入口、物理 END、最终标签 |

## 怎样证实全核窗口

计时从所有实际活跃核中最早的入口发起时钟至最晚的指令完成时钟。所有时间戳都在同一设备时域。`CA_us = (max(last_tick) − min(first_tick)) / 1650`，1650 是 1.65 GHz 对应的每微秒时钟数。不会把不同核的 duration 相加。运行时 SKU、HBM/LPDDR 配置、同名安装包 `ops_transformer` ELF 的 `.text`、设备退出状态、任务完成、逐元素数值检查各自单独核实。

`Pre` 的原始指令日志提供每核的发起、完成和程序计数器。取每个活跃核第一条 `MOV_XD_SPR` 的 popped 时钟和最后一条实际完成记录，用 ELF 符号识别 AIC / AIV 的入口地址，并核对两份记录里物理 END 的指令字、地址与事件 ID。AIC 在物理 END 后还能有已在途的指令完成，AIV 有最终 `END_LABEL`。原始文件、线程表与 trace 的核名单必须相等。各回执保留原始文件哈希。[逐核核验](integrity/)

`Pre [64,4,4096]` 的 950PR 旧 trace 能读到 JSON 尾，却将六个 AIV 的末时钟截短了 3–108 个周期，其中五个核已丢失 END。归档中的 per-core duration 表也沿用了被截短的时长。恢复时在同 SKU / 同 ELF 的 `[56,4,4096]` 上先逐一验证全部 84 个核的原始首末时钟与完整导出相等，再读取目标形状全部 66 个核的原始文件。该形状的总窗可用于校准，旧导出的流水指令计数只代表实际导出的子集。[目标与对照逐核时钟及哈希](mhc_pre_sinkhorn_64_4_4096_Ascend950PR_957b.json)

`Post [253,4,2048]` 旧采集关闭了原始指令 dump。DT 导出缺五个核的物理 END，PR 缺一个，两端均使用同一宿主与输入重测。单个 CA 槽串行，每端上限 3600 秒和 8192 MiB，保留完成设备窗和数值结果。实际宿主向 `aclnnMhcPostGetWorkspaceSize` 传入 `x, res, h, post` 四个输入张量及一个数值输出。登记文字中的“two named inputs”有误，可执行文件的 SHA、参数和冻结的调用合同共同确定实际测试内容。[重测登记](retry/registration.json) · [队列状态](retry/status.json)

8 个端点全部取得完整设备窗。6 个端点从归档中恢复，新增 CA 为 0 秒。`Post [253,4,2048]` 两端串行重测分别用了 308.39 秒（DT）和 298.13 秒（PR），总计 606.53 秒，两次输出各覆盖 2,072,576 个元素且 `bad=0`。

本机原始采集保留在主 checkout 的 `runs/fusion21`、`runs/dispatch21`。本轮新采集在 `build/v022-checkout/build/mhc-window-recovery`，主 checkout 通过 `build/mhc-window-recovery` 相对符号链接访问同一份原始文件。回执的 `trace`、原始回执和安装包 ELF 路径均可在主 checkout 解析。仓库中保存哈希与核证结果；原始大文件由本机采集目录保存。
