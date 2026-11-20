# Kimi AttnResFwd 真实前序混合

## 计时与适用范围

| 数据口径 | 芯片 | 窗 | WAPE / 最大 |
|---|---|---:|---:|
| DT-phase-development | Ascend950DT_9582 | 11 | 2.84% / 5.27% |
| archived-sibling-phase-replay | Ascend950DT_9582 | 10 | 3.55% / 6.92% |
| archived-sibling-phase-replay | Ascend950PR_957b | 21 | 3.55% / 9.10% |
| 新合同冻结后独立整窗 | Ascend950DT_9582 | 0 | N/A / N/A |
| 新合同冻结后独立整窗 | Ascend950PR_957b | 0 | N/A / N/A |

前三行只比较 materialized=True 且会写 bank 的 40020 归档中两个变体共有的阶段。共同阶段去掉了旧形物化写与另一条 gamma 读门控的间隔，不等同于已测新算子总时间。各阶段训练来源为 11 个 DT 已观测窗；10 个其它 DT 窗与 21 个 PR 窗只作旧合同回放。DT 到 PR 的阶段价格按每核峰值/频率与端口换算。

| 输入输出契约 | 设备行为 | 计时口径 |
|---|---|---|
| 归档 materialize=True、write_idx>=0 | 独立写 materialized、指定 bank、prefixOut、Y | 已有旧完整 CA；上表前三行仅提取共有阶段 |
| 真实 Kimi prefill，materialize=False、write_idx=-1 | materialized 与 Y 别名，仅写 prefixOut、Y，bank 保持原值 | 下表按新合同、数据角色和芯片报告端到端时间 |

| 真实调用 [T,H,S,N] | 芯片 | 模型 µs | 合格 CA µs | 误差 % | 证据 |
|---|---|---:|---:|---:|---|
| [65, 7168, 8, 1] | Ascend950DT_9582 | 14.896 | N/A | N/A | registered; no qualifying receipt |
| [65, 7168, 8, 1] | Ascend950PR_957b | 14.994 | N/A | N/A | registered; no qualifying receipt |
| [65, 7168, 8, 1] | Ascend970DT | 8.514 | N/A | N/A | 970DT extrapolation |
| [121, 7168, 8, 2] | Ascend950DT_9582 | 17.557 | N/A | N/A | registered; no qualifying receipt |
| [121, 7168, 8, 2] | Ascend950PR_957b | 24.114 | N/A | N/A | registered; no qualifying receipt |
| [121, 7168, 8, 2] | Ascend970DT | 16.089 | N/A | N/A | 970DT extrapolation |
| [512, 7168, 8, 1] | Ascend950DT_9582 | 45.078 | N/A | N/A | registered; no qualifying receipt |
| [512, 7168, 8, 1] | Ascend950PR_957b | 55.623 | N/A | N/A | registered; no qualifying receipt |
| [512, 7168, 8, 1] | Ascend970DT | 27.478 | N/A | N/A | 970DT extrapolation |

### B8 整输入的单卡外推

| 每请求上下文 | 输入行 | DT / PR / 970DT 单次设备 µs |
|---:|---:|---:|
| 32768 | 262144 | 20609.3 / 23782.7 / 10075.5 |
| 131072 | 1048576 | 82422.7 / 95101.2 / 40283.9 |
| 262144 | 2097152 | 164840.5 / 190197.6 / 80558.8 |

这些场景假设单卡一次调度全部 token，valid=1、UB 可驻留。单解释器内 60 个不同大形、预加载后未缓存的 API 计时中位 0.056 ms，最大 0.065 ms。所有 970DT 时间均外推。

真实 Kimi text_config 有 93 层、残差块跨度 12。按框架派发枚举，上述真实 add/mix/norm、write_idx=-1、materialize=False 合同共有170个层/位置，在 950 UB 253952B 下 valid 1–3 合计 66 个位置可驻留。valid 4–8 的104个位置进入 reload 路线，公开入口仍可给出分核、UB 分支与重读量，耗时缺直接证据。模型不覆盖解码图中的所有别名和多样标志。

新采集调用固定 vLLM-Ascend 自定义库的 ACLNN AttnResFwd，发布源 c09d3e20。每个 SKU 最小跨核块边界 T=65,N=1；第二项 T=121,N=2 在 DT/PR 上分别2与3个忙核迭代。前者每端最多2小时、20GiB，后者3小时、28GiB，在同一既有串行锁逐个放行。512 形保持原有队列。成功需同时过实际 SKU、安装 ELF、全部输出与未改写 bank、供应商全核时长和唯一 END，分核/读写量另对照。没有新的128Ki或256Ki同合同设备窗口。

[冻结](freeze.json) · [阶段标定及每条原始 SHA](phase-calibration.json) · [逐形与开销](score.json) · [实时槽状态](state.json) · [安装 wheel 逐项核验](wheel-check.json) · [公式与算例](../../../docs/examples/attn_res_fwd.md)
