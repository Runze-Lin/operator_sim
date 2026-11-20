# 融合与路由算子的大形验证

| 算子 / 形状 | 真实部署几何、替代实现或库合同 | 950DT 模型 / 合格 CA µs | 950PR 模型 / 合格 CA µs | 970DT 外推 µs | DT / PR 取证 |
|---|---|---:|---:|---:|---|
| mhc_pre [512, 4, 5120] | Flash 实际 D5120，单个 512-token chunk；内置三段替代链 | 2512.514 / N/A | 2982.251 / N/A | 909.174 | pending / pending |
| mhc_sinkhorn [512, 4] | Flash 实际 D5120，单个 512-token chunk；内置三段替代链 | N/A / N/A | N/A / N/A | N/A | pending / pending |
| mhc_post [512, 4, 5120] | Flash 实际 D5120，单个 512-token chunk；内置三段替代链 | 29.610 / N/A | 36.854 / N/A | 17.814 | pending / pending |
| mhc [512, 4, 5120] | Flash 实际 D5120，单个 512-token chunk；内置三段替代链 | 3205.144 / N/A | 3865.850 / N/A | 1316.935 | pending / pending |
| vllm_hc_post [1, 512, 4, 5120] | 实际 Flash 宽度、512-token chunk 及 vLLM HcPost 实现 | 24.355 / N/A | 28.967 / N/A | 14.653 | pending / pending |
| lightning_indexer [1, 9, 32768, 8, 128] | Q9、KV32768、G8、D128 的密集 BF16 索引库合同，top-512 | 110.653 / N/A | 114.075 / N/A | 100.686 | pending / pending |
| mhc_pre_sinkhorn [512, 4, 4096] | 库合同 D4096、I20，T≥128 分组 | 43.892 / N/A | 54.095 / N/A | 34.296 | pending / pending |
| mhc_sinkhorn [4096, 4] | 8 请求 × 512 token，FP32 N4/I20 单调度 | N/A / N/A | N/A / N/A | N/A | pending / pending |
| mhc_pre [4096, 4, 5120] | 8 请求 × 512 token，实际 Flash 宽度、内置替代链 | 20100.112 / N/A | 23219.433 / N/A | 6767.275 | pending / pending |
| mhc_sinkhorn [4096, 4] | 8 请求 × 512 token，实际 Flash 宽度、内置替代链 | N/A / N/A | N/A / N/A | N/A | pending / pending |
| mhc_post [4096, 4, 5120] | 8 请求 × 512 token，实际 Flash 宽度、内置替代链 | 219.229 / N/A | 256.563 / N/A | 110.884 | pending / pending |
| mhc [4096, 4, 5120] | 8 请求 × 512 token，实际 Flash 宽度、内置替代链 | 25641.154 / N/A | 29774.539 / N/A | 9655.858 | pending / pending |
| mhc_sinkhorn [262144, 4] | 8 请求 × 32768 token，FP32 N4/I20 单调度 | N/A / N/A | N/A / N/A | N/A | pending / pending |

## 旧证据的尺度

| 算子 | 本次登记前 950DT 的最大张量 shape / CA µs | 950PR 的最大张量 shape / CA µs |
|---|---|---|
| mhc | [24, 4, 1536] / 21.986 | [9, 4, 1024] / 19.199 |
| mhc_pre | [24, 4, 1536] / 12.201 | [9, 4, 1024] / 9.865 |
| mhc_sinkhorn | [4097, 4] / 8.964 | [4097, 4] / 9.112 |
| mhc_post | [253, 4, 2048] / 6.767 | [253, 4, 2048] / 7.583 |
| mhc_pre_sinkhorn | [80, 4, 4096] / 13.782 | [80, 4, 4096] / 13.653 |
| vllm_hc_post | [1, 33, 4, 7168] / 6.101 | [1, 33, 4, 7168] / 6.508 |
| lightning_indexer | [1, 9, 4096, 8, 128] / 15.334 | [1, 8, 1536, 8, 128] / 8.829 |

mHC 以每 rank 的实际 token 数 T 为行数。512 行、D5120 时，vLLM HcPost 的 950DT 64 个 AIV 最多每核 8 行，PR 56 个 AIV 最多 10 行；每行再按 4096+1024 通道分片。4096 行对应 64/74 行每 AIV。旧归档中的最大输入或最多 token 均在冻结记录逐项列出，单个完整窗不代表已覆盖其它实现。内置链只做一次真实三段派发，以各段独立核窗之和给出链端时间，排除 host 中间间隔，链中 Post 的前序来自同一真实流。

仅 shape 的 FP32 Sinkhorn 对大 T 可能没有时间。独立冷启动可另查显式 resident_aiv=0；4096 行和完整 262144 行均已将两种调用的可用性保存在逐端记录中。完整大形如果跨多个 UB 分块而没有发起价格，模型时间写 N/A。Q9/KV32Ki 索引测试的输入、所有输出 top-512 集合和整芯片核窗可核对。Flash 上游的压缩页与过滤 top-k，以及 Pro 的 top-1024，均需要独立运行时证据。

宿主新增实验只调整 smoke 输入大小限制，并沿用同一输出构造和数值 golden。所有 950 端点需要全量预定数值组、准确编译目标及 DRAM 配置、对应安装 ELF 的 .text、每核物理 END、每个核与官方 duration 表的完整窗口。整行或整链的 CA 只有全部合格后进入表格。队列与其他大实验共用单槽，每端点完成采集即释放，解析期间其它队列可以采集。每端预算 1.5 到 6 小时、16 到 96 GiB raw，状态与达预算原因单列。

B8 × 32Ki、N4、D5120 的全张量有 5,368,709,120 元素，超出原宿主 golden 的有符号 32 位索引。表内明确分开了完整 token、chunk 和替代合同。970DT 全域仅填模型外推。

[冻结三卡价格、各端口径、预算和来源](freeze.json) · [逐形状态、误差和真实回执](score.json) · [实时队列](state.json)
