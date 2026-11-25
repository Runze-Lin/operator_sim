# 950DT 系统预设及真实调用输入

本次三种 Vector 策略分别在 183 张真实前向图上通过全部 122,383 个特征签名的 950DT 系统报价检查，覆盖 119 个原始调用名、766,282 次节点调用。其中 321,396 次为合法的结构操作或空工作量。116 个原始调用至少有十种不同的物理 shape；其余三个均来自 Block Diffusion 固定 16 维注意力头的 RoPE，十二组任务中对应张量总是 FP32 [8]。

本目录保留用户列出的 12 个模型预设，另收录同一系统注册的纯文本 Block Diffusion，后者采用源码自带 `source_default`。模型注册名相同、预设不同的 V4 Flash / Pro、Qwen3.5 27B / 397B 分别构图。数据来自 SIMPlatform preview 的 meta 张量前向与图适配，权重、激活值和专家实际路由均未从生产服务采样。

| 文件 | 内容与单位 |
|---|---|
| [manifest.json](manifest.json) | 工具版本、输入哈希、预设数、任务数、节点数、特征数和原始调用名 |
| [model-configs.json.gz](model-configs.json.gz) | 每个生产者预设解析后的模型配置，包含真实的层数、通道和注意力头等维度 |
| [cases.json.gz](cases.json.gz) | 各份任务的输入条件、可复用 TOML 和各阶段节点数。每个阶段记录特征 SHA-256 及出现次数。TOML 中的 950DT.jsonc 可替换为系统配置的实际路径 |
| [features.json.gz](features.json.gz) | 以 SHA-256 为键的完整 OpFeatures，包括实际输入输出、归一化名称、原始调用名、精度属性、逻辑/物理 shape、布局和集合通信事实 |

每个具名自回归预设各有以下 10 个共同前向任务。下列序列长度指图输入的历史位置与本轮处理 token 数，两者相加为本次任务预留的 KV 容量。验证阶段为主模型 Q=5 或 9 的单次校验，不含单独的草稿模型图。B 表示 attention DP 副本的 batch。TP=8，稠密 Qwen3.5 27B 使用 DP=1，其他自回归预设采用 DP=4 并开启专家并行。捕获选择 exact rank 0，未验证其它 rank 的负载倾斜、实际整网显存和设备端端到端耗时。

| 阶段 | 历史 token | 本轮 Q | B | 请求的精度配方 |
|---|---:|---:|---:|---|
| decode | 32768 | 1 | 1 | 混合 |
| prefill | 28672 | 4096 | 4 | 混合 |
| decode | 131056 | 1 | 16 | BF16 |
| prefill | 0 | 512 | 1 | BF16 |
| verify | 32768 | 9 | 1 | 混合 |
| verify | 65536 | 5 | 4 | FP8 |
| decode | 65536 | 1 | 32 | FP8 |
| decode | L−1024 | 1 | 4 | FP4 |
| prefill | L−2048 | 2048 | 8 | FP4 |
| prefill | 130048 | 1024 | 16 | FP8 |

L 对 V3、V3.2 取 131072，对 Qwen 取 262144，对 V4、Kimi、GLM 取 1048576。混合配方请求 attention 投影 A8W4、attention compute FP8、路由及共享专家 FP4、Vector BF16。FP8 和 FP4 配方分别请求对应的投影权重/激活、attention compute 和专家精度，Vector 仍采用 BF16。数据表中的精度是调用方请求，各算子被生产者实际 lower 后仍需检查对应签名的 dtype、打包方式和 attributes。同一算子还会出现整型索引、FP32 归约等类型。

Block Diffusion 的源预设采用 FP32、TP1、无专家并行。分别用 prefill (B,Q)=(1,128)/(4,512)，denoise 和 commit 各用 (B,Q,历史)=(1,8,512)/(4,16,2048)。该预设缺少官方 checkpoint，设备执行和精度覆盖不能由这六张图推定。

在共同任务外，再按低频算子补采 57 张图。DeepSeek V3 与 Kimi K3 的 BF16 任务包含转置 BMM 和稠密 attention V2；Qwen3.5 27B 追加不同 B 的 decode、Q=3～17 的 verify 、不齐整的 prefill 长度，并在没有 KV 历史时以七种额外 batch 检查零状态初始化；GLM、V4.1 和其余 Kimi/Qwen 测试补充索引、卷积、注意力或递归路径。Block Diffusion 另取 31 至 2048 个输入 token。逐模型 shape 数以实际张量尺寸去重。[精度与 shape 索引](shape-index.md)

复现需要与清单一致的系统和推理模型包。下例在本机发布工作区运行。先设置 CANN 环境，再调用安装了模拟前端的 Python。第二条命令核对预设元数据，只需 CPU。

```bash
source toolchains/cann/9.2.1/ascend-toolkit/set_env.sh
SIM=system-sim-workspace/preview-0.1.0-preview.1
PYTHONPATH=src:.:$PYTHONPATH "$SIM/.venv-runtime/bin/python" -m calibration.examples.catalog_capture /tmp/catalog-v24
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:. "$SIM/.venv-runtime/bin/python" -m calibration.examples.catalog_capture /tmp/catalog-v24 --publish results/v2.4/catalog
PYTHONPATH=src:. "$SIM/.venv-runtime/bin/python" -m calibration.examples.catalog_shapes
PYTHONPATH=src:. "$SIM/.venv-runtime/bin/python" -m calibration.examples.catalog_replay /tmp/catalog-prices --data results/v2.4/catalog
```

## 如何读报价

[默认策略逐任务覆盖](prices/tasks.csv)和[逐算子覆盖](prices/operators.csv)展示实际捕获的调用次数及相同特征合并后的签名数。`serial_us` 是逐节点设备预测微秒数乘以出现次数后相加，系统 DES 还要按依赖、并行和通信重叠重新调度。`median_query_us` 是本次离线并发回放逐签名的单次主机查询墙钟。硬件资源来自系统 950DT 配置；即使某个库合同有 CA 锚点，也不能把这里的系统报价作为实测。

[默认报价明细](prices/quotes.json.gz)的 fields 给出各列顺序；可按特征哈希索引。A5 候选 Indexer 的键另附冒号与 PA 表页数（按本任务 KV 容量计算）。Cube/Vector 工作按 dtype 以 op 计，读写量以逻辑 B 计。每个非结构、非通信节点都检查了精度对齐的计算与冷读写峰值下界；所有非结构零耗时节点均检查了空张量边界及零计算/零逻辑读写。

[默认策略核验](prices/audit.json) · [原法核验](calibrated/audit.json) · [指令类别法核验](instruction_classes/audit.json)。核验文件包含同一冻结源码摘要、输入哈希及实际成功/失败签名数。两种显式模式各自也有逐任务、逐算子汇总。本次回放检查方法的可调用性与资源自洽性，独立计时误差需使用完整 CA 窗口。

[超过 100 ms 的并发查询复核](prices/slow-query-recheck.csv)对本轮默认策略中超过预算的签名单进程连续重复五次，记录每个签名的中位数、最大墙钟，并检查设备报价可复现。可用 catalog_replay 的 `--recheck-slow` 参数重算。

三种策略同时运行、每种使用 12 个工作进程，含载入和导出账本的独立墙钟分别为默认 55.64 秒、原法 52.64 秒、指令法 60.91 秒。默认策略查询中位数 0.226 ms、P95 2.417 ms、P99 21.200 ms，最大 265.545 ms；80 个并发慢样本的单进程复测中，各签名中位数最大 10.327 ms，所有单次最大 16.714 ms。在线服务若并发争用主机仍可能超过 100 ms。

[119 个调用名的三策略并列表](method-comparison.csv)按相同图、shape 和出现次数加总各模式的设备报价，单位为 µs。末列是指令类别法相对原法的百分比变化。原法合计为零时留空。各行合计包含跨图、跨层的多次调用，不宜当成单次调用或系统排程后的延迟。
