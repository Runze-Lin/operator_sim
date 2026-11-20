# Kimi-K3 prefill 的真实 FLA 分派与设备成本

同样 8 个 32k prompt、attention TP=8，一次交满时走 V2，每条本次只交 2,048 token 时走 V1。系统仿真可将本步实际交给 `run_chunk_kda` 的各请求 token 数传给 `plan_kimi_kda_prefill`。模型配置的 96 个 KDA 头在 TP8 下每卡 12 头，Q/K/V 每头宽 128，gate 下界 −5。这里的数值覆盖一层、一个 rank 和一次 prefill 调用。模型的另 24 层 MLA、decode 和投机校验有各自的执行路径。[固定模型配置](../../../configs/models/Kimi-K3.json) · [vLLM Ascend Kimi prefill 与状态更新](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/ops/kimi_kda.py#L472-L524)

|本次单卡 B8，每条提交的 token|工作项|官方路由|最小张量载荷 GiB¹|950DT 三主核 ms|950PR 三主核 ms|970DT 三主核 ms|
|---|---:|---|---:|---:|---:|---:|
|512|768|V1|0.08|N/A|N/A|N/A|
|2,048|3,072|V1|0.29|N/A|N/A|N/A|
|4,096|6,144|V2|1.14|4.157|5.068|1.963|
|32,768|49,152|V2|9.02|33.161|39.396|16.450|
|131,072|196,608|V2|36.06|132.600|157.121|65.403|
|262,144|393,216|V2|72.11|265.186|314.017|130.614|
|524,288|786,432|V2|144.20|超容量|超容量|261.036|

表内所有 ms **都是资源模型外推**。950 两端有较小直接 V2 的逐核有效 CA，其中一组可以逐时间块查看 PR 的 `FwdH` 模型和 CA 对照。950 端超出已有样本的分块次数、多链共核与 Finalize 的 AIV 搬运分支仍需验证；表中的 970DT 额外采用同一 950 host 分派与实现假设。层完整耗时取 `full_dispatch_us`，非空场景当前均为 `None`；仿真器可在 `main_kernels_us` 上叠加自己已有的前后处理成本。首次与重复调用合计 399 次，该机本次最高查询墙钟约 4.2 ms，逐场景中位数与最大值见 JSON。[各核阶段和查询成本](dispatch.json) · [PR FwdH 三时间块，模型及原始 CA](../../v0.23/kda-multiblock/) · [十形双端三主核直接采集](../../kda-v23/mechanisms.md)

## 系统输入、选路与费用

```python
from operator_model.kimi_kda import plan_kimi_kda_prefill

p = plan_kimi_kda_prefill([32768] * 8, attention_tp=8,
                          initialized_sequences=3, chip="Ascend950PR_957b")
api = p["aclnn"]                 # aclnnChunkKdaFwdV2
main_us = p["main_kernels_us"]  # 39396.01 µs，三个串行主核
stage_us = p["main_stages_us"]
state_reads = p["graph_io"]["state_gather_read_bytes_range"]  # (2359296, 2359296)
```

令每条本步有效长度为 $s_i$。vLLM 会压实空序列，生成固定长度 64 的规范块号。各条独立取整得到 $C=\sum_i\lceil s_i/64\rceil$。本卡 head 数 $H=96/TP$，FLA 的工作项是 $W=CH$。Kimi 已在 FLA 外分别做 Q/K 的 L2Norm，且 beta 已预处理；采用 BF16、K=V=128、默认精度开关与显式严格递增的累计长度时，$W\ge4096$ 走 V2，否则走 V1。此阈值属于 **26.9.1 包的分派策略**，其注释把阈值选取归于 A2 实验。每个 Kimi future-chip 算例沿用相同软件策略。遇到不均匀的请求，工具仍给出正确路由、块数与逻辑费用；原冻结资源模型仅在均匀的非空长度上给出三主核估时。比如 `[21824]` 与 `[1,21823]` 的 token 总量相同，TP8 下分别得到 W=4092/V1 和 W=4104/V2。[公开 stable dispatcher](https://github.com/flashserve/flash-linear-attention-npu/blob/5cc699069c7f91a35aabd583032c8720a0b626fa/torch_custom/fla_npu/csrc/src/stable_chunk_kda_fwd.cpp#L165-L198) · [Kimi 的公开调用参数](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/ops/kda.py#L60-L101)

在均匀 B8、TP8 下，首次到达 V2 的每请求长度是 **2,689 token**。2,688 恰好为 42 个满块，$8\times12\times42=4032$；多一个 token 后每条都多一个尾块，工作项为 4,128。若调度参数给的是全卡 `max_num_batched_tokens`，先把本步分摊到各请求后再算。例如每条各 2,048 token 时，attention TP=1/2/4 分别有 24,576/12,288/6,144 个工作项，均走 V2；TP=8/16/32 时有 3,072/1,536/768 个工作项，均走 V1。部署调整 TP 会改变 kernel 身份，即使每请求长度相同。

¹ 令 $T=\sum_i s_i$，$B$ 为非空请求数，$D=128$，$X=THD$ 为单个 BF16 Q 张量的元素数，$P=4HD^2$ 为一条请求的 FP32 VK 状态字节数。公开接口的输入输出同时存在的载荷至少为 $L=10X+260TH+4H(1+D)+2BP$ 字节。第一项来自 Q/K/V/raw gate/输出 O 五张 BF16 张量，第二项来自 beta 与两个 64 列 BF16 辅助输出，另含 A_log、dt_bias、初态与终态。V1 栏只计这部分。V2 在 Prepare 与 FwdH 之间存在 FP32 Gk 及四张 BF16 中间张量，共 $12X$ 字节；FwdH 与 Finalize 间至少存在 QgScaled、vNew、每个块头的 BF16 状态，共 $4X+2W D^2$ 字节。故两道边界中至少一道要求 $L+\max(12X,4X+2WD^2)$ 字节。这个下界用于排除不可能的 dispatch。可用 HBM 还取决于权重、cache、分配器、格式对齐和额外 workspace。此处沿用 Kimi 的 rank-4 BSND、独立输入输出缓冲与规范连续辅助输出；物理 HBM 流量和 workspace 峰值要另测。[发行版 V2 host 的中间张量、形状和 dtype](https://github.com/flashserve/flash-linear-attention-npu/blob/5cc699069c7f91a35aabd583032c8720a0b626fa/fla/ops/ascendc/kda/chunk_kda_fwd/op_host/op_api/chunk_kda_fwd_v2.cpp#L176-L271)

两次外部 Q/K L2Norm 的逻辑读和写分别是 $4X$ 字节。有效初态标志数为 $B_0$ 时，gather 从 cache 读取 $B_0P$，写出 $BP$；scatter 读取整份终态、写回有效 cache 行，各为 $BP$。不知道 $B_0$ 时输入可省略，返回从零至 $BP$ 的读量范围。这里假定有效 cache 索引和 950 FP32 dense VK 状态；metadata、重复读、临时 contiguous 拷贝和 pipeline 等待另计。四个简单字节量可让系统侧接已有内存模型。[Kimi 950 状态拷贝实现](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/ops/triton/kda_state_copy_kernel.py#L94-L119)

定向回放从原 A5 wheel 提取三个 metadata 函数与 `4096` 常量。dtype 只需要身份标记，未加载 torch/NPU kernel。19 组独立分派输入覆盖 W=4095/4096、空槽压实、不等长、不同 TP 及大 batch，用同一组逐个对照原包函数的块数和选择。它验证宿主元数据判断，端到端模型图的实际执行与设备时间仍需实测。[可复核选路脚本](../../../development/validation/kda27/dispatch_audit.py) · [19 组结果及原始函数 SHA](dispatch.json)

## 默认 V1 的正式四段同步实验

重新读取 FLA 官方 GitHub release 清单，A5/aarch64 的三个现存包的 sha256 和前期逐 ELF 核验均一致。已经实测的单序列 T2 控制走一份 FULL 主核，DT/PR 四组数值均过，但出现双置旗。此次查到了相同公开 V1 ABI 的另外一个实际分支。在 950 上显式传 `cu_seqlens`、且两条序列各自形成一块时，L2 会用 phase 0、1、2、3 分别发起同族 `ChunkKdaFwd`。上游在此处明确依靠物理 launch 重置事件。V2 另有不同的入口与三个不同的设备核。[官方发布清单与 SHA 复核](release-check.json) · [供应商 950 host 代码](https://github.com/flashserve/flash-linear-attention-npu/blob/5cc699069c7f91a35aabd583032c8720a0b626fa/fla/ops/ascendc/kda/chunk_kda_fwd/op_host/op_api/aclnn_chunk_kda_fwd.cpp#L879-L898) · [上一单序列双端同合同证据](../../v0.24/kda-v1/paired-varlen.json)

本轮将同一 `[1,2,1,1,128,128]` 输入描述符的两行分给两条独立序列，`cu=[0,1,2]`，规范下标 `[0,0,1,0]`。两条共享 Q/K，第二条取相反的 V 与非零初始 VK。按发布的混合精度 ATK，输出 O/VK 应反号，短块 Aqk/Akk 相同，从而能发现未重置的跨序列依赖。核验每组原始数组、初态、FLA 载入库、四个 V1 ELF、全辅助 launch、官方全核 roster、物理 END、合并 B/E 指令窗口及错误回调。DT 在独立 8-CPU 单槽执行，采集限 8 GiB/60 分钟。仅在 DT 全部合格后同样采 PR。模型的设备成本取决于每次实际提交的 token。[采前参数与输入/参考 SHA](v1-register.json) · [逐卡原生结果](v1-outcome.md) · [状态](v1-state.json) · [原生 host](../../../development/validation/kda27/packed_v1.cpp)

另做过一次只用无存储 `aclTensor` 描述符的原生 V2 `GetWorkspaceSize` 控制。两 token 的宿主调用返回 561103、无 executor，runtime 记录 107002（当前线程缺少 context），因此未扩大到大 shape；独立尝试显式初始化原生模拟器在 90 s 内没有取到 workspace 回执。表里的内存下界来自发行版张量生命周期，`workspace_bytes=0` 在该失败状态下没有测量含义。真实数据缓冲与完整 CA 的 V2 小形另有前述历史独立样本。[宿主限制及库指纹](workspace-query.json)

离线宿主调度另有一次采中附录。原 worker 的 103 个线程在 8 个逻辑 CPU 上运行，约 769% CPU；第一个 248 周期的启动任务耗费 117.33 s，历史采集中同长启动任务约 12.51 s。为让 Pro 256Ki PR 优先入槽，只在它实际运行且能够观测到其工作线程、或它已结束之后，才考虑将同一 PID 的 CA worker 扩展到最多 24 个 CPU。额外 16 个 CPU 要连续四次采样低于 35% 利用率、避开 Pro 活跃线程的物理同核兄弟，并保有整机余量；一旦后续采到冲突就回到原 8 个 CPU。设备 ELF、输入、完整窗要求和 60 min 截止保持采前合同。记录会给出是否真的执行、哪些物理核、对应进程起始 tick 与源 ELF 哈希；这是影响墙钟的宿主调度条件。[宿主附录与当前状态](host-scheduling.json)

原 DT 探针在 3,600.95 s 墙钟限额退出。首个辅助 Transpose 的已验证加载、AIV0 的 block start 与 kick 先后出现，设备导航文件内只有该任务的 BEGIN。从头至尾没有一份 V1 主核发起或首末窗。950PR 未执行。现已用相同的单次公开调用、四组金标和正式设备计时协议登记 DT 宿主对照，原失败单独保留。对照在长任务释放资源后使用实测空闲物理核，给其他作业预留至少 96 个空闲核；墙钟上限 24 分钟。最新的原始窗口、同步错误、实际输出及资格由对照表自动更新。[V1 逐项宿主对照](host-control/report.md)

Kimi B8×32ki 的另一项独立实验已经完成。它实际调用 `aclnnDynamicMxQuant`，`[262144,7168]` 的 1,879,048,192 个量化值和 58,720,256 个 scale 全部达标，64 个 AIV 的原生 wrapper 起止完整，包络为 **1,778.988 µs**。墙钟 4.888 h，输出 6.782 GiB。该低记录量实验未导出标准逐指令窗口，950PR 未采同形。这一输入量化实验有独立的采前冻结与小形口径控制；其数值、包络和耗时均只归于量化。[原生回执](../../v0.20/full-quant/native.json) · [采前登记与短控](../../v0.20/full-quant/capture-registration.json)

DT 宿主对照已完成，墙钟 **331.9 s**，目录峰值 **1.677 GiB**。四组原始数值逐项通过，全部 7 次真实发起都取得官方全核窗，4 次 V1 主核各有 32 AIC + 64 AIV。官方融合 `.text` 与此前单序列控制一致，但 phase 0（Gate/Prepare）与 phase 3（Finalize）各报告 114、18 次重复 SET，所有故障 PC 均映射到 `SET_FLAG.M.MTE1`。在两个异常阶段内，都能逐核证实 2 个 AIC 的三个事件各自在第一个匹配 WAIT 之前已有两次 SET。这个最小输入区分了两件事：原 8 CPU 时没有到达目标核；在充足宿主资源下正式四段均执行完毕，物理 launch 的间隔仍无法修复各段内部的同步违规。**本分支的 DT 标定耗时为 N/A，PR 没有运行。** [逐阶段窗口与数值](host-control/report.md) · [逐 PC 机器码及首 WAIT 前二次置旗](host-control/fault-sites.json)
