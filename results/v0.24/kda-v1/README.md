# Kimi Chunk KDA 默认融合 V1 的数值与同步复核

本页验证 FLA A5 26.9.1 公开入口 `aclnnChunkKdaFwd`，使用 B=1、BF16 Q/K/V 与 raw gate、FP32 beta 与非零 VK 初态、K=V=128。这里的数值来自固定的混合符号输入。真实调用在卡上分得几个头、一次调度送入多少 token、调用前 Q/K 的归一化结果，取决于系统的分片和数据。大场景自动路由见下节。

|芯片|T × 本地头数|元数据|四组数值|超差元素 O / Aqk / Akk / VK|同步错误条数|全部物理 launch 完整窗|融合主核观测窗 µs¹|可校准主核 µs|离线 CA 墙钟 / 目录采样峰值|
|---|---:|---|---|---|---:|---|---:|---:|---|
|950DT|2 × 1|varlen|全部通过|0/0/0/0|167|是（4 个表）|103.0267|N/A|386.6 s / 1.365 GiB|
|950DT|64 × 2|dense|数值超差|0/935/101/0|207|是（4 个表）|73.7794|N/A|340.3 s / 1.257 GiB|
|950PR|2 × 1|varlen|全部通过|0/0/0/0|155|是（4 个表）|97.9600|N/A|344.5 s / 0.731 GiB|
|950PR|64 × 2|dense|未触发条件|—|—|—|—|—|—|

¹ 只有捕获 ELF 的聚合 trace、全部官方逐核表、分核 shard、每核 END 和指令签名均一致时才列观测窗。该列便于调试。四组数值、完整接口、动态库和 device ELF 身份、同步检查也全部通过，方可填入可校准列。时间从首条指令分别计到同一设备 launch 的最后退休事件。墙钟包含采集；进程结束后独立 JSON 核验所花时间另计。通过 [冻结清单](register.json) 可核对选样时间、全部 SHA256 及 6 GiB / 20 min / 单并发上限。

## 为什么更换元数据

FLA 的 prefill 自动分派以本次调度的序列段计算工作项 $W=H_v\sum_i\lceil S_i/64\rceil$。正常 BF16、K=V=128 输入在 $W<4096$ 时选择融合 V1。Kimi 包装按显式 `cu_seqlens` 及 chunk indices 调用，此路径在此例的宿主 tiling 中设置 `isVarLen=1`，执行业务代码中的通用后端。T=2/H=1 使工作项降至 1。旧版 DT T=64/H=2 工作项为 2，真实主核有 96 个物理 END、26 条 MMAD 和 240 次重复 SET，之后包整体因 1 GiB 上限退出，未取得四项数值。[旧探针与来源](../../chunk-recovery/fla2691/report.md) · [固定的 vLLM 调用](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/ops/kimi_kda.py) · [FLA 发行版](https://github.com/flashserve/flash-linear-attention-npu/releases/tag/v26.9.1-release)

第二个 DT 实验保持和旧 DT T64/H2 完全相同的混合符号 tensor。B=1 时省略 `cu` 和 `indices` 与传 `[0,64]` 在数学上等价。该改动在 950 上令 `denseAligned=true`，进入 `useDenseFwdH`；偶数头还会启用专门的 `fusePostWu`。它可以检验替换后端是否会消除重复置旗。因 raw gate 是 BF16，`computeGateInPrepare` 与 `fusePostWuIntoFwdH` 均为 false。这里把缺省元数据实验标为 dense，Kimi 本身的实际调用列为 varlen。底层 MIX 仍是发行版单体设备代码。三段 V2 的通过数据另归其自身合同。[发行版 950 tiling](https://github.com/flashserve/flash-linear-attention-npu/blob/v26.9.1-release/fla/ops/ascendc/kda/chunk_kda_fwd/op_host/arch35/chunk_kda_fwd_tiling_impl.h)

## 数值与同步检查

O 是 attention 输出，Aqk 是因果 QK 得分，Akk 是单位下三角系统的逆，VK 是按 V、K 排列的最终状态。四组都做逐元素检查，同时检查初始状态没有被改写。融合 V1 版官方 ATK 使用每 16 行最后一行的累计门值作为数值平移，以 BF16 形成矩阵乘法操作数，在对应边界舍入中间张量。CPU 实现使用同一算法和冻结门限，另外输出高精度数学参考的误差。BF16 判定为 $|x-r|\le 2\times10^{-6}+2^{-6}|r|$，FP32 state 判定为 $|x-r|\le2^{-16}+2^{-10}|r|$，任一元素超差即整组失败。对应绝对硬上限是 1 和 0.01；全部输出都必须有限。[发行版 ATK](https://github.com/flashserve/flash-linear-attention-npu/blob/v26.9.1-release/tests/atk/chunk_kda_fwd/executor_chunk_kda_fwd.py) · [本次可执行计算过程](../../../development/validation/kda_v1_24/golden.py)

同步错误计数取自原始模拟器回调。每次触发按实际加载地址折回 ELF 偏移；可用时，从同核 shard 列出同一生产/消费方向、同一 flag 的连续两次 SET 和两者之间的 WAIT 检查。异常回调本身不能证明输出一定损坏，也不能证明某个内联 C++ 对象是唯一根因。该方案不依赖修改厂商内核。


**dt-t2-h1-varlen**。完成 4 个已启动的库内 launch，进程退出码 0，停止原因 自行退出。发布库符号 已验证，单体设备代码 已验证。四项与发行版参考的最大绝对误差 {'attention': 4.76837e-07, 'aqk': 0.0, 'akk': 0.0, 'state_vk': 1.09896e-07}。 主核窗口内有 32 个 AIC、64 个 AIV，合计 11 条 MMAD。实际错误出现在 42 个 ELF 偏移，反汇编匹配 42 个，其中 3 个与相同 ELF 的旧 DT T64/H2 重合；32 个 AIC 的 shard 可给出同事件两次连续 SET，中间没有 WAIT。 其中 31 个 AIC 在整个窗口没有发出 MMAD，也出现了重复置旗；矩阵计算本身的工作轮数不能单独解释这个现象。 [完整回执](dt-t2-h1-varlen.json) · [逐核事件及指令偏移](dt-t2-h1-varlen-sync.json) · [实际值、坐标及阈值](dt-t2-h1-varlen-arrays.json)

**dt-t64-h2-dense**。完成 4 个已启动的库内 launch，进程退出码 0，停止原因 自行退出。发布库符号 已验证，单体设备代码 已验证。四项与发行版参考的最大绝对误差 {'attention': 7.15256e-07, 'aqk': 0.00012207, 'akk': 0.000488281, 'state_vk': 3.17837e-06}。 主核窗口内有 32 个 AIC、64 个 AIV，合计 26 条 MMAD。实际错误出现在 34 个 ELF 偏移，反汇编匹配 34 个，其中 0 个与相同 ELF 的旧 DT T64/H2 重合；32 个 AIC 的 shard 可给出同事件两次连续 SET，中间没有 WAIT。 其中 30 个 AIC 在整个窗口没有发出 MMAD，也出现了重复置旗；矩阵计算本身的工作轮数不能单独解释这个现象。 [完整回执](dt-t64-h2-dense.json) · [逐核事件及指令偏移](dt-t64-h2-dense-sync.json) · [实际值、坐标及阈值](dt-t64-h2-dense-arrays.json)

**pr-t2-h1-varlen**。完成 4 个已启动的库内 launch，进程退出码 0，停止原因 自行退出。发布库符号 已验证，单体设备代码 已验证。四项与发行版参考的最大绝对误差 {'attention': 4.76837e-07, 'aqk': 0.0, 'akk': 0.0, 'state_vk': 1.09896e-07}。 主核窗口内有 28 个 AIC、56 个 AIV，合计 11 条 MMAD。实际错误出现在 42 个 ELF 偏移，反汇编匹配 42 个，其中 3 个与相同 ELF 的旧 DT T64/H2 重合；28 个 AIC 的 shard 可给出同事件两次连续 SET，中间没有 WAIT。 其中 27 个 AIC 在整个窗口没有发出 MMAD，也出现了重复置旗；矩阵计算本身的工作轮数不能单独解释这个现象。 [完整回执](pr-t2-h1-varlen.json) · [逐核事件及指令偏移](pr-t2-h1-varlen-sync.json) · [实际值、坐标及阈值](pr-t2-h1-varlen-arrays.json)

## 为什么双头满块的 Aqk 和 Akk 与发布参考不同

在 T64/H2 的实测数组上，冻结的 V1 ATK 有 935 个 Aqk 和 101 个 Akk 元素超过逐元素门限，各组均有 8,192 个元素。直接读取 wheel 配套的 arch35 代码，可以看到 `KDA_SAFE_SCORE_REF_BC=32`，而 `ScoreRefToken` 选择 32 行块的中间行。由于 Q/K 因子先按 BF16 舍入，不同的平移行会产生不同的得分。按该源码规则离线重算，Aqk 剩余 2 个、Akk 剩余 59 个超差。该模板的双头分支还把 beta 提前乘入 K，再舍入 BF16 后送 MMAD；加上这一条，Akk 的超差降至 0。保留的 2 个 Aqk 差异没有对应到单条指令。O 和 VK 对冻结参考均通过。以上重放仅用于定位，原始 ATK 失败与重复 SET 继续保留。[三组同数据重放、坐标与源码指纹](operand-replay.json)


## 两款芯片为什么相差 12 条错误

T2/H1 默认融合 V1 的 4 组输出在 DT 和 PR 逐字节一样。两端相同设备代码上的 42 个故障偏移完全重合，只有 3 个偏移的次数变化。`0x1abb2c`、`0x1abb38`、`0x1abb44` 在 DT 各出现 32 次，在 PR 各出现 28 次，恰好与各自参加 launch 的 AIC 数相同。三者合计解释 167 减至 155 的全部差异。DT 有 31 个 AIC、PR 有 27 个 AIC 在窗口内没有发出 MMAD，同一初始化事件仍影响这些核。这个观察把供应商排查范围缩到单体 MIX 的每核事件初始化和复用；这些重复置旗对正常芯片执行时间的影响还未知。[同 ELF 的逐偏移计数及四组原始数据 SHA256](paired-varlen.json)


## 复核与范围

本轮使用已校验的 FLA A5/aarch64 wheel、CANN 9.2.1 和两款芯片各自的 CA 内存模型。可在原登记环境检查 `development/validation/kda_v1_24/investigate.py run` 的单次采集与资格规则，已有回执可用 `inspect_sync.py`、`numerical_details.py` 复核。后者使用项目的 CANN conda Python + NumPy。主核观察时长只保留作异常样本诊断，计时合格的样本才可以进入将来的资源建模。另一个实际未测边界是变长输入且 $N_{seq}H_v>N_{AIC}$：arch35 源码会先释放外层 pipe，再进入通用后端。它有生产意义，例如 8 条序列和 12 个本地头；本轮单序列样本不能断言该分支是否有同样的数值和同步结果。

