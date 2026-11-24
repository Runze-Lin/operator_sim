# 精度对齐后，精细模型和 Roofline 差多少？

27 组 950DT 场景里，精细模型有 **8 组**比历史 Roofline 快。让两边执行同一份分精度工作、使用相同芯片峰值、通信和调度后，**0 组**比按 100% 峰值计算的逻辑 Roofline 快。单算子仍有 **42 次 BF16→FP32 Cast** 低于逻辑流量时间，需要同窗的片外流量来区分缓存、写回边界和模型误差。

本轮还修正了精细模型的一个精度问题。v1.9.0 的 A8W4 投影为权重展开和 FP32 重缩放统一调用 16 位 Vector 峰值；v1.9.1 按两类服务分别取 8 位和 32 位峰值。27 组总时间从 3992.324 变为 3992.261 ms，相差 0.063 ms。旧版比历史 Roofline 快的主要原因由下面的同图重放定位。[分精度排程公式和算例](examples/system-cube-routes.md)

## 端到端对照

每行是同一模型 3 种并行 × 每卡 batch 1、16、64 的 **9 个单轮时长之和**，单位 ms。每轮含 target 与 draft，128 Ki 历史、推测 8、接受长度 7，捕获 rank 0。原图和任务的哈希、每个节点的 ID、原始名、输入输出 shape 和存储 dtype 均逐对核验。四列均来自系统离散事件调度（DES），沿用同一组节点依赖和资源令牌。精细模型还在节点内部估计分核、流水和尾波。

| 模型 | 历史 Roofline | 同精度、峰值 100% | 同精度、原效率 | 精细模型 |
|---|---:|---:|---:|---:|
| DeepSeek-V4 Pro | 859.311 | 555.189 | 747.250 | 963.157 |
| DeepSeek-V4.1 Flash | 2216.538 | 556.239 | 722.766 | 997.344 |
| Kimi K3.1 Pro | 1739.003 | 1352.930 | 1749.884 | 2031.760 |
| **27 组** | **4814.851** | **2464.358** | **3219.899** | **3992.261** |
| 其中满足 144 GB 容量的 19 组 | 2597.658 | 1149.642 | 1513.536 | 2059.385 |

“同精度”两列对 95,689 个本地计算节点用精细模型的有用工作量、执行引擎、各引擎的精度和逻辑字节重新计算 Roofline。另有 7,326 个通信节点沿用相同的精细通信价格，68,217 个元数据节点沿用零设备时间。所有本地计算节点均可计算这两列。按原效率比较，只有 Kimi DEP64、B=64 的精细模型更快（607.202 对 620.759 ms，同精度峰值为 462.642 ms），该配置的静态显存峰值为 256.24 GB。27 组均无整机实测，汇总表示预测方法之间的差距。[逐配置与反事实 DES](../calibration/reference/roofline-audit-cases.csv) · [原始配置结果和容量](system-full-comparison.md)

### 单独换回一个算子的计时

保持精细模型的其他节点、通信、资源令牌和依赖，只将指定名称的每次调用耗时替换为同节点的历史值，再调度全图。表内数字为 ms。

| 比较对象 | 历史 Roofline | 精细模型 | 只换回候选索引器 | 只换回 GroupedMatmul |
|---|---:|---:|---:|---:|
| 全部 27 组之和 | 4814.851 | 3992.261 | **5394.750** | 4389.018 |
| Flash TP8/EP32，B=64 | 555.703 | 223.615 | **596.038** | 225.207 |
| V4 Pro DEP64，B=64 | 301.436 | 251.239 | 251.239 | **336.585** |
| Kimi DEP64，B=64 | 680.007 | 607.202 | 607.202 | **750.954** |

Flash 的 54 次候选索引器调用合计预测共缩短 **1402.489 ms**（相对历史方案）。V4 Pro 和 Kimi 的两个 DEP64 大 batch 情景则在换回 GroupedMatmul 后翻转。也有反方向的项，例如 17,748 次动态 MX 量化的精细计时合计增加 557.069 ms。把 27 组所有节点各自相加，原方案为 4814.851 ms，精细模型为 4061.058 ms；重新调度使后者再减少 68.797 ms。由此，净差 822.590 ms 中有 753.793 ms 来自节点价格，68.797 ms 来自两次调度的重叠差。端到端反事实与逐节点差额分别保存。[80 个原始调用名账单](../calibration/reference/roofline-audit-operators.csv)

## 同一精度怎么定义

1. **存储**决定字节数。FP4 两个数占一字节，E8M0 的块 scale 和 FP32 的 token scale 各自按存储格式计数。最后写 BF16 或 FP32 也会改变输出流量。
2. **执行**决定计算峰值。设某精度上必要的有效矩阵操作数为 W，峰值为 P TOP/s，则计算项为 `W / (P × 10⁶)` µs，一次乘加计两次操作。对 BF16×BF16→FP32 的 MMAD，乘加按 BF16 Cube 计；另行执行的 FP32 归约或缩放进入 Vector。FP32 累加器本身不把整个乘法变成 FP32 Vector 运算。低精度 attention 还需说明 P×V 前概率以什么精度存放和参与乘法。
3. **根据所选实现分项。** 对同一引擎累计各精度时间，再对 Cube、Vector 和 `逻辑读写字节 / 带宽` 取最大。100% 列使用任务配置的峰值，原效率列分别乘以 0.6、0.6 和 0.8 作为有效吞吐。该任务 Cube 4/8/16/32 位为 2000/1000/500/125 TOP/s，Vector 8/16/32 位为 62/31/16 TOP/s，HBM 为 4 TB/s，固定首尾为 0。独立 CA 对照使用各自的 SKU 配置，如原生 DT BF16 Cube 为 432 TOP/s。两套配置分开核算。
4. **证据决定能给数字的含义。** 精细排程可计入补齐的 MMAD、权重展开、页间同步和尾波；Roofline 使用同合同的有用工作账本。等价 Vector 操作如 exp、比较、转换的服务时间取决于具体指令，用同位宽 FMA 峰值折算只能做诊断。硬件必需的字节是本窗口实际穿过 HBM 的字节。只有工作、精度、资源、字节层级和计时边界都匹配，超过 100% 才构成物理矛盾。新列沿用精细模型声明的计算路径。25,101 个有正 Cube 工作的节点中，18,504 个捕获了累加精度，另有 6,597 个在上游此字段为空，按已选合同或实现假设计价。缺少融合指令证据的路径仍属外推。[按调用和精度分组的完整账单](../calibration/reference/roofline-audit-precision.csv) · [实现选择及 CA 身份](system-full-evidence.md)

### Flash 候选索引器，为什么一个节点相差 27 倍

TP8/EP32、B=64 的 target 节点 2588 接到 BF16 Q[576,32,128]、K[64,131081,128]、BF16 权重[576,32]、int64 位置[576]、int32 候选块[576,2048]。两条路径都给全部 key 打分，每个请求校验 9 行，没有用候选优先剪枝取得加速。

| 同一节点 | 计算方法 | 时间 µs |
|---|---|---:|
| 历史 Roofline | 将 618,518,937,600 次等价工作全部放在 FP32 Vector，除以 16 TOP/s × 0.6 | 64,429.056 |
| 只将历史 Vector 效率提到 100% | 同一分子除以 16 TOP/s | 38,657.434 |
| 对齐实现的峰值 Roofline | 点积 `2×64×9×131081×32×128` 除以 500 TOP/s 得 1,237.036；ReLU、权重和头间归约等价 448.297；逻辑读写 539.572；取最大 | **1,237.036** |
| 对齐实现、用原效率 | 以上三项分别除以 0.6、0.6、0.8 并取最大 | 2,061.726 |
| 精细模型 | 单请求按 4+4+1 拆查询，计入 Cube 补齐、扫页、AIV 归约和筛选、尾部排空 | **2,359.845** |

这个节点的有用 BF16 点积吞吐为整片场景峰值的 52.42%。区别主要来自点积派给哪个引擎。相同形状的 4 个 target 调用之外，单轮还含 draft，端到端表已分别重放。完整融合实现缺少 128 Ki 同窗 CA；小形上有真实的分解实现，例如 DT 的 800 个 key 选 600 候选时，13 个库 kernel 合计 37.057 µs，假设的融合实现预测 6.094 µs。该证据核对算法和组件，同时保留融合排程可能偏乐观的风险。[算法、流水和 CA](examples/candidate-filtered-indexer.md)

### 同名 GroupedMatmul 的两个精度合同

| 27 组内的调用 | 历史乘加精度 | 对齐后乘加与辅助工作 | 调用次数 |
|---|---|---|---:|
| 专家 A4W4，双侧 E8M0，BF16 输出 | BF16 Cube | 原生 MXFP4 × MXFP4；库的 GroupedMatmulV5 有双端代表 CA 窗 | 3,582 |
| 分组投影 A8W4，token FP32 scale，BF16 输出 | BF16 Cube | 权重从 packed W4 展开，按 FP8 Cube 乘加、FP32 累加，再做 token 重缩放。这个布局按设计排程 | 963 |

例如同一 Flash target 节点 67，A 为 FP8[576,4096]，逻辑权重 W4 的 K=4096、N=1024。两边矩阵操作数都是 4,831,838,208。历史 BF16 有效峰值给出 16.106 µs，按 FP8 峰值的计算项为 4.832 µs，保留 0.6 效率则为 8.053 µs。精细模型含权重展开和尾部，得 11.588 µs（v1.9.0 为 11.596 µs）。另一种 `quantized_batched_matmul` 有 432 次调用，捕获声明 FP4 计算及 FP32 累加。权重 scale 沿 N，直接送入要求沿 K 分组的原生 MX 内核会改变矩阵积。当前实现假设两侧先按各自 scale 解码到 BF16，再用 BF16 Cube。因此对齐基准也按 BF16 计。这个方案假设解码值落在 BF16 表示范围内，缺同实现整窗与逐值数值验证，原生 FP4 路线耗时为 N/A。使用 MXFP4 存储的路径并不共享一种计算价格。以上均为设备时间预测；CA 的资格仍取决于匹配的 kernel、布局和 scale。[分组投影和展开路径](examples/system-cube-routes.md) · [原生路由专家的 CA](errors/routed_grouped_matmul_mxfp4_v5.md)

## 逐节点仍有哪些越界？

相对历史预测，171,232 次调用中 8,363 次更快、94,652 次更慢，另有 68,217 次相等，均为零设备时间节点。改用精度对齐的 100% 峰值，25,101 个有正 Cube 工作的节点均未越过该计算项；82,230 个有正且可定价的 Vector 等价工作节点也均未越过其名义项。两组计算节点可以重叠。

95,689 个本地计算节点有 **42 个**低于对齐后的逻辑基准，全部是 BF16→FP32 的 `aten._to_copy.default`。最大一例是 Kimi TP8/EP8、B=16，draft 节点 303，[128,163840] 读 40 MiB、写 80 MiB。按 4 TB/s 需 31.457 µs，预测 21.968 µs，系数 **143.20%**。若系统的 4 TB/s 成立，要在预测窗内完成，至少 36.20 MiB 逻辑流量须被缓存服务或位于该设备窗外；另一种可能是该 shape 的预测偏短。缺同窗控制器计数，这两项无法分摊。[最严重节点及原始输入](../calibration/reference/roofline-audit-exceptions.json)

另对独立算子库的 950DT/PR 归档重算。2,180 个 CA 整窗中，2,068 个可算逻辑系数，69 个模型结果大于 100%，全部受逻辑访存项控制。把分母换成 CA 的实际设备窗，13 个回到 100% 内，56 个仍在其上。可核实的稠密 Cube 物理子项覆盖 281 窗，模型、CA 均无越界。前面的 42 是系统图的调用次数，这里的 69 是独立算子归档的窗口数。[含 CA 的四个具体反例及逐 shape 报告](error-analysis.md#为什么有些-u-很小有些超过-100)

## 借鉴 Uni3Sim 的 910B 微基准

阅读的版本为 `52ffdb18aef705b03e27534ddd999312a9ec5ca2`。有三项直接适合这个项目。迁移到 950DT/PR 的是实验设计；970DT 的频率、执行宽度、缓存与发起间隔仍从目标配置或显式继承假设取值。[固定版本目录](https://github.com/thu-nics/uni3sim/tree/52ffdb18aef705b03e27534ddd999312a9ec5ca2/microbench/ascend_910b) · [参数和证据的定义](https://github.com/thu-nics/uni3sim/blob/52ffdb18aef705b03e27534ddd999312a9ec5ca2/docs/kernel_performance_model.md#10-parameters-and-evidence)

| 910B 微基准怎样做 | 在 950 上可以回答什么 |
|---|---|
| 向量流固定 dtype、lane 和操作种类。FP32 Compare/Select 用不同目的缓冲、短长循环，在同一 AIV 计时并等待 PIPE_V 完成。报告单位是每次 API 调用的时间。 | 对 quant 的 cast/pack、RMSNorm 的 reduce/rsqrt 分开测“依赖链延迟”和“独立流发起间隔”；先确认实际发出的指令。这样扩到不同 Vector/Cube 比例时可替换相应价格。[Typed Vector 设计](https://github.com/thu-nics/uni3sim/blob/52ffdb18aef705b03e27534ddd999312a9ec5ca2/microbench/ascend_910b/compute/vector_pipe/typed_services.md) |
| AIC 的 ND2NZ 搬入测有限个描述符，每笔通过 MTE2→MTE1 事件确认完成。DRAM 带宽 sweep 读取 msprof 的 kernel 设备窗，以载荷除以时间。 | 针对 42 个 Cast，保持 BF16→FP32 和活动核数，用能留在 L2 和远超 L2 的工作集做匹配对，分只读、只写、并发读写。记录首响应、发起间隔、完成时间与实际 HBM 读写量。若只能得到逻辑字节，继续只报逻辑系数。CANNBot 的 950PR 记录也指出单向带宽之和会高估共享总线并发带宽，该记录的 DT 适用性待测。[完成事件内核](https://github.com/thu-nics/uni3sim/blob/52ffdb18aef705b03e27534ddd999312a9ec5ca2/microbench/ascend_910b/memory/dram/mte2_tile/finite_drain/aic_mte2_nd2nz_finite_drain_kernel.cpp) · [带宽 sweep](https://github.com/thu-nics/uni3sim/blob/52ffdb18aef705b03e27534ddd999312a9ec5ca2/microbench/ascend_910b/memory/dram/bandwidth/dram_bw_sweep.py) |
| FIA 对 C1→V1、V1→C2、C2→V2 逐边做 1/2/3 轮前缀差分，再用剩余任务求稳态间隔，另留验证 run。 | 对融合 Indexer/FA，先核对首次发射、页界停顿、尾部退休，再用冻结的参数预测整窗。微基准要能识别依赖，单个总时延不足以推回三个参数。[FIA 前缀差分实现](https://github.com/thu-nics/uni3sim/blob/52ffdb18aef705b03e27534ddd999312a9ec5ca2/microbench/ascend_910b/runtime/fia_edge_handoff/fia_edge_handoff_report.py) |

用现有 trace 做了第三项的初步检查。原生 `QuantLightningIndexerV2` 的同一 MXFP4 小窗 [B=1,Q=1,L=2176,H=32,D=128,topk=2048] 发出 17 条 MMAD。DT 的 16 个发起间隔中位数为 315 cycles，直接乘 16 得 5,040，而逐项间隔之和是 **10,135**，其中最长一段达 4,097。完整窗可复算为 `8,568（首发）+10,135（发起间距之和）+131（末条退休）+4,614（至整窗结束）=23,448 cycles`。PR 对应 `7,211+10,094+131+3,864=21,300 cycles`。这说明单一中位间隔会漏掉长停顿；要给停顿归因，仍需对应页或同步事件。这个例子是 MXFP4 量化索引器，Flash 的 BF16 融合路径需要自己的阶段证据。本轮复用归档，新增 CA 调用 0 次。[两端的逐次发起间隔](../calibration/system_attention_ca/mxindexer-static.json)

## 复算

在本仓根目录，用安装有同版 SIMPlatform 和 operator-sim 后端的系统仿真环境运行。脚本从原任务读档，核验发布的图和时长，按相同调度器重放所有对照；逐节点价格取校正后的 DES 输入。CSV 中时间单位为 µs。精度账单中的空白代表上游未声明该项或该引擎没有正工作，`aligned_vec_proxy_types` 表示等价计数。每个调用名的每种越界指标留一个比值最大的原始输入。[审计脚本](../calibration/roofline_audit.py)

```bash
sim="$PWD/system-sim-workspace/preview-0.1.0-preview.1"
env -u PYTHONPATH TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    LD_LIBRARY_PATH="$sim/compiler-runtime" "$sim/.venv-runtime/bin/python" \
    -m calibration.roofline_audit "$sim/experiments/decode-950dt-128k" \
    /tmp/operator-roofline-replay --output /tmp/operator-roofline-report
```
