# 在系统仿真器里使用 operator-sim

在系统框架中找到原来根据算子及单卡 shape 返回 Roofline 耗时的函数。在这里调用 `operator_model.backend.estimate`，将微秒数交给原有事件调度器。每次查询无需分配设备张量、读取 trace 或启动 CA。缺省参考平台为 `Ascend950DT_9582`，其他设备通过静态配置传入。

## 安装及一次查询

在项目根目录使用 Python 3.10 或更新版本安装。源码和 wheel 均带有当前模型参数，运行时依赖仅为 Python 标准库。

```sh
python -m pip install .
operator-sim --op rmsnorm --shape 2 6144
```

部署到其他仿真主机时，可在构建机运行 `python -m pip wheel --no-deps . -w dist`，传输生成的 wheel 后用目标环境的 `python -m pip install 路径/文件名.whl` 安装。命令行返回含 `predicted_us` 的诊断 JSON，适合手动检查。批量系统仿真直接在同一 Python 进程调用下述接口，模型参数在进程内复用。

```python
from operator_model.backend import estimate

result = estimate("rmsnorm", (2, 6144))
print(result.latency_us, result.status)
```

## 在系统仿真里选择计时模式

| `timing_mode` | 适用范围与语义 |
|---|---|
| `auto`（默认） | FA、Matmul、列偏置 Addmm、RMSNorm、mHC Post 及已建模的资源路线在所选芯片上定价。Quant 在参考 DT 的原有范围内采用校准路径，合并 token 超过 1,048,576 行或换到其他芯片配置时用目标资源模型。其余路线见[计时方法](modeling-strategy.md)。 |
| `resource` | 请求该算子的目标资源排程，可在 DT、PR 或自定义芯片上调用。没有相应模型的算子返回 `unsupported`。 |
| `calibrated` | 使用已登记配置的校准路线。DT 上沿用原接口，PR 上的三条路线复现 v0.15 的专属预测及其分支覆盖。自定义配置无对应校准时返回 `unsupported`。 |

例如 `estimate("fa", (9, 32768, 64), chip=chip, timing_mode="resource")`。CLI 对应 `--timing-mode resource`。`projection.method == "target_resource_schedule"` 表示按目标资源计价，`stages_us` 是可相加的关键路径阶段，`timing_diagnostics` 内含操作数、搬运量、局部吞吐或流水约束。具体单位由字段和算子公式给出。目标资源路径不调用 DT 时延预测，`reference_latency_us` 为 `None`。[三类算法、参数和算例](modeling-strategy.md)

## 维护芯片配置

`load_chip("Ascend950DT_9582")`、`load_chip("Ascend950PR_957b")` 读取随包发布的 950 配置，`load_chip("Ascend970DT")` 读取 120 AIC / 120 AIV 的架构情景。自定义芯片传 JSON 路径或同结构的 Python 字典。建议每个系统情景只载入一次，复用返回的只读 `ChipConfig`。运行时无需安装 CANN、读取芯片信息或执行 profiler。

| JSON 字段 | 单位与含义 |
|---|---|
| `schema_version`、`name` | 格式版本 1、调用方给定的设备或架构情景名 |
| `frequency_ghz` | 一个调度域的统一工作频率，GHz |
| `cube_cores`、`vector_cores` | 此调度域的 Cube、Vector 核数。双 die 设备的核数、峰值和带宽需取同一范围 |
| `cube_tops`、`vector_tops` | 两张表各有字符串键 `32`、`16`、`8`、`4`，每项是该单元全体核的稠密峰值，单位为 $10^{12}$ 次标量操作/秒，FMA=2。位宽以该算子的计算格式解释。无法确定的项填 `null`。如果仅有整卡矩阵乘算力，将其填入 Cube 表，Vector 对应位置填 `null` |
| `hbm_capacity_gb`、`hbm_bandwidth_gbs` | 设备外存容量（十进制 GB）和总带宽（十进制 GB/s）。字段名沿用 HBM，PR 的 CA 外存类型另有记录；容量供系统层放置张量 |
| `details` | 可选的 `ub_bytes`、`l1_bytes`、`l2_bytes`、`l0{a,b,c}_bytes`，以及 `l2_bandwidth_gbs`（整片）、`mte1_bandwidth_gbs`、`mte2_bandwidth_gbs`、`mte3_bandwidth_gbs`（每核）。省略的细项继承 950DT 有记录的值，`chip.inherited_fields` 列出来源。显式 `null` 保留该字段的未知状态，算子可能继续使用模型缺省值。资源法可直接使用目标端口带宽，可按算子覆盖发起、等待、局部有效带宽等参数；缺少细节时用 DT 阶段测量的默认值。该类模型默认值记录在各算子参数表中，独立于芯片字段继承清单。其余路线的相对带宽投影需要两端同口径数据 |
| `sources` | 可选，字符串到字符串的来源/假设说明 |

仅有芯片宏观规格时，保持未知细项为空并记录继承来源。970 的 288 GB、14.4 TB/s 与 200 MB、30 TB/s L2 都按十进制解释；算法假设每个忙 AIC 最多配 1 个 AIV（`details.fa_aiv_per_aic=1`），配置中已计入 Vector 双发射。直接使用给定的 TOP/s，避免在调用方再次乘二。[970 全部参数及来源](../configs/chips/Ascend970DT.json)

例如将仓库里的示例复制成自己的 `960.json`、`970.json`、`980.json`，用各自的参数替换。示例数值仅用于演示。在 CLI 可直接运行 `operator-sim --op fa --shape 9 32768 64 1 1 --dtype bf16 --chip configs/chips/scenario-example.json`。[完整示例](../configs/chips/scenario-example.json) · [DT 配置及来源](../configs/chips/Ascend950DT_9582.json) · [PR 配置及来源](../configs/chips/Ascend950PR_957b.json)

```python
from operator_model import load_chip, estimate

chip = load_chip("Ascend970DT")
event = estimate("fa", (9, 32768, 64, 1, 1), chip=chip, dtype="bf16")
if event.status in ("estimated", "extrapolated"):
    duration_s = event.latency_us * 1e-6
    cube_ops = event.work["cube_ops"]
    vec_ops_by_precision = event.work["vec_ops_by_dtype"]
    input_bytes = event.work["logical_read_bytes"]
```

显式 `dtype` 指主输入张量的存储类型，省略时使用该库实现默认值。也兼容在 `attrs["dtype"]` 中传入；两个位置同时提供时须一致。该 dtype 与累加或 Vector 中间计算的 dtype 可以不同。举例说，BF16 RMSNorm 的算术计入 FP32，MXFP8 MoE 的激活与权重进入 Cube，SwiGLU 在 Vector 上另计。无对应计时合同的格式返回 `unsupported`。

**返回值**包含以下几组信息。

| 字段 | 调度器可如何使用 |
|---|---|
| `latency_us`、`status`、`reason`、`chip_name` | 设备时间、适用状态和原因，区分参考估计、外推、缺少计时与输入不支持 |
| `active_aic`、`active_aiv` | 模型估计的活跃核数，缺少判断依据时为 `None`。串行组合的目标核数取各段的最大值 |
| `work` | `cube_ops`、`vec_ops` 及各自的 `*_by_dtype`，`vec_ops_by_class` 拆分运算类别。每个 exp、除法等按一次有用标量运算计，实际每周期服务能力随指令而变。见 `scope` 和 `notes` 中的已统计范围 |
| `work["logical_read_bytes"]`、`work["logical_write_bytes"]` | 每个 kernel 输入、输出张量的逻辑字节。未知方向返回 `None`。实际外存字节 `physical_hbm_bytes` 当前保留为 `None`。系统内存容量约束需单独核算活跃张量和 workspace |
| `rates` | `useful_cube_tops`、`counted_vector_tops` 和 `logical_io_gbs`，均为工作量除以预测的设备时间。逻辑接口吞吐可以大于外存带宽；另给出 `cube_nominal_peak_fraction`、`vector_nominal_peak_fraction`，即已计工作按对应位宽峰值完成所需的理想时间除以预测时间。缺少相应峰值时为 `None`。用于比较有用吞吐占整片名义峰值的比例，实际硬件忙碌率需要独立流水证据 |
| `projection` | 目标资源路径给出目标周期或字节诊断和阶段微秒数；DT 阶段投影另保留参考微秒数、目标阶段时间、继承字段、迁移方法和假设。FA、Quant、RMSNorm 与 Matmul 命中分段路径时，`timing_diagnostics` 提供首读、计算和写回等阶段的 µs 估计；通用路线可返回空字典。系统调度按 `latency_us` 记一次耗时。高水位已知时，两种核数间按本次活跃核比例折算，并记录在 `resident_aiv_mapping` |

CLI 用 `predicted_us` 表示相同微秒数，用 `estimate_status` 表示系统接口状态。Python 接口保留上一版的 `soc=` 参数。芯片名拼错或配置不合法会返回 `unsupported`；在初始化时直接用 `load_chip` 可以及早发现配置问题。

## 接入现有调度器

下例假定系统事件队列以秒计时，原有 `roofline_us` 接口返回微秒。调用方可以自行决定是否采纳外推结果，并记录回退原因。

```python
from operator_model.backend import estimate

def duration_seconds(op, local_shape, attrs, roofline_us, soc="Ascend950DT_9582"):
    result = estimate(op, local_shape, attrs, soc=soc)
    if result.status == "estimated":
        return result.latency_us * 1e-6
    return roofline_us(op, local_shape, attrs) * 1e-6
```

| 返回状态 | 含义 | latency_us |
|---|---|---|
| estimated | 当前模型接受这次调用的属性并给出设备时间估计。新 shape 可能没有逐点 CA；适用范围和独立误差见校准报告。 | µs |
| extrapolated | 目标芯片与参考配置不同，或已显式标记的 shape 外推，例如逻辑读写总量超过 L2 的 DT 参考 Quant、KV 超过已完整验证的 128k 的 DT 参考 FA，或 T>256 的 mHC 拆分链兼容路径；显式资源法在 DT 上也返回此状态。 | µs |
| unsupported | 算子名、目标 SoC、shape 或调用属性未被该路线接受。 | None |
| unavailable | 有结构模型，缺少可信的完整耗时；例如默认的融合 KDA chunk V1，或实际内核铺开后末片只有 16 列且数值失败的 vLLM HcPost。 | None |

`reason` 给出不可用或显式外推的原因。各模型能识别的外推分支不同；需要精确判别某个输入是否已测时，查[逐 shape 校准结果](error-analysis.md)。`latency_us` 是同一 kernel 的跨核设备指令窗口，串行组合取实际 kernel 窗口之和。调度器单独处理主机发射、设备排队、kernel 间空闲、通信和跨流重叠。后端在 CPU 上计算这个估计的墙钟用于衡量仿真效率，请勿加入设备时间线。

## 把框架节点映射到调用参数

传入已经过 TP/DP/EP 切分的每卡输入。shape 使用整数序列，张量维度通常为正数。AttnResFwd 的有效历史行数可以是 0。可选属性使用字典，省略时采用当前已建模的调用合同。相同数学算子的 dtype、布局、融合和可选输出不同时，请先核对[逐算子方法和算例](modeling-strategy.md)，再决定是否路由到该模型。下表列出默认算子名和维度顺序。

| op | 每卡 shape | 关键合同 |
|---|---|---|
| [quant](examples/quant.md) | (R,C) | dynamicMXquant，BF16 输入、block32 |
| [moe_gating_top_k](examples/moe_gating_top_k.md) | (T,E,K) | FP32 logits，单分组 sigmoid 加 bias，挑选 K 个专家 |
| [sigmoid_fp32](examples/sigmoid_fp32.md) | (T,H_local) | 独立连续 FP32 Sigmoid；Kimi KDA 的 beta 宽是本地 head 数，TP8 时为 12 |
| [cast_bf16_fp32](examples/cast_bf16_fp32.md) | (T,D) | 连续 BF16 输入、FP32 输出 |
| [attn_res_fwd](examples/attn_res_fwd.md) | (T,H,S,N) | Kimi BF16 prefix/addend、S 个 bank slot、N 个有效历史行 |
| [rmsnorm](examples/rmsnorm.md) | (R,C) | BF16，输出含 rstd |
| [matmul](examples/matmul.md)、[addmm](examples/addmm.md) | (M,N,K) | BF16；Addmm 的 bias=[N] 且 α=β=1 |
| [fa](examples/fa.md) | (Q,KV,D,B,H) | 普通非因果 SBH；Q 是本次校验长度，H 是本卡头数 |
| [lightning_indexer](examples/lightning_indexer.md) | (B,Q,L,G,D) | BSND，G=8、D=128、top-512 |
| [mhc_pre](examples/mhc_pre.md)、[mhc_post](examples/mhc_post.md)、[mhc_pre_sinkhorn](examples/mhc_pre_sinkhorn.md)、[mhc](examples/mhc.md) | (T,N,D) | N=4；mhc 代表 Pre、Sinkhorn、Post 三段，外部层单独调度 |
| [mhc_sinkhorn](examples/mhc_sinkhorn.md) | (T,N) | N=4，`num_iters` 为实际迭代数 |
| [kda_recurrent](examples/kda_recurrent.md)、[kda_chunk](examples/kda_chunk.md) | (B,S,Hq,Hv,K,V) | 递推有 CA 校准；分块默认 V1 仅工作量，显式 FLA V2 可估计三个设备窗 |
| [grouped_matmul_swiglu_quant_v2](examples/grouped_matmul_swiglu_quant_v2.md)、[gmm2](examples/gmm2.md)、[moe_ffn](examples/moe_ffn.md) | (M,K,F,E) | `expert_counts` 为 E 个本地专家各自收到的有效 token 数，总和为 M |

例如 `estimate("moe_ffn", (32, 256, 128, 2), {"expert_counts": [1, 31]})` 预测整个两段计算链。调度该节点后不要再把两个 GMM 组件的估计加到同一事件上。EP 大小不能替代本地专家直方图。推测 8 个 token 时，普通 FA 的主模型校验长度常为 Q=9，实际 B、H、打包方式仍由框架提供。

系统有序流能够提供前一算子时，可用两个条件接口做诊断；这里所有 `us` 都是设备微秒。

```python
from operator_model import estimate_post_history, rmsnorm_pipeline

post = estimate_post_history([1, 4, 192], chip="Ascend950DT_9582")
first_us = post.scenarios["first"].latency_us
repeat_us = post.for_predecessor(post.active_aiv, previous_shape=post.shape).latency_us
# 这里的概率是调用方给定的示例场景。
mean_us = post.expected_us({"first": 0.2, "repeat": 0.8})

rms = rmsnorm_pipeline([4, 784], chip="Ascend950PR_957b",
                       resident_aiv=4, immediate_repeat=True, diagnostics=True)
elapsed_us = rms["us"]
```

常规 `estimate("matmul", ...)` 已按输入与目标芯片比较单输出矩形的普通和分 K 排程。选中分 K 时，`work["implementation_work"]` 给出附加归约与中间传输，逻辑输入和输出字节仍对应应用张量。需要查看候选定价时可调用 `from operator_model import estimate_matmul_splitk`，返回普通与分 K 的耗时、活跃核数、归约工作量和选择理由。[前瞻几何与使用示例](../results/v0.28/matmul/streamk-analysis.md)

对 Stream-K 与全输出分块识别尚不确定的 Matmul，应保留 `[M,N,K]`、dtype、转置及权重格式，分别统计真实库分支。四个 batch 取 1、9、11、16 时，普通 decode 的 M 为 1、9、11、16；同一 rank 全部请求同时校验 9 个 token 的 M 为 9、81、99、144。混合批次以调度器实际提交（含 padding）的行数为准。[三模型投影的输入和两端 CA](../results/v0.28/matmul/report.md)

RMSNorm 与 mHC Post 可用 shape、dtype 和芯片独立预测。RMSNorm 在最忙核最多两轮 UB 且前序未知时，当前默认使用归约与双写流水的折中值；如需分清首遇与立即复跑，可调用上面的显式接口。一次查询没有提供同流历史时，返回未知启动状态的点估计，同时返回首次、扩核和热态的时间供系统做敏感性分析。选择多种调度方案时，调用本身不会改变后续预测。如果明确是单独首次启动，可传入 `resident_aiv=0`。

在整网事件循环中，每卡每条实际流持有一个 `OperatorStream`。试排候选时用 `preview`，只有成功进入该流的调用才用 `advance`。下面的首次和紧邻复跑由调度序列决定，无需目标芯片的 CA：

```python
from operator_model import OperatorStream

stream = OperatorStream("Ascend950PR_957b", track_sinkhorn_history=True)
first = stream.advance("mhc_post", [65, 4, 5120])
repeat = stream.advance("mhc_post", [65, 4, 5120])
```

对于只提供 shape 的内置 Post，混合首遇/复跑样本在 DT、PR 上的最优静态 WAPE 下界都超过 25%。如果系统没有可恢复的派发历史，应把不同前序的估时作为情景输出；服务平均值取决于真实状态频率。[逐芯片数学下界与条件调用](../results/v0.28/post/report.md)

当调度器知道隔离序列刚开始时，令 `resident = 0`。以下三行只由同一流、同类 kernel 的实际调度事件驱动；同一个待选方案反复试算时保留原状态。返回的 `active_aiv` 是本次预计活跃核数，缺少此信息的路线返回 None。

```python
result = estimate("rmsnorm", (2, 6144), {"resident_aiv": resident})
if result.active_aiv is not None:
    resident = max(resident, result.active_aiv)
```

## 检查资源瓶颈与设备情景

对同一算子比较设备时，在具有目标资源调度的路线上对两端都指定 `timing_mode="resource"`，以保持算法口径一致。默认的 DT 参考校准仍可用于和系统原有结果衔接。

```python
from operator_model import estimate, load_chip

target = load_chip("Ascend970DT")
r = estimate("matmul", (1024, 2048, 32), chip=target, timing_mode="resource")
diag = r.projection["timing_diagnostics"]
print(r.latency_us, diag["target_plan"]["tail_split_mn"], diag["mmads_total"])
print(diag["shared_traffic"]["hbm_payload_fraction_of_nominal"])
```

| 路线 | 可用于诊断的返回量 | 目标配置与敏感性 |
|---|---|---|
| Quant | `target_plan` 的分核、UB tile 和迭代数；`first_tile_cycles` 的首读、读发起间隔、VF、数据写和 scale 写；`scale_peak_writers` 为同一 32 B scale 区段的最多写核数 | UB 容量、每核 MTE2/MTE3 带宽及 BF16 Vector 峰值会改变服务；全片冷读写需满足 `chip_io_floor_cycles` |
| RMSNorm | `target_plan.rows_per_core`、`ub_tile_rows` 和 `conditional_us` 给出目标分核、UB 轮次和首次／扩核／热态时间。 | 已知前序驻留 AIV 数可传 `attrs={"resident_aiv": n}`，其中 0 指首次。各状态是模型估计，不能当作实测误差界。 |
| Matmul | `target_plan` 的基本 tile、K 片、DMA 页、波数和尾波分片；`wave_cost_cycles` 区分发起与完成；`shared_traffic` 给本设备指令窗暴露 HBM 字节、最终写回字节、L2/HBM 有效 GB/s 及峰值占比 | 可设置 `matmul_hbm_efficiency`、`matmul_l2_efficiency` 在 (0,1] 内限定峰值，`matmul_output_retained_fraction` 在 [0,1] 内指定可留脏比例；设 0 表示输出均在窗口内落片外 |
| FA | `wave_types` 列出每类波的忙核数及每 AIC 的 AIV 数。`first.demands_cycles` 与 `first.steady_bottleneck` 给一块 KV 的资源需求与最长项，`steady_kv_read_fraction_of_hbm_peak` 是各忙核的稳态冷读需求占整片片外峰值比例 | `fa_aiv_per_aic` 可设置 1 或 2，比较固定绑定、可借用空闲 AIV 两种库情景；长序列可能由读发起间隔主导 |

设备时延与有效吞吐按不同口径解释。对单核端口，`payload/(1000 × µs × GB/s/核)` 表示这段窗口中有效传输占该端口名义容量的比例；整片多个端口的分母需乘相应核数。芯片级带宽只取一次整片值。模型返回的冷读、重复读、窗口内写回和 `work` 的逻辑接口字节各有不同用途；物理 HBM 字节没有实测时仍为空。可通过各波瓶颈、芯片配置和 CA 的指令窗口逐项核对。[三种资源模型的公式和算例](modeling-strategy.md) · [FA 场景与核绑定敏感性](../results/v0.17/fa-scenarios.md)


### 模型主干中独立调用的 Vector 算子

Kimi 的单组 sigmoid 专家选择使用 FP32 输入，调用 `estimate("moe_gating_top_k", (tokens,experts,topk), dtype="fp32", chip=chip)`。DeepSeek 回退路径中的 BF16→FP32 转换使用 `estimate("cast_bf16_fp32", (tokens,hidden), dtype="bf16", chip=chip)`。二者输出包含设备时间、分核、逻辑读写字节和 Vector 工作量；物理 HBM 流量需要缓存/事务级证据。系统将路由选择、专家 FFN、跨卡派发分别排入事件图。Flash 的后归一化只计一次，再接类型转换，避免同时计融合辅助函数与其叶节点。[模型假跑与路由合同](workload-coverage.md)


Kimi 的融合 Attention Residual 接收 `(tokens, hidden, bank_slots, valid_bank_rows)`。例如 `estimate("attn_res_fwd", (1,7168,8,1), {"addend":True,"write_idx":1,"materialize":True}, dtype="bf16", chip=chip)` 表示每个 token 使用一个历史 slot，并将结果写至下一个 slot。工作量、逻辑字节以及目标 tiling 随属性变化。只有已定价的单 token/核驻留分支返回设备时间。系统调度器遇到 `status="unavailable"` 应保留工作量并标出缺失时间。`latency_us=None` 不能作为零时长加入整网统计。[逐阶段算例与支持条件](examples/attn_res_fwd.md)


在系统延迟的敏感性分析中，注意两个已量化的误差。对于 [2048,5120] 的 RMSNorm，PR 全写回情景高估 77.64%，默认按 L2 余量估计为 16.669 µs，比完整 CA 的 17.820 µs 低 6.46%。可通过芯片配置 `rms_output_retained_fraction` 做敏感性分析。Kimi AttnResFwd 在 10 个驻留 shape 上，DT/PR 的耗时 WAPE 分别为 29.95%/33.10%，点估计未判中两张卡的相对方向。一个可计时分支仍有特定适用范围；模型的设备微秒数与置信程度分别查看。[全体逐算子误差与逐 shape 数据](error-analysis.md)

### 从系统的调用顺序恢复启动状态

系统知道流从空闲开始、并连续提交同类算子时，可以为每个 rank 和流建一个 `OperatorStream`。首次、扩核、热态只依赖已提交的核数，调用方仍只需传入 shape 和 dtype。试排多个候选 shape 用 `preview`，确认派发用 `advance`。重置流用 `reset`。在中间执行另一族 kernel 后，首次回到该家族使用未知历史点估计。此后连续同类调用给出已预热的核数下界；需求在该下界内按热态计，需求超过已知下界时，RMSNorm 会将 shape 估计限制在仍可能的前序时长范围内，其他路线使用未知状态估计。达到整卡核数或显式传入驻留核数后，高水位重新确定。其他 kernel 对指令缓存的具体作用目前缺少足够证据。

```python
from operator_model import OperatorStream

rank0 = OperatorStream("Ascend950PR_957b")
first = rank0.advance("mhc_post", [2, 4, 6144], dtype="bf16")
candidate = rank0.preview("mhc_post", [1, 4, 128], dtype="bf16")
executed = rank0.advance("mhc_post", [1, 4, 128], dtype="bf16")
assert candidate == executed
```

真实例子是同一流连续派发两次 BF16 mHC Post [3,4,5376]。DT 上 `advance` 两次给出 4.855、2.380 µs，对应 CA 5.370、2.122 µs。只用形状的 `estimate` 两次都给出 3.195 µs；即使换成数学上最优的单一值，这两次的最坏误差仍至少 43.34%。PR 的下界也达到 41.08%。如果系统需要把最大误差控制在 40% 以内，至少要保留能区分这两个执行状态的信息。[两端的预测与 CA 流水图](examples/mhc_post.md)

同形状、同平台、同一 kernel 的两次采集仍可能给出不同设备时间。验收的默认主表向无状态 `estimate` 只传 shape、dtype 和算子属性。已知调用顺序的成绩单列。[同输入实测上下界](../results/v0.21/dispatch/post-identifiability.json)

对于 RMSNorm，前序覆盖已知且每核只执行一个 UB 块时，`projection.timing_diagnostics.lane_schedule` 提供末核通道类型、竞争扇区和获胜的写回路径。可以在相同输入下对 `stream.preview()` 与 `advance()` 核对，只用实际提交的调用推进状态。跨族 Post 的实测表明，直接继承前一算子的活动 AIV 数会把 PR 的 WAPE 从 6.93% 提高到 24.29%。[逐行 RMS 例子](examples/rmsnorm.md) · [混合序列的实测](../results/v0.22/post-stream/report.md)


### 中间运行了纯 AIC 内核

系统调度记录能确认中间的实现只发射到 AIC 时，给对应的 `advance` 传 `aic_only=True`，下一次 RMSNorm 仍使用之前的 AIV 覆盖。该值属于本次内核的执行合同，与算子名称、Cube FLOPs 和模型返回的预测核数独立。带 Vector 归并的 Stream-K Matmul 应走普通调用。首次接入运行中的 rank 先调用 `forget_history()`；失败或仅试算的候选不会改变记录。

```python
rank0.advance("rmsnorm", [2, 2944])
rank0.advance("addmm", [16, 64, 256], {"self_shape": [64]}, aic_only=True)
after_cube = rank0.preview("rmsnorm", [2, 2944])
```

这条纯 AIC 声明在包含 AIV 的已建模路线上会触发 `ValueError`。未提供声明时，跨族按未知前序处理。已验证的跨族例子来自 CANN 的 BF16 列偏置 Addmm，采集时另外用实际核掩码和全输出数值独立复核。[计算示范及逐形 CA](examples/rmsnorm.md)

### 按实际库选择扩展

对于锁定的 FLA 26.9.1 三段 KDA，传 `estimate("kda_chunk", [B,S,H,H,128,128], {"implementation":"fla_26_9_1_v2"}, chip=...)`；每段的工作组、核数、分层字节和耗时见 `projection["target_plan"]["stages"]`。返回状态为 `extrapolated`。对 Flash/Pro 的 batched HcPost，传 `estimate("mhc_post", [1,T,4,D], {"implementation":"vllm_hc_post"}, chip=...)`，可查询目标分核、有效 FP32 算量、读写字节与双槽竞争后的设备时间。适用 BF16 路径返回 `extrapolated`；真实内核涉及短尾片数值风险或缺少必要资源时，时延为 `None`、工作量仍可用。默认三维 `mhc_post` 对应 CANN 安装包的 `aclnnMhcPost`。对 BF16 RMSNorm，`estimate_rms_history_band([T,D], chip=...).by_resident_aiv_us` 返回各个可区分历史 H 下的场景值。该区间用于敏感性分析；需要一条连续执行轨迹时使用 `OperatorStream`。[KDA V2](examples/kda_chunk.md) · [vLLM HcPost](examples/hc_post_vllm.md) · [RMSNorm](examples/rmsnorm.md)


从中途接入一条已执行的流，或外部后端确实执行了一次未建模的 kernel，调用 `rank0.forget_history()`。随后第一个可见 RMSNorm 按未知历史估计，`advance` 记录它实际要求的活跃核数。若下次增加用核数，可查 `result.projection["partial_history"]` 中的 `minimum_resident_aiv`、`possible_us`、`shape_only_us` 和 `endpoint_scenario_aiv`。范围描述模型的可行前序，CA 波动仍可能超出。新进程明确从空流开始时使用 `reset()`；仅用 `preview()` 试算不会推进状态。

### 读取 Cast 的发起时刻与使用 Post 的有序重复

```python
from operator_model import OperatorStream, estimate

cast = estimate("cast_bf16_fp32", [2097152, 32], dtype="bf16", chip="Ascend950PR_957b")
issue_us = cast.projection["timing_diagnostics"]["initial_read_issue_us"]

stream = OperatorStream("Ascend950DT_9582")
first = stream.advance("mhc_post", [65, 4, 5120], dtype="bf16")
repeat = stream.advance("mhc_post", [65, 4, 5120], dtype="bf16")
```

`issue_us` 是忙核第一个 UB 块（以及第二个空槽，如有）的读发起时间，相对全片首条设备指令计时。其余时间由 `latency_us` 和阶段诊断描述。若换了指令实现，在芯片 config 的 `details.cast_initial_read_pair_cycles` 覆写同一核连续发出两笔读的周期数；不确定时可传 null。在同一系统流上只有紧邻成功重复的内置 token 分核 Post 会默认使用重复派发的读写时序。前序丢失时调用 `forget_history()`，分支探索使用 `preview()`。[Cast 同轴读图](../results/v0.26/memory/README.md) · [Post 适用分支和独立验证](../results/v0.27/post/report.md)

独立 Sinkhorn 可传入同一库实现此前已覆盖的 AIV 数，例如 `estimate("mhc_sinkhorn",(67,4),{"resident_aiv":0,"num_iters":10},chip="Ascend950DT_9582")`。0 表示确定首次，省略表示前序未知。其他算子的活跃核数无法说明本实现的前序。[调用边界与实测](../results/v0.27/fusion/public-sink.md)

有序系统也可从已提交的调用恢复 Sinkhorn 前序。每个 rank 和 stream 各持有一个实例，调用方按实际顺序传入 shape 和迭代数即可。

```python
sink_stream = OperatorStream("Ascend950PR_957b", track_sinkhorn_history=True)
for tokens, iters in [(13, 3), (46, 8), (67, 10)]:
    result = sink_stream.advance("mhc_sinkhorn", (tokens, 4), {"num_iters": iters})
```

若接入时上一段历史未知，先调 `forget_history()`。它仅在能证明当前 AIV 已全部被覆盖时恢复热态，跨库实现时重新判断。两端 30 个实际 CA 窗和 970DT 的冻结预测都重放了相同的调用顺序。[使用边界与验证](../results/v0.27/fusion/stream.md)

## Kimi-K3 在一次 prefill 调度中的 KDA

可以从调度器的实际 token 数选择 FLA 调用。在 TP8、单卡 B8、每条 32Ki 的未切块例子中，模型配置给出每卡 12 个头和 128 维状态。结果包含 `aclnn`、工作项、三个主核各自的耗时和外围张量的读写字节。

```python
from operator_model import plan_kimi_kda_prefill

p = plan_kimi_kda_prefill([32768] * 8, attention_tp=8,
                          initialized_sequences=3, chip="Ascend950DT_9582")
main_us = p["main_kernels_us"]
dispatch_us = p["full_dispatch_us"]
```

这个例子选择 V2，`main_us` 约为 33,161 µs，属于大 shape 外推。系统只应把这个量用于三个主核，Q/K 归一化、状态 gather/scatter 和 launch 间隔另行建模。缺少合格计时的 V1 或不等长的 packed V2 返回相应的形状、路由和可用字节字段，耗时返回 `None`。[各代输入与输出示例](../results/v0.27/kda/report.md)
