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
| `auto`（默认） | 原 950DT 配置使用参考校准。其他配置的 Quant、RMSNorm、Matmul、FA 使用按目标芯片计价的资源模型。其余算子保留现有 DT 阶段投影。 |
| `resource` | 对四条已实现路线强制使用同一套资源调度，可在 DT、PR 或自定义芯片上调用。没有相应模型的算子返回 `unsupported`。 |
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
| extrapolated | 目标芯片与参考配置不同，或已显式标记的 shape 外推，例如 KV 超过已完整验证的 128k 的 DT 参考 FA，或 T>256 的 mHC 拆分链兼容路径；显式资源法在 DT 上也返回此状态。 | µs |
| unsupported | 算子名、目标 SoC、shape 或调用属性未被该路线接受。 | None |
| unavailable | 有结构模型，缺少可信的完整耗时；例如 KDA chunk。 | None |

`reason` 给出不可用或显式外推的原因。各模型能识别的外推分支不同；需要精确判别某个输入是否已测时，查[逐 shape 校准结果](error-analysis.md)。`latency_us` 是同一 kernel 的跨核设备指令窗口，串行组合取实际 kernel 窗口之和。调度器单独处理主机发射、设备排队、kernel 间空闲、通信和跨流重叠。后端在 CPU 上计算这个估计的墙钟用于衡量仿真效率，请勿加入设备时间线。

## 把框架节点映射到调用参数

传入已经过 TP/DP/EP 切分的每卡输入。shape 使用正整数序列；可选属性使用字典，省略时采用当前已建模的调用合同。相同数学算子的 dtype、布局、融合和可选输出不同时，请先核对[逐算子方法和算例](modeling-strategy.md)，再决定是否路由到该模型。下表列出默认算子名和维度顺序。

| op | 每卡 shape | 关键合同 |
|---|---|---|
| [quant](examples/quant.md) | (R,C) | dynamicMXquant，BF16 输入、block32 |
| [rmsnorm](examples/rmsnorm.md) | (R,C) | BF16，输出含 rstd |
| [matmul](examples/matmul.md)、[addmm](examples/addmm.md) | (M,N,K) | BF16；Addmm 的 bias=[N] 且 α=β=1 |
| [fa](examples/fa.md) | (Q,KV,D,B,H) | 普通非因果 SBH；Q 是本次校验长度，H 是本卡头数 |
| [lightning_indexer](examples/lightning_indexer.md) | (B,Q,L,G,D) | BSND，G=8、D=128、top-512 |
| [mhc_pre](examples/mhc_pre.md)、[mhc_post](examples/mhc_post.md)、[mhc_pre_sinkhorn](examples/mhc_pre_sinkhorn.md)、[mhc](examples/mhc.md) | (T,N,D) | N=4；mhc 代表 Pre、Sinkhorn、Post 三段，外部层单独调度 |
| [mhc_sinkhorn](examples/mhc_sinkhorn.md) | (T,N) | N=4，`num_iters` 为实际迭代数 |
| [kda_recurrent](examples/kda_recurrent.md)、[kda_chunk](examples/kda_chunk.md) | (B,S,Hq,Hv,K,V) | 分别为递推和分块路线，仅递推有完整时延校准 |
| [grouped_matmul_swiglu_quant_v2](examples/grouped_matmul_swiglu_quant_v2.md)、[gmm2](examples/gmm2.md)、[moe_ffn](examples/moe_ffn.md) | (M,K,F,E) | `expert_counts` 为 E 个本地专家各自收到的有效 token 数，总和为 M |

例如 `estimate("moe_ffn", (32, 256, 128, 2), {"expert_counts": [1, 31]})` 预测整个两段计算链。调度该节点后不要再把两个 GMM 组件的估计加到同一事件上。EP 大小不能替代本地专家直方图。推测 8 个 token 时，普通 FA 的主模型校验长度常为 Q=9，实际 B、H、打包方式仍由框架提供。

对 RMSNorm 和 mHC Post，默认采用调用上下文未知时的估计，不保存上次预测的设备状态。系统若维护着**同一类 kernel 连续执行时前序调用的最大活跃 AIV 数**，可用 `{"resident_aiv": 32}` 显式传入，取值 0…目标 Vector 核数，0 表示该隔离序列的首次调用。发生换流或并发干扰而无法确定时，省略该属性。该量用于描述 CA 观察到的启动上下文，硬件缓存状态仍待独立确认。

当调度器知道隔离序列刚开始时，令 `resident = 0`。以下三行只由同一流、同类 kernel 的实际调度事件驱动；同一个待选方案反复试算时保留原状态。返回的 `active_aiv` 是本次预计活跃核数，缺少此信息的路线返回 None。

```python
result = estimate("rmsnorm", (2, 6144), {"resident_aiv": resident})
if result.active_aiv is not None:
    resident = max(resident, result.active_aiv)
```

mHC Post 在少 token 且切通道时，默认估计在首次、扩核与热态之间平衡相对误差。已知连续调用处于热态时，传入 `resident_aiv` 可避免把额外的启动时间累积到每个事件。其余分核分支用历史条件时间的加权中位数。在线查询本身不改变设备历史。[不同状态的同集误差与大 shape 核查](../results/v0.12/audit/summary.md)

## 检查资源瓶颈与设备情景

对同一算子比较设备时，在四条已实现的资源路线上对两端都指定 `timing_mode="resource"`，以保持算法口径一致。默认的 DT 参考校准仍可用于和系统原有结果衔接。

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
