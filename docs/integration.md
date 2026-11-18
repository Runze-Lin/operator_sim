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

## 维护芯片配置

`load_chip("Ascend950DT_9582")` 与 `load_chip("Ascend950PR_957b")` 读取随包发布的常量。自定义芯片传 JSON 路径或同结构的 Python 字典。建议每个系统情景只载入一次，复用返回的只读 `ChipConfig`。运行时无需安装 CANN、读取芯片信息或执行 profiler。

| JSON 字段 | 单位与含义 |
|---|---|
| `schema_version`、`name` | 格式版本 1、调用方给定的设备或架构情景名 |
| `frequency_ghz` | 一个调度域的统一工作频率，GHz |
| `cube_cores`、`vector_cores` | 此调度域的 Cube、Vector 核数。双 die 设备的核数、峰值和带宽需取同一范围 |
| `cube_tops`、`vector_tops` | 两张表各有字符串键 `32`、`16`、`8`、`4`，每项是该单元全体核的稠密峰值，单位为 $10^{12}$ 次标量操作/秒，FMA=2。位宽以该算子的计算格式解释。无法确定的项填 `null`。如果仅有整卡矩阵乘算力，将其填入 Cube 表，Vector 对应位置填 `null` |
| `hbm_capacity_gb`、`hbm_bandwidth_gbs` | 设备外存容量（十进制 GB）和总带宽（十进制 GB/s）。字段名沿用 HBM，PR 的 CA 外存类型另有记录；容量供系统层放置张量 |
| `details` | 可选的 `ub_bytes`、`l1_bytes`、`l2_bytes`、`l0{a,b,c}_bytes`，以及 `l2_bandwidth_gbs`、`mte2_bandwidth_gbs`、`mte3_bandwidth_gbs`。省略的细项继承 950DT 有记录的值，`chip.inherited_fields` 列出来源。显式 `null` 保留未知。当前 MTE 带宽倍率需要两端都有同口径值；L2、UB 信息的使用范围见建模方法 |
| `sources` | 可选，字符串到字符串的来源/假设说明 |

例如将仓库里的示例复制成自己的 `960.json`、`970.json`、`980.json`，用各自的参数替换。示例数值仅用于演示。在 CLI 可直接运行 `operator-sim --op fa --shape 9 32768 64 1 1 --dtype bf16 --chip configs/chips/scenario-example.json`。[完整示例](../configs/chips/scenario-example.json) · [DT 配置及来源](../configs/chips/Ascend950DT_9582.json) · [PR 配置及来源](../configs/chips/Ascend950PR_957b.json)

```python
from operator_model import load_chip, estimate

chip = load_chip("960.json")
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
| `projection` | 跨芯片时提供参考微秒数、目标阶段时间、继承字段、迁移方法和假设。Quant 与 Matmul 命中分段路径时，`timing_diagnostics` 提供首读、计算和写回等阶段的 µs 估计；通用路线可返回空字典。系统调度按 `latency_us` 记一次耗时。高水位已知时，两种核数间按本次活跃核比例折算，并记录在 `resident_aiv_mapping` |

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
| extrapolated | 目标芯片与参考配置不同，或已显式标记的 shape 外推，例如 KV 超过已完整验证的 128k 的普通 FA，或 T>256 的 mHC 拆分链兼容路径。 | µs |
| unsupported | 算子名、目标 SoC、shape 或调用属性未被该路线接受。 | None |
| unavailable | 有结构模型，缺少可信的完整耗时；例如 KDA chunk。 | None |

`reason` 给出不可用或显式外推的原因。各模型能识别的外推分支不同；需要精确判别某个输入是否已测时，查[逐 shape 校准结果](error-analysis.md)。`latency_us` 是同一 kernel 的跨核设备指令窗口，串行组合取实际 kernel 窗口之和。调度器单独处理主机发射、设备排队、kernel 间空闲、通信和跨流重叠。后端在 CPU 上计算这个估计的墙钟用于衡量仿真效率，请勿加入设备时间线。

## 把框架节点映射到调用参数

传入已经过 TP/DP/EP 切分的每卡输入。shape 使用正整数序列；可选属性使用字典，省略时采用当前已建模的调用合同。相同数学算子的 dtype、布局、融合和可选输出不同时，请先核对[逐算子方法和算例](modeling-strategy.md)，再决定是否路由到该模型。下表列出默认算子名和维度顺序。

| op | 每卡 shape | 关键合同 |
|---|---|---|
| quant | (R,C) | dynamicMXquant，BF16 输入、block32 |
| rmsnorm | (R,C) | BF16，输出含 rstd |
| matmul、addmm | (M,N,K) | BF16；Addmm 的 bias=[N] 且 α=β=1 |
| fa | (Q,KV,D,B,H) | 普通非因果 SBH；Q 是本次校验长度，H 是本卡头数 |
| lightning_indexer | (B,Q,L,G,D) | BSND，G=8、D=128、top-512 |
| mhc_pre、mhc_post、mhc_pre_sinkhorn、mhc | (T,N,D) | N=4；mhc 代表 Pre、Sinkhorn、Post 三段，外部层单独调度 |
| mhc_sinkhorn | (T,N) | N=4，`num_iters` 为实际迭代数 |
| kda_recurrent、kda_chunk | (B,S,Hq,Hv,K,V) | 分别为递推和分块路线，仅递推有完整时延校准 |
| grouped_matmul_swiglu_quant_v2、gmm2、moe_ffn | (M,K,F,E) | `expert_counts` 为 E 个本地专家各自收到的有效 token 数，总和为 M |

例如 `estimate("moe_ffn", (32, 256, 128, 2), {"expert_counts": [1, 31]})` 预测整个两段计算链。调度该节点后不要再把两个 GMM 组件的估计加到同一事件上。EP 大小不能替代本地专家直方图。推测 8 个 token 时，普通 FA 的主模型校验长度常为 Q=9，实际 B、H、打包方式仍由框架提供。

对 RMSNorm 和 mHC Post，默认采用调用上下文未知时的估计，不保存上次预测的设备状态。系统若维护着**同一类 kernel 连续执行时前序调用的最大活跃 AIV 数**，可用 `{"resident_aiv": 32}` 显式传入，取值 0…目标 Vector 核数，0 表示该隔离序列的首次调用。发生换流或并发干扰而无法确定时，省略该属性。该量用于描述 CA 观察到的启动上下文，硬件缓存状态仍待独立确认。

当调度器知道隔离序列刚开始时，令 `resident = 0`。以下三行只由同一流、同类 kernel 的实际调度事件驱动；同一个待选方案反复试算时保留原状态。返回的 `active_aiv` 是本次预计活跃核数，缺少此信息的路线返回 None。

```python
result = estimate("rmsnorm", (2, 6144), {"resident_aiv": resident})
if result.active_aiv is not None:
    resident = max(resident, result.active_aiv)
```

mHC Post 在少 token 且切通道时，默认估计在首次、扩核与热态之间平衡相对误差。已知连续调用处于热态时，传入 `resident_aiv` 可避免把额外的启动时间累积到每个事件。其余分核分支用历史条件时间的加权中位数。在线查询本身不改变设备历史。[不同状态的同集误差与大 shape 核查](../results/v0.12/audit/summary.md)
