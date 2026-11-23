# 系统仿真接入

在系统分析框架中，每个待调度的计算节点提供一张卡上的实际输入。算子耗时包括模型覆盖的设备流水和核间尾部，系统层负责宿主发起、通信、跨卡同步和跨流资源争用。原先给 Roofline 传全局张量的接入点，需要先应用 TP/EP 分片、batch 打包和 chunk 调度。

SIMPlatform 通过任务的 `prediction.backends` 选择精细模型和原 Roofline 的优先顺序。适配器直接读取捕获图给出的本卡张量。[SIMPlatform 接入及端到端对比](sim-platform.md)

## 一次查询

```python
from operator_model import estimate, load_chip

chip = load_chip("Ascend950PR_957b")
r = estimate("rmsnorm", (64, 2048), dtype="bf16", chip=chip)
if r.latency_us is not None:
    event_duration_s = r.latency_us * 1e-6
```

`estimate(op, shape, attrs=None, *, chip=None, dtype=None, timing_mode="auto")` 返回 `Estimate`。默认芯片为 `Ascend950DT_9582`。dtype 是主输入的存储格式；例如 BF16 RMSNorm 的归约使用 FP32。可省略 dtype 以使用该算子的默认格式，也可从 attrs 传入，两处同时指定须相符。CLI 通过 `--attrs-json` 接受同一组属性，输出 JSON 使用 `predicted_us` 和 `estimate_status`。

| 状态 | 调用方的处理 |
|---|---|
| `estimated` | 模型返回所选实现与设备下的设备时间。查误差表可了解相近 shape 的实测覆盖。 |
| `extrapolated` | 返回跨芯片、跨覆盖范围或实验性实现的时间。可由系统策略决定是否纳入情景，并记录原因。 |
| `unavailable` | 能识别该工作，但缺少所需时间或资源证据。可读取已算出的工作量，时间为空。 |
| `unsupported` | 输入、属性、类型或所选计时路线不受支持。按 reason 修正输入或进入框架自己的回退路线。 |

`auto` 依算子和目标配置选择发布模型，适合系统集成。对支持目标资源调度的算子，`resource` 显式请求按目标芯片定价，适合做核数、带宽和频率的敏感性分析。`calibrated` 请求已登记的专门路线，具体覆盖随实现而异。所有模式都应检查状态和可空时间。复合节点的设备窗若已包括内部步骤，同一计算图应只计一次。

| 返回字段 | 单位与含义 |
|---|---|
| `latency_us`、`chip_name`、`status`、`reason` | 微秒、解析后的芯片与有效性说明。 |
| `active_aic`、`active_aiv` | 活跃 Cube/Vector 核数估计，缺少证据可为 None。 |
| `kernel_count`、`dispatch_overhead_us`、`scheduled_us` | 当前实现的物理发射次数、配置产生的总附加微秒、串行系统排程微秒。缺设备时间时三者为空。已单独模拟发射或异步队列的框架使用 `latency_us`。 |
| `work` | `cube_ops` 和 `vec_ops` 为有用标量运算计数，另按 dtype、Vector 运算类型细分。`logical_read_bytes`、`logical_write_bytes` 按张量边界计数，实际 HBM 流量须另有证据。 |
| `rates` | 工作量除以设备时间所得的 TOP/s 和逻辑 GB/s。`rates["roofline"]` 给出纯 Roofline 的各项理想 µs、控制项和等效利用率（比例）。[公式和分母](error-analysis.md#纯-roofline-要用多少利用率) |
| `projection` | 路线可提供分块方案、波数、阶段耗时、流水瓶颈及继承参数。诊断键随算子变化，通常调度只依赖上面的公共字段。 |

## 给出正确的本卡 shape

| 算子 | 维度顺序 | 输入约束 | 主输入、计算与输出精度 |
|---|---|---|---|
| [quant](examples/quant.md) | (R,C) | dynamicMXquant，BF16 输入、block32 | BF16 → E4M3FN + E8M0 scale |
| [moe_gating_top_k](examples/moe_gating_top_k.md) | (T,E,K) | FP32 logits，单分组 sigmoid 加 bias，挑选 K 个专家 | FP32 → FP32 权重、INT32 索引 |
| [sigmoid_fp32](examples/sigmoid_fp32.md) | (T,H_local) | 独立连续 FP32 Sigmoid；Kimi KDA 的 beta 宽是本地 head 数，TP8 时为 12 | FP32 → FP32 |
| [cast_bf16_fp32](examples/cast_bf16_fp32.md) | (T,D) | 连续 BF16 输入、FP32 输出 | BF16 → FP32 |
| [attn_res_fwd](examples/attn_res_fwd.md) | (T,H,S,N) | Kimi BF16 prefix/addend、S 个 bank slot、N 个有效历史行 | BF16，评分/归一化按 FP32 计；输出 BF16 |
| [rmsnorm](examples/rmsnorm.md) | (R,C) | BF16，输出含 rstd | BF16，FP32 归约 → BF16 Y、FP32 rstd |
| [matmul](examples/matmul.md)、[addmm](examples/addmm.md) | (M,N,K) | BF16；Addmm 的 bias=[N] 且 α=β=1 | BF16 A/B（Addmm 含 BF16 bias）→ BF16 |
| [Matmul，原生 A4W4](examples/mxfp4-design.md) | (M,N,K) | 传 `dtype="mxfp4"`，稠密 ND，W 已按 [N,K] 打包，沿 K 每 32 值共享 E8M0，目标为无 bias 的 BF16 输出 | A/W 均为 MXFP4 E2M1，FP32 累加 → BF16 |
| [Matmul，权重解包 A8W4](examples/mxfp4-design.md) | (M,N,K) | 与上行相同，传 `dtype="fp8_e4m3fn"` 和 `attrs={"weight_dtype":"mxfp4"}` | MXFP8 E4M3FN × MXFP4 E2M1，权重在核内展开，FP32 累加 → BF16 |
| [fa](examples/fa.md) | (Q,KV,D,B,H) | 普通非因果 SBH；Q 是本次校验长度，H 是本卡头数 | BF16 Q/K/V，FP32 softmax → BF16、FP32 统计 |
| [lightning_indexer](examples/lightning_indexer.md) | (B,Q,L,G,D) | BSND，G=8、D=128、top-512 | BF16 Q/K/权重 → INT32 |
| [mhc_pre](examples/mhc_pre.md)、[mhc_post](examples/mhc_post.md)、[mhc_pre_sinkhorn](examples/mhc_pre_sinkhorn.md)、[mhc](examples/mhc.md) | (T,N,D) | N=4；mhc 代表 Pre、Sinkhorn、Post 三段，外部层单独调度 | BF16 激活、FP32 参数；Post 输出 BF16，Pre 还返回 FP32 系数 |
| [mhc_sinkhorn](examples/mhc_sinkhorn.md) | (T,N) | N=4，`num_iters` 为实际迭代数 | FP32 → FP32 |
| [kda_recurrent](examples/kda_recurrent.md)、[kda_chunk](examples/kda_chunk.md) | (B,S,Hq,Hv,K,V) | 递推有 CA 校准；分块默认 V1 仅工作量，显式 FLA V2 可估计三个设备窗 | recurrent / FLA V2 的 Q/K/V 为 BF16、状态 FP32；V1 接受 BF16/FP16 结构输入，计时 N/A |
| [grouped_matmul_swiglu_quant_v2](examples/grouped_matmul_swiglu_quant_v2.md)、[gmm2](examples/gmm2.md)、[moe_ffn](examples/moe_ffn.md) | (M,K,F,E) | `expert_counts` 为 E 个本地专家各自收到的有效 token 数，总和为 M | MXFP8 E4M3FN × E4M3FN、E8M0 scale；GMM1 输出 MXFP8，GMM2/整链输出 BF16 |
| [vLLM HcPost](examples/hc_post_vllm.md) | (B,S,N,D) | 调用 mhc_post 并传 implementation="vllm_hc_post"，使用与内置 Post 不同的 kernel 与布局。 | BF16 激活、FP32 系数 → BF16 |
| [quant_mxfp4](examples/quant-mxfp4.md) | (...,D) | 末轴 block32，round，E8M0 成对存放。省略 `input_stride` 时按连续布局定价，可传真实元素步长，非连续视图仅返回工作量 | BF16 → packed MXFP4 E2M1、E8M0 |
| [quant_token_fp8](examples/quant-token-fp8.md) | (...,D) | 对称逐行量化，全行驻留 UB；连续输入，无 offset/smoothing | BF16 → E4M3FN、每行一个 FP32 scale |
| [cache_load](examples/cache_load.md) | (...,D)，物化输出的逻辑维度 | 传实际存储 dtype；可校验物理 value、scale 形状。连续 FP4 显式选 `implementation="aclnnAntiMxQuant"` 可计时。FP8 逐行参考可查离线 CA；默认仅工作量 | MXFP4/E8M0 或 FP8 E4M3FN/每行 FP32 → BF16 |
| [routed_grouped_matmul](examples/routed-grouped-matmul.md) | (有效本地行 R, N, K) | 路由专家的单个投影。传 `expert_counts`，可提供 `dispatch_capacity_rows`。匹配转置权重和 scale 时传 `implementation="aclnnGroupedMatmulV5"` 计时，默认仅工作量 | MXFP4 双侧 block32、成对 E8M0 → BF16 |
| [candidate_filtered_indexer](examples/candidate-filtered-indexer.md) | (B,Q,H,D,L,C,K) | 默认全表打分；候选优先须传块大小，可传每查询有效 key 数。返回工作量，耗时 N/A | BF16 查询、key、头权重，INT64 位置、INT32 候选与输出 |

MoE 的 MXFP8 接口传 `dtype="fp8_e4m3fn"`，两侧的 E8M0 scale 和输出格式由该实现固定。芯片配置里的 4-bit/8-bit 峰值表示资源能力。本表列出在线模型实际接入的组合。精度盘点另列出 FA 的 Q/KV cache、量化/反量化以及 MoE 权重格式对应的已安装实现和 CA 证据。[低精度路线清单](precision-coverage.md)

例如 TP 切 head 后传入本卡 H；普通 decode 每请求 1 行，MTP 推 8 个 token、主模型同时校验时每请求可有 9 行。对于 MoE，EP 大小给出专家所在的卡，实际 token 路由决定本地的 expert_counts。可按模型配置和指定并行方式导出 prefill 几何，再与框架实际调度的张量对照。[三模型示例](../examples/workloads/README.md)

## 给有序流保留历史

```python
from operator_model import OperatorStream

stream = OperatorStream("Ascend950PR_957b")
shape = (65, 4, 5120)
candidate = stream.preview("mhc_post", shape)  # 评估候选调度
first = stream.advance("mhc_post", shape)      # 实际发起
repeat = stream.advance("mhc_post", shape)     # 同一流的紧邻复跑
```

新流从已知的首次发起开始。中途接管正在运行的流或遇到无法跟踪的派发时，调用 `forget_history()`；确定从空流重新开始时调用 `reset()`。没有顺序信息时使用无状态 `estimate`，内置 mHC Post 的同形冷热差异会使该入口出现较大误差。[系统时钟示例](../examples/system_sim.py) · [mHC Post 算例](examples/mhc_post.md)

## 配置目标芯片

`chip` 可接受内置名称、JSON 路径、同结构字典或 `load_chip` 返回的配置。在同一仿真进程复用已解析的芯片。必填规格包括调度域的频率（GHz）、AIC/AIV 数、Cube/Vector 按 32/16/8/4 位的整域 TOP/s（FMA 计两次，未知峰值可填 null）、片外容量（十进制 GB）和总带宽（GB/s）。核数和峰值须来自同一调度域。

细项可包括 UB、L1、L2 容量（字节）、L2 总带宽以及每核 MTE 端口吞吐。省略的细项按配置加载规则继承 950DT 的记录值，`inherited_fields` 可查看清单；算子级成本另有各自的参数来源。在 `sources` 写明给定规格、推导及继承假设。970DT 是独立情景，所有 970DT 耗时都属于外推。[内置配置](../src/operator_model/data/chips/) · [自定义配置](../examples/chip.json)

```sh
operator-sim --op fa --shape 9 32768 64 1 1 --dtype bf16 --chip examples/chip.json
```

## 每次发射的额外开销和片上容量

这四项都可省略。容量位于 JSON 的 `details`，`kernel_overhead_us` 位于顶层。以下为 Python 等价示例，保留示例中指定 SKU 的其余参数。在自己的完整 JSON 文件中填同名字段，即可用 `--chip 路径.json` 调用。

```python
from dataclasses import replace
from operator_model import estimate, load_chip

base = load_chip("Ascend950DT_9582")
# 这里保留 DT 自身容量，只演示一次发射额外暴露 1.5 µs 的系统假设。
chip = replace(base, kernel_overhead_us=1.5)
r = estimate("matmul", (1280, 2048, 512), dtype="bf16", chip=chip)
print(r.kernel_count, r.latency_us, r.dispatch_overhead_us, r.scheduled_us)
# 1, 12.4491…, 1.5, 13.9491… µs
```

| 配置字段 | 单位和作用范围 | 缺省与依据 |
|---|---|---|
| `kernel_overhead_us` | 每个实际发射的 kernel，在 CA 指令设备窗之外、串行暴露的固定微秒数。可设为零 | 默认 0。由系统的 dispatch 实测或情景假设给定。CA 核内启动、首读和收尾继续由算子模型计价。 |
| `details.ub_bytes` | 字节 / AIV，用于 Quant、RMSNorm、Cast 等的核内切块或驻留判断 | DT 253,952 B（248 KiB） |
| `details.l1_bytes` | 字节 / AIC，用于支持该参数的 Matmul 页深和操作数常驻判断 | DT 524,288 B（512 KiB） |
| `details.l2_bytes` | 字节 / 芯片调度域，用于工作集驻留、输出回写及共享传输估计 | DT 134,217,728 B（128 MiB），PR 内置配置为 112 MiB |

同一调度域上，配置的附加时间为 `kernel_count × kernel_overhead_us`。已计时实现中，普通 Matmul、融合 PreSinkhorn 各发射 1 次，MoE FFN 发射 2 次，拆分 mHC 与 FLA V2 的三个主体 kernel 各计 3 次。tile、波次和参与的核数不增加发射次数。设置附加时间不会改变设备工作量、峰值利用率、校准资格或 CA 误差。设备窗内如需调整可辨识的启动阶段，Matmul、Quant 部分资源路线另有 `details.matmul_launch_cycles`、`details.quant_launch_cycles` 等周期价格，各分支的首发或预取参数见对应算例。

例如，下面是一份芯片 JSON 中需要合并的字段（其余必填规格沿用该文件原值）。

```json
{
  "kernel_overhead_us": 1.5,
  "details": {
    "ub_bytes": 253952,
    "l1_bytes": 524288,
    "l2_bytes": 134217728
  }
}
```

使用完整自定义芯片 JSON 时，只需在原 `details` 中修改希望改变的容量并保留其他已知项。新 JSON 省略的细项由加载器继承 950DT；从 PR 配置做敏感性分析时复制 PR 原始配置及其细项，以保留 PR 自身的 L2。固定实现（例如当前稠密 FA）沿用其已有块策略，尚无证据根据任意 UB/L1 容量重排所有算子。[配置示例](../examples/chip.json) · [资源变化的数值算例](modeling-strategy.md#容量如何改变一次预测)
