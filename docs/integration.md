# 系统仿真接入

在系统分析框架中，每个待调度的计算节点提供一张卡上的实际输入。算子耗时包括模型覆盖的设备流水和核间尾部，系统层负责宿主发起、通信、跨卡同步和跨流资源争用。原先给 Roofline 传全局张量的接入点，需要先应用 TP/EP 分片、batch 打包和 chunk 调度。

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
| `work` | `cube_ops` 和 `vec_ops` 为有用标量运算计数，另按 dtype、Vector 运算类型细分。`logical_read_bytes`、`logical_write_bytes` 按张量边界计数，实际 HBM 流量须另有证据。 |
| `rates` | 工作量除以设备时间所得的 TOP/s 和逻辑 GB/s。名义峰值比例与芯片配置有关。 |
| `projection` | 路线可提供分块方案、波数、阶段耗时、流水瓶颈及继承参数。诊断键随算子变化，通常调度只依赖上面的公共字段。 |

## 给出正确的本卡 shape

| 算子 | 维度顺序 | 输入约束 |
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
| [vLLM HcPost](examples/hc_post_vllm.md) | (B,S,N,D) | 调用 mhc_post 并传 implementation="vllm_hc_post"，使用与内置 Post 不同的 kernel 与布局。 |

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
