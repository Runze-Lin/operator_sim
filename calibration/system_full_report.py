"""Render the 27-task comparison and the exhaustive, evidence-labelled operator inventory."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import gzip
import json

ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/"calibration/reference"
DOC=ROOT/"docs"
# Each label applies to a concrete route and input contract. Mixed names can have multiple labels.
REUSE={
    "quant_mxfp4":"原生 DynamicMxQuant 双端",
    "slice_then_quant_mxfp4":"Slice＋原生量化双端组件",
    "cache_load/aclnnAntiMxQuant":"原生 AntiMxQuant 双端",
    "fp8_cache_dequant":"AscendAntiQuantV2 双端代表窗", "mx4_cache_unpack":"异布局 MX 解包设计",
    "matmul/A4W4":"原生 A4W4 双端", "routed_grouped_matmul/aclnnGroupedMatmulV5":"原生 GMM V5 双端",
    "quant_token_fp8/aclnnDynamicQuantV2":"原生 DynamicQuantV2 双端",
    "mhc_post/vllm_hc_post":"vLLM 扩展双端代表窗",
    "rmsnorm":"原生 RmsNorm 双端", "weighted_rms_rows":"非标准权重/布局设计",
    "cast_bf16_fp32":"原生 Cast 双端", "sigmoid_fp32":"原生 FP32 Sigmoid 双端",
    "mhc_split_k_sinkhorn":"MhcPreSinkhorn 等价实现，宽通道验证",
    "matmul/transposed_weight_bf16":"原生转置权重 aclnnMm 双端", "matmul/transposed_weight_fp32":"原生严格 FP32 aclnnMm 双端",
    "addmm/transposed_weight_fp32":"原生 FP32 aclnnAddmm 双端",
    "matmul/router_bf16_to_fp32":"BF16→FP32 Router，独立库探针见矩阵例页",
    "matmul/token_fp8_mxfp4":"token FP32 scale＋MX 权重设计", "grouped_matmul/token_fp8_mxfp4":"分头 token scale＋MX 权重设计",
    "batch_matmul/last_axis_mxfp4_to_bf16":"异轴双 MX scale 展开设计",
    "candidate_filtered_indexer/full_scan/li_pipeline_projection":"Gather/LI/Sort 组件双端；全窗 N/A",
    "strided_view":"张量别名，设备时间为零","allocator_reservation":"空分配，设备时间为零",
    "add.Tensor":"原生 Add BF16 代表窗", "mul.Tensor":"原生 Mul FP32 广播代表窗",
    "moe_gating_top_k":"原生 MoeGatingTopK 双端代表窗","score_full_row_then_hash_gather":"Hash TopKV2 原生代表窗","sqrtsoftplus_bias_rank":"无 TID 的带偏置排名设计",
    "rope_interleaved_pair":"原生 partial RoPE 双端","rope_pair":"RoPE 核心组件校准",
    "swiglu":"SwiGlu 双端", "swiglu_clipped":"带裁剪设计，SwiGlu 组件双端",
    "system_attention/quant_lightning_indexer_reference":"BF16 普通 LI 组件；量化参考调用按图精度",
    "system_attention/lightning_indexer":"普通 LI 双端组件；候选生产侧输出为设计",
    "system_attention/kimi_dsa_indexer_score":"FP4 混排缓存，MX 原生接口边界见例页",
    "system_attention/npu_fused_infer_attention_score":"FIA V5 Q9 代表窗；历史+临时 KV 长形外推",
    "system_attention/causal_conv1d_update_reference":"原生 CausalConv1dUpdate 双端",
}
NETWORK={"ring_reduce_scatter_allgather","ring_allgather","ep_weighted_combine","ep_valid_row_dispatch"}
DESIGN={"prefix_bank_resident_mix","prefix_bank_two_pass_mix","clone.default","situ_gate_and_linear_branch",
    "quantize_scatter_current_rows","grouped_rms_sigmoid","mx4_query_and_head_weights","sigmoid_times_value",
    "layernorm_mx4_scatter_current_rows","engram_norm_dot_sigmoid_mix","engram_unpack_scales",
    "engram_sharded_row_gather","ngram_hash_modulo","system_attention/kimi_gated_dsa_attention",
    "system_attention/recurrent_kda_reference","system_attention/sparse_attn_sharedkv_reference",
    "system_attention/compressor","system_attention/compressor_metadata","system_attention/c2_ring_compressor",
    "system_attention/candidate_blocks","system_attention/compressor_source_positions","masked_fill.Scalar"}
MODELS={"v4-pro":"DeepSeek-V4 Pro","v41-flash":"DeepSeek-V4.1 Flash","kimi-k31":"Kimi K3.1 Pro"}


def load(name):
    with (REF/name).open() as f:
        return list(csv.DictReader(f))


def total(rows,key):
    return sum(float(x[key]) for x in rows)


def evidence(raw,method,primitive):
    if method in REUSE:
        return REUSE[method]
    if method in NETWORK:
        return "通信场景；互连整窗 N/A"
    if raw in primitive:
        return primitive[raw]+" 双端代表基元；异精度/布局外推"
    if method in DESIGN:
        return "按实现设计；匹配整窗 N/A"
    raise ValueError(f"Unclassified {raw}, {method}")


def main():
    rows=load("system-full-comparison.csv");ops=load("system-full-operators.csv")
    with gzip.open(REF/"system-full-shapes.csv.gz","rt") as f:
        samples=list(csv.DictReader(f))
    times=sorted(float(x["first_uncached_query_ms"]) for x in samples)
    original,hybrid,current=(total(rows,k)/1000 for k in ("original_us","hybrid_v18_us","full_us"))
    lines=["# 128Ki decode · 27 组完整算子后端对照","",
        "数据来自 3 个模型 × 3 种并行 × 每卡 B=1、16、64，均为 rank 0。target/draft 的调用、张量、依赖、任务配置和显存峰值逐对核对。128Ki 个历史 token，推测长度 8，每轮校验 9 个输入，统计口径为原任务的接受长度 7。这里的毫秒是 DES 预测的单轮时长。","",
        f"精细后端命中 **{len(ops)} 个调用名、{int(total(rows,'nodes')):,} 次调用**，含 {int(total(rows,'zero_duration')):,} 个零设备时间节点。每组只配置一个后端；校验器会拒绝任何一个未成功定价的节点。原始已计费预算覆盖率由 v1.8 的 67.63% 达到 100%。",
        "",f"27 组时长求和为原 Roofline {original:.3f} ms、v1.8 混合 {hybrid:.3f} ms、全量 {current:.3f} ms。求和用于跨配置比较，单个请求的时延要看所在行。"
        f"相对原方案变化 {(current/original-1)*100:+.2f}%，相对 v1.8 变化 {(current/hybrid-1)*100:+.2f}%。只有与标明精度和布局匹配的代表 CA 窗可用于评价算子误差；尚无这 27 组的整机时延真值。该表保留 v1.9.0 完整后端，后续结果见[当前同精度 Roofline 和端到端重放](roofline-audit.md)。",
        "", "| 模型（9 组求和） | 原 Roofline ms | v1.8 ms | 完整后端 ms | 相对原方案 |","|---|---:|---:|---:|---:|"]
    for key,label in MODELS.items():
        subset=[r for r in rows if r["model"]==key]
        a,b,c=[total(subset,k)/1000 for k in ("original_us","hybrid_v18_us","full_us")]
        lines.append(f"| {label} | {a:.3f} | {b:.3f} | {c:.3f} | {(c/a-1)*100:+.1f}% |")
    lines+=["","![每种模型在三种并行和 batch 下的预测耗时变化](figures/system-full-decode.png)","",
        "红色表示全量后端预测时间增加，蓝色表示减少。格内为相对原 Roofline 的百分比，星号表示显存峰值超出 144 GB。",
        "","## 逐配置单轮耗时","",
        "单轮含 target 和 draft。显存超过任务配置 144 GB 的行以 † 标记，仍按平台的 compare 模式算出设备工作时间。它们用于容量之外的对照；实际部署需要调整放置、并行或容量。用户原表中的 114.35、33.49 等数值对应原始静态显存峰值，单位 GB。","",
        "| 模型 | 并行 | B | 原 Roofline ms | v1.8 ms | 完整 ms | 变化 | target / draft ms | 峰值 GB |","|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        values=[float(r[k])/1000 for k in ("original_us","hybrid_v18_us","full_us","target_us","draft_us")]
        a,b,c,t,d=values; mark=" †" if r["fits_144gb"]=="False" else ""
        lines.append(f'| {MODELS[r["model"]]} | {r["layout"]} | {r["batch"]} | {a:.3f} | {b:.3f} | **{c:.3f}** | {float(r["delta_pct"]):+.1f}% | {t:.3f} / {d:.3f} | {float(r["peak_gb"]):.2f}{mark} |')
    lines+=["","## 差异来自哪些调用","",
        f"完整模型逐节点串行求和 {total(rows,'full_serial_us')/1000:.3f} ms，其中四类 TP/EP 通信算子预算 {total(rows,'comm_operator_us')/1000:.3f} ms。DES 允许部分工作重叠，串行和超过实际轮次和 {total(rows,'overlap_us')/1000:.3f} ms。以下按所有节点的串行预算归因，各行的差额不能直接当作单算子端到端加速。",
        "","| 主要调用 | 原 ms | 完整 ms | 差额 ms |","|---|---:|---:|---:|"]
    for op in sorted(ops,key=lambda x:-abs(float(x["delta_us"])))[:10]:
        lines.append(f'| `{op["raw_name"]}` | {float(op["roofline_us"])/1000:.2f} | {float(op["full_us"])/1000:.2f} | {float(op["delta_us"])/1000:+.2f} |')
    lines+=["","Flash 的最大差额来自 candidate_filtered_indexer。完整模型区分实际扫描的压缩 key 和只在候选集合上执行的后处理，并利用同一请求跨查询行的重叠。具体边界见[候选 Indexer 算例](examples/candidate-filtered-indexer.md)。Kimi 中 FP4 DSA 的低位解包、打分和筛选，V4 中多轮 MX 量化与真实库式启动，以及原价为零的通信，均能推高单轮估计。DEP64 的静态超容和 rank 之间的稀疏路由不平衡会影响部署表现；这里没有多 rank 机器运行结果。",
        "","## 调用成本和复算","",
        f"在每个任务/阶段/原始名/路线/输入输出物理几何的首次出现时，显式绕过系统预测结果缓存，再调用一次已初始化的公开后端，共 {len(times):,} 次。墙钟 P50 {times[len(times)//2]:.3f} ms、P95 {times[int(.95*len(times))]:.3f} ms、P99 {times[int(.99*len(times))]:.3f} ms、最大 {times[-1]:.3f} ms。保留库内部常驻参数与公式缓存，冷进程导入、图捕获和离线 CA 单列。每一组的完整任务墙钟见 CSV。",
        "","在 preview.1 系统虚拟环境中从本仓根目录运行以下命令。矩阵目录需包含原 task.toml 和 results/l1；scratch 单独保存两套重放的中间输出。","",
        '```sh','python -m calibration.system_full_compare "$MATRIX" "$SCRATCH" roofline --jobs 3',
        'python -m calibration.system_full_compare "$MATRIX" "$SCRATCH" full --jobs 3',
        'python -m calibration.system_full_compare "$MATRIX" "$SCRATCH" compare','python -m calibration.system_full_report','```',
        "","[80 个调用名及证据](system-full-evidence.md) · [逐物理几何账本](../calibration/reference/system-full-shapes.csv.gz) · [逐配置 CSV](../calibration/reference/system-full-comparison.csv) · [逐调用 CSV](../calibration/reference/system-full-operators.csv) · [通信公式和场景假设](examples/system-communication.md)"]
    (DOC/"system-full-comparison.md").write_text("\n".join(lines)+"\n")
    primitive={r["raw_name"]:r["api"] for r in csv.DictReader((ROOT/"calibration/system_vector_ca/primitives.csv").open())}
    grouped=defaultdict(list)
    for v in samples: grouped[v["raw_name"]].append(v)
    inventory=["# 80 个捕获调用的精度、覆盖和设备证据","",
        "按完整后端的累计串行预算排序。每行覆盖该名称全部实例；“代表窗”只覆盖列明的实现与实验范围。双端指 950DT 和 950PR。全量 27 组用任务的 950DT-system 资源，970DT 只作为算子级外推。缺匹配整窗的设计没有 CA 误差。",
        "","[Vector 每个调用的例子与误差](examples/system-vector-routes.md) · [矩阵逐 shape 双端误差](examples/system-cube-routes.md) · [注意力与状态更新逐 shape 误差](examples/system-attention-routes.md) · [独立算子的逐窗误差](errors/README.md) · [通信算例](examples/system-communication.md)",
        "","| 原始调用 | 次数 | 输入→输出（第一个实图例） | 模型路径与 CA 边界 | 最大查询 ms |","|---|---:|---|---|---:|"]
    def tensors(raw):
        ts=json.loads(raw.replace("torch.",""))
        return ", ".join(t[0].replace("bfloat16","BF16").replace("float8_e4m3fn","FP8").replace("float4_e2m1fn_x2","FP4×2").replace("float8_e8m0fnu","E8M0")+"["+",".join(map(str,t[1]))+"]" for t in ts) or "∅"
    from operator_model.system_routes import STRUCTURAL,attention,communication,cube,vector
    families=(("Vector、量化和状态",vector.NAMES),("矩阵和分组乘",cube.NAMES),
              ("注意力、压缩和时序",attention.NAMES),
              ("Flash 候选过滤",{"inference::candidate_filtered_indexer"}),
              ("TP/EP 通信",communication.NAMES),("视图和空分配",STRUCTURAL))
    coverage=["| 全图调用组 | 原始名数 | 节点数 | 原串行 ms | 完整串行 ms |","|---|---:|---:|---:|---:|"]
    for label,names in families:
        subset=[row for row in ops if row["raw_name"] in names]
        coverage.append(f"| {label} | {len(subset)} | {int(total(subset,'nodes')):,} | {total(subset,'roofline_us')/1000:.2f} | {total(subset,'full_us')/1000:.2f} |")
    inventory=inventory[:-2]+coverage+[""]+inventory[-2:]
    for op in ops:
        name=op["raw_name"];g=grouped[name];rep=g[0]
        notes="；".join(dict.fromkeys(evidence(name,m,primitive) for m in json.loads(op["methods"])))
        inventory.append(f'| `{name}` | {int(op["nodes"]):,} | {tensors(rep["physical_inputs"])} → {tensors(rep["physical_outputs"])} | {notes} | {max(float(r["first_uncached_query_ms"]) for r in g):.3f} |')
    inventory+=["","所有输入及输出示例为本卡物理存储形状，FP4×2 的一个字节存两个数；kernel 选择还读取逻辑宽度、stride、量化 scale 和属性。全量 CSV 保留每个任务和阶段、具体方法、相同几何不同属性对应的设备时间范围。它可以用 pandas.read_csv 直接解压读取。[逐几何明细](../calibration/reference/system-full-shapes.csv.gz) · [调用次数与路线分布](../calibration/reference/system-full-operators.csv)"]
    (DOC/"system-full-evidence.md").write_text("\n".join(inventory)+"\n")


if __name__=="__main__":
    main()
