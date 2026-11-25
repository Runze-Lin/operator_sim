"""Rank-local Vector schedules for the published inference captures.

Each route chooses operands, storage boundary, arithmetic and an independent
work unit before the scheduler is given a chip. Packed shapes use byte slots.
"""
from dataclasses import dataclass, replace
from math import ceil, log2, prod

from . import vf
from operator_model.backend import estimate
from operator_model.chips import load_chip
from operator_model.operators.attention.residual_work import attention_residual_work
from operator_model.operators.quant.mxfp4 import contiguous_input
from operator_model.system_adapter import NodeEstimate
from operator_model.workload import _counts


_POINT = {
    "aten.add.Tensor": 1, "aten.mul.Tensor": 1, "aten.sub.Tensor": 1,
    "aten.pow.Tensor_Scalar": 1, "aten.neg.default": 1, "aten.rsqrt.default": 1,
    "aten.sigmoid.default": 4, "aten.lt.Scalar": 1, "aten.ge.Scalar": 1,
    "aten.bitwise_or.Tensor": 1, "aten.masked_fill_.Scalar": 1, "aten.masked_fill.Scalar": 1,
    "aten._to_copy.default": 1, "aten.clone.default": 0, "aten.cat.default": 0,
    "aten.constant_pad_nd.default": 0,
}
NAMES = frozenset(_POINT) | {
    "aten.embedding.default", "aten.index.Tensor", "aten.mean.dim", "aten.sum.dim_IntList",
    "inference::cache_load", "inference::cache_store", "inference::state_update",
    "inference::attention_residual", "inference::grouped_gated_rms_norm_reference",
    "inference::situ", "inference::mla_output_gate", "npu::npu_swiglu",
    "npu::npu_rotary_mul", "_C_ascend::inplace_partial_rotary_mul",
    "_C_ascend::moe_gating_top_k", "_C_ascend::moe_gating_top_k_hash",
    "_C_ascend::npu_hc_pre_v2", "_C_ascend::npu_hc_post", "npu::npu_rms_norm",
    "inference::kimi_dsa_indexer_query_quant", "npu::npu_dynamic_mx_quant",
    "inference::dynamic_mx_quant_fp8_reference", "inference::engram_hash",
    "inference::engram_lookup", "inference::engram_dequantize", "inference::engram_gate",
}
_WIDTH = {"float32": 4, "float64": 8, "float16": 2, "bfloat16": 2, "int64": 8,
          "int32": 4, "uint32": 4, "int16": 2, "int8": 1, "uint8": 1, "bool": 1,
          "float4_e2m1fn_x2": 1, "float8_e4m3fn": 1, "float8_e8m0fnu": 1}


def _size(tensor, *, distinct=False):
    shape = tensor.shape
    if distinct:
        shape = tuple(1 if size and stride == 0 else size
                      for size, stride in zip(shape, tensor.stride))
    return prod(shape) * _WIDTH[tensor.dtype.removeprefix("torch.")]


@dataclass(frozen=True)
class Tiles:
    algorithm: str
    elements: int
    grain: int              # independent contiguous output segments, or one token/group row
    read: int
    write: int
    ops: int
    precision: str = "fp32"
    setup: str = "stream"
    transaction_read: int = 0     # extra overfetch for indirect accesses
    traffic_read: int = 0         # repeated DMA beyond distinct tensor bytes
    register_bits: int = 0
    class_visits: dict = None
    known_precision: bool = False
    work_source: str = "capture semantics; proposed arch35 register/UB implementation"


# Fixed non-payload service in cycles, shared across chips. The reference
# subtraction and geometry of each anchor are explained in the worked example.
_SETUP = {"stream": 4400, "broadcast": 6100, "dequant": 7380, "swiglu": 5560,
          "rope": 7260, "slice": 4500}
_ANCHOR = {"stream": "aclnnAdd BF16 9x7168", "broadcast": "aclnnMul FP32 9x4",
           "dequant": "aclnnAscendAntiQuantV2 FP8 1x128x512",
           "swiglu": "aclnnSwiGlu BF16 9x576", "rope": "aclnnInplacePartialRotaryMul BF16 9x1x8x512",
           "slice": "aclnnSlice BF16 strided 9x128"}
_PRIMITIVE_API = {
    "aten.lt.Scalar":"aclnnLtScalar", "aten.ge.Scalar":"aclnnGeScalar",
    "aten.bitwise_or.Tensor":"aclnnBitwiseOrTensor", "aten.sub.Tensor":"aclnnSubs",
    "aten.masked_fill.Scalar":"aclnnInplaceMaskedFillScalar", "aten.masked_fill_.Scalar":"aclnnInplaceMaskedFillScalar",
    "aten.neg.default":"aclnnNeg", "aten.pow.Tensor_Scalar":"aclnnPowTensorScalar",
    "aten.sum.dim_IntList":"aclnnReduceSum", "aten.mean.dim":"aclnnMean", "aten.rsqrt.default":"aclnnRsqrt",
    "aten.embedding.default":"aclnnEmbedding", "aten.index.Tensor":"aclnnGatherV2",
    "aten._to_copy.default":"aclnnCast", "aten.cat.default":"aclnnCat", "aten.constant_pad_nd.default":"aclnnConstantPadNd",
}
# First launch/last retire residual after removing the modeled payload on 9 rows (DT, PR cycles).
_PRIMITIVE_CONTROL = {
    "aclnnAddRmsNorm": (6506.0, 6327.0),
    "aclnnMoeGatingTopKV2/hash": (8632.873797, 8021.299883),
    "aclnnLtScalar": (4818.159516, 4353.157603),
    "aclnnGeScalar": (2306.159516, 2123.157603),
    "aclnnBitwiseOrTensor": (1621.276142, 1590.274229),
    "aclnnSubs": (1909.190766, 1788.188853),
    "aclnnInplaceMaskedFillScalar": (2202.004662, 2010.002749),
    "aclnnNeg": (3372.925860, 3142.913877),
    "aclnnPowTensorScalar": (2438.851720, 2128.729285),
    "aclnnReduceSum": (4678.797120, 4215.676598),
    "aclnnMean": (3405.726720, 3356.604285),
    "aclnnRsqrt": (1913.851720, 1779.729285),
    "aclnnEmbedding": (4317.439490, 3710.437790),
    "aclnnGatherV2": (3201.439490, 2871.437790),
    "aclnnCast": (1795.673060, 1780.550625),
    "aclnnCat": (3228.357320, 3565.357320),
    "aclnnConstantPadNd": (4882.627792, 4608.228571),
}
_DIV = {"fp32": "32", "bf16": "16"}


def _schedule(p, chip):
    work = _counts(p.precision, p.read, p.write, vec={p.precision: p.ops} if p.ops else {})
    n = p.elements
    if n == 0:
        work["schedule"] = {"algorithm": p.algorithm, "active_aiv": 0, "waves": 0,
                            "evidence": "empty logical update; no work scheduled"}
        return NodeEstimate(p.algorithm, "extrapolated", 0., work=work)
    grain = max(1, min(n, p.grain))
    active = min(chip.vector_cores, ceil(n / grain))
    per_core = ceil(ceil(n / grain) / active) * grain
    payload_read = p.read + p.transaction_read + p.traffic_read
    # A single row/group may span multiple UB panels; the partial accumulator
    # stays in UB. Ping-pong pages bound both live data and an FP32 temporary.
    bytes_per_item = (payload_read + p.write) / n + 4
    ub = chip.details.get("ub_bytes") or 253952
    panel = max(1, int(ub / (2 * bytes_per_item)) // 16 * 16)
    waves = ceil(per_core / panel)
    peak = chip.vector_tops[_DIV[p.precision]]
    if p.ops and not peak:
        return NodeEstimate(p.algorithm, "unavailable", reason="Missing target Vector peak")
    clock = 1000 * chip.frequency_ghz
    # Peak FMA has two arithmetic ops; model ordinary SIMD lanes as one per
    # cycle. Nonlinear schedules expose their arithmetic sequence in the route.
    lanes = (peak or 1) * 1000 / (chip.vector_cores * chip.frequency_ghz * 2)
    rd = (chip.details.get("mte2_bandwidth_gbs") or 48.36*chip.frequency_ghz) / chip.frequency_ghz
    wr = (chip.details.get("mte3_bandwidth_gbs") or 32*chip.frequency_ghz) / chip.frequency_ghz
    item = min(panel, per_core)
    read = payload_read/n/rd
    vec = vf.panel(chip,p,item,item*p.ops/n/lanes)/item
    write = p.write/n/wr
    # Overlap the three resources between pages. Equal full pages have a
    # max-stage initiation interval. A proportionate tail is an approximation.
    pipe_cycles = item*(read+vec+write) + (per_core-item)*max(read,vec,write)
    hbm_cycles = (payload_read + p.write) * chip.frequency_ghz/chip.hbm_bandwidth_gbs
    setup = _SETUP.get(p.setup)
    anchor = _ANCHOR.get(p.setup)
    if setup is None:
        dt, pr = _PRIMITIVE_CONTROL[p.setup]
        setup = pr if chip.sources.get("timing_template",chip.name) == "Ascend950PR_957b" else dt
        anchor = p.setup + " 9-row CA residual; other chip inherits DT control cycles"
    us = (setup + max(pipe_cycles, hbm_cycles)) / clock
    work["schedule"] = {
        "algorithm": p.algorithm, "evidence": p.work_source, "control_anchor": anchor,
        "setup_cycles": setup, "active_aiv": active, "tile_elements": item,
        "waves": waves, "partition_grain": grain, "critical_elements": per_core,
        "lane_ops_per_cycle": lanes, "mte2_bytes_per_cycle": rd, "mte3_bytes_per_cycle": wr,
        "first_panel_read_cycles": item*read, "first_panel_vector_cycles": item*vec,
        "first_panel_write_cycles": item*write, "core_pipeline_cycles": pipe_cycles,
        "shared_payload_floor_cycles": hbm_cycles, "extra_read_transactions_bytes": p.transaction_read,
        "repeated_dma_read_bytes": p.traffic_read,
    }
    return NodeEstimate(p.algorithm, "extrapolated", us, reason=p.work_source, work=work)


def _anchor(op, shape, chip, attrs=None, *, qualifier):
    result = estimate(op, shape, vf.native_attrs(attrs), chip=chip)
    if result.latency_us is None:
        return None
    work = dict(result.work)
    work["schedule"] = {"algorithm": op, "active_aiv": result.active_aiv,
                        "active_aic": result.active_aic, "evidence": qualifier,
                        "vector_model": (result.projection or {}).get("vector_model") or vf.selected(True),
                        "target_plan": (result.projection or {}).get("target_plan", {})}
    return NodeEstimate(op, "extrapolated", result.latency_us, reason=qualifier, work=work)


# Input/output storage types of the archived complete system-window fixtures.
# A new dtype under the same ATen spelling is a different implementation prior.
_KNOWN_INPUT = {
    'aclnnLtScalar':'int64','aclnnGeScalar':'int64','aclnnBitwiseOrTensor':'bool',
    'aclnnSubs':'int64','aclnnInplaceMaskedFillScalar':'int64','aclnnNeg':'bfloat16',
    'aclnnPowTensorScalar':'float32','aclnnReduceSum':'float32','aclnnMean':'float32',
    'aclnnRsqrt':'float32','aclnnEmbedding':'bfloat16','aclnnGatherV2':'bfloat16',
    'aclnnCast':'bfloat16','aclnnCat':'bfloat16','aclnnConstantPadNd':'bfloat16',
    'aten.add.Tensor':'bfloat16','aten.mul.Tensor':'float32'}


def _primitive(f):
    p=_primitive_plan(f)
    api=_PRIMITIVE_API.get(f.raw_name,f.raw_name)
    ref=_KNOWN_INPUT.get(api); src=f.inputs[0].dtype.removeprefix('torch.')
    dst=f.outputs[0].dtype.removeprefix('torch.')
    ref_dst={'aclnnCast':'float32','aclnnLtScalar':'bool','aclnnGeScalar':'bool'}.get(api,ref)
    return replace(p,known_precision=src==ref and dst==ref_dst)


def _primitive_plan(f):
    name = f.raw_name
    inputs, out = f.inputs, f.outputs[0]
    n = prod(out.shape)
    read = sum(_size(t, distinct=True) for t in inputs)
    write = _size(out)
    precision = "bf16" if out.dtype.endswith("bfloat16") else "fp32"
    setup = _PRIMITIVE_API.get(name, "stream")
    if name in {"aten.index.Tensor", "aten.embedding.default"}:
        ids = sum(_size(t) for t in inputs[1:])
        width = out.shape[-1] if len(out.shape)>1 else 1
        rows = n//max(1,width)
        row_bytes = write//max(1,rows)
        overfetch = rows*((-row_bytes) % 32)
        return Tiles("selected_row_gather", n, min(width*8,1024), ids+write, write, rows,
                     setup=setup, transaction_read=overfetch)
    if name in {"aten.mean.dim", "aten.sum.dim_IntList"}:
        terms = prod(inputs[0].shape)
        operations = terms-n + (n if name=="aten.mean.dim" else 0)
        return Tiles("strided_group_reduce", n, 256, read, write, operations, setup=setup)
    ops = _POINT[name]*n
    broadcast = name=="aten.mul.Tensor" and len(inputs)>1 and _size(inputs[1],distinct=True)<write
    grain = 2048 if broadcast else 1792   # 9×7168 Add uses 36 payload AIV
    if name == "aten.constant_pad_nd.default":
        grain = max(1,32//_WIDTH[out.dtype.removeprefix("torch.")])
    return Tiles(name.removeprefix("aten."), n, grain, read, write, ops, precision,
                 "broadcast" if broadcast else setup, register_bits=8*_WIDTH[out.dtype.removeprefix("torch.") if name not in ("aten.lt.Scalar","aten.ge.Scalar") else inputs[0].dtype.removeprefix("torch.")])


def _activation(f, sem, n, read, write):
    name = f.raw_name
    if name.endswith("grouped_gated_rms_norm_reference"):
        width = f.inputs[0].shape[-1]
        groups = sem.get("num_groups") or 1
        group_len = sem.get("group_size") or width//groups
        # square, sum, mean/eps/rsqrt, normalize, gamma, sigmoid and gate.
        return Tiles("grouped_rms_sigmoid", n, group_len, read, write, 9*n+3*ceil(n/group_len))
    if name.endswith("mla_output_gate"):
        return Tiles("sigmoid_times_value", n, 1024, read, write, 5*n)
    if name.endswith("npu_swiglu"):
        act = sem.get("activation", ())
        if isinstance(act, tuple) and len(act)==2 and act[0]=="mapping":
            act=act[1]
        limit = dict(act).get("limit") if isinstance(act,(tuple,dict)) else None
        # exp, sign, add, reciprocal, gate multiply and up multiply. The
        # optional gate ceiling and two-sided up clamp add three comparisons.
        return Tiles("swiglu_clipped" if limit is not None else "swiglu", n, 4096, read, write,
                     (6+3*int(limit is not None))*n, setup="swiglu", known_precision=limit is None and f.inputs[0].dtype.endswith("bfloat16"))
    return Tiles("situ_gate_and_linear_branch", n, 2048, read, write, 10*n, setup="swiglu")


def _project_hc_pre(shape, sem, chip, read, write):
    # The shipping smoke schedule restricts D to 4096. Keep the same 950
    # split-K mechanism and price the actual K, including partial M panels.
    from operator_model.operators.mhc.pre_sinkhorn import _cost
    from operator_model.operators.mhc.sink_resources import partition
    t, n, d = shape
    iters = sem["sinkhorn_iterations"]
    if n != 4 or iters != 20:
        raise ValueError("PreSinkhorn requires four streams and 20 iterations")
    ref = load_chip()
    plan = partition(shape, chip)
    model = estimate("mhc_pre_sinkhorn", (1,4,4096),
                     {"need_backward":False,"num_iters":20}, chip=chip)
    if model.latency_us is None:
        raise ValueError("PreSinkhorn requires a usable CV chip template")
    baseline,_ = _cost((1,4,4096),chip,partition((1,4,4096),chip),ref)
    start = max(0.,model.latency_us-sum(baseline.values()))
    local_m = ceil(t/plan["m_blocks"])
    panels, tail = divmod(local_m, 128)
    cast = projection = 0.
    for count, m in ((panels,128),(int(tail>0),tail)):
        if count:
            stage,info = _cost((m,n,d),chip,plan,ref)
            elements=info["critical_aiv_stage1_elements"]
            clock=1000*chip.frequency_ghz
            cast_us=vf.stage(chip,'mhc/partition_cast_reduce',stage["cast"]*clock,elements,known=True,
                             load=elements,convert=elements,arithmetic=elements,reduce=elements,store=elements)/clock
            dma=chip.details.get("mte2_bandwidth_gbs")
            cast += count*max(cast_us,2*elements/(1000*dma) if dma else 0)
            projection += count*stage["projection"]
    rows=plan["rows_per_aiv"]; cells=rows*n*n; payload=rows*n*d
    sink = vf.stage(chip,'mhc/sinkhorn_mix',stage["post"]*clock,payload,known=True,load=payload+cells*(2*iters+1),
                    exp=cells,reduce=2*iters*cells,divide=2*iters*cells,arithmetic=2*payload,store=rows*d+cells)/clock
    k=n*d
    c=n*n+2*n
    groups=t*n*n
    vec=t*((2*k+1)+1+3*c+2*n+2*n+n*d+(n-1)*d)
    vec += t*n*(n-1)+2*groups+2*iters*t*n*(n-1)+groups+(2*iters-1)*t*n+2*iters*groups
    work=_counts("bf16",read,write,cube={"fp32":2*t*k*c},vec={"fp32":vec})
    evidence="950 PreSinkhorn 4096-wide CA stage transfers to extended K; HF32 uses per-AIC 16-bit throughput ratio"
    hbm=(read+write)/(1000*chip.hbm_bandwidth_gbs)
    work["schedule"]={**plan, "algorithm":"mhc_split_k_sinkhorn", "evidence":evidence,
                      "control_us":start,"cast_us":cast,"cube_us":projection,"sink_us":sink,
                      "hbm_floor_us":hbm,"critical_m_panels":panels+int(tail>0)}
    return NodeEstimate("mhc_split_k_sinkhorn","extrapolated",
                        max(hbm,start+cast+projection+sink),reason=evidence,work=work)


def _propose(f, sem, chip):
    name = f.raw_name
    xs, ys = f.inputs, f.outputs
    x = xs[0]
    read = sum(_size(t, distinct=True) for t in xs)
    write = sum(_size(t) for t in ys)
    n = prod(ys[0].shape) if ys else 0
    width = x.shape[-1] if x.shape else 1
    if name.startswith("aten."):
        return _primitive(f)
    if name.endswith(("grouped_gated_rms_norm_reference","mla_output_gate","situ","npu_swiglu")):
        return _activation(f, sem, n, read, write)
    if name.endswith(("inplace_partial_rotary_mul","npu_rotary_mul")):
        lo,hi = sem.get("partial_slice",(0,width))
        rot = hi-lo
        if not 0 < rot <= width:
            raise ValueError("Rotary slice exceeds head width")
        n = prod(x.shape[:-1])*rot
        write = n*_WIDTH[x.dtype.removeprefix("torch.")]
        return Tiles("rope_interleaved_pair" if "partial" in name else "rope_pair",
                     n, rot*2, write+sum(_size(t,distinct=True) for t in xs[1:]),
                     write, 3*n, "bf16", "rope", known_precision="partial" in name and x.dtype.endswith("bfloat16"))
    if name == "inference::cache_load":
        fp4 = sem["storage_precision"]=="fp4"
        if sem["storage_precision"] not in ("fp4","fp8"):
            raise ValueError("Unsupported cache encoding")
        # Reading valid rows of sliced batches may skip rows between batches.
        # Coalesced within each row; no copy of the whole cache allocation.
        return Tiles("mx4_cache_unpack" if fp4 else "fp8_cache_dequant", n,
                     ys[0].shape[-1], read, write, n*(3 if fp4 else 2), setup="dequant", known_precision=not fp4 and x.dtype.endswith("float8_e4m3fn") and ys[0].dtype.endswith("bfloat16"))
    if name == "inference::cache_store":
        # Position IDs precede aliased cache value/scale tensors. Output tensors
        # describe backing capacity, not the number of updated slots.
        # Some compressed producers already return consecutive rows and have
        # no explicit positions operand; cache tensors always occupy the tail.
        payload_inputs = xs[:-len(ys)]
        rows = prod(x.shape[:-1])
        logical = sem["logical_cache_shape"]
        d = logical[-1]
        if sem.get("layout") == "K/V,B,capacity,local_kv_heads,head_dim":
            rows *= 2
        read = sum(_size(t,distinct=True) for t in payload_inputs)
        if sem["storage_precision"]=="fp4":
            payload = ceil(d/2)+2*ceil(d/64)
        elif sem["storage_precision"]=="fp8":
            payload = d+4
        else:
            raise ValueError("Unsupported cache-store codec")
        if sem.get("verify_state_strategy") == "paired encoded value/scale prefix snapshots":
            axis = sem["snapshot_query_axis"]
            steps = sem["snapshot_query_len"]
            bases = xs[-len(ys):]
            if any(y.shape != (*base.shape[:axis],steps,*base.shape[axis:])
                   for base,y in zip(bases,ys)):
                raise ValueError("Invalid encoded prefix snapshot extent")
            # The reference materializes one complete image per prefix. Charge
            # base bytes once logically, and stream the unmodified slots again
            # as needed. Encoding only visits the newly supplied elements.
            base_bytes = sum(_size(t) for t in bases)
            visits = dict(load=rows*d,arithmetic=2*rows*d,reduce=rows*d,
                          divide=rows,convert=rows*d,store=rows*d)
            return Tiles("materialize_encoded_prefix_states",write,512,read+base_bytes,
                         write,4*rows*d,class_visits=visits,
                         traffic_read=max(0,write-base_bytes-rows*payload),
                         work_source="capture materializes each encoded prefix; unchanged base slots are streamed into snapshots")
        return Tiles("quantize_scatter_current_rows", rows*d, d, read, rows*payload,
                     4*rows*d, setup="dequant")
    if name=="inference::state_update":
        if "row_bytes" not in sem:
            from .catalog_vector import state_rows
            return state_rows(f, sem)
        rows = prod(x.shape[:-1])
        return Tiles("layernorm_mx4_scatter_current_rows", rows*width, width,
                     sum(_size(t,distinct=True) for t in xs[:-1]), rows*sem["row_bytes"],
                     rows*(11*width+3), setup="dequant")
    if name=="inference::kimi_dsa_indexer_query_quant":
        rows = prod(x.shape[:-1])
        return Tiles("mx4_query_and_head_weights", rows*width, width, read, write,
                     4*rows*width+rows, setup="dequant")
    if name.endswith(("npu_dynamic_mx_quant","dynamic_mx_quant_fp8_reference")):
        q = "quant_mxfp4" if name.startswith("npu::") else "quant_token_fp8"
        logical = tuple(sem["logical_shape"])
        anchor = None
        if prod(logical):
            anchor = _anchor(q, logical, chip, qualifier="CANN contiguous quantizer component; input layout priced separately")
        if anchor is None:
            return Tiles("quantize_last_axis", prod(logical), width, read, write, 4*prod(logical), setup="dequant")
        if contiguous_input(x.shape, x.stride):
            return anchor
        gather = _schedule(Tiles("materialize_input_slice",prod(logical),512,read,read,0,setup="slice"),chip)
        work=dict(anchor.work)
        work.update(logical_read_bytes=work["logical_read_bytes"]+read,
                    logical_write_bytes=work["logical_write_bytes"]+read,
                    schedule={"algorithm":"slice_then_quantize", "slice_us":gather.latency_us,
                              "quant_us":anchor.latency_us, "quant":anchor.work["schedule"],
                              "slice":gather.work["schedule"],"evidence":"aclnnSlice then DynamicMxQuant in measured strided fixture"})
        return NodeEstimate("slice_then_"+q,"extrapolated",gather.latency_us+anchor.latency_us,work=work)
    if name=="_C_ascend::npu_hc_pre_v2":
        p=_anchor("mhc_pre_sinkhorn", tuple(x.shape[-3:]), chip,
                  {"need_backward":False,"num_iters":sem["sinkhorn_iterations"]},
                  qualifier="PreSinkhorn native equivalent with three inference outputs; extension dispatch unobserved")
        if p is not None:
            return p
        return _project_hc_pre(x.shape, sem, chip, read, write)
    if name in ("npu::npu_rms_norm","_C_ascend::npu_hc_post"):
        if name.endswith("rms_norm"):
            return Tiles("weighted_rms_rows",prod(x.shape),width,read,write,4*prod(x.shape)+3*prod(x.shape[:-1]))
        branches = xs[1].shape[-2]
        return Tiles("mhc_weighted_residual_mix", n, width, read, write, 2*n*(branches+1))
    if name=="_C_ascend::moe_gating_top_k_hash":
        rows,k = ys[0].shape
        full = bool(sem.get("out_flag"))
        if len(xs)==2 and sem.get("topk_method")=="topk_bias":
            # This use of the same extension symbol has no token IDs or table.
            # It scores and ranks all experts with a bias as the normal router does.
            base=estimate('moe_gating_top_k',(rows,width,k),{'vector_model':'calibrated'},chip=chip)
            if base.latency_us is not None:
                projection=base.projection; plan=projection['target_plan']; stages=dict(projection['stages_us'])
                clk=1000*chip.frequency_ghz; local=plan['rows_per_busy_core']; visits=local*width; levels=ceil(log2(k+1))
                stages['rank_and_normalize']=vf.stage(chip,'sqrtsoftplus_bias_rank',clk*stages['rank_and_normalize'],visits,
                        load=3*visits,exp=visits,log=visits,rsqrt=visits,arithmetic=3*visits,
                        compare=levels*visits,permute=levels*visits,store=2*local*k)/clk
                work=dict(base.work); evidence='Native router startup/read/write proxy; proposed biased sqrt(softplus) scoring/ranking, no matched ELF.'
                work['schedule']={**plan, 'algorithm':'sqrtsoftplus_bias_rank','stages_us':stages,'evidence':evidence}
                return NodeEstimate('sqrtsoftplus_bias_rank','extrapolated',sum(stages.values()),reason=evidence,work=work)
        if len(xs)!=3:
            raise ValueError("Hash routing requires token IDs and tid2eid")
        selected = rows*k
        visits = rows*width
        # The 950 arch35 implementation reads every logit, scores the row in
        # UB, fetches tid2eid[token], then gathers k already-computed scores.
        # 7 lanes/logit account for the exponent/log/sqrt handling and scale.
        route=Tiles("score_full_row_then_hash_gather",visits,width,
                    _size(x)+_size(xs[1])+selected*4,selected*8+(4*visits if full else 0),
                    7*visits+2*selected,setup="aclnnMoeGatingTopKV2/hash", known_precision=x.dtype.endswith("float32"),
                    transaction_read=rows*((-4*k)%32),
                    work_source="arch35 HashCompute, native aclnnMoeGatingTopKV2 complete DT/PR windows")
        result=_schedule(route,chip)
        if result.latency_us is None:
            return result
        plan=result.work["schedule"]
        repeat=ceil(rows/plan["active_aiv"])-1
        # Two dependent MTE2 fetches (token then mapping row) on the critical
        # path. The first pair is already included in the startup calibration.
        delay=chip.details.get("quant_read_latency_cycles") or 1030.
        plan.update(hash_lookup_waves=repeat+1,score_rows=rows,experts_per_row=width,
                    selected_per_row=k,dependent_lookup_cycles=2*delay*repeat)
        return NodeEstimate(result.contract,result.status,result.latency_us+2*delay*repeat/(1000*chip.frequency_ghz),
                            reason=result.reason,work=result.work)
    if name=="_C_ascend::moe_gating_top_k":
        rows,k=ys[0].shape
        if sem.get("scoring_func")=="sigmoid" and sem.get("n_group")==1 and not sem.get("out_flag"):
            anchored=_anchor("moe_gating_top_k",(rows,width,k),chip,
                             qualifier="native sigmoid, one expert group and no full score output")
            if anchored is not None:
                return anchored
        return Tiles("router_score_topk_merge", rows*width, width, read,
                     sum(_size(t) for t in ys[:2])+(_size(ys[2]) if sem.get("out_flag") else 0),
                     rows*(width*(6+ceil(log2(k+1)))+2*k))
    if name=="inference::attention_residual":
        rows,d=x.shape
        capacity=xs[1].shape[0]
        valid=sem["valid_blocks"]
        idx=sem["write_slot"]
        counts=attention_residual_work(rows,d,capacity,valid,chip=chip,
                                       output_norm=False,write_idx=idx)
        ops=counts["fp32_basic_ops"]+counts["exp_elements"]+counts["rsqrt_elements"]+counts["reciprocal_elements"]
        return Tiles("prefix_bank_resident_mix" if counts["resident"] else "prefix_bank_two_pass_mix",
                     rows*d,d,counts["distinct_input_bytes"],counts["dma_write_bytes"],ops,
                     class_visits=dict(arithmetic=counts["fp32_basic_ops"], exp=counts["exp_elements"], divide=counts["rsqrt_elements"]+counts["reciprocal_elements"], load=counts["dma_read_bytes"]//2,store=counts["dma_write_bytes"]//2),
                     traffic_read=max(0,counts["dma_read_bytes"]-counts["distinct_input_bytes"]),
                     work_source="AttnResFwd arch35 bank reads and valid prefix; inference normalization is a separate node")
    if name=="inference::engram_lookup":
        ids=_size(x)
        # Each selected segment carries signed-byte weights and per-32 FP32 scales.
        return Tiles("engram_sharded_row_gather", n, ys[0].shape[-1], ids+write, write,
                     prod(x.shape), transaction_read=0)
    if name=="inference::engram_dequantize":
        return Tiles("engram_unpack_scales", n, 32, read, write, 2*n, setup="dequant")
    if name=="inference::engram_hash":
        ngram=sem["ngram_size"]
        ids=prod(x.shape)
        # The vocabulary remap is probed once per token, not scanned.
        return Tiles("ngram_hash_modulo", n, ys[0].shape[-1],
                     read-_size(xs[1])+4*ids, write, n*(3*ngram+2), transaction_read=28*ids)
    if name=="inference::engram_gate":
        return Tiles("engram_norm_dot_sigmoid_mix", n, width, read, write, 11*n+4*ceil(n/width))
    raise ValueError("Unknown rank-local vector contract")


def price(features, chip):
    if features.raw_name not in NAMES:
        return NodeEstimate(None,"unsupported",reason="Unknown Vector call")
    target=load_chip(chip)
    try:
        sem=dict(dict(features.attributes).get("semantic",()))
        proposal=_propose(features,sem,target)
        return _schedule(proposal,target) if isinstance(proposal,Tiles) else proposal
    except (IndexError, KeyError, StopIteration, ValueError, TypeError) as error:
        return NodeEstimate(None,"unsupported",reason=f"{features.raw_name}: {error}")
