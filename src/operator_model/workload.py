"""Logical work of the accepted single-rank kernel contracts.

Counts refer to useful scalar-equivalent math and the tensor interface of each
launch. They exclude padding, cache misses, DMA replay and ISA expansion.
The timing adapter validates the actual dispatch before calling this module.
"""
from collections import Counter
from math import prod

from operator_model.scheduling.graph import build_graph, plan_from_case
from operator_model.scheduling.reference import ALIASES

_GMM1 = "grouped_matmul_swiglu_quant_v2"
_MOE = {_GMM1, "gmm2", "moe_ffn"}


def default_dtype(op):
    """Primary input storage dtype used in the reference kernel call."""
    op = ALIASES.get(op, op)
    if op == 'routed_grouped_matmul':
        return 'mxfp4'
    if op in _MOE:
        return "fp8_e4m3fn"
    return "fp32" if op in ("mhc_sinkhorn", "moe_gating_top_k", "sigmoid_fp32") else "bf16"


def _counts(dtype, read, write, cube=None, vec=None, *, classes=None, notes=()):
    cube, vec = cube or {}, vec or {}
    return {
        "dtype": dtype, "cube_ops": sum(cube.values()), "vec_ops": sum(vec.values()),
        "cube_ops_by_dtype": cube, "vec_ops_by_dtype": vec,
        "vec_ops_by_class": classes or {},
        "logical_read_bytes": read, "logical_write_bytes": write,
        "physical_hbm_bytes": None,
        "scope": "Useful scalar-equivalent work (FMA=2), one logical read/write per tensor per kernel; excludes padding, cache traffic and ISA expansion.",
        "notes": list(notes),
    }


def _chain(dtype, children):
    cube, vec, classes = Counter(), Counter(), Counter()
    for child in children:
        cube.update(child["cube_ops_by_dtype"])
        vec.update(child["vec_ops_by_dtype"])
        classes.update(child["vec_ops_by_class"])
    notes = ["Kernel-interface byte sum includes materialized intermediates at both ends."]
    notes.extend(dict.fromkeys(note for child in children for note in child["notes"]))
    return _counts(dtype, sum(c["logical_read_bytes"] for c in children),
                   sum(c["logical_write_bytes"] for c in children),
                   dict(cube), dict(vec), classes=dict(classes), notes=notes)


def _moe(op, shape, attrs):
    dtype = default_dtype(op)
    if op == "moe_ffn":
        return _chain(dtype, [_moe(part, shape, attrs) for part in (_GMM1, "gmm2")])
    m, k, f, e = shape
    histogram = attrs["expert_counts"]
    if len(histogram) != e or sum(histogram) != m or any(x < 0 for x in histogram):
        raise ValueError("Expected a valid local expert histogram")
    active = sum(x > 0 for x in histogram)
    reduction, width = (k, 2*f) if op == _GMM1 else (f, k)
    # Each 64-element reduction group carries two E8M0 scale bytes.
    scale_bytes = 2*((reduction + 63)//64)
    read = (m + active*width)*(reduction + scale_bytes) + 8*e
    cube = {dtype: 2*m*reduction*width}
    if op == "gmm2":
        return _counts(dtype, read, 2*m*k, cube,
                       notes=["MX scale application is folded into the grouped matrix product."])
    elements = m*f
    classes = {opname: elements for opname in ("negate", "exp", "add", "divide", "multiply")}
    return _counts(dtype, read, elements + m*2*((f+63)//64), cube,
                   {"fp32": sum(classes.values())}, classes=classes,
                   notes=["Counts the SwiGLU math; MX conversion, exponent selection and casts are additional."])


def _structured(op, shape, attrs, dtype):
    # Normalize tensor attributes separately from non-tensor launch metadata.
    if op.startswith("kda"):
        params = {k: v for k, v in attrs.items() if k not in ("beta_dtype", "safe_gate", "lower_bound")}
    elif op == "mhc_sinkhorn":
        params = {"num_iters": attrs.get("num_iters", 20)}
    else:
        # These validated routes fix the remaining tensor attributes, including
        # split Pre's auxiliary outputs and the indexer's causal top-512 mask.
        params = {"return_intermediates": True} if op == "mhc_pre" else {}
    params["dtype"] = dtype
    plan = plan_from_case(op, shape, params)
    if plan["status"] == "unsupported":
        raise ValueError(plan.get("reason") or plan.get("unsupported_reason"))
    graph = build_graph(plan)
    cube, classes = Counter(), Counter()
    for stage in graph["stages"]:
        resource = stage["resource"]
        if resource.startswith("cube_"):
            precision = resource.removeprefix("cube_")
            cube["fp32" if precision == "hf32" else precision] += stage["work"]
        elif resource.startswith("vector_"):
            classes[resource.removeprefix("vector_")] += stage["work"]
    work = plan["work"]
    notes = []
    if op == "lightning_indexer":
        b, q, length, heads, d = shape
        write = 4*b*q*plan["attrs"]["sparse_count"]
        notes.append("Counts visible causal pairs; data-dependent top-k selection is excluded.")
    elif op == "mhc_sinkhorn":
        t, n = shape
        write = work["logical_tensor_bytes"] - 4*t*n*n
    elif op == "mhc_post":
        t, n, d = shape
        write = 2*t*n*d
        notes.append("Output cast counted once per element against FP32 source precision.")
    elif op in ("mhc_pre", "mhc_pre_sinkhorn"):
        t, n, d = shape
        # All inference outputs except h_in use FP32 storage.
        write = 2*t*d
        write += 4*sum(prod(dims) for name, dims in plan["output_shapes"].items()
                       if name != "h_in" and dims is not None)
        if op == "mhc_pre":
            notes.append("Projection work counts 32-bit operands. In the audited split Pre kernel "
                         "the FP32 MMAD reduction fragments have depth 32.")
        else:
            notes.append("Projection work is grouped by its 32-bit operand width. "
                         "The fused kernel's internal math mode depends on the implementation.")
    else:
        b, s, hq, hv, k, v = shape
        data_width = {"fp32": 4, "bf16": 2, "fp16": 2}
        gate_width = data_width.get(str(attrs.get("gate_dtype", "fp32")).lower())
        beta_width = data_width.get(str(attrs.get("beta_dtype", "fp32")).lower())
        state = work["state_tensor_bytes"]
        read = None
        prepared_gate = not attrs.get("use_gate_in_kernel") and params.get("gate_mode", "preactivated") == "preactivated"
        if gate_width and beta_width and prepared_gate:
            read = 2*b*s*(2*hq*k+hv*v) + b*s*hv*(gate_width*k + beta_width) + state
        if op == "kda_chunk" and gate_width and beta_width:
            if prepared_gate and not attrs.get("initial_state", True):
                read -= state
            if not prepared_gate and "dt_bias" in attrs:
                read = (2*b*s*(2*hq*k+hv*v) + b*s*hv*(gate_width*k+beta_width)
                        + 4*hv + (4*hv*k if attrs["dt_bias"] else 0)
                        + (state if attrs.get("initial_state", True) else 0))
            if "cu_seqlens" in attrs:
                notes.append("Counts tensor bytes only; host-side sequence metadata and internal workspaces are excluded.")
        write = 2*b*s*hv*v + (state if attrs.get("output_final_state", True) else 0)
        if op == "kda_chunk":
            chunk = plan["attrs"]["chunk_size"]
            write += 4*b*s*hv*chunk  # two 16-bit attention outputs
            if attrs.get("export_gk", True):
                write += 4*b*s*hv*k
            if attrs.get("export_intermediates", False):
                write = None
            notes.append("Chunk dot counts mix useful triangular and dense-equivalent terms; triangular solve work is unknown.")
        notes.append("Core recurrence/state work only; gate processing, casts and normalization excluded.")
        return _counts(dtype, read, write, dict(cube), {"fp32": sum(classes.values())},
                       classes=dict(classes), notes=notes)
    return _counts(dtype, work["logical_tensor_bytes"] - write, write, dict(cube),
                   {"fp32": sum(classes.values())}, classes=dict(classes), notes=notes)


def account_work(op, shape, attrs=None):
    """Account an accepted dispatch. Counts are chip-independent; FMA counts as 2.

    Each vector lane operation counts as one, including special functions.
    See vec_ops_by_class and notes for the scope of partial algebraic counts.
    physical_hbm_bytes is unknown without a cache/transaction model.
    """
    op = ALIASES.get(op, op)
    attrs = attrs or {}
    dtype = str(attrs.get("dtype", default_dtype(op))).lower()
    if op in ('cache_load', 'candidate_filtered_indexer', 'routed_grouped_matmul'):
        call = {key: value for key, value in attrs.items() if key != 'dtype'}
        if op == 'routed_grouped_matmul':
            from operator_model.operators.moe.routed_grouped_matmul import describe
            return describe(shape, dtype, call)[0]
        if op == 'cache_load':
            from operator_model.operators.vector.cache_load import describe
            return describe(shape, dtype, call)[0]
        from operator_model.operators.attention.candidate_indexer import describe
        if dtype not in ('bf16', 'bfloat16'):
            raise ValueError('candidate_filtered_indexer requires BF16 scoring inputs')
        return describe(shape, call)[0]
    if op == "quant_mxfp4":
        from operator_model.operators.quant.mxfp4 import mx4_work
        if dtype not in ("bf16", "bfloat16"):
            raise ValueError("quant_mxfp4 requires BF16 input")
        return mx4_work(shape, {k: v for k, v in attrs.items() if k != "dtype"})
    if op == "mhc_post" and attrs.get("implementation") == "vllm_hc_post":
        if dtype != "bf16":
            raise ValueError("vLLM HcPost accounting expects BF16 activations and FP32 coefficients")
        from operator_model.operators.mhc.post_layout import hc_post_work
        return hc_post_work(shape, {k: v for k,v in attrs.items() if k != "dtype"})
    if op == "attn_res_fwd":
        from operator_model.chips import load_chip
        from operator_model.operators.attention.residual_timing import resource_attention
        return resource_attention(shape, load_chip(), attrs)[1]
    if op == "sigmoid_fp32":
        from operator_model.operators.vector.sigmoid import sigmoid_work
        return sigmoid_work(shape)
    if op == "cast_bf16_fp32":
        from operator_model.operators.vector.cast import cast_work
        return cast_work(shape)
    if op == "moe_gating_top_k":
        from operator_model.operators.vector.router import router_work
        return router_work(shape)
    if op == "mhc":
        t, n, _ = shape
        children = [account_work("mhc_pre", shape),
                    account_work("mhc_sinkhorn", (t, n), {"num_iters": attrs.get("num_iters", 20)}),
                    account_work("mhc_post", shape)]
        return _chain(dtype, children)
    if op in _MOE:
        return _moe(op, shape, attrs)
    if op == "matmul" and (dtype not in ("bf16", "bfloat16") or "weight_dtype" in attrs):
        from operator_model.operators.matmul.mx import mx_work
        return mx_work(shape, dtype, {k: v for k, v in attrs.items() if k != "dtype"})
    if op in ("addmm", "matmul"):
        m, n, k = shape
        read = 2*(m*k+k*n)
        out = _counts(dtype, read + (2*n if op == "addmm" else 0), 2*m*n, {dtype: 2*m*n*k})
        if op == "addmm":
            out["fused_bias_additions"] = m*n
            out["notes"].append("Column bias additions use the fused matrix epilogue.")
        return out
    if op == "fa":
        if len(shape) == 3:
            shape = (*shape, 1, 1)
        q, kv, d, b, h = shape
        rows = b*h*q
        pairs = rows*kv
        classes = {"score_scale": pairs, "max_comparison": pairs-rows, "subtract": pairs,
                   "exp": pairs, "sum_add": pairs-rows, "divide": pairs}
        return _counts(dtype, 2*b*h*d*(q+2*kv), 2*rows*d+64*rows,
                       {dtype: 4*pairs*d}, {"fp32": sum(classes.values())}, classes=classes,
                       notes=["Stable softmax algebra; online tile merges and padding excluded.",
                              "Output includes two FP32 [B,H,Q,8] statistic tensors."])
    if op == "quant":
        rows, cols = shape
        x = rows*cols
        groups = rows*((cols+31)//32)
        classes = {"abs": x, "max_comparison": x-groups}
        return _counts(dtype, 2*x, x+rows*2*((cols+63)//64),
                       vec={dtype: sum(classes.values())}, classes=classes,
                       notes=["Input-precision magnitude and block maximum only; exponent formation, scaling and FP8 conversion excluded."])
    if op == "rmsnorm":
        rows, cols = shape
        x = rows*cols
        classes = {"multiply": 3*x+rows, "add": x, "rsqrt": rows}
        return _counts(dtype, 2*(x+cols), 2*x+4*rows,
                       vec={"fp32": sum(classes.values())}, classes=classes)
    return _structured(op, shape, attrs, dtype)
