"""Dense, batched and grouped projections in the extended system catalog."""

from math import ceil

from operator_model.system_adapter import (
    BF16,
    E8,
    F32,
    FP4,
    FP8,
    NodeEstimate,
    _arity,
    _dtype,
    _fields,
    _layout,
    _packed,
    _require,
    _shape,
)
from operator_model.workload import _counts
from . import vf
from .cube import _align, _schedule


def _matrix_view(t, dtype, dims):
    """Direct ND panel loads for positive, non-overlapping row/column-major views."""
    _layout(t, dtype, dims, strides=t.stride)
    axes = sorted((pitch, size) for pitch, size in zip(t.stride, dims) if size > 1)
    span = 1
    for pitch, size in axes:
        _require(pitch >= span, "Overlapping matrix batch strides")
        span += (size - 1) * pitch
    _require(not axes or axes[0][0] == 1, "No contiguous panel dimension")


def capture_bmm(f, sem):
    quant = f.raw_name.startswith("inference::")
    transpose = "transpose" in f.raw_name
    _arity(f, 4 if quant else 2, 1)
    if transpose:
        _require(
            sem.get("perm_x1") == sem.get("perm_y") == (1, 0, 2),
            "Expected swapped head and token axes",
        )
    shapes = sem.get("logical_input_shapes", tuple(t.shape for t in f.inputs[:2]))
    _require(len(shapes) == 2, "Expected two logical BMM inputs")
    x, w = (_shape(shape, 3) for shape in shapes)
    g, m, k = (x[1], x[0], x[2]) if transpose else x
    _require(w[:2] == (g, k), "Incompatible batch or K dimensions")
    n = w[2]
    output = BF16 if quant else _dtype(f.inputs[0])
    _require(output in (BF16, F32), "Expected BF16 or FP32 dense BMM")
    _layout(f.outputs[0], output, (m, g, n) if transpose else (g, m, n))
    elements = g * k * (m + n)
    if not quant:
        for t, shape in zip(f.inputs, shapes):
            _matrix_view(t, output, shape)
        width = 4 if output == F32 else 2
        return (
            (m,) * g,
            n,
            k,
            "dense32" if output == F32 else "batch16",
            width * elements,
            width * g * m * n,
            {},
        )
    _require(
        sem.get("input_layout") == "values_then_corresponding_scales"
        and sem.get("accumulator_dtype") == F32
        and sem.get("group_size") == 32,
        "Unexpected BMM scale/accumulation metadata",
    )
    precision = sem.get("compute_precision")
    if precision == "fp4":
        for i, dims in enumerate(shapes):
            _packed(f, "inputs", i, dims)
            _layout(f.inputs[i + 2], E8, (*dims[:-1], ceil(dims[-1] / 64), 2))
        read = elements // 2 + 2 * g * (m * ceil(k / 64) + k * ceil(n / 64))
        return (
            (m,) * g,
            n,
            k,
            "batch_mx",
            read,
            2 * g * m * n,
            {"unpack": elements, "scale_bf16": elements},
        )
    _require(precision == "fp8", "Unknown BMM input precision")
    for i, shape in enumerate(shapes):
        _matrix_view(f.inputs[i], FP8, shape)
        _layout(f.inputs[i + 2], F32, shape[:-1])
    # B is scaled by row K, so scaling only the final output cannot implement this program.
    return (
        (m,) * g,
        n,
        k,
        "batch_row8",
        elements + 4 * g * (m + k),
        2 * g * m * n,
        {"convert": elements, "scale_bf16": elements},
    )


def _roles(f):
    records = _fields(dict(f.attributes).get("tensor_semantics", ())).get("inputs", ())
    return tuple((_fields(r).get("index"), _fields(r).get("role")) for r in records)


def padded_mx(f, sem, chip):
    from operator_model import estimate

    _arity(f, 4, 1)
    m, k = _shape(sem.get("logical_input_shape"), 2)
    n, kw = _shape(sem.get("logical_weight_shape"), 2)
    physical = _align(k, 32)
    _require(
        k == kw
        and sem.get("padding") == (0, physical - k)
        and sem.get("group_size") == 32
        and sem.get("activation_precision") == sem.get("weight_precision") == "fp4"
        and sem.get("accumulator_dtype") == F32
        and sem.get("activation_scale_layout") == "mx_e8m0",
        "Unexpected MX padding or precision",
    )
    _require(
        _roles(f)
        == tuple(
            enumerate(("activation", "weight", "weight_scale", "activation_scale"))
        ),
        "Unexpected MX roles",
    )
    hints = dict(f.attributes).get("input_logical_shapes")
    _require(
        hints is None or hints == ((m, k), (k, n), None, None),
        "Conflicting MX logical shapes",
    )
    _layout(f.inputs[0], FP4, (m, physical // 2))
    _layout(f.inputs[1], FP4, (physical // 2, n), strides=(1, physical // 2))
    pages = ceil(physical / 64)
    _layout(f.inputs[2], E8, (pages, n, 2), strides=(2, 2 * pages, 1))
    _layout(f.inputs[3], E8, (m, pages, 2))
    _layout(f.outputs[0], BF16, (m, n))
    result = estimate("matmul", (m, n, physical), chip=chip, dtype="mxfp4")
    if result.latency_us is None:
        return NodeEstimate(None, "unsupported", reason=result.reason)
    dtype = result.work["dtype"]
    work = {
        **result.work,
        "cube_ops": 2 * m * n * k,
        "cube_ops_by_dtype": {dtype: 2 * m * n * k},
        "schedule": {
            "logical_k": k,
            "issued_k": physical,
            "active_aic": result.active_aic,
            "target_plan": (result.projection or {}).get("target_plan"),
        },
        "evidence": "analytical_design",
    }
    return NodeEstimate(
        "matmul/A4W4/padded_k",
        "extrapolated",
        result.latency_us,
        "Issued K includes stored padding; useful FLOPs exclude padding.",
        work,
    )


def grouped_mlp(f, sem, chip):
    m, k = _shape(sem.get("logical_input_shape"), 2)
    up, down = (_fields(sem.get(key, ())) for key in ("gate_up", "down"))
    groups, n, h = _shape(down.get("logical_weight_shape"), 3)
    counts = sem.get("group_counts")
    _require(
        isinstance(counts, tuple)
        and len(counts) == groups
        and all(type(x) is int and x >= 0 for x in counts)
        and sum(counts) == m
        and up.get("logical_weight_shape") == (groups, 2 * h, k),
        "Inconsistent expert histogram or projections",
    )
    precision = up.get("scheme")
    _require(
        precision in ("fp4", "fp8", "bf16")
        and all(
            p.get(key) == precision
            for p in (up, down)
            for key in ("scheme", "activation_precision", "weight_precision")
        ),
        "Unmatched expert projection precisions",
    )
    quant = precision != "bf16"
    _arity(f, 6 if quant else 3, 1)
    roles = (
        (
            "activation",
            "gate_up_weight",
            "gate_up_weight_scale",
            "activation_scale",
            "down_weight",
            "down_weight_scale",
        )
        if quant
        else ("activation", "gate_up_weight", "down_weight")
    )
    _require(
        _roles(f) == tuple(enumerate(roles))
        and sem.get("fusion_components")
        in (
            ("grouped_matmul", "swiglu", "grouped_matmul"),
            ("grouped_matmul", "situ", "grouped_matmul"),
        )
        and sem.get("intermediate_quantization")
        == {"fp4": "mxfp4", "fp8": "fp8", "bf16": "none"}[precision]
        and sem.get("internal_hbm_intermediates")
        == "not_materialized; workspace is separately declared"
        and sem.get("workspace_bytes") in (None, 0),
        "Unpriced workspace or unexpected fused contract",
    )
    live = sum(c > 0 for c in counts)
    _layout(f.outputs[0], BF16, (m, n))
    reduction_w = []
    if precision == "fp8":
        _require(
            all(
                p.get("activation_scale_layout") == "token_fp32"
                and p.get("scale_layout") == "channel_fp32"
                for p in (up, down)
            ),
            "Expected FP32 token/channel scales",
        )
        _layout(f.inputs[0], FP8, (m, k))
        _layout(f.inputs[3], F32, (m,))
        input_bytes = m * (k + 4)
        for wi, si, red, width in ((1, 2, k, 2 * h), (4, 5, h, n)):
            _matrix_view(f.inputs[wi], FP8, (groups, red, width))
            _layout(f.inputs[si], F32, (groups, width))
            reduction_w.append(live * width * (red + 4))
    elif quant:
        _require(
            up.get("group_size") == down.get("group_size") == 32
            and k % 32 == h % 32 == 0,
            "Expected complete MX groups",
        )
        _layout(f.inputs[0], FP4, (m, k // 2))
        _layout(f.inputs[3], E8, (m, ceil(k / 64), 2))
        input_bytes = m * (k // 2 + 2 * ceil(k / 64))
        for wi, si, red, width in ((1, 2, k, 2 * h), (4, 5, h, n)):
            pg = ceil(red / 64)
            _layout(
                f.inputs[wi],
                FP4,
                (groups, red // 2, width),
                strides=(red // 2 * width, 1, red // 2),
            )
            _layout(
                f.inputs[si],
                E8,
                (groups, pg, width, 2),
                strides=(2 * pg * width, 2, 2 * pg, 1),
            )
            reduction_w.append(live * width * (red // 2 + 2 * pg))
    else:
        _layout(f.inputs[0], BF16, (m, k))
        for t, red, width in ((f.inputs[1], k, 2 * h), (f.inputs[2], h, n)):
            _layout(t, BF16, (groups, width, red))
            reduction_w.append(2 * live * width * red)
        input_bytes = 2 * m * k
    activation = _fields(sem.get("activation", ()))
    kind = activation.get("kind")
    _require(kind in ("silu", "situ"), "Unknown gated activation")
    # Each phase uses the real local histogram. Microtile residence in UB/L1 avoids GM intermediates.
    mode = {"fp4": "native4", "fp8": "native8", "bf16": "batch16"}[precision]
    first, a = _schedule(
        counts, 2 * h, k, mode, input_bytes + reduction_w[0], 0, {}, chip, False
    )
    last, b = _schedule(counts, n, h, mode, reduction_w[1], 2 * m * n, {}, chip, False)
    lanes = min(chip.vector_cores, max(1, m * ceil(h / 128)))
    visits = dict(load=2, arithmetic=4, exp=1, divide=1, store=1)
    if kind == "situ":
        visits.update(arithmetic=9, exp=2, divide=2)
    if activation.get("limit") is not None:
        visits["compare"] = 3
    if quant:
        visits.update(reduce=1, convert=1)
        visits["arithmetic"] += 2
    count = m * h
    compute = sum(v for key, v in visits.items() if key not in ("load", "store"))
    peak = chip.vector_tops.get("16")
    _require(peak, "Missing BF16 vector throughput")
    cycles = vf.stage(
        chip,
        "moe/" + kind,
        compute
        * count
        * chip.frequency_ghz
        * chip.vector_cores
        / (1000 * peak * lanes),
        count / lanes,
        bits=16,
        **{key: value * count / lanes for key, value in visits.items()},
    )
    rescale_work = 0
    if precision == "fp8":
        rescale_work = 2 * m * (2 * h + n)
        units = rescale_work / lanes
        peak32 = chip.vector_tops.get("32")
        _require(peak32, "Missing FP32 rescale throughput")
        cycles += vf.stage(
            chip,
            "moe/channel_token_rescale",
            rescale_work
            * chip.frequency_ghz
            * chip.vector_cores
            / (1000 * peak32 * lanes),
            units,
            bits=32,
            load=units,
            arithmetic=units,
            store=units / 2,
        )
    middle = cycles / (1000 * chip.frequency_ghz)
    waves = min(a["waves"], b["waves"])
    # One launch. The Cube stages contend for the same engine; only VF can overlap across waves.
    launch = b["resource_cycles"]["first_issue"] / (1000 * chip.frequency_ghz)
    cube_us = first + last - launch
    body = max(cube_us, middle) + min(cube_us, middle) / waves
    dtype = {"fp4": "mxfp4", "fp8": "fp8_e4m3fn", "bf16": "bf16"}[precision]
    work = _counts(
        dtype,
        input_bytes + sum(reduction_w),
        2 * m * n,
        {dtype: 2 * m * h * (2 * k + n)},
        {"bf16": count * compute, **({"fp32": rescale_work} if rescale_work else {})},
        classes={
            key: v * count for key, v in visits.items() if key not in ("load", "store")
        },
    )
    work.update(
        evidence="analytical_design",
        schedule=dict(
            gate_up=a,
            down=b,
            active_aic=max(a["active_aic"], b["active_aic"]),
            active_aiv=lanes,
            vector_us=middle,
            overlap_waves=waves,
        ),
    )
    return NodeEstimate(
        "grouped_mlp/" + precision + "/" + kind,
        "extrapolated",
        body,
        "Capture declares a hypothetical fusion; on-chip handoff, unknown workspace excluded.",
        work,
    )


def head_group_mx(f, sem, chip):
    _arity(f, 4, 1)
    m, k = _shape(sem.get("logical_input_shape"), 2)
    g, n, kw = _shape(sem.get("logical_weight_shape"), 3)
    counts = sem.get("group_counts")
    _require(
        kw == k
        and k % 32 == 0
        and sem.get("weight_precision") == "fp4"
        and isinstance(counts, tuple)
        and len(counts) == g
        and all(type(c) is int and c >= 0 for c in counts)
        and sum(counts) == m
        and _roles(f)
        == tuple(
            enumerate(("activation", "weight", "weight_scale", "activation_scale"))
        ),
        "Expected local MX head partitions",
    )
    pg = ceil(k / 64)
    _packed(f, "inputs", 0, (m, k))
    _packed(f, "inputs", 1, (g, k, n), axis=-2, strides=(n * k // 2, 1, k // 2))
    _layout(f.inputs[2], E8, (g, pg, n, 2), strides=(2 * n * pg, 2, 2 * pg, 1))
    _layout(f.inputs[3], E8, (m, pg, 2))
    _layout(f.outputs[0], BF16, (m, n))
    reads = (m + sum(c > 0 for c in counts) * n) * (k // 2 + 2 * pg)
    t, s = _schedule(counts, n, k, "native4", reads, 2 * m * n, {}, chip, False)
    work = _counts("mxfp4", reads, 2 * m * n, {"mxfp4": 2 * m * n * k})
    work.update(schedule=s, evidence="analytical_design")
    return NodeEstimate(
        "grouped_matmul/output_heads_mxfp4",
        "extrapolated",
        t,
        "Native MX page costs with local head histogram.",
        work,
    )


def token_channel(f, sem, chip):
    _arity(f, 4, 1)
    m, k = _shape(sem.get("logical_input_shape"), 2)
    grouped = f.raw_name == "npu::npu_grouped_matmul"
    shape = _shape(sem.get("logical_weight_shape"), 3 if grouped else 2)
    n, kw = shape[-2:]
    _require(
        k == kw
        and sem.get("activation_precision") == sem.get("weight_precision") == "fp8"
        and sem.get("padding", (0, 0)) == (0, 0)
        and sem.get("accumulator_dtype", F32) == F32
        and _roles(f)
        == tuple(
            enumerate(("activation", "weight", "weight_scale", "activation_scale"))
        ),
        "Expected unpadded FP8 token/channel-scaled matmul",
    )
    if grouped:
        g = shape[0]
        counts = sem.get("group_counts")
        _require(
            sem.get("gemm_grouping") == "output_head_groups"
            and isinstance(counts, tuple)
            and len(counts) == g
            and all(type(c) is int and c >= 0 for c in counts)
            and sum(counts) == m,
            "Missing local head histogram",
        )
        _matrix_view(f.inputs[1], FP8, (g, k, n))
        _layout(f.inputs[2], F32, (g, n, 1))
    else:
        counts = (m,)
        _require(
            sem.get("activation_scale_layout") == "token_fp32"
            and sem.get("scale_layout") == "channel_fp32",
            "Missing dense scale contract",
        )
        _matrix_view(f.inputs[1], FP8, (k, n))
        _layout(f.inputs[2], F32, (n,))
    _layout(f.inputs[0], FP8, (m, k))
    _layout(f.inputs[3], F32, (m,))
    _layout(f.outputs[0], BF16, (m, n))
    reads = (m + sum(c > 0 for c in counts) * n) * (k + 4)
    classes = {"token_channel_scale": 2 * m * n, "cast": m * n}
    t, s = _schedule(counts, n, k, "native8", reads, 2 * m * n, classes, chip, False)
    # The first output tile must wait for its accumulator before applying the two scales.
    cycles = s["resource_cycles"]
    stages = [
        max(cycles["cube"], cycles["hbm"], cycles["panel_mte2"]),
        cycles["vector"],
    ]
    t += (min(stages) / s["waves"]) / (1000 * chip.frequency_ghz)
    work = _counts(
        "fp8_e4m3fn",
        reads,
        2 * m * n,
        {"fp8_e4m3fn": 2 * m * n * k},
        {"fp32": 3 * m * n},
        classes=classes,
    )
    work.update(schedule=s, evidence="analytical_design")
    return NodeEstimate(
        "grouped_matmul/output_heads_fp8" if grouped else "matmul/token_channel_fp8",
        "extrapolated",
        t,
        "FP8 Cube plus per-token and per-channel rescaling.",
        work,
    )
