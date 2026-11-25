"""First and continuation chunks in the Kimi prefill capture, using rank-local tensors."""

from dataclasses import replace
from math import prod

from . import vf
from operator_model.system_adapter import NodeEstimate
from operator_model.system_routes.vector import Tiles, _schedule
from operator_model.workload import _counts

CONV = "vllm_ascend::causal_conv1d_fn_reference"
NORM = "vllm_ascend::l2norm_fwd_reference"
KDA = "vllm_ascend::chunk_kda_fwd_reference"
INSERT = "inference::kimi_dsa_indexer_k_norm_quant_insert"
ZERO = "aten.zeros_like.default"
COPY = "aten.copy_.default"
NAMES = frozenset({CONV, NORM, KDA, INSERT, ZERO, COPY})
BF16 = "bfloat16"
FP32 = "float32"
FP4 = "float4_e2m1fn_x2"
E8 = "float8_e8m0fnu"
_BYTES = {"float8_e4m3fn": 1, BF16: 2, FP32: 4, FP4: 1, E8: 1, "uint8": 1, "int64": 8}


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


def _dtype(tensor):
    return tensor.dtype.removeprefix("torch.")


def _bytes(tensor):
    return prod(tensor.shape) * _BYTES[_dtype(tensor)]


def _ceil(n, divisor):
    return (n + divisor - 1) // divisor


def _tensor(tensor, dtype, shape):
    _require(
        _dtype(tensor) == dtype and tuple(tensor.shape) == tuple(shape),
        f"expected {dtype}{tuple(shape)}, received {_dtype(tensor)}{tensor.shape}",
    )


def _vector(chip, plan, **extra):
    result = _schedule(plan, chip)
    if result.latency_us is not None:
        result.work["schedule"].update(extra)
        result = replace(result, latency_us=result.latency_us + chip.kernel_overhead_us)
    return result


def _state_io(f, chip):
    (out,) = f.outputs
    _require(
        _dtype(out) == FP32 and len(out.shape) == 4, "Expected FP32 BHVK state tensor"
    )
    dest = f.inputs[0]
    _tensor(dest, FP32, out.shape)
    copying = f.raw_name == COPY
    _require(
        len(f.inputs) == (2 if copying else 1), "Unsupported state initialization/copy"
    )
    if copying:
        _tensor(f.inputs[1], FP32, dest.shape)
    n = prod(out.shape)
    plan = Tiles(
        "state/cache_copy" if copying else "state/zero_fill",
        n,
        128,
        4 * n if copying else 0,
        4 * n,
        0,
        setup="stream",
        work_source="Independent contiguous state stripes, write destination once; FP32 device kernel projection.",
    )
    return _vector(chip, plan, state_elements=n, reads_previous_destination=False)


def _norm(f, chip, sem):
    (x,) = f.inputs
    (y,) = f.outputs
    _tensor(x, BF16, y.shape)
    _tensor(y, BF16, x.shape)
    depth = x.shape[-1]
    _require(
        sem.get("normalization") == "l2" and depth == sem["head_dim"],
        "Expected last-axis L2 normalization",
    )
    rows = prod(x.shape[:-1])
    # x², reduction, reciprocal norm and rescale. The row stays in registers/UB.
    ops = rows * (3 * depth + 2)
    return _vector(
        chip,
        Tiles(
            "norm/l2_last_axis",
            rows * depth,
            depth,
            _bytes(x),
            _bytes(y),
            ops,
            work_source="BF16 row resident in UB; FP32 dot, rsqrt, rescale; uncalibrated.",
        ),
        rows=rows,
        depth=depth,
        reductions_per_row=1,
    )


def _insert(f, chip, sem):
    key, gamma, bias, positions, cache = f.inputs
    (out,) = f.outputs
    rows, depth = key.shape
    _require(
        sem.get("key_norm") == "layernorm"
        and sem.get("storage_precision") in ("fp4","fp8")
        and sem.get("writes") == "current query positions only"
        and depth == sem["logical_head_dim"]
        and depth % 32 == 0,
        "Expected fused LayerNorm, block32 FP4 and current-row insertion",
    )
    fp4 = sem["storage_precision"] == "fp4"
    record = depth // 2 + depth // 32 if fp4 else depth + 4
    _require(
        len(cache.shape) == 3
        and cache.shape[-1] == record == sem["row_bytes"]
        and rows == sum(sem["query_lens"]),
        "Invalid FP4 row/scale extent",
    )
    _tensor(key, BF16, (rows, depth))
    _tensor(gamma, FP32, (depth,))
    _tensor(bias, FP32, (depth,))
    _tensor(positions, "int64", (rows,))
    _tensor(out, "uint8", cache.shape)
    _tensor(cache, "uint8", out.shape)
    read = _bytes(key) + _bytes(gamma) + _bytes(bias) + _bytes(positions)
    # FP32 mean, squared deviations + variance, scale/shift, max/quantize/pack.
    ops = rows * (11 * depth + 5 * (_ceil(depth, 32) if fp4 else 1) + 4)
    return _vector(
        chip,
        Tiles(
            "indexer/key_layernorm_mx4_insert" if fp4 else "indexer/key_layernorm_fp8_insert",
            rows * depth,
            depth,
            read,
            rows * record,
            ops,
            work_source="One row per task, fused per-row reductions and indexed stores.",
        ),
        updated_rows=rows,
        cache_record_bytes=record,
        cache_capacity_rows=prod(cache.shape[:-1]),
        ops_per_group="LayerNorm + max(abs(x)) + scale + pack",
    )


def _conv(f, chip, sem):
    x, kernel, state = f.inputs
    (y,) = f.outputs
    rows, channels = x.shape
    taps = sem["kernel_size"]
    lens = sem["query_lens"]
    history = sem["has_initial_state"]
    _require(
        sem["phase"] == "prefill"
        and sem["activation"] == "silu"
        and sum(lens) == rows
        and len(history) == len(lens)
        and all(type(h) is bool for h in history)
        and taps > 0
        and all(n > 0 for n in lens),
        "Expected valid ragged BF16 prefill convolution",
    )
    for t, shape in (
        (x, (rows, channels)),
        (kernel, (taps, channels)),
        (state, (len(lens), taps - 1, channels)),
        (y, x.shape),
    ):
        _tensor(t, BF16, shape)
    # Tokens are independent given a K-1 token halo. Split the sequence to expose
    # work even when there are fewer channel stripes than cores.
    ct = min(channels, 256)
    tokens = min(64, max(1, chip.details["ub_bytes"] // (2 * ct * 8) - (taps - 1)))
    blocks = sum(_ceil(length, tokens) for length in lens)
    unique_read = _bytes(x) + _bytes(kernel) + sum(history) * (taps - 1) * channels * 2
    repeat = (blocks - len(lens)) * (taps - 1) * channels * 2
    writes = _bytes(y) + _bytes(state)
    plan = Tiles(
        "kda/causal_depthwise_conv_prefill",
        rows * channels,
        min(rows, tokens) * ct,
        unique_read,
        writes,
        rows * channels * (2 * taps + 4),
        traffic_read=repeat,
        work_source="Parallel time/channel tiles share K-1 halo; update only the final per-request state, SiLU on AIV.",
    )
    return _vector(
        chip,
        plan,
        channel_tile=ct,
        token_tile=tokens,
        time_blocks=blocks,
        initial_states=sum(history),
        duplicate_halo_bytes=repeat,
        logical_state_write_bytes=_bytes(state),
    )


def _chunk(f, chip, s):
    # Capture packs equal or ragged requests into [1,total,local_heads,D].
    y, final = f.outputs
    batch, heads, v, k = final.shape
    _, total, yh, yv = y.shape
    lengths = tuple(s["query_lens"])
    bits = {"bf16":16,"fp8":8,"fp4":4}[s["compute_precision"]]
    base = 3 if bits == 16 else 6
    _require(
        len(f.inputs) == base+5
        and s.get("accumulator_dtype",FP32) == FP32
        and s.get("algorithm") == "chunk"
        and s.get("state_layout") == "BHVK"
        and s.get("qk_normalization_location") == "before_chunk"
        and (bits != 4 or s.get("group_size") == 32)
        and len(lengths) == batch
        and sum(lengths) == total
        and all(n > 0 for n in lengths)
        and (yh, yv) == (heads, v)
        and k % 64 == v % 64 == 0,
        "Expected packed MXFP4/FP32 prefill chunk KDA; L2 normalization occurs separately",
    )
    c = s["chunk_size"]
    _require(c == 64, "Expected Kimi 64-token chunk algorithm")
    _tensor(y, BF16, (1, total, heads, v))
    _tensor(final, FP32, (batch, heads, v, k))
    for i, d in enumerate((k, k, v)):
        if bits == 4:
            _tensor(f.inputs[i], FP4, (1,total,heads,d//2))
            _tensor(f.inputs[i+3], E8, (1,total,heads,d//64,2))
        elif bits == 8:
            _tensor(f.inputs[i], "float8_e4m3fn", (1,total,heads,d))
            _tensor(f.inputs[i+3], FP32, (1,total,heads))
        else:
            _tensor(f.inputs[i], BF16, (1,total,heads,d))
    _tensor(f.inputs[base], BF16, (1,total,heads,k))
    _tensor(f.inputs[base+1], FP32, (1,total,heads))
    _require(_dtype(f.inputs[base+2]) == FP32 and f.inputs[base+2].shape in ((heads,), (1,1,heads,1)), "Expected broadcastable per-head gate bias")
    _tensor(f.inputs[base+3], FP32, (heads*k,))
    _tensor(f.inputs[base+4], FP32, final.shape)
    chunks = [_ceil(n, c) for n in lengths]
    chunk_heads = sum(chunks) * heads
    rows = total * heads
    square = heads * sum((n // c) * c * c + (n % c) ** 2 for n in lengths)
    cubic = heads * sum((n // c) * c**3 + (n % c) ** 3 for n in lengths)
    # QK/KK causal triangle + complementary diagonal, before block/WY inversion.
    dot4 = 2 * k * square
    wy16 = 2 * (k + v) * square
    chain16 = 4 * rows * k * v
    out16 = 2 * rows * k * v + 2 * v * square
    # Triangular substitution and gating in FP32. State is stored/accumulated in
    # FP32; BF16 operands are rounded before using 16-bit MMAD in the state products.
    vector_prepare = rows * (10 * k + 3 * v) + cubic // 3
    vector_chain = rows * v + 2 * chunk_heads * k * v
    dtype = {16:"bf16",8:"fp8_e4m3fn",4:"mxfp4"}[bits]
    cube = {"bf16":wy16+chain16+out16}
    cube[dtype] = cube.get(dtype,0)+dot4
    vec = vector_prepare + vector_chain
    # Split (request,head) across V stripes. A given stripe processes the chunks in order.
    vt = min(64, v, max(1, (chip.details["ub_bytes"] - 8192) // (12 * k)))
    chains = batch * heads * _ceil(v, vt)
    independent = min(chunk_heads, chip.cube_cores)
    ordered = min(chains, chip.cube_cores)
    lanes = min(chip.vector_cores, 2 * independent)
    chain_lanes = min(chip.vector_cores, 2 * ordered)

    def matrix_us(ops, bits, live):
        return ops / (1e6 * chip.cube_tops[bits] * live / chip.cube_cores)

    def vector_us(ops, live):
        return ops / (1e6 * chip.vector_tops["32"] * live / chip.vector_cores)

    unit=1000*chip.frequency_ghz
    pre=vf.stage(chip,'kda/chunk_prepare',unit*vector_us(vector_prepare,lanes),rows*k/lanes,
                 load=(3*rows*k+cubic/3)/lanes,convert=3*rows*k/lanes,exp=rows*k/lanes,
                 arithmetic=(6*rows*k+3*rows*v+cubic/3)/lanes,store=(rows*k+rows*v)/lanes)/unit
    chain=vf.stage(chip,'kda/chunk_state',unit*vector_us(vector_chain,chain_lanes),vector_chain/chain_lanes,
                   load=vector_chain/chain_lanes,arithmetic=vector_chain/chain_lanes,store=(rows*v+chunk_heads*k*v)/chain_lanes)/unit
    prep = max(
        matrix_us(dot4,str(bits),independent) + matrix_us(wy16, "16", independent),
        pre,
    )
    # Explicit critical chain bound matters for ragged tails as well as B=1.
    chain_share = max(chunks) * batch / sum(chunks)
    recur = (
        max(matrix_us(chain16, "16", ordered), chain)
        * chain_share
    )
    finish = matrix_us(out16, "16", independent)
    # First chunk has an explicit zero/copy node; the caller passes materialized
    # initial state even when all its entries are zero.
    read = sum(_bytes(t) for t in f.inputs)
    write = _bytes(y) + _bytes(final)
    hbm = (read + write) / (1000 * chip.hbm_bandwidth_gbs)
    stages = 3
    setup = stages * 4587 / (1000 * chip.frequency_ghz)
    work = _counts(dtype, read, write, cube, {"fp32": vec})
    work["schedule"] = {
        "algorithm": "three-stage chunked gated delta rule",
        "evidence": f"Resource projection of {dtype} Kimi chunk KDA; BF16 state products; no same-contract CA window.",
        "active_aic": independent,
        "active_aiv": lanes,
        "state_phase_aic": ordered,
        "chunk_tokens": c,
        "chunk_head_tasks": chunk_heads,
        "max_serial_chunks": max(chunks),
        "state_v_stripe": vt,
        "prepare_us": prep,
        "ordered_state_us": recur,
        "finalize_us": finish,
        "setup_us": setup,
        "shared_hbm_floor_us": hbm,
        "initial_state_bytes": _bytes(f.inputs[-1]),
        "scratch_assumption": "on-chip forwarding and tiled reuse; off-chip scratch unknown",
        "cube_input_dot_ops": dot4,
        "cube_fp4_dot_ops": dot4 if bits==4 else 0,
        "cube_bf16_state_ops": chain16,
        "state_accumulator": "fp32",
    }
    return NodeEstimate(
        f"kda/{dtype}_chunk_prefill",
        "extrapolated",
        setup + max(prep + recur + finish, hbm) + stages * chip.kernel_overhead_us,
        work["schedule"]["evidence"],
        work,
    )


def price(features, chip):
    name = features.raw_name
    if name in (ZERO, COPY):
        return _state_io(features, chip)
    sem = dict(dict(features.attributes)["semantic"])
    if name == KDA:
        return _chunk(features, chip, sem)
    if name == CONV:
        return _conv(features, chip, sem)
    if name == NORM:
        return _norm(features, chip, sem)
    if name == INSERT:
        return _insert(features, chip, sem)
    raise ValueError("Unknown prefill route")
