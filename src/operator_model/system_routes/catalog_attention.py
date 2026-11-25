"""Projected schedules for grouped-query, latent, pooled-index and gated-delta attention."""

from math import ceil

from . import vf
from . import attention_resources as attention

GDN = {"recurrent_gated_delta_rule_reference", "chunk_gated_delta_rule_fwd_reference"}
POOLED = {"kpool_indexer", "qsa_indexer", "qsa_compress"}
NAMES = (
    GDN
    | POOLED
    | {
        "sparse_mla",
        "qsa_attention",
        "low_precision_attention",
        "npu_fused_infer_attention_score_v2",
    }
)
BITS = {"bf16": 16, "fp4": 4, "fp8": 8}


def _inputs(f, sem):
    precision = sem["compute_precision"]
    bits = BITS[precision]
    shapes = tuple(sem["logical_input_shapes"])
    expected = {
        16: {"bfloat16"},
        8: {"float8_e4m3fn"},
        4: {"float4_e2m1fn_x2", "uint8"},
    }[bits]
    attention._check(
        shapes
        and all(
            t.dtype.removeprefix("torch.") in expected for t in f.inputs[: len(shapes)]
        ),
        "Operand dtype disagrees with declared compute precision",
    )
    for t, shape in zip(f.inputs, shapes):
        attention._check(
            t.shape[:-1] == tuple(shape[:-1])
            and t.shape[-1]
            == (attention.divup(shape[-1], 2) if bits == 4 else shape[-1]),
            "Invalid packed operand extent",
        )
    if bits < 16:
        attention._check(
            sem["operand_count"] == len(shapes) and len(f.inputs) >= 2 * len(shapes),
            "Expected explicit scale operands",
        )
    return bits, shapes


def _qsa_visits(query, keys, selected, ratio, budget):
    """Sum capped completed-group tokens and raw tail per causal row in O(1)."""
    attention._check(
        0 < query <= keys and 0 < selected <= keys, "Invalid frozen QSA selection"
    )
    cutoff = attention.divup(budget, ratio) * ratio

    def prefix(n):
        # Inclusive positions [0,n]; below cutoff every token is selected.
        full = min(n, cutoff - 1)
        total = full * (full + 1) // 2
        if n >= cutoff:
            count = n - cutoff + 1
            groups, tail = divmod(count, ratio)
            total += (
                budget * count
                + groups * ratio * (ratio - 1) // 2
                + tail * (tail - 1) // 2
            )
        return total

    start = keys - query
    stop = min(keys, selected)
    frozen = keys - max(start, stop)
    return (
        prefix(stop)
        - prefix(min(start, stop))
        + frozen * (min(stop // ratio * ratio, budget) + stop % ratio)
    )


def _dense(f, chip, s, name):
    bits, logical = _inputs(f, s)
    rows, heads, value = f.outputs[0].shape
    lens, keys = tuple(s["query_lens"]), tuple(s["kv_lens"])
    attention._check(
        sum(lens) == rows and len(keys) == len(lens) and all(k >= 0 for k in keys),
        "Invalid local Q/K lengths",
    )
    sparse = name in ("sparse_mla", "qsa_attention")
    latent = (
        name == "sparse_mla"
        or s.get("mla_decode", False)
        or name == "npu_fused_infer_attention_score_v2"
    )
    dim = logical[0][-1]
    if len(logical) == 5:
        dim += logical[3][-1]
    elif latent:
        dim += logical[1][-1]
    kvheads = s.get("num_kv_heads", 1 if latent else logical[1][-2])
    attention._check(
        0 < kvheads <= heads and heads % kvheads == 0, "Invalid grouped KV heads"
    )
    pairs = s.get("effective_qk_pairs")
    if name == "qsa_attention":
        # Budget counts selected individual tokens, plus at most ratio-1 tail tokens.
        ratio, budget = s["compression_ratio"], s["selection_budget"]
        attention._check(ratio > 0 and budget > 0, "Invalid QSA budget")
        pairs = sum(
            _qsa_visits(q, k, selected, ratio, budget)
            for q, k, selected in zip(lens, keys, s["selection_kv_lens"], strict=True)
        )
    elif name == "sparse_mla":
        pairs = s["selected_pair_bound"]
    attention._check(
        pairs is not None and 0 <= pairs <= sum(q * k for q, k in zip(lens, keys)),
        "Missing valid pair bound",
    )
    qt = min(16, max(lens))
    ht = min(16, heads // kvheads)
    groups = (
        sum(attention.divup(q, qt) for q in lens)
        * kvheads
        * attention.divup(heads // kvheads, ht)
    )
    pages = max(1, ceil(pairs / max(1, rows * 128)))
    splits = min(pages, max(1, chip.cube_cores // groups))
    jobs = groups * splits
    cube = 2 * heads * pairs * (dim + value)
    visits = heads * pairs
    vec = visits * (6 + 3 * (bits < 16)) + rows * heads * value * 3 * pages
    rc = attention._rate(chip.cube_tops[str(bits)], chip.cube_cores, chip.frequency_ghz)
    rv = attention._rate(chip.vector_tops["32"], chip.vector_cores, chip.frequency_ghz)
    divisor = jobs
    lanes = visits / divisor
    soft = vf.stage(
        chip,
        "attention/catalog_softmax",
        vec / (rv * divisor),
        lanes,
        exp=lanes,
        reduce=2 * lanes,
        arithmetic=vec / divisor - 3 * lanes,
        convert=(bits < 16) * 2 * lanes,
        load=lanes,
        store=lanes,
    )
    # A page pays all three dependent stages once, then the two engines overlap.
    count = attention.divup(pages, splits)
    matrix = cube / (rc * divisor)
    cycles = max(matrix, soft) + min(matrix, soft) / count
    merge_visits = (splits - 1) * qt * ht * (value + 2)
    merge = vf.stage(
        chip,
        "attention/split_merge",
        2 * merge_visits / rv,
        merge_visits,
        load=2 * merge_visits,
        arithmetic=2 * merge_visits,
        store=merge_visits,
    )
    cycles += merge
    vec += 2 * (splits - 1) * rows * heads * (value + 2)
    fraction = min(1, pairs / max(1, sum(keys))) if sparse else 1
    # Exclude repeated compressed latent K/V descriptor in the five-operand MLA route.
    q_ids = {0, 3} if len(logical) == 5 else ({0, 1} if latent else {0})
    read = 0
    for i, t in enumerate(f.inputs):
        op = i % len(logical)
        if (
            latent
            and len(logical) == 5
            and op == 2
            and i < len(logical) * (1 + (bits < 16))
        ):
            continue
        read += attention.nbytes(t) * (
            1 if op in q_ids or (sparse and i == len(f.inputs) - 1) else fraction
        )
    return attention._schedule(
        chip,
        name,
        cube,
        vec,
        bits,
        ceil(read),
        sum(map(attention.nbytes, f.outputs)),
        jobs,
        dict(query=qt, heads=ht, keys=128, kv_splits=splits),
        cycles,
        4587,
        "Projected online QK/softmax/PV; selected sparse pairs use the capture bound; no same-contract CA.",
        vector_partners=1,
        phases=dict(
            effective_pairs=pairs,
            qk_width=dim,
            value_width=value,
            pages=pages,
            merge_cycles=merge,
            cube_cycles_per_wave=matrix,
            vf_cycles_per_wave=soft,
        ),
    )


def _gdn(f, chip, s, name):
    bits, logical = _inputs(f, s)
    attention._check(
        len(logical) == 3 and s["state_layout"] == "BHVK",
        "Expected GDN Q,K,V and BHVK state",
    )
    b, h, v, k = f.inputs[-1].shape
    lens = s["query_lens"]
    n = sum(lens)
    attention._check(
        len(lens) == b
        and logical[0][-1] == k
        and logical[2][-2:] == (h, v)
        and f.outputs[0].shape == (1, n, h, v),
        "Invalid grouped head or state layout",
    )
    stripe = min(v, 64, max(1, (chip.details["ub_bytes"] - 8192) // (12 * k)))
    chunked = name.startswith("chunk_")
    c = s["chunk_size"] if chunked else 1
    tiles = b * h * attention.divup(v, stripe)
    if chunked:
        square = h * sum((q // c) * c * c + (q % c) ** 2 for q in lens)
        # QK/KK and WY update, BF16 operands used for the FP32 state-products.
        dot = 2 * k * square
        state = 6 * n * h * k * v + 2 * square * (k + 2 * v)
        vec = (
            n * h * (10 * k + 4 * v + 8)
            + h * sum((q // c) * c**3 + (q % c) ** 3 for q in lens) // 3
        )
    else:
        dot = state = 0
        vec = n * h * (7 * k * v + 8 * k + 4 * v + 8)
    factor = max(lens) * b / n / tiles
    rv = attention._rate(chip.vector_tops["32"], chip.vector_cores, chip.frequency_ghz)
    visits = n * h * factor
    simd = vf.stage(
        chip,
        "gdn/gated_delta",
        vec * factor / rv,
        visits,
        exp=3 * visits,
        divide=2 * visits,
        reduce=visits * (2 * k + v),
        load=visits * (k + v + 3),
        store=visits * v,
        convert=visits * (2 * k + v) * (bits < 16),
        arithmetic=vec * factor,
    )

    def rc(width):
        return attention._rate(
            chip.cube_tops[str(width)], chip.cube_cores, chip.frequency_ghz
        )

    compute = (dot / rc(bits) + state / rc(16)) * factor if chunked else 0
    reads = sum(map(attention.nbytes, f.inputs))
    writes = attention.nbytes(f.outputs[0]) + attention.nbytes(f.inputs[-1])
    result = attention._schedule(
        chip,
        name,
        dot + state,
        vec,
        bits,
        reads,
        writes,
        tiles,
        dict(state_v=stripe, state_k=k, chunk=c),
        max(simd, compute),
        4587,
        "Projected grouped delta rule; independent value stripes, serial per-request state; CA N/A.",
        cv=chunked,
        vector_partners=1,
        phases=dict(
            state_dtype="fp32",
            gate="scalar per value head",
            serial_chunks=max(attention.divup(q, c) for q in lens),
        ),
    )
    if chunked and bits != 16:
        result.work["cube_ops_by_dtype"] = {
            {4: "mxfp4", 8: "fp8_e4m3fn"}[bits]: dot,
            "bf16": state,
        }
    return result


def _pooled(f, chip, s, name):
    q = f.inputs[0]
    write = sum(map(attention.nbytes, f.outputs))
    if name == "qsa_compress":
        count, _, width = f.outputs[0].shape
        ratio = s["ratio"]
        values = count * width * ratio
        tiles = max(1, count * attention.divup(width, 128))
        cv = False
        cube = 0
        vec = values
        read = 2 * values
    else:
        rows, heads, width = q.shape
        key = f.inputs[1 if name == "qsa_indexer" else 2]
        batch, capacity = key.shape[:2]
        ratio = s.get("pool_size", s.get("ratio"))
        attention._check(
            ratio > 0 and rows % batch == 0, "Invalid pooled index geometry"
        )
        length = s.get("kv_lens")
        selected = (
            sum(attention.divup(x, ratio) for x in length)
            if length
            else batch * capacity
        )
        pairs = rows // batch * selected
        cube = 2 * pairs * heads * width
        vec = pairs * (3 * heads + 8)
        read = attention.nbytes(q) + min(attention.nbytes(key), selected * width * 2)
        if name == "kpool_indexer":
            read += attention.nbytes(f.inputs[1])
        tiles = (
            batch
            * attention.divup(rows // batch, 4)
            * max(1, attention.divup(capacity, 256))
        )
        cv = True
    slots = tiles
    rv = attention._rate(chip.vector_tops["32"], chip.vector_cores, chip.frequency_ghz)
    rc = attention._rate(chip.cube_tops["16"], chip.cube_cores, chip.frequency_ghz)
    lanes = vec / slots
    service = vf.stage(
        chip,
        name,
        lanes / rv,
        lanes,
        load=lanes,
        reduce=lanes / 2,
        arithmetic=lanes / 2,
        store=lanes / 4,
    )
    return attention._schedule(
        chip,
        name,
        cube,
        vec,
        16,
        read,
        write,
        tiles,
        dict(pool_tokens=ratio),
        max(cube / (slots * rc), service),
        4587,
        "Pooled index projection; absent per-query history uses allocated pool count as an upper bound; ranking and merges projected.",
        cv=cv,
        vector_partners=1,
        phases=dict(
            history_bound="visible_lengths" if s.get("kv_lens") else "capacity"
        ),
    )


def price(f, chip, s, name):
    if name in GDN:
        return _gdn(f, chip, s, name)
    if name in POOLED:
        return _pooled(f, chip, s, name)
    return _dense(f, chip, s, name)
