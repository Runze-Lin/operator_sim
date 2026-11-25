"""Per-device schedules for the attention and sequence-state nodes in the decode graph."""

from math import ceil, prod

from operator_model.operators.attention.indexer import _prices
from . import vf
from operator_model.system_adapter import NodeEstimate


NAMES = frozenset(
    {
        "vllm_ascend::quant_lightning_indexer_reference",
        "vllm_ascend::sparse_attn_sharedkv_reference",
        "npu::npu_fused_infer_attention_score",
        "inference::kimi_dsa_indexer_score",
        "inference::lightning_indexer",
        "inference::kimi_gated_dsa_attention",
        "_C_ascend::compressor",
        "_C_ascend::compressor_metadata",
        "vllm_ascend::causal_conv1d_update_reference",
        "inference::c2_ring_compressor",
        "inference::candidate_blocks",
        "inference::compressor_source_positions",
        "vllm_ascend::recurrent_kda_reference",
    }
)
BYTES = {
    "bfloat16": 2,
    "float32": 4,
    "float8_e4m3fn": 1,
    "float8_e8m0fnu": 1,
    "float4_e2m1fn_x2": 1,
    "int64": 8,
    "int32": 4,
    "uint8": 1,
}
DTYPE = {4: "mxfp4", 8: "fp8_e4m3fn", 16: "bf16"}
INDEXERS = {
    "quant_lightning_indexer_reference",
    "lightning_indexer",
    "kimi_dsa_indexer_score",
}


def divup(n, d):
    return (n + d - 1) // d


def nbytes(t):
    # Packed storage dimensions from SIMPlatform, not capture's separate logical_shape.
    return prod(t.shape) * BYTES[t.dtype.removeprefix("torch.")]


def _check(condition, reason):
    if not condition:
        raise ValueError(reason)


def _rate(tops, cores, clock):
    """One core's operations/cycle from a full chip peak in TOP/s."""
    return tops * 1000 / (cores * clock)


def _schedule(
    chip,
    name,
    cube,
    vec,
    bits,
    reads,
    writes,
    tiles,
    tile,
    pipeline,
    launch,
    evidence,
    *,
    cv=True,
    phases=None,
    hbm=None,
    status="extrapolated",
    active=None,
):
    clock = chip.frequency_ghz
    cores = chip.cube_cores if cv else chip.vector_cores
    available = min(cores, chip.vector_cores) if cv else cores
    live = min(available, tiles) if active is None else active
    waves = divup(tiles, live)
    total_bytes = reads + writes if hbm is None else hbm
    hbm_us = total_bytes / (chip.hbm_bandwidth_gbs * 1000)
    local = waves * pipeline / (clock * 1000)
    startup = launch / (clock * 1000)
    duration = chip.kernel_overhead_us + startup + max(local, hbm_us)
    work = {
        "cube_ops": cube,
        "vec_ops": vec,
        "cube_ops_by_dtype": {DTYPE[bits]: cube} if cube else {},
        "vec_ops_by_dtype": {"fp32": vec} if vec else {},
        "logical_read_bytes": reads,
        "logical_write_bytes": writes,
        "physical_hbm_bytes": None,
        "schedule": {
            "tile": tile,
            "tiles": tiles,
            "waves_on_critical_core": waves,
            "active_aic": live if cv else 0,
            "active_aiv": min(
                chip.vector_cores,
                live
                * (min(2, max(1, chip.vector_cores // chip.cube_cores)) if cv else 1),
            ),
            "dependency_path_cycles_per_wave": pipeline,
            "launch_us": startup,
            "additional_kernel_overhead_us": chip.kernel_overhead_us,
            "local_pipeline_us": local,
            "assumed_shared_hbm_us": hbm_us,
            "stage_budget": phases or {},
            "evidence": evidence,
        },
    }
    return NodeEstimate("system_attention/" + name, status, duration, evidence, work)


def _index(f, chip, sem, name):
    q, key = f.inputs[0], f.inputs[2 if name == "kimi_dsa_indexer_score" else 1]
    rows, heads, logical_d = q.shape
    packed = name == "kimi_dsa_indexer_score"
    bits = 4 if packed else 16
    d = sem["index_head_dim"] if packed else logical_d
    if packed:
        _check(
            q.dtype.endswith("uint8")
            and sem["storage_precision"] == "fp4"
            and q.shape[-1] * 2 == d,
            "Expected packed MXFP4 query with separate E8M0 scale",
        )
    else:
        _check(
            q.dtype.endswith("bfloat16")
            and key.dtype.endswith("bfloat16")
            and key.shape[-1] == d,
            "Expected BF16 contiguous shared-key projection",
        )
    b = key.shape[0]
    _check(
        rows % b == 0 and d == 128 and f.outputs[0].dtype.endswith("int32"),
        "Expected local shared-key top-k indices",
    )
    nq = rows // b
    lens = sem.get(
        "kv_lens", (sem.get("valid_key_rows_per_request", key.shape[1]),) * b
    )
    _check(
        len(lens) == b and all(0 < n <= key.shape[1] for n in lens),
        "Invalid per-request visible K length",
    )
    k = f.outputs[0].shape[-1]
    _check(k == sem.get("index_topk", k) and k > 0, "Invalid requested top-k extent")
    source = (
        name == "lightning_indexer"
        and sem.get("index_source") == sem.get("kv_source") == 20
        and sem.get("reference")
        == "vllm_ascend/models/deepseek_v41/model.py:DeepseekV41Attention"
    )
    p = _prices()
    rc = _rate(chip.cube_tops[str(bits)], chip.cube_cores, chip.frequency_ghz)
    rv = _rate(chip.vector_tops["32"], chip.vector_cores, chip.frequency_ghz)
    partners = min(2, max(1, chip.vector_cores // chip.cube_cores))
    # Four queries share a key stream; H64 uses the library's split-M path.
    r = min(nq, 4)
    m = 2 * heads * divup(r, 2)
    length = max(lens)
    pages = divup(length, 128)
    read_row = (d // 2 + divup(d, 32)) if packed else 2 * d
    dma = chip.details.get("mte2_bandwidth_gbs")
    read_page = (
        128 * read_row * chip.frequency_ghz / dma
        if dma
        else p["page_ii_cycles"] * read_row / (2 * 128)
    )
    # H64 Q4 emits two M128 MMADs per K page (verified on the ordinary LI ELF).
    mma_m = min(m, 128)
    mmads_per_page = divup(m, mma_m)
    visits = divup(r, partners)*128*heads
    weighted = vf.stage(chip,'indexer/weighted_relu_reduce',divup(r,partners)*128*(3*heads-1)/rv,visits,
                       known=bits==16,load=2*visits,compare=visits,arithmetic=visits,reduce=visits,store=visits/heads)
    page = max(
        read_page,
        mmads_per_page * 2 * mma_m * 128 * d / rc,
        weighted,
    )
    scans = divup(length, 8192)
    blocks = divup(length, 256) + (scans - 1) * divup(k, 256)
    sf, sg = p["histogram_span"]
    gf, gg = p["histogram_gap"]
    rank = (
        scans * sf
        + blocks * sg
        + max(0, scans - 1) * gf
        + max(0, blocks - divup(min(length, 8192) + k, 256)) * gg
    )
    rank *= p["vector_ops_cycle"] / rv
    rank = vf.stage(chip,'indexer/histogram_rank',rank,length,known=bits==16,load=8*blocks*256,
                    compare=8*blocks*256,logical=8*blocks*256,reduce=8*blocks*256,store=k)
    sort = divup(r, partners) * rank
    producer = 0.0
    # A side-output of the full candidate-source layer, published separately.
    if source:
        # Same logits before top-512. Independently rank all block maxima.
        num_blocks = divup(length, 8)
        producer = divup(r, partners) * (
            (length + num_blocks) / rv
            + (sf + divup(num_blocks, 256) * sg) * p["vector_ops_cycle"] / rv
        )
    if source:
        producer=vf.stage(chip,'indexer/source_block_rank',producer,length,load=length,reduce=length,
                          compare=8*num_blocks,store=num_blocks)
    cycles = (
        pages * page
        + p["mmad_fixed_cycles"]
        + p["sort_handoff_cycles"]
        + sort
        + producer
        + p["store_drain"][0]
    )
    pairs = nq * sum(lens)
    # Weighted ReLU score, eight histogram passes per 256-key block, and
    # optional extrema/ranking on the producer's eight-position blocks.
    vector_work = (3 * heads - 1) * pairs + 8 * 256 * blocks * rows
    if source:
        vector_work += nq * sum(length + 9 * divup(length, 8) for length in lens)
    read = sum(nbytes(x) for x in f.inputs if x is not key) + sum(lens) * read_row
    # The final input in Kimi is a reusable output/scratch for top-k, not keys to score.
    if packed:
        read -= nbytes(f.inputs[-1])
    write = sum(nbytes(x) for x in f.outputs)
    ev = "Ordinary 950 BF16 LightningIndexer H8/H32 and H64/top1024 kernel evidence; long-K loop, low-bit extension and optional block selection projected."
    return _schedule(
        chip,
        name,
        2 * pairs * heads * d,
        vector_work,
        bits,
        read,
        write,
        b * divup(nq, 4),
        {
            "queries": 4,
            "page_keys": 128,
            "physical_m": mma_m,
            "mmads_per_key_page": mmads_per_page,
            "trunk": 8192,
            "top_k": k,
        },
        cycles,
        p["first_issue_cycles"],
        ev,
        phases={
            "page_interval_cycles": page,
            "scan_pages": pages,
            "topk_cycles": sort,
            "source_block_max_and_rank_cycles": producer,
            "layer20_produces_block_ids": source,
            "query_key_pairs": pairs,
        },
    )


def _attention(f, chip, sem, name):
    rows, h, dout = f.outputs[0].shape
    qlens = sem["query_lens"]
    b = len(qlens)
    _check(
        sum(qlens) == rows and all(n > 0 for n in qlens),
        "Query length metadata mismatch",
    )
    pairs = sem["effective_qk_pairs"]
    bits = {"bf16": 16, "fp8": 8, "fp4": 4}[sem["compute_precision"]]
    dense = name == "npu_fused_infer_attention_score"
    kimi = name == "kimi_gated_dsa_attention"
    _check(
        len(sem["kv_lens"]) == b
        and h > 0
        and pairs > 0
        and pairs <= sum(q * length for q, length in zip(qlens, sem["kv_lens"])),
        "Attention pair count exceeds valid local K rows",
    )
    _check(
        dense and bits == 16 or not dense and bits in (4, 8),
        "Unexpected attention precision",
    )
    keys = divup(pairs, rows)
    dk = dout + sem["qk_rope_head_dim"] if kimi else dout
    kvheads = sem["num_kv_heads"] if dense else 1
    _check(h % kvheads == 0 and pairs > 0, "Invalid GQA geometry")
    partners = min(2, max(1, chip.vector_cores // chip.cube_cores))
    # QK and PV with online softmax in between. Independent Q rows and KV splits need
    # partial-state merge only when multiple cores split the same KV row.
    qtile = min(max(qlens), 4 if not dense else 16)
    headtile = min(16, h // kvheads)
    kvtile = 128
    groups = (
        sum(divup(n, qtile) for n in qlens) * kvheads * divup(h // kvheads, headtile)
    )
    pages = divup(keys, kvtile)
    splits = min(pages, max(1, chip.cube_cores // groups))
    jobs = groups * splits
    rc = _rate(chip.cube_tops[str(bits)], chip.cube_cores, chip.frequency_ghz)
    rv = _rate(chip.vector_tops["32"], chip.vector_cores, chip.frequency_ghz) * partners
    useful = qtile * headtile
    padded = 16 * divup(useful, 16)
    # QK and P@V use the same Cube. For low precision, quantize probabilities
    # back to the input precision before P@V; model counts this vector work.
    vector_per_pair = 6 + (3 if bits < 16 else 0)
    qk = 2 * padded * kvtile * dk / rc
    pv = 2 * padded * kvtile * dout / rc
    soft = (vector_per_pair * useful * kvtile + 3 * useful * dout) / rv
    visits=useful*kvtile/partners; acc=useful*dout/partners
    soft=vf.stage(chip,'attention/softmax',soft,visits,known=dense,load=visits+acc,exp=visits,
                  reduce=2*visits,arithmetic=2*visits+3*acc,convert=2*visits if bits<16 else 0,
                  divide=visits if bits<16 else 0,store=visits+acc)
    drow = dk + dout if dense else dk
    key_bytes = drow * bits / 8 + (
        divup(drow, 32) if bits == 4 else (4 if bits == 8 else 0)
    )
    port = (
        chip.details.get("mte2_bandwidth_gbs")
        or 2 * 128 * 128 * chip.frequency_ghz / 416
    )
    fetch = kvtile * key_bytes * chip.frequency_ghz / port
    local_pages = divup(pages, splits)
    cycles = fetch + qk + soft + pv + (local_pages - 1) * max(fetch, qk + pv, soft)
    merge = 0 if splits == 1 else 2 * (splits - 1) * useful * (dout + 2) / rv
    if splits>1:
        count=(splits-1)*useful*(dout+2)/partners
        merge=vf.stage(chip,'attention/split_merge',merge,count,known=dense,load=2*count,arithmetic=2*count,store=count)
    read = nbytes(f.inputs[0]) + int(
        (sum(sem["kv_lens"]) * kvheads if dense else pairs) * key_bytes
    )
    if kimi:
        read += sum(nbytes(f.inputs[i]) for i in (1, 3, 4, -1))
    if not dense and not kimi:
        read += sum(
            nbytes(t)
            for t in f.inputs
            if t.dtype.endswith("int32") or t.dtype.endswith("int64")
        )
    if not dense and not kimi:
        read += nbytes(f.inputs[sem["operand_count"]]) + nbytes(
            f.inputs[-1]
        )  # query scales and head sink
    write = sum(nbytes(x) for x in f.outputs)
    evidence = "Resource projection of KV gather, QK–online softmax–PV reused across heads; materialization and head/KV partitions assumed."
    if dense:
        evidence = "CANN 9.2.1 aclnnFusedInferAttentionScoreV5 BF16 GQA Y/LSE DT/PR component CA; streaming interleaved history plus query-tail segments and production length projected."
    return _schedule(
        chip,
        name,
        2 * pairs * h * (dk + dout),
        pairs * h * vector_per_pair + 3 * rows * h * dout * pages,
        bits,
        read,
        write,
        jobs,
        {"query": qtile, "heads": headtile, "keys": kvtile, "kv_splits": splits},
        cycles + merge,
        20123 if dense else 4587,
        evidence,
        phases={
            "qk_cycles": qk,
            "softmax_cycles": soft,
            "pv_cycles": pv,
            "fetch_cycles": fetch,
            "merge_cycles": merge,
            "pages_per_shard": local_pages,
            "effective_pairs": pairs,
        },
    )


def _state(f, chip, sem, name):
    output = f.outputs[0]
    ins = f.inputs
    rv = _rate(chip.vector_tops["32"], chip.vector_cores, chip.frequency_ghz)
    bw = chip.details.get("mte2_bandwidth_gbs") or 78 * chip.frequency_ghz
    ticks_per_byte = chip.frequency_ghz / bw
    ub = chip.details["ub_bytes"]
    label = "Designed AIV schedule; no matching full device timing."
    if name == "recurrent_kda_reference":
        b, q, h, d = output.shape
        v, k = ins[-1].shape[-2:]
        _check(
            ins[-1].dtype.endswith("float32") and d == v and k > 0,
            "Expected FP32 BHVK state",
        )
        vt = min(v, max(1, (ub - 8192) // (3 * 4 * k)))
        # Keep a stripe of state resident, iterate the token chain in order.
        vt = min(vt, 32)
        tiles = b * h * divup(v, vt)
        statebytes = b * h * v * k * 4
        read = sum(nbytes(x) for x in ins[:-1]) + statebytes
        write = nbytes(output) + statebytes
        vec = (
            q * b * h * (7 * v * k + 5 * k + 4 * v) + 3 * b * h * q * k
        )  # unpack/dequantize and norm
        tileops = q * (7 * vt * k + 8 * k + 4 * vt)
        setup = 4 * vt * k * ticks_per_byte
        recur=vf.stage(chip,'kda/recurrent_step',tileops/(q*rv),vt*k,load=3*vt*k+3*k,
                      convert=3*k,reduce=2*vt*k+k,arithmetic=5*vt*k+4*k+4*vt,store=vt*k+vt)
        cycles = (
            2 * setup
            + q * max(recur, 2 * (3 * k + vt) * ticks_per_byte)
            + q * 2 * ceil(k / 64).bit_length() * 8
        )
        return _schedule(
            chip,
            name,
            0,
            vec,
            16,
            read,
            write,
            tiles,
            {"state_v": vt, "state_k": k, "serial_steps": q},
            cycles,
            3750,
            label,
            cv=False,
        )
    if name == "causal_conv1d_update_reference":
        rows, d = output.shape
        w = ins[1].shape[0]
        b, state, _ = ins[2].shape
        _check(
            rows % b == 0
            and sem.get("activation") == "silu"
            and ins[0].dtype.endswith("bfloat16"),
            "Expected verify BF16 SiLU",
        )
        q = rows // b
        ct = min(d, 512)
        tiles = b * divup(d, ct)
        # Each independent channel stripe loads old state and kernel once.
        read = 2 * (rows * d + w * d + b * (w - 1) * d)
        write = 2 * (rows * d + b * min(state, q + w - 1) * d)
        vec = rows * d * (2 * w + 4)
        move = 2 * ct * (q + w + w - 1 + q + min(state, q + w - 1)) * ticks_per_byte
        issue = q * ct * (2 * w + 4) / rv
        visits=q*ct
        issue=vf.stage(chip,'conv/causal_silu',issue,visits,known=True,load=2*w*visits,arithmetic=(w+2)*visits,
                       exp=visits,divide=visits,store=visits)
        # One core services each 512-channel stripe in the two-SKU fixture.
        return _schedule(
            chip,
            name,
            0,
            vec,
            16,
            read,
            write,
            tiles,
            {"channels": ct, "steps": q, "kernel": w},
            max(move, issue) + 2 * w * 32,
            16418,
            "BF16 SiLU aclnnCausalConv1dUpdate; B1 Q9 D4608 K4 same-geometry DT/PR CA, B/D tiling extrapolated.",
            cv=False,
            phases={"transfer_cycles": move, "simd_cycles": issue},
        )
    if name == "c2_ring_compressor":
        rows, d = ins[0].shape
        out_rows = output.shape[0]
        b = ins[3].shape[0]
        read = 8 * rows * d + 8 * min(2 * b, rows) * d
        write = 8 * rows * d + nbytes(output)
        vec = (6 * 2 + 2) * out_rows * d
        groups = b
        channels = min(128, d)
        tiles = groups * divup(d, channels)
        per = divup(rows, b)
        ring=vf.stage(chip,'ring_compress/pair',8*channels/rv,channels,load=4*channels,arithmetic=4*channels,
                      exp=2*channels,divide=2*channels,store=2*channels)
        cycles = (
            per * max(ring, 16 * channels * ticks_per_byte)
            + 2 * channels * ticks_per_byte
        )
        return _schedule(
            chip,
            name,
            0,
            vec,
            16,
            read,
            write,
            tiles,
            {"channels": channels, "history": 2, "serial_steps": per},
            cycles,
            2200,
            label,
            cv=False,
        )
    if name == "compressor_metadata":
        count, _, _, d = output.shape
        read = count * 4 * d + sum(nbytes(x) for x in ins[2:])
        write = sum(nbytes(x) for x in f.outputs)
        vec = 8 * count
    elif name == "compressor_source_positions":
        count = output.shape[0]
        d = 1
        read = count * 8
        write = count * 8
        vec = 2 * count
    else:
        _check(
            name == "candidate_blocks"
            and sem.get("block_size") == 8
            and sem.get("source_layer") == 20
            and output.shape[-1] == 2048,
            "Candidate side output requires the captured eight-key blocks and 2048-block budget",
        )
        # The upstream LI produces ranked IDs; this graph primitive publishes them.
        count, d = output.shape
        read = 4 * count * d
        write = read
        vec = count * d
        label = "Block maxima + latest pin + block top-k charged to source layer 20 LI while logits are live. This node only publishes its side output."
    tiles = max(1, count)
    visits=vec/tiles
    meta=vf.stage(chip,name,visits/rv,visits,load=(read/4)/tiles,arithmetic=0 if name=='candidate_blocks' else visits,
                  store=(write/4)/tiles,bits=32)
    cycles = max((read + write) * ticks_per_byte / tiles, meta) + 64
    return _schedule(
        chip,
        name,
        0,
        vec,
        16,
        read,
        write,
        tiles,
        {"metadata_rows": 1, "width": d},
        cycles,
        1200,
        label,
        cv=False,
    )


def _compress(f, chip, sem, name):
    x, w1, w2 = f.inputs[:3]
    m, k = x.shape
    n = w1.shape[0]
    ratio = sem["cmp_ratio"]
    coff = sem["coff"]
    pooled, d = f.outputs[0].shape
    _check(
        w1.shape == w2.shape
        and n == coff * d
        and sem.get("activation_precision") == "fp8",
        "Unsupported compressor fusion contract",
    )
    bits = 8
    # Two projections share each input tile, independent output column stripes.
    bm, bn, bk = 16, 64, 128
    rc = _rate(chip.cube_tops["8"], chip.cube_cores, chip.frequency_ghz)
    rv = _rate(chip.vector_tops["32"], chip.vector_cores, chip.frequency_ghz)
    groups = divup(m, bm) * divup(n, bn)
    npage = divup(k, bk)
    expand = 2 * bn * bk / rv
    expand=vf.stage(chip,'compressor/unpack_weight',expand,2*bn*bk,load=2*bn*bk,permute=2*bn*bk,
                    convert=2*bn*bk,store=2*bn*bk,bits=8)
    mac = 4 * bm * bn * bk / rc
    pool = pooled * d * (8 * ratio * coff + 8)
    u=pooled*d/groups; a=m*k/groups
    pool_cycles=vf.stage(chip,'compressor/pool_rope',(pool+3*m*k)/(rv*groups),u+a,load=2*u*ratio*coff+a,
                         exp=u*ratio*coff,reduce=2*u*ratio*coff,arithmetic=5*u*ratio*coff+6*u+2*a,
                         convert=a,divide=u,permute=u,store=u+a)
    cycles = (
        expand
        + (npage - 1) * max(mac, expand)
        + mac
        + pool_cycles
    )
    read = (
        nbytes(x)
        + n * k
        + sum(nbytes(x) for x in f.inputs[-2:])
        + pooled * ratio * 2 * n * 4
    )
    # State stores full projection values + gate; callers consume only the pooled output.
    if pooled:
        read += sum(nbytes(t) for t in f.inputs[4:8])
    read += sum(nbytes(t) for t in f.inputs[8:11])
    write = 2 * m * n * 4 + nbytes(f.outputs[0])
    return _schedule(
        chip,
        name,
        4 * m * n * k,
        2 * n * k + pool + 3 * m * k,
        bits,
        read,
        write,
        groups,
        {"m": bm, "n": bn, "k": bk, "reduction": ratio, "completed_rows": pooled},
        cycles,
        4587,
        "Configured fused A8W4 dual projection with inline weight expansion, rolling state, pooled RMSNorm and RoPE. No matching low-bit compressor ELF.",
        phases={"expand_cycles": expand, "dual_matmul_cycles": mac, "k_pages": npage},
    )


def price(features, chip):
    name = features.raw_name.split("::")[-1]
    _check(features.raw_name in NAMES, "No matching attention route")
    sem = dict(dict(features.attributes).get("semantic", ()))
    if name in INDEXERS:
        return _index(features, chip, sem, name)
    if name in (
        "kimi_gated_dsa_attention",
        "sparse_attn_sharedkv_reference",
        "npu_fused_infer_attention_score",
    ):
        return _attention(features, chip, sem, name)
    if name == "compressor":
        return _compress(features, chip, sem, name)
    return _state(features, chip, sem, name)
