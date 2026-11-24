"""DeepSeek V4.1 Flash A5 PA_BBND sparse-indexer consumer resource projection.

Geometry follows arenanet_kernels' QSLI. Coefficients borrowed from other
kernels price an uncalibrated fused device window. No cache slab is scanned.
"""
from bisect import bisect_right
from functools import lru_cache
from math import isfinite

from operator_model.operators.attention.candidate_indexer import _candidate_prefix
from operator_model.operators.attention.indexer import _ceil, _prices
from operator_model.resources import load_parameters

IMPLEMENTATION = "a5_qsli_projection"
TILE = 512
BLOCK = 8
CAPACITY = 2048
RECORD = 544  # eight packed D128 keys and 4 E8M0 scale bytes per key


def options(call):
    if set(call) - {"implementation", "pa_block_size", "table_pages", "scalar_lookup_cycles", "candidate_limit"}:
        raise ValueError("Unknown A5 QSLI option")
    return {k:v for k,v in call.items() if k != "implementation"}


def _geometry(shape, candidate_limit=None):
    if len(shape) != 7 or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError("Expected positive B,Q,H,D,L,C,K")
    b, q, h, d, length, capacity, topk = shape
    if (h, d, capacity, topk) != (32, 128, CAPACITY, 512) or length < q:
        raise ValueError("Flash A5 QSLI requires H32/D128/C2048/K512 and a contiguous suffix Q<=L")
    if candidate_limit is not None:
        if type(candidate_limit) is not int or not 1 <= candidate_limit <= capacity:
            raise ValueError("candidate_limit must be in [1,2048]")
        capacity = candidate_limit  # width of the allocated input remains 2048
    # Prefixes with equal padded candidate tile counts are contiguous. There
    # are at most 32 runs even for a million query rows.
    ends, row_ends, groups = [], [], []
    first = length - q + 1
    tiles = min(_ceil(first, TILE), _ceil(capacity * BLOCK, TILE))
    while first <= length:
        end = length if tiles == _ceil(capacity * BLOCK, TILE) else min(length, tiles * TILE)
        count = end - first + 1
        groups.append((tiles, count))
        ends.append((ends[-1] if ends else 0) + tiles * count)
        row_ends.append((row_ends[-1] if row_ends else 0) + count)
        tiles += 1
        first = end + 1
    useful = b * (_candidate_prefix(length, capacity, BLOCK) - _candidate_prefix(length-q, capacity, BLOCK))
    return groups, ends, row_ends, useful


def _work(shape, tiles, useful):
    b,q,h,d,length,capacity,topk = shape
    rows = b*q
    cube = 2*h*d*useful
    vec = 3*h*useful  # vmax then vmadd into a zero-initialized BF16 accumulator
    # Candidate slots repeat on the tail; native gather still executes all 64
    # copies. Ring traffic is kept separate from external logical operands.
    gathered = tiles*(TILE//BLOCK)*RECORD
    return dict(dtype="mxfp4", cube_ops=cube, cube_ops_by_dtype={"mxfp4":cube},
                vec_ops=vec, vec_ops_by_dtype={"bf16":vec},
                logical_read_bytes=gathered+rows*(h*(d//2+4)+4*h+4*capacity+4),
                logical_write_bytes=rows*topk*4, physical_hbm_bytes=None,
                gathered_record_bytes=gathered, staging_write_plus_read_bytes=2*gathered,
                key_table_capacity_bytes=b*length*RECORD//BLOCK)


@lru_cache(maxsize=1)
def _attention_rates():
    return load_parameters("attention_resources")


def work_for(shape, candidate_limit=None):
    _, ends, _, useful = _geometry(shape,candidate_limit)
    return _work(shape, shape[0]*ends[-1], useful)


def price(shape, chip, *, pa_block_size=128, table_pages=None, scalar_lookup_cycles=64., candidate_limit=None):
    """Price packed Q, folded K and QSLI. PA page-table width can exceed current L."""
    if type(pa_block_size) is not int or pa_block_size <= 0 or pa_block_size % BLOCK:
        raise ValueError("PA block size must be a positive multiple of eight")
    b,q,h,d,length,capacity,topk = shape
    groups, ends, row_ends, useful = _geometry(shape,candidate_limit)
    required = _ceil(length,pa_block_size)
    if table_pages is None:
        table_pages = required
    if type(table_pages) is not int or table_pages < required:
        raise ValueError("Page table must cover the key extent")
    if not isinstance(scalar_lookup_cycles,(int,float)) or not isfinite(scalar_lookup_cycles) or scalar_lookup_cycles < 0:
        raise ValueError("Expected a nonnegative finite scalar lookup price")
    tiles = b*ends[-1]
    rows = b*q
    work = _work(shape,tiles,useful)
    teams = min(chip.cube_cores,chip.vector_cores//2)
    if teams < 1 or not chip.cube_tops.get("4") or not chip.vector_tops.get("16"):
        raise ValueError("QSLI projection requires one AIC with two AIVs, MXFP4 Cube and BF16 Vector rates")
    live = min(teams,tiles)
    clock = chip.frequency_ghz*1000
    cube_rate = chip.cube_tops["4"]*1000/(chip.cube_cores*chip.frequency_ghz)
    vec_rate = chip.vector_tops["16"]*1000/(chip.vector_cores*chip.frequency_ghz)
    params = _attention_rates()
    dma = chip.details.get("mte2_bandwidth_gbs")
    read_rate = dma/chip.frequency_ghz if dma else params["mte_bytes_per_cycle"]
    scalar = table_pages > 2048
    # Two AIVs gather then publish 32 records each. The Cube then reads
    # the contiguous GM ring. All three passes consume different DMA ports.
    payload = TILE*RECORD//BLOCK
    lookup = (TILE//(2*BLOCK))*(scalar_lookup_cycles if scalar else 0)
    gather = payload/(2*read_rate) + lookup
    publish = payload/(2*read_rate)
    cube_read = payload/read_rate
    cube = 2*h*TILE*d/cube_rate
    fix = 2*h*TILE/params["fix_bytes_per_cycle"]
    vector = 3*h*TILE/(2*vec_rate)
    # Shared read port on each core, independent Cube and AIV DMA engines.
    stages = dict(gather=gather, publish=publish, cube_read=cube_read, cube=cube,
                  fix=fix, vector=vector)
    interval = max(stages.values())
    # Adjacent ordinary QLI MX kernel DT first MMAD (8,568 cycles);
    # apply once to the window, not once per query row. A SKU clock converts cycles.
    startup = 8568
    hist = _prices()
    # uint16 histogram selector in the DSL. BF16 ordinary LI is a service
    # proxy here; decode shards may merge partial top-k rows across teams.
    hist_fixed, hist_per256 = hist["histogram_span"]
    vec_scale = hist["vector_ops_cycle"]*2/vec_rate
    # A cut through the linear (document,row,key tile) stream may split a
    # row across teams. Map just the endpoints, without constructing B*Q rows.
    def row_at(tile):
        doc, offset = divmod(tile,ends[-1])
        i = bisect_right(ends,offset)
        base = ends[i-1] if i else 0
        before = row_ends[i-1] if i else 0
        return doc*q+before+(offset-base)//groups[i][0]

    slow = 0.
    split_rows = {}
    first_tile = 0
    for lane in range(live):
        stop = (tiles*(lane+1)+live//2)//live
        n = stop-first_tile
        first_row, last_row = row_at(first_tile), row_at(stop-1)
        fragments = last_row-first_row+1
        # Histogram on AIV0, register pipeline with fill/drain per fragment.
        price_cycles = (n*interval + fragments*(sum(stages.values())-interval)
                        + (fragments*hist_fixed + 2*n*hist_per256)*vec_scale)
        slow = max(slow,price_cycles)
        if stop < tiles and row_at(stop) == last_row:
            split_rows[last_row] = split_rows.get(last_row,1)+1
        first_tile = stop
    merge = 0.
    if split_rows:
        # At most 8 partial lists plus streaming carry in the LD merge pass.
        parts = max(split_rows.values())
        passes = _ceil(parts-1,7)
        merge = (passes*hist_fixed + (parts+passes-1)*_ceil(topk,256)*hist_per256)*vec_scale
    local = (startup+slow+merge)/clock
    io = (work["logical_read_bytes"]+work["logical_write_bytes"])/(1000*chip.hbm_bandwidth_gbs)
    l2_bw = chip.details.get("l2_bandwidth_gbs")
    ring = 3*work["gathered_record_bytes"]/(1000*l2_bw) if l2_bw else 0.
    us = max(local,io,ring)
    plan = dict(implementation=IMPLEMENTATION, timing_qualification="extrapolated",
                reason="A5 QSLI source geometry; stage rates are proxies. Packed consumer device projection.",
                score_policy="candidate_first", target_plan=dict(active_aic=live, active_aiv=2*live),
                query_tile=1, kv_page=TILE, candidate_block_size=BLOCK,
                effective_candidate_limit=capacity if candidate_limit is None else candidate_limit,
                candidate_key_pairs=useful, scored_query_key_pairs=tiles*TILE,
                tile_runs_per_request=groups, query_groups=rows, candidate_tiles=tiles,
                split_rows=len(split_rows), largest_row_shards=max(split_rows.values(),default=1),
                pa_block_size=pa_block_size, table_pages=table_pages, address_path="scalar" if scalar else "UB",
                scalar_lookup_cycles=scalar_lookup_cycles,
                tile_service_cycles=stages, interval_cycles=interval,
                stages_us=dict(startup=startup/clock, critical_team=slow/clock, merge=merge/clock,
                               shared_service=us-local), io_us=io, l2_us=ring,
                unpriced=["AICPU metadata", "prepare_indexer_indices", "upstream Q quantization and folded K production"])
    return us, work, plan
