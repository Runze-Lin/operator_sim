"""DeepSeek V4.1 Flash A5 PA_BBND sparse-indexer consumer resource projection.

Geometry follows arenanet_kernels' QSLI. Coefficients borrowed from other
kernels price an uncalibrated fused device window. No cache slab is scanned.
"""
from bisect import bisect_right
from collections import Counter
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
    if set(call) - {"implementation", "pa_block_size", "table_pages", "scalar_lookup_cycles", "candidate_limit", "candidate_order_overlap"}:
        raise ValueError("Unknown A5 QSLI option")
    _order_overlap(call.get("candidate_order_overlap"))
    return {k:v for k,v in call.items() if k != "implementation"}


def _order_overlap(value):
    if value is None:
        return 0.5
    if type(value) not in (int,float) or not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("candidate_order_overlap must be a finite fraction in [0,1]")
    return value


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


@lru_cache(maxsize=1)
def _native_rates():
    return load_parameters("flash_qsli")


def _wide_issue_interval():
    measured = _native_rates()
    return measured["wide_table_team_issue_span_cycles"] / measured["adjacent_issues"]


def _merge_service(parts, topk, read_rate, vec_scale):
    """Wheel LD owner streams at most seven new lists plus one carry per pass."""
    prices = _native_rates()["split_row"]
    passes = _ceil(parts,7)
    entries = (parts + passes - 1)*topk
    read = passes*prices["read_latency_cycles"] + 6*entries/read_rate
    select = (passes*prices["select_fixed_cycles"]
              + _ceil(topk,256)*(parts+passes-1)*prices["select_cycles_per_256"])*vec_scale
    write = passes*(prices["write_latency_cycles"] + 6*topk/read_rate)
    return passes, entries, read, select, write


def work_for(shape, candidate_limit=None):
    _, ends, _, useful = _geometry(shape,candidate_limit)
    return _work(shape, shape[0]*ends[-1], useful)


def price(shape, chip, *, pa_block_size=128, table_pages=None, scalar_lookup_cycles=None,
          candidate_limit=None, candidate_order_overlap=None):
    """Price packed Q, folded K and QSLI. PA page-table width can exceed current L."""
    if type(pa_block_size) is not int or pa_block_size <= 0 or pa_block_size % BLOCK:
        raise ValueError("PA block size must be a positive multiple of eight")
    b,q,h,d,length,capacity,topk = shape
    overlap = _order_overlap(candidate_order_overlap)
    groups, ends, row_ends, useful = _geometry(shape,candidate_limit)
    required = _ceil(length,pa_block_size)
    if table_pages is None:
        table_pages = required
    if type(table_pages) is not int or table_pages < required:
        raise ValueError("Page table must cover the key extent")
    use_measured_issue = scalar_lookup_cycles is None
    if use_measured_issue:
        scalar_lookup_cycles = 64.  # unmeasured address/fill budget on the first tile
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
    resource_interval = max(stages.values())
    interval = resource_interval
    if scalar and use_measured_issue:
        # The measured interval already includes address handling and pipeline
        # waits. Avoid adding the scalar-address allowance a second time.
        interval = max(_wide_issue_interval(), payload/(2*read_rate),
                       publish, cube_read, cube, fix, vector)
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
    def position(tile):
        doc, offset = divmod(tile,ends[-1])
        i = bisect_right(ends,offset)
        base = ends[i-1] if i else 0
        before = row_ends[i-1] if i else 0
        width = groups[i][0]
        row, rank = divmod(offset-base,width)
        return doc*q+before+row, rank, width

    # Source ASC scheduler recomputes the cost target after every assignment.
    # Full 512-key tiles cost 128 scheduling units; their half-tile tolerance
    # rounds the remaining target to nearest (ties up). Ragged per-row final
    # tiles can have smaller source weights; this path approximates those
    # boundaries with the same full-tile unit. Always force one tile.
    weighted_tail = (min(_ceil(length-q+1,BLOCK), capacity if candidate_limit is None else candidate_limit)
                     % (TILE//BLOCK) != 0 or
                     (q > 1 and length-q+1 < (capacity if candidate_limit is None else candidate_limit)*BLOCK))
    slow = 0.
    split_rows = {}
    first_tile = 0
    owners = []
    for lane in range(live):
        remaining_lanes = live-lane
        n = max(1, (2*(tiles-first_tile)+remaining_lanes)//(2*remaining_lanes))
        stop = first_tile+n
        first_row, first_rank, first_width = position(first_tile)
        last_row, last_rank, last_width = position(stop-1)
        fragments = last_row-first_row+1
        left_split = first_rank > 0
        right_split = last_rank+1 < last_width
        partials = int(left_split) + int(right_split and (not left_split or first_row != last_row))
        if right_split:
            split_rows[last_row] = split_rows.get(last_row,1)+1
        doc_changes = last_row//q-first_row//q
        # The next-row gather is prefetched inside a document. The first
        # fragment in every new document has to fill its own slots again.
        first_gaps = min(n,first_width-first_rank)-1 + doc_changes*(groups[0][0]-1)
        if doc_changes and last_row%q == 0:
            first_gaps -= last_width-last_rank-1
        owners.append((first_tile,n,fragments,partials,first_gaps,doc_changes,
                       min(n,last_rank+1),right_split))
        first_tile = stop

    native = _native_rates()["split_row"]
    if split_rows:
        startup = native["startup_control_cycles"]
    # The native drain landmark covers one tile's Fix/Vector/local selection
    # and public partial write. Credit service already present in the fragment.
    # A larger fragment retains its extra per-tile histogram work.
    ordinary_tail = fix + vector + (hist_fixed+2*hist_per256)*vec_scale
    drain_extra = max(0., native["one_tile_score_drain_cycles"] - ordinary_tail)
    wave = scalar and use_measured_issue and q>1 and live>1 and tiles>live
    phase_fraction = 0.
    first_interval = carry_interval = interval
    handoff = 0.
    if wave:
        # Compare lock-step *logical* issue positions. Different requests have
        # different KV; equal rank alone does not create a shared record.
        # Three bounded samples also cover ragged and multi-document owners.
        common = min(owner[1] for owner in owners)
        samples = {0,(common-1)//2,common-1}
        for step in samples:
            phases = Counter()
            for first, *_ in owners:
                row, rank, _ = position(first+step)
                phases[(row//q,rank)] += 1
            phase_fraction += sum(n*(n-1) for n in phases.values())/(live*(live-1))
        phase_fraction /= len(samples)
        pressure = phase_fraction*overlap
        wave_rates = _native_rates()["parallel_row_issue"]
        floor = max(publish,cube_read,cube,fix,vector)
        first_interval = max(floor, (1-pressure)*wave_rates["first_disperse_cycles"]
                             + pressure*wave_rates["first_aligned_cycles"])
        # A team starts the following row's gather before its previous local
        # select. Consecutive MMADs after that handoff use the primed-row rate.
        carry_interval = max(floor,wave_rates["prefetched_row_cycles"])
        interval = first_interval
        handoff = max(gather+publish+cube_read,
                      (hist_fixed+2*hist_per256)*vec_scale, wave_rates["same_document_handoff_cycles"])
        startup = wave_rates["first_control_cycles"]
    for _, n, fragments, partials, first_gaps, doc_changes, final_tiles, right_split in owners:
        if wave:
            carry_gaps = n-fragments-first_gaps
            joins = fragments-1-doc_changes
            # The measured last-MMAD -> next-first issue already includes
            # local selection and the next-row fill. Charge that tail once.
            price_cycles = (sum(stages.values()) + first_gaps*first_interval + carry_gaps*carry_interval
                            + joins*handoff + doc_changes*(handoff+gather+publish+cube_read)
                            + (hist_fixed+2*final_tiles*hist_per256)*vec_scale
                            + int(right_split or (fragments==1 and partials))*drain_extra)
        else:
            price_cycles = ((n-fragments)*interval + fragments*sum(stages.values())
                            + (fragments*hist_fixed + 2*n*hist_per256)*vec_scale
                            + partials*drain_extra)
        slow = max(slow,price_cycles)
    merge = 0.
    passes, entries, merge_read, merge_select, merge_write = 0, 0, 0., 0., 0.
    if split_rows:
        # Metadata assigns each split row to a final AIV; multiple rows may
        # merge concurrently. Its longest chain sets the local FD critical path.
        passes, entries, merge_read, merge_select, merge_write = _merge_service(
            max(split_rows.values()),topk,read_rate,vec_scale)
        merge = native["barrier_setup_cycles"] + merge_read + merge_select + merge_write
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
                team_partition="remaining cost, half-tile tolerance" if not weighted_tail else "remaining cost, full-tile approximation to unequal tail weights",
                merge_passes=passes, merge_scored_entries=entries,
                merge_service_cycles=dict(read=merge_read, radix_and_remap=merge_select, write=merge_write,
                                          barrier=native["barrier_setup_cycles"] if split_rows else 0.),
                pa_block_size=pa_block_size, table_pages=table_pages, address_path="scalar" if scalar else "UB",
                scalar_lookup_cycles=scalar_lookup_cycles,
                tile_service_cycles=stages, interval_cycles=interval, resource_interval_cycles=resource_interval,
                interval_source=("multi-team first / carried same-AIC MMAD issue cycles" if wave else
                                 "DT consecutive MMAD_MX issues" if scalar and use_measured_issue else "resource schedule"),
                parallel_row_issue=wave, candidate_order_overlap=overlap,
                order_overlap_source="unknown: endpoint midpoint" if candidate_order_overlap is None else "caller",
                team_phase_pair_fraction=phase_fraction, ordered_record_pressure=phase_fraction*overlap,
                first_fragment_interval_cycles=first_interval, carried_fragment_interval_cycles=carry_interval,
                same_document_handoff_cycles=handoff,
                stages_us=dict(startup=startup/clock, critical_team=slow/clock, merge=merge/clock,
                               shared_service=us-local), io_us=io, l2_us=ring,
                unpriced=["AICPU metadata", "prepare_indexer_indices", "upstream Q quantization and folded K production"])
    return us, work, plan
