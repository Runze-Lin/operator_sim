"""Single wave of the 950 BF16 BSND LightningIndexer, with query-owned KV loops."""
from functools import lru_cache
from pathlib import Path
import json


@lru_cache(maxsize=1)
def _prices():
    path = Path(__file__).resolve().parents[1]/'results/v0.34/indexer/prices.json'
    return json.loads(path.read_text(encoding='utf-8'))


def _ceil(value, divisor):
    return (value + divisor - 1)//divisor


def applies(shape, attrs, chip):
    if len(shape) != 5 or any(type(x) is not int or x <= 0 for x in shape):
        return False
    b, q, length, g, d = shape
    contract = {'layout':'BSND', 'sparse_count':512, 'sparse_mode':3, 'return_values':False}
    return (b == 1 and g == 8 and d == 128 and length-q+1 >= 512
            and _ceil(q, 4) <= min(chip.cube_cores, chip.vector_cores//2)
            and all(key in contract and type(value) is type(contract[key]) and value == contract[key]
                    for key, value in attrs.items()))


def _sort_row(length, prices, vector_scale):
    # The next 8192-key scan includes the 512 winners from its predecessor.
    scans = _ceil(length, 8192)
    blocks = _ceil(length, 256) + (scans-1)*2
    final_length = (length-1)%8192+1 + (512 if scans>1 else 0)
    last_blocks = _ceil(final_length, 256)
    hist_fixed, hist_group = prices['histogram_span']
    gap_fixed, gap_group = prices['histogram_gap']
    # Responses between successive scans use the observed row-to-row handoff as
    # a proxy. Multiple scans of a single row have no complete reference yet.
    hist = (scans*hist_fixed + hist_group*blocks)*vector_scale
    gaps = (scans-1)*gap_fixed + gap_group*(blocks-last_blocks)*vector_scale
    return hist+gaps, last_blocks, 8*blocks


def schedule(shape, chip):
    _, queries, length, heads, depth = shape
    p = _prices()
    clk = chip.frequency_ghz * 1000
    cube = chip.cube_tops.get('16')
    vec = chip.vector_tops.get('32')
    cube_rate = cube*1000/(chip.frequency_ghz*chip.cube_cores) if cube else p['cube_ops_cycle']
    vec_rate = vec*1000/(chip.frequency_ghz*chip.vector_cores) if vec else p['vector_ops_cycle']
    vec_scale = p['vector_ops_cycle']/vec_rate
    dma = chip.details.get('mte2_bandwidth_gbs')
    # A measured AIC fetches one 128x128 BF16 key tile per steady interval.
    # The effective issue rate also includes contention and descriptor scheduling.
    page_bytes = 2*128*depth
    page_service = page_bytes*chip.frequency_ghz/dma if dma else p['page_ii_cycles']
    key_reads = mmads = histograms = sorting_aiv = 0
    geometry = {}
    critical_cycles = 0.
    for start in range(0,queries,4):
        rows = min(4,queries-start)
        m = 16*_ceil(rows*heads,16)
        visible = length-queries+start+rows
        pages = _ceil(visible,128)
        # The mask clips page count. Only the tensor's final physical page has a short N.
        last_n = min(128,length-(pages-1)*128)
        for n,count in ((128,pages-1),(last_n,1)):
            if count:
                geometry[(m,n)] = geometry.get((m,n),0)+count
        cube_interval = max(page_service, 2*m*128*depth/cube_rate)
        stages = {'first_mmad':p['first_issue_cycles'], 'page_intervals':(pages-1)*cube_interval,
                  'last_mmad':p['mmad_fixed_cycles']+2*m*last_n*depth/cube_rate,
                  'score_to_histogram':p['sort_handoff_cycles']}
        key_reads += ((pages-1)*128+last_n)*depth*2
        mmads += pages
        half = _ceil(rows,2)
        sorting_aiv += min(rows,2)
        slowest_lane = 0.
        for lane_start in (0,half):
            lane_cycles = 0.
            lane_rows = range(lane_start,min(rows,lane_start+half))
            for j in lane_rows:
                row_len = length-queries+start+j+1
                span, blocks, count = _sort_row(row_len,p,vec_scale)
                histograms += count
                lane_cycles += span
                costs = p['store_drain'] if j==lane_rows.stop-1 else p['histogram_gap']
                lane_cycles += costs[0]+costs[1]*blocks*vec_scale
            slowest_lane = max(slowest_lane,lane_cycles)
        stages['histogram_and_output'] = slowest_lane
        stage_sum = sum(stages.values())
        if stage_sum>critical_cycles:
            critical_cycles, critical, critical_group = stage_sum, stages, start//4
    local_us = critical_cycles/clk
    kv_tensor = length*depth*2
    q_and_out = queries*(heads*(depth+1)*2+512*4)
    # All query groups stream the same key tensor. When L2 fits it, assign one
    # compulsory HBM read. Otherwise price the full per-group read demand.
    hbm_keys = kv_tensor if kv_tensor <= (chip.details.get('l2_bytes') or 0) else key_reads
    hbm_us = (hbm_keys+q_and_out)/(1000*chip.hbm_bandwidth_gbs)
    l2_bw = chip.details.get('l2_bandwidth_gbs')
    l2_us = key_reads/(1000*l2_bw) if l2_bw else 0.
    duration = max(local_us,hbm_us,l2_us)
    stages_us = {name:cycles/clk for name,cycles in critical.items()}
    stages_us['additional_shared_service'] = duration-local_us
    plan = dict(query_rows_per_tile=4,kv_page_keys=128,histogram_chunk_keys=8192,
                active_aic=_ceil(queries,4),active_aiv=sorting_aiv,
                active_aiv_scope='AIVs issuing histogram instructions; the runtime also launches idle partners',
                launched_aic_950=chip.cube_cores, launched_aiv_950=chip.vector_cores,
                critical_query_group=critical_group,mmad_count=mmads,
                mmads_by_shape=[[*key,depth,n] for key,n in sorted(geometry.items())],
                modeled_histogram_instructions=histograms,key_reads_across_aic_bytes=key_reads)
    return duration, plan, dict(stages_us=stages_us, local_us=local_us, hbm_floor_us=hbm_us,
                                l2_read_floor_us=l2_us, reference_page_ii_cycles=p['page_ii_cycles'],
                                target_page_service_cycles=page_service,
                                unvalidated_multiple_histogram_chunks=length>8192)
