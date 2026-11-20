"""Contiguous BF16 MatMulV3: choose an output grid before dividing the K reduction."""
from math import log2

from operator_model.operators.matmul.resources import _costs
from operator_model.operators.matmul.split_k import resource_matmul_dispatch as price_original_grid
from operator_model.profiles.stage_model import align, ceildiv


def resource_matmul_dispatch(shape, target):
    """Return device µs, launched AICs/AIVs and work for the chosen implementation."""
    previous = price_original_grid(shape, target)
    _, _, _, previous_diag = previous
    if previous_diag['streamk_selection']['route'] == 'one_output_split_k':
        return previous

    m, n, k = shape
    cores = target.cube_cores
    k_threshold = max(8192, 256 * cores) // 2
    if align(k, 256) < k_threshold or not target.vector_tops.get('32'):
        return previous
    details = target.details
    l0a = details.get('l0a_bytes') or 65536
    l0b = details.get('l0b_bytes') or 65536
    l0c = details.get('l0c_bytes') or 262144
    l1 = details.get('l1_bytes') or 524288
    # Two live slots per BF16 operand and a minimum 64-deep L0 tile.
    m_cap = 16 * (l0a // (2 * 2 * 64 * 16))
    n_cap = 16 * (l0b // (2 * 2 * 64 * 16))
    if not m_cap or not n_cap:
        return previous
    m_tiles, n_tiles = ceildiv(m, m_cap), ceildiv(n, n_cap)
    groups = m_tiles * n_tiles
    splits = cores // groups
    if splits < 2:
        return previous
    bm = align(ceildiv(m, m_tiles), 16)
    bn = align(ceildiv(n, n_tiles), 16)
    if 4 * bm * bn > l0c:
        return previous
    bk = 64 * (min(l0a // (4 * bm), l0b // (4 * bn)) // 64)
    if not bk:
        return previous
    page = bk * (min(l1 // (4 * (bm + bn)), l1 // (8 * max(bm, bn))) // bk)
    segment = ceildiv(k, splits)
    tail = k - (splits - 1) * segment
    if not page or tail <= 0:
        return previous

    params = _costs()
    clock = target.frequency_ghz
    cycles_per_us = clock * 1000

    def cycles(name):
        return details.get('matmul_' + name) or params[name]

    def bytes_per_cycle(engine, default):
        bandwidth = details.get(engine + '_bandwidth_gbs')
        return bandwidth / clock if bandwidth else params[default]

    mte2 = bytes_per_cycle('mte2', 'mte2_bytes_per_cycle')
    mte1 = bytes_per_cycle('mte1', 'mte1_bytes_per_cycle')
    fixp = bytes_per_cycle('mte3', 'write_bytes_per_cycle')
    peak = target.cube_tops.get('16')
    cube_rate = peak * 1000 / (cores * clock) if peak else params['default_bf16_ops_per_cycle']
    operand_edge = max(min(bm, m), min(bn, n))

    def page_work(depth):
        # MTE2 and MTE1 carry the two operands concurrently on the critical core.
        first = min(depth, bk)
        last = (depth - 1) % bk + 1
        read = 2 * depth * operand_edge / mte2
        load = 2 * first * operand_edge / mte1
        cube_cycles_per_k = 2 * bm * bn / cube_rate
        issue = max(cycles('mmad_ii_cycles') + cube_cycles_per_k * align(first, 16), load)
        last_retire = cycles('mmad_latency_cycles') + cube_cycles_per_k * align(last, 16)
        return read, load, (ceildiv(depth, bk) - 1) * issue + last_retire

    read, load, mac = page_work(min(segment, page))
    last_mac = page_work((segment - 1) % page + 1)[2]
    read_latency = cycles('read_latency_cycles')
    interval = max(2 * cycles('read_ii_cycles'), read, (read_latency + read) / 2, mac)
    local_us = (read_latency + read + cycles('l0_latency_cycles') + load
                + (ceildiv(segment, page) - 1) * interval + last_mac
                + cycles('drain_cycles') + 4 * min(m, bm) * min(n, bn) / fixp) / cycles_per_us

    input_bytes = 2 * k * (m + n)
    l2_input_bytes = 2 * k * (m * n_tiles + n * m_tiles)
    l2_gbs = details.get('l2_bandwidth_gbs') or clock * 100 * cores
    source_us = max(input_bytes / (1000 * target.hbm_bandwidth_gbs), l2_input_bytes / (1000 * l2_gbs))
    launch = (cycles('launch_cycles') + params['arrival_log2_cycles'] * log2(cores)) / cycles_per_us
    vec_ops_per_cycle = target.vector_tops['32'] * 1000 / (target.vector_cores * clock)
    merge_issue = (1000 + 60 * splits * 256 / vec_ops_per_cycle) / cycles_per_us
    partial_bytes = 4 * m * n * splits
    # Relative to a BF16 output already charged, count the two FP32 partial passes.
    incremental_bytes = 2 * partial_bytes - 2 * m * n
    stages = dict(launch=launch, partial_pipeline=local_us, exposed_shared_input=max(0., source_us - local_us),
                  merge_issue=merge_issue, merge_transfer=incremental_bytes / (1000 * l2_gbs))
    productive = groups * splits
    mmads = groups * ((splits - 1) * ceildiv(segment, bk) + ceildiv(tail, bk))
    plan = dict(branch='retiled_output_split_k', tile=[bm, bn, bk], output_tiles=groups,
                m_tiles=m_tiles, n_tiles=n_tiles, waves=1, active_aic=cores, active_aiv=target.vector_cores,
                productive_aic=productive, k_splits=splits, max_segment_k=segment, tail_segment_k=tail,
                read_page_k=page, mmads_total=mmads, iterations=ceildiv(segment, bk))
    internal = dict(fp32_partial_write_bytes=partial_bytes, fp32_partial_read_bytes=partial_bytes,
                    fp32_reduction_adds=m * n * (splits - 1), priced_incremental_fixup_bytes=incremental_bytes)
    diag = dict(model='target_resource_matmul_retiled_split_k', target_plan=plan, stages_us=stages,
                streamk_selection=dict(route=plan['branch'], threshold_aligned_k=k_threshold,
                                       max_groups_for_split=cores // 2, original_output_tiles=previous_diag['target_plan']['output_tiles']),
                internal_work=internal, logical_cube_ops=2*m*n*k, logical_read_bytes=input_bytes,
                write_bytes=2*m*n, l2_to_l1_read_bytes=l2_input_bytes, mmads_total=mmads,
                qualification='Full BF16 ND output grid split across K. Assumes the 950 mixed kernel and its FP32 merge policy on the target.')
    return sum(stages.values()), cores, target.vector_cores, diag
