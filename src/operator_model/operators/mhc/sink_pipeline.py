"""FP32 N4 Sinkhorn. Price the extra register passes and the kernel's UB loop."""
from operator_model.resources import load_parameters
from functools import lru_cache

from operator_model.operators.mhc.sink_history import _epochs, _prices, price as single_register_price

PROFILE = 'sinkhorn_pipeline'


@lru_cache(maxsize=1)
def _register_prices():
    return load_parameters(PROFILE)


def _block(chip, plan, rows, chip_rows):
    """One kernel-loop body on the critical AIV, with its concurrent chip traffic."""
    single = _prices()
    g = (rows + 7) // 8
    state = single['phase_prices'][plan['history_state']]
    work = dict(plan, rows_per_core=rows, register_passes=g, logical_read_bytes=64*chip_rows)
    e = _epochs(chip, single, state, work, plan['iterations'])
    if g == 1:
        return e
    p = _register_prices()
    clock = 1000 * chip.frequency_ghz
    # Vector reference cycles per microsecond of target FP32 work. DMA retains
    # the target clock and bandwidth bounds already used in _epochs.
    unit = chip.vector_tops['32'] * 1e6 / chip.vector_cores / single['reference_vec_fp32_ops_per_cycle']
    def phase(key):
        intercept, slope = state[key]
        return intercept + slope * plan['active_aiv']
    write_duration = e['write_end']-e['write_begin']
    retirement = e['end']-e['write_end']
    soft = max(p['soft_one_floor'], p['soft_start_to_first_col_linear_intercept']+p['soft_extra_cycles']*(g-1))
    e['first_col_vf'] = max(e['softmax_vf']+soft/unit, e['setup_vf']+phase('first_col_from_setup')/clock)
    e['first_row_vf'] = e['first_col_vf']+(p['col_row_second_pass']+p['col_row_increment']*(g-2))/unit
    e['first_pair_period'] = (p['first_pair']+p['extra_pair_cycles']*(g-1))/unit
    e['steady_pair_period'] = (p['steady_pair']+p['extra_pair_cycles']*(g-1))/unit
    col_tail = (p['column_tail_two']+p['column_tail_inc']*(g-2))/unit
    e['first_row_to_col'] = max(p['row_col_one']/unit, e['first_pair_period']-col_tail)
    e['steady_row_to_col'] = e['steady_pair_period']-col_tail
    iterations = plan['iterations']
    e['scatter_vf'] = e['first_row_vf']
    if iterations >= 2:
        e['scatter_vf'] += e['first_pair_period']+max(0,iterations-2)*e['steady_pair_period']
    e['write_begin'] = e['scatter_vf']+(phase('write_gap')+p['scatter_additional']*(g-1))/unit
    e['write_end'] = e['write_begin']+write_duration
    e['end'] = e['write_end']+retirement
    return e


def price(shape, chip, attrs):
    """Conditional on an explicit same-family resident AIV high-water mark."""
    old = single_register_price(shape, chip, attrs)
    latency, aic, active, diag = old
    plan = dict(diag['target_plan'])
    if plan['register_passes'] == 1 and latency is not None:
        return old
    ub = chip.details.get('ub_bytes')
    if not ub or not chip.vector_tops['32']:
        return old
    # The host chooses the smaller of forward and backward UB factors. For
    # N=4, a 32-byte UB block and no snapshots, the limiting backward footprint
    # is 896 bytes/matrix including the two slots, plus 832 bytes once.
    rows = plan['rows_per_core']
    fits = (ub-832)//896
    tile = rows if rows <= fits else (fits//8)*8
    if tile <= 0:
        return old
    blocks = (rows+tile-1)//tile
    last_core_rows = shape[0]-(active-1)*rows
    # All regular cores have the same work. At most one short core stops early.
    # Visit only the distinct bodies even if the full batch spans millions.
    boundaries = {0, 1, blocks-1, blocks, last_core_rows//tile, (last_core_rows+tile-1)//tile}
    bounds = sorted(i for i in boundaries if 0 <= i <= blocks)
    segments = []
    additional = 0.
    for left, right in zip(bounds,bounds[1:]):
        nr = min(rows-left*tile,tile)
        tail = max(0,min(tile,last_core_rows-left*tile))
        epochs = _block(chip, plan, nr, (active-1)*nr+tail)
        if left == 0:
            first = epochs
        # The repeated body is priced from existing mask/setup to its write
        # retirement. This serialized schedule is an explicit multi-UB
        # scenario; possible inter-block overlap has no measured issue price.
        body = epochs['write_end']-epochs['setup_vf']
        additional += (right-left-int(left==0))*body
        segments.append(dict(first_block=left, count=right-left, rows=nr, concurrent_rows=(active-1)*nr+tail,
                             register_passes=(nr+7)//8, repeated_body_us=body))
    breaks = [('tiling_and_setup','setup_vf'),('input_and_softmax_issue','softmax_vf'),
              ('first_column','first_col_vf'),('normalization_loop','scatter_vf'),
              ('scatter_to_output_issue','write_begin'),('output_and_retirement','end')]
    prev = 0.
    stages = {}
    for name, mark in breaks:
        stages[name] = first[mark]-prev
        prev = first[mark]
    if blocks > 1:
        stages['additional_ub_bodies'] = additional
    end = first['end']+additional
    plan.update(ub_tile_rows=tile, ub_blocks=blocks, required_ub_bytes=tile*896+832,
                register_passes=(tile+7)//8, vf_calls_per_core=1+blocks*(2*plan['iterations']+1))
    info = dict(diag, target_plan=plan, stages_us=stages, parameter_source=PROFILE,
                schedule='sinkhorn_register_pipeline', history_timing_status='extrapolated',
                issue_epochs_us=dict(first, end=end), first_block_epochs_us=first, block_schedule=segments,
                qualification=('Single-UB per-pass issue prices from 950DT VF instructions; forward validation is reported separately'
                               if blocks==1 else 'Multiple UB blocks with persistent mask and serialized block bodies; inter-block overlap uncalibrated'))
    return end, aic, active, info
