"""Standalone 4×4 FP32 Sinkhorn issue schedule with explicit same-family AIV history."""
from operator_model.resources import load_parameters
from functools import lru_cache
from math import ceil

from operator_model.chips import load_chip

_PROFILE = 'sinkhorn_history'


@lru_cache(maxsize=1)
def _prices():
    return load_parameters(_PROFILE)


def _epochs(chip, prices, phase, plan, iterations):
    """Issue and retirement epochs relative to the first instruction, in µs."""
    active = plan['active_aiv']
    local_rows = plan['rows_per_core']
    # Control/descriptor waits use target clock cycles. Vector work scales with
    # the per-AIV FP32 peak per clock relative to the original instruction prices.
    clock = chip.frequency_ghz * 1000
    per_cycle = chip.vector_tops['32'] * 1e12 / chip.vector_cores / (chip.frequency_ghz * 1e9)
    rate = per_cycle / prices['reference_vec_fp32_ops_per_cycle']
    groups = plan['register_passes']
    phase_cost = {key: fixed + slope * active for key, (fixed, slope) in phase.items()}

    def dma_cycles(byte_count, name):
        bandwidth = chip.details.get(name + '_bandwidth_gbs')
        return byte_count * chip.frequency_ghz / bandwidth if bandwidth else 0.

    shared = plan['logical_read_bytes'] * chip.frequency_ghz / chip.hbm_bandwidth_gbs
    read_floor = max(shared, dma_cycles(64 * local_rows, 'mte2'))
    write_floor = max(shared, dma_cycles(64 * local_rows, 'mte3'))
    tiling = phase_cost['tiling_issue']
    tiling_end = tiling + max(phase_cost['tiling_wait'], dma_cycles(96, 'mte2'))
    setup = tiling_end + phase_cost['setup_from_tile']
    read = setup + phase_cost['input_issue']
    read_end = read + max(phase_cost['input_wait'], read_floor)
    soft = read_end + phase_cost['soft_after_input']
    first_col = max(soft + prices['softmax_col_floor'] * groups / rate,
                    setup + phase_cost['first_col_from_setup'])
    first_row = first_col + prices['first_col_to_row'] * groups / rate
    first_pair = prices['first_pair'] * groups / rate
    steady_pair = prices['steady_pair'] * groups / rate
    scatter = first_row + (first_pair + max(0, iterations-2) * steady_pair if iterations >= 2 else 0.)
    write = scatter + phase_cost['write_gap'] * groups / rate
    write_end = write + max(phase_cost['write_wait'], write_floor)
    epochs = dict(tiling_begin=tiling, tiling_end=tiling_end, setup_vf=setup, read_begin=read,
                  read_end=read_end, softmax_vf=soft, first_col_vf=first_col, first_row_vf=first_row,
                  first_pair_period=first_pair, steady_pair_period=steady_pair,
                  first_row_to_col=prices['row_col'] * groups / rate,
                  steady_row_to_col=prices['steady_row_col'] * groups / rate,
                  scatter_vf=scatter, write_begin=write, write_end=write_end, end=write_end + phase_cost['end'])
    return {name: cycles / clock for name, cycles in epochs.items()}


def price(shape, chip, attrs):
    """Price one explicit first/expanded/resident same-family launch; never infer H."""
    if (len(shape) != 2 or any(type(x) is not int for x in shape) or shape[0] <= 0 or shape[1] != 4
            or set(attrs) - {'resident_aiv', 'num_iters', 'return_intermediates'}
            or attrs.get('return_intermediates', False) is not False):
        raise ValueError('Explicit Sinkhorn history requires [T,4], T>0, FP32 inference without intermediates')
    iterations = attrs.get('num_iters', 20)
    if type(iterations) is not int or not 1 <= iterations <= 100:
        raise ValueError('Sinkhorn num_iters must be an integer from 1 to 100')
    h = attrs['resident_aiv']
    if type(h) is not int or not 0 <= h <= chip.vector_cores:
        raise ValueError('resident_aiv must be a same-family high-water mark within the target AIV count')
    t = shape[0]
    rows = ceil(t / chip.vector_cores)
    active = ceil(t / rows)
    prices = _prices()
    state = 'first' if h == 0 else ('resident' if h >= active else 'expand')
    plan = dict(active_aiv=active, active_aic=0, rows_per_core=rows,
                register_passes=ceil(rows / prices['aligned_rows_per_vreg']), iterations=iterations,
                vf_calls_per_core=2 * iterations + 2, logical_read_bytes=64*t, logical_write_bytes=64*t,
                resident_aiv=h, history_state=state, required_ub_bytes=896 * rows + 64 + 3 * 256)
    diag = dict(target_plan=plan, schedule='standalone_sinkhorn_known_history', stages_us={},
                parameter_source=_PROFILE, kernel_api='aclnnMhcSinkhorn')
    ub = chip.details.get('ub_bytes')
    if not ub or plan['required_ub_bytes'] > ub or not chip.vector_tops['32']:
        diag['qualification'] = ('Unknown UB capacity' if not ub else 'Multiple UB blocks have no validated inter-block issue schedule'
                                 if plan['required_ub_bytes'] > ub else 'Missing target FP32 Vector throughput')
        return None, 0, active, diag

    known_chip = chip in (load_chip(), load_chip('Ascend950PR_957b'))
    scoped = t <= 256 and 2 <= iterations <= 32 and known_chip
    if not known_chip:
        qualification = 'New hardware: inherited local issue prices and assumed 950 kernel layout; no target CA calibration'
    elif t > 256:
        qualification = 'One-UB extrapolation: T>256 has four separate holdout geometries per 950 SKU'
    elif not 2 <= iterations <= 32:
        qualification = 'Iteration count outside the 2..32 measured range; steady issue period extrapolated'
    else:
        qualification = 'Explicit same-family H: ten independent T<=256 geometries on each 950 SKU'
    epoch = _epochs(chip, prices, prices['phase_prices'][state], plan, iterations)
    # Adjacent boundaries make the additive stages telescope to the whole end.
    marks = [('tiling_and_setup', 'setup_vf'), ('input_and_softmax_issue', 'softmax_vf'),
             ('first_column', 'first_col_vf'), ('normalization_loop', 'scatter_vf'),
             ('scatter_to_output_issue', 'write_begin'), ('output_and_retirement', 'end')]
    begin = 0.
    for name, end in marks:
        diag['stages_us'][name] = epoch[end] - begin
        begin = epoch[end]
    diag.update(issue_epochs_us=epoch, qualification=qualification,
                history_timing_status='estimated' if scoped else 'extrapolated',
                outside_development_local_grain=rows > 4)
    return epoch['end'], 0, active, diag
