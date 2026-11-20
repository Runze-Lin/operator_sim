"""Target RMSNorm partition, isolated-call startup, UB tile waves and shared I/O."""
from math import ceil
from operator_model.operators.rmsnorm import writeback as rms_writeback
from operator_model.operators.vector import wide_rows as wide_rows

STATES = ('first', 'expand', 'reuse')
# Cycles at the 950DT reference; reproducible from per-phase measurements with
# The startup and transfer prices use reference-device instruction windows.
ARRIVAL_PER_AIV = 14.625
FIRST_SCALAR = 2933.5
FIRST_PAYLOAD_RELEASE = 2970.0
FIRST_READ = 1388.654
READ_PER_KIB = 18.39635
WRITE_PER_KIB = 16.01826
STAT_SOLO = 133.0
STAT_PER_NEIGHBOUR = 147.58824


def _write_window(width, rows, active, cap, full, tail, rates, neighbours):
    """Price a periodic full-tile train from its steady increment and its tail.

    Only the first two identical tiles and boundary tile are materialized.
    Consecutive equal tile launches have the same separation in this schedule.
    """
    def run(sequence):
        return rms_writeback.schedule_writes(width, rows, active, sequence, rates, neighbours)
    edge = [tail] if tail else []
    count = min(2, full)
    raw, hidden = run([cap] * count + edge)
    if full > count:
        raw1, hidden1 = run([cap])
        raw2, hidden2 = run([cap, cap])
        raw += (full-count)*(raw2-raw1)
        hidden += (full-count)*(hidden2-hidden1)
    return raw, hidden


def resource_rms(shape, chip, attrs=None):
    call = dict(attrs or {})
    resident = call.pop('resident_aiv', None)
    if call:
        raise ValueError('RMSNorm resource route accepts BF16 full rows and optional resident_aiv')
    if resident is not None and (type(resident) is not int or not 0 <= resident <= chip.vector_cores):
        raise ValueError('resident_aiv must be within the target Vector core count')
    if len(shape) != 2 or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError('RMSNorm requires two positive integer dimensions')
    tokens, width = shape
    if width > 8192 or width % 16:
        raise ValueError('Outside full-row BF16 channel range')
    rows = (tokens + chip.vector_cores - 1) // chip.vector_cores
    active = (tokens + rows - 1) // rows
    if width <= 16 and rows > 512:
        raise ValueError('Short-channel transpose route is outside full-row RMSNorm')
    cap = rms_writeback.tile_capacity(width, chip.details.get('ub_bytes') or 253952)
    full, tail = divmod(rows, cap)
    waves = full + bool(tail)
    cycles_per_us = 1000 * chip.frequency_ghz
    params = rms_writeback.parameters()
    coeffs = wide_rows.parameters()['phases']['rmsnorm']
    dt_ops = 27000 / (64 * 1.65)
    target_ops = ((chip.vector_tops['32'] or 0) * 1000 /
                  (chip.vector_cores * chip.frequency_ghz)) or dt_ops
    vec_price = dt_ops / target_ops
    groups = rows * ceil(width / 128)
    rates = dict(params)
    for name in ('normalize', 'reduce'):
        fixed, per_group = params[name]
        rates[name] = [fixed, per_group * vec_price]
    write_gbs = chip.details.get('mte3_bandwidth_gbs')
    write_price = 1024 * chip.frequency_ghz / write_gbs if write_gbs else WRITE_PER_KIB
    rates['y'] = [params['y'][0], write_price]
    rates['rstd'] = [STAT_SOLO, STAT_PER_NEIGHBOUR]
    neighbours = max(0., min(active, params['window_bytes'] / (4*rows)) - 1)
    raw_write, possible_overlap = _write_window(width, rows, active, cap, full, tail, rates, neighbours)
    local_read = 2 * width * (rows + 1)
    first_tile_bytes = 2 * width * min(rows, cap)
    read_gbs = chip.details.get('mte2_bandwidth_gbs')
    read_price = 1024 * chip.frequency_ghz / read_gbs if read_gbs else READ_PER_KIB
    read_floor = local_read * read_price / 1024
    x_and_gamma = 2 * width * (tokens + 1)
    outputs = 2 * width * tokens + 4 * tokens
    # Both ping-pong input tiles of every active AIV and the shared weight
    # compete for L2. Dirty output that fits may retire before external drain.
    input_reserve = 2 * width * (1 + min(tokens, 2 * min(rows, cap) * active))
    cache_bytes = chip.details.get('l2_bytes') or 0
    retention = chip.details.get('rms_output_retained_fraction')
    retained = (1. if retention is None else retention) * min(outputs, max(0, cache_bytes - input_reserve))
    logical_bytes = x_and_gamma + outputs
    external_bytes = logical_bytes - retained
    external_cycles = external_bytes * chip.frequency_ghz / chip.hbm_bandwidth_gbs
    l2_rate = chip.details.get('l2_bandwidth_gbs')
    l2_cycles = logical_bytes * chip.frequency_ghz / l2_rate if l2_rate else 0.
    io_floor = max(external_cycles, l2_cycles)

    def conditional(state):
        simd = params['simd'][STATES.index(state)] + params['simd'][3]*groups*vec_price
        times = {q:coeffs[q][state] * params['ticks_per_us']
                 for q in ('arrival','SCALAR','other','MTE2')}
        if state == 'first':
            times['arrival'] = ARRIVAL_PER_AIV * (active-1)
            times['SCALAR'] = FIRST_SCALAR
            times['MTE2'] = FIRST_READ
        # Gamma and first payload issue close together. Subsequent payload can
        # overlap useful vector work; serialize only the remaining port demand.
        times['MTE2'] += read_price * first_tile_bytes / 1024
        times['MTE2'] += max(0., read_floor - times['MTE2'] - simd)
        times['SIMD'] = simd
        port_cycles = rows*(2*width+4)*write_price/1024
        times['MTE3'] = max(raw_write - min(simd, possible_overlap), port_cycles-simd)
        # The same start-to-end window spans all AIVs. A core can transfer
        # while other cores still execute their own setup or reduction.
        release = FIRST_PAYLOAD_RELEASE * times['SCALAR'] / FIRST_SCALAR
        times['shared_io'] = max(0., release + io_floor - sum(times.values()))
        return {q:v/cycles_per_us for q,v in times.items()}

    conditions = {s:conditional(s) for s in STATES}
    totals = {s:sum(t.values()) for s,t in conditions.items()}
    state = 'first' if resident is None else wide_rows.occupancy(active, resident)
    times = conditions[state]
    # A compact run length represents arbitrarily many equal full UB tiles.
    plan = dict(active_aic=0, active_aiv=active, rows_per_core=rows, iterations=waves,
                full_tile_rows=cap, full_tile_count=full, tail_tile_rows=tail, vector_groups=groups)
    if waves <= 32:
        plan['ub_tile_rows'] = [cap]*full + ([tail] if tail else [])
    diag = dict(model='target_resource_rmsnorm',target_plan=plan,stages_us=times,
                conditional_stages_us=conditions,conditional_us=totals,residency=state,
                residency_assumption='standalone_first_launch' if resident is None else 'provided',
                resident_aiv=resident,vector_price_cycles_per_reference_cycle=vec_price,
                store_union_cycles=raw_write,store_eligible_overlap_cycles=possible_overlap,
                statistic_neighbours=neighbours,local_read_bytes=local_read,
                local_read_port_floor_cycles=read_floor,read_cycles_per_kib=read_price,
                write_cycles_per_kib=write_price,cold_read_bytes=x_and_gamma,
                write_bytes=outputs,shared_io_floor_cycles=io_floor,l2_input_reserve_bytes=input_reserve,
                assumed_retained_write_bytes=retained,modeled_external_bytes=external_bytes,
                external_floor_cycles=external_cycles,l2_floor_cycles=l2_cycles,
                calibrated_on='950DT critical-AIV phases, startup spread and descriptor intervals')
    return sum(times.values()),0,active,diag
