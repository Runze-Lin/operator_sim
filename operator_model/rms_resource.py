"""RMSNorm on the target's row partition, full-row UB path and two output streams."""
from math import ceil
from . import rms_v12, wide_v10
from .vector_v8 import features

STATES = ('first', 'expand', 'reuse')


def resource_rms(shape, chip, attrs=None):
    call = dict(attrs or {})
    resident = call.pop('resident_aiv', None)
    if call:
        raise ValueError('RMSNorm resource route accepts BF16 full rows and optional resident_aiv')
    if resident is not None and (type(resident) is not int or not 0 <= resident <= chip.vector_cores):
        raise ValueError('resident_aiv must be within the target Vector core count')
    features('rmsnorm', shape, call)
    tokens, width = shape
    rows = ceil(tokens / chip.vector_cores)
    active = ceil(tokens / rows)
    tiles = rms_v12.tile_rows(rows, width, chip.details.get('ub_bytes') or 253952)
    cycles_per_us = 1000 * chip.frequency_ghz
    params = rms_v12.parameters()
    coeffs = wide_v10.parameters()['phases']['rmsnorm']
    # 950DT reference whole-chip FP32 dense peak, core count and frequency.
    dt_ops = 27000 / (64 * 1.65)
    target_ops = ((chip.vector_tops['32'] or 0) * 1000 /
                  (chip.vector_cores * chip.frequency_ghz)) or dt_ops
    vec_price = dt_ops / target_ops
    groups = rows * ceil(width / 128)
    # Only byte/group work responds to the corresponding throughput; preserve
    # the fixed first-issue and dependency parts of each DT phase in cycles.
    rates = dict(params)
    for name in ('normalize', 'reduce'):
        fixed, per_group = params[name]
        rates[name] = [fixed, per_group * vec_price]
    fixed, per_kib = params['y']
    write_gbs = chip.details.get('mte3_bandwidth_gbs')
    rates['y'] = [fixed, 1024 * chip.frequency_ghz / write_gbs if write_gbs else per_kib]
    raw_write, possible_overlap = rms_v12.schedule_writes(width, rows, active, tiles, rates)
    local_read = 2 * width * (rows + 1)  # critical core loads its own gamma and x
    local_read_gbs = chip.details.get('mte2_bandwidth_gbs')
    read_floor = local_read * chip.frequency_ghz / local_read_gbs if local_read_gbs else 0.
    x_and_gamma = 2 * width * (tokens + 1)
    outputs = 2 * width * tokens + 4 * tokens
    bandwidth = min(chip.hbm_bandwidth_gbs, chip.details.get('l2_bandwidth_gbs') or float('inf'))
    io_floor = (x_and_gamma + outputs) * chip.frequency_ghz / bandwidth

    def conditional(state):
        simd = params['simd'][STATES.index(state)] + params['simd'][3]*groups*vec_price
        times = {q:coeffs[q][state] * params['ticks_per_us']
                 for q in ('arrival','SCALAR','other','MTE2')}
        times['MTE2'] = max(times['MTE2'], read_floor)
        times['SIMD'] = simd
        times['MTE3'] = raw_write - min(simd, possible_overlap)
        # Shared cold-payload floor applies across the complete data path, after
        # arrival and control. No extra charge when already hidden by compute.
        useful_window = times['MTE2'] + times['SIMD'] + times['MTE3']
        times['shared_io'] = max(0., io_floor - useful_window)
        return {q:v/cycles_per_us for q,v in times.items()}

    conditions = {s:conditional(s) for s in STATES}
    totals = {s:sum(t.values()) for s,t in conditions.items()}
    if resident is None:
        # Equal relative loss at either endpoint, without assuming the simulated
        # application's unknown cold-launch frequency equals a DT capture batch.
        fast, slow = min(STATES,key=totals.get), max(STATES,key=totals.get)
        weight = totals[slow] / (totals[fast] + totals[slow])
        times = {q:weight*v + (1-weight)*conditions[slow][q] for q,v in conditions[fast].items()}
        state = 'unknown_relative_minimax'
    else:
        state = wide_v10.occupancy(active, resident)
        times = conditions[state]
    plan = dict(active_aic=0,active_aiv=active,rows_per_core=rows,ub_tile_rows=tiles,
                iterations=len(tiles),vector_groups=groups)
    diag = dict(model='target_resource_rmsnorm',target_plan=plan,stages_us=times,
                conditional_stages_us=conditions,conditional_us=totals,residency=state,
                resident_aiv=resident,vector_price_cycles_per_reference_cycle=vec_price,
                store_union_cycles=raw_write,store_eligible_overlap_cycles=possible_overlap,
                local_read_bytes=local_read,local_read_port_floor_cycles=read_floor,
                cold_read_bytes=x_and_gamma,write_bytes=outputs,shared_io_floor_cycles=io_floor,
                calibrated_on='950DT phase intercepts/byte service and critical-core write intervals')
    return sum(times.values()),0,active,diag
