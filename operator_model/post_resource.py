"""Four-stream BF16 mHC Post. Predict the launch state before issuing the kernel."""
from math import ceil, log2

from .chips import load_chip
from .post_v12 import parameters
from .wide_v10 import occupancy


def resource_post(shape, chip, attrs):
    if len(shape) != 3 or any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError('mHC Post expects positive T,N,D')
    tokens, streams, width = shape
    if streams != 4 or width % 16 or width > 8192:
        raise ValueError('mHC Post timing requires N=4 and 16-aligned D <= 8192')
    if attrs.keys() - {'has_h_res','resident_aiv'} or attrs.get('has_h_res', True) is not True:
        raise ValueError('mHC Post timing requires the residual input')
    cores = chip.vector_cores
    resident = attrs.get('resident_aiv')
    if resident is not None and (type(resident) is not int or not 0 <= resident <= cores):
        raise ValueError('resident_aiv must be within the target AIV count')

    rows = ceil(tokens / cores)
    slices = 1
    route = 'post_channel' if 2*tokens < cores else 'post_token'
    if route == 'post_channel':
        while ceil(width/slices) > 256 and tokens*2*slices <= cores:
            slices *= 2
    active = ceil(tokens/rows)*slices
    fragment = ceil(width/slices)
    units = dict(arrival=log2(active), SCALAR=rows-1,
                 SIMD=rows*ceil(fragment/64)*streams*(streams+1),
                 MTE2=rows*fragment*streams*2/1024,
                 MTE3=tokens*width*streams*2/1024, other=0.)
    ref = load_chip()
    clock = ref.frequency_ghz / chip.frequency_ghz
    scales = {}
    if chip.vector_tops['32'] and ref.vector_tops['32']:
        scales['SIMD'] = (ref.vector_tops['32']/ref.vector_cores)/(chip.vector_tops['32']/cores)
    for queue in ('MTE2','MTE3'):
        key = queue.lower() + '_bandwidth_gbs'
        if ref.details.get(key) and chip.details.get(key):
            scales[queue] = ref.details[key]/chip.details[key]
    costs = parameters()['phases'][route]
    states = ('first', 'expand', 'reuse')
    phases = {s:{q: c[s]*clock + c['service']*units[q]*scales.get(q,clock)
                    + c['per_extra_row']*(rows-1)*clock for q,c in costs.items()} for s in states}
    # Independent chip-wide ingress floor. The output may still reside in L2.
    logical_in = 2*tokens*width*(streams+1) + 4*tokens*streams*(1+streams)
    total_io = logical_in + 2*tokens*width*streams
    l2_bw = chip.details.get('l2_bandwidth_gbs') or 100*cores*chip.frequency_ghz
    hbm = max(logical_in, total_io - (chip.details.get('l2_bytes') or 134217728))
    floor = max(total_io/l2_bw, hbm/chip.hbm_bandwidth_gbs)/1000
    for stages in phases.values():
        stages['shared_link_floor'] = max(0.,floor-sum(stages.values()))
    totals = {s:sum(stages.values()) for s,stages in phases.items()}
    if resident is not None:
        state = occupancy(active, resident)
        stages = phases[state]
    else:
        # State frequencies in a calibration harness depend on launch order.
        # With no caller history use equal relative loss at hot/cold endpoints.
        feasible = ('first','reuse') if active == 1 else states
        low = min(feasible, key=totals.get)
        high = max(feasible, key=totals.get)
        w = totals[low]/(totals[low]+totals[high])
        stages = {q:(1-w)*us + w*phases[high][q] for q,us in phases[low].items()}
        state = 'unknown'
    plan = dict(active_aiv=active, active_aic=0, route=route, channel_slices=slices,
                fragment_width=fragment, rows_per_core=rows)
    return sum(stages.values()), 0, active, dict(model='post_target_critical_aiv',
        target_plan=plan, stages_us=stages, context=state, context_predictions_us=totals,
        state_phases_us=phases, shared_floor_us=floor)
