"""RMSNorm startup, known-stream lane/sector scheduling and unknown-history policy."""
from . import rms_lane_schedule
from .rms_target_cost import configured_rms

# Median excess of the exposed MTE2 stage on 19 previously recorded reusable
# critical AIVs (v0.12 development). Calibrated in core cycles, independently
# of kernel time. Future architectures inherit the 950 issue pipeline.
REUSE_READ_LEAD_CYCLES = 228.


def price_rms(shape, chip, attrs):
    total, aic, active, diag = configured_rms(shape, chip, attrs)
    phases = {state:dict(costs) for state,costs in diag['conditional_stages_us'].items()}
    hot = phases['reuse']
    cycles_per_us = chip.frequency_ghz * 1000
    # A warm input can issue earlier. Preserve both the remaining local read
    # service and the existing whole-chip read/write bound.
    hot['MTE2'] = max(0.,hot['MTE2']-REUSE_READ_LEAD_CYCLES/cycles_per_us,
                      diag['local_read_port_floor_cycles']/cycles_per_us-hot['SIMD'])
    shared_min = total_hot_bound(diag, cycles_per_us)
    hot['shared_io'] = max(0.,shared_min-sum(t for q,t in hot.items() if q!='shared_io'))
    totals = {state:sum(costs.values()) for state,costs in phases.items()}
    state = diag['residency']
    probability = None
    lane_diag = None
    basis = {s:'v0.21 resource pipeline' for s in phases}
    if attrs.get('resident_aiv') is None:
        # Unknown preceding dispatch size H ~ uniform {1,...,C}, conditional
        # on an already-started stream. A prefix of A cores expands if H<A.
        probability = (active-1)/chip.vector_cores
        running = {q:(1-probability)*hot[q]+probability*phases['expand'][q] for q in hot}
        running_us = sum(running.values())
        # Initial vs running stream is unobserved. Their harmonic center has
        # equal opposite relative residuals at the two candidate latencies.
        w = running_us/(running_us+totals['first'])
        stages = {q:w*phases['first'][q]+(1-w)*running[q] for q in hot}
        state = 'unknown'
    elif attrs['resident_aiv'] > 0 and diag['target_plan']['iterations'] == 1:
        stages, lane_diag = rms_lane_schedule.schedule(shape, chip, attrs['resident_aiv'], diag)
        # Expansion depends on the exact previous prefix. The other states
        # describe the existing v0.21 scenario baselines, not this launch.
        phases[state] = stages
        totals[state] = sum(stages.values())
        basis[state] = 'known AIV prefix, single UB tile, rstd sector and y retirement'
    else:
        stages = phases[state]
    return sum(stages.values()), aic, active, {**diag,'stages_us':stages,
        'conditional_stages_us':phases,'conditional_us':totals,'residency':state,
        'residency_assumption':'unknown_stream_history' if state=='unknown' else 'provided',
        'running_stream_expand_probability':probability,'lane_schedule':lane_diag,
        'conditional_model_basis':basis,'reuse_read_lead_cycles':REUSE_READ_LEAD_CYCLES}


def total_hot_bound(diag, cycles_per_us):
    from .rms_resource import FIRST_PAYLOAD_RELEASE, FIRST_SCALAR
    scalar = diag['conditional_stages_us']['reuse']['SCALAR']
    return scalar*FIRST_PAYLOAD_RELEASE/FIRST_SCALAR + diag['shared_io_floor_cycles']/cycles_per_us
