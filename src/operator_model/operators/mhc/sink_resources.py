"""Target partitions for N=4 Sinkhorn and its BF16 Pre fusion."""
from math import ceil, log2
from statistics import median

from operator_model.chips import load_chip
from operator_model.operators.fused.templates import FUSED, work
from operator_model.scheduling.reference import _profile
from operator_model.operators.vector.partition import parameters, predict


def compute_scale(ref, target, kind, bits):
    a = getattr(ref, kind + '_tops')[bits]
    b = getattr(target, kind + '_tops')[bits]
    cores = kind + '_cores'
    if a and b:
        return (a / getattr(ref, cores)) / (b / getattr(target, cores))
    return ref.frequency_ghz / target.frequency_ghz


def sink(shape, target, attrs):
    reference = predict('mhc_sinkhorn', shape, attrs)
    if reference['predicted_us'] is None:
        raise ValueError(reference['reason'])
    ref = load_chip()
    tokens, n = shape
    rows = ceil(tokens / target.vector_cores)
    active = ceil(tokens / rows)
    units = {'SIMD': ceil(rows * n*n / 64) * attrs.get('num_iters', 20),
             'MTE2': rows*n*n*4/1024, 'MTE3': tokens*n*n*4/1024,
             'arrival': log2(active), 'SCALAR': rows-1, 'other': 0}
    clock = ref.frequency_ghz / target.frequency_ghz
    rates = {'SIMD': compute_scale(ref, target, 'vector', '32')}
    for phase in ('MTE2', 'MTE3'):
        key = phase.lower() + '_bandwidth_gbs'
        rates[phase] = ref.details[key]/target.details[key] if ref.details.get(key) and target.details.get(key) else clock
    stages = {q: fixed*clock + slope*units[q]*rates.get(q, clock)
              for q, (fixed, slope) in parameters()['phases']['mhc_sinkhorn'].items()}
    return sum(stages.values()), 0, active, dict(stages_us=stages,
        target_plan=dict(active_aiv=active, active_aic=0, rows_per_core=rows,
                         packed_chunks=ceil(rows*n*n/64), iterations=attrs.get('num_iters', 20)))


def partition(shape, target):
    t, n, d = shape
    m_blocks = min(target.cube_cores, ceil(t/256))
    local_m = 16*ceil(ceil(t/m_blocks)/16)
    local_k = 256*ceil(ceil(n*d/(target.cube_cores//m_blocks))/256)
    k_blocks = ceil(n*d/local_k)
    rows = ceil(t/target.vector_cores)
    stage2 = ceil(t/rows)
    ratio = target.vector_cores / target.cube_cores
    launch = min(target.cube_cores, max(m_blocks*k_blocks, ceil(stage2/ratio))) if ratio==2 else target.cube_cores
    return dict(active_aic=launch, active_aiv=min(target.vector_cores, ceil(launch*ratio)),
                m_blocks=m_blocks, k_blocks=k_blocks, tile=[local_m,local_k],
                stage2_payload_aiv=stage2, rows_per_aiv=rows)


def fused(shape, target, attrs):
    # The original stages are disjoint intervals on three representative core
    # classes. Their queue labels permit independent repricing of overlaps.
    work(FUSED, shape, attrs)
    model = _profile('fusion_overlap')
    if any(model['contracts'][FUSED].get(k, object()) != v for k,v in attrs.items()):
        raise ValueError('Expected inference with 20 Sinkhorn iterations')
    ref = load_chip()
    clock = ref.frequency_ghz / target.frequency_ghz
    plan = partition(shape, target)
    m, k = plan['tile']
    estimates = []
    for template in model['templates'][FUSED]:
        base = partition(template['shape'], ref)
        rm, rk = base['tile']
        rows = plan['rows_per_aiv']/base['rows_per_aiv']
        cube = (m*k)/(rm*rk)
        # AIC local fragments scale with the aligned M/K block, whereas the
        # AIV tail handles rows and merges the partials produced by K blocks.
        vs = compute_scale(ref, target, 'vector','32')
        vc = compute_scale(ref, target, 'cube','16')
        amounts = {'CUBE':cube*vc, 'MTE1':cube*clock, 'FIXP':(m/rm)*clock,
                   'SIMD':rows*vs,'MTE2':rows*clock,'MTE3':rows*clock}
        estimates.append(max(sum(us*max(amounts.get(q,clock) for q in mask.split('+'))
                                 for mask,us in core.items()) for core in template['cores']))
    pipe = median(estimates)
    # Traffic floor uses distinct tensors for one cold launch, including shared weights.
    t,n,d = shape
    total_bytes = 2*t*n*d + 4*n*d*(n*n+2*n) + 2*t*d
    transfer = total_bytes / (target.hbm_bandwidth_gbs*1000)
    stages = {'overlap_window':pipe, 'exposed_hbm_floor':max(0.,transfer-pipe)}
    return sum(stages.values()),plan['active_aic'],plan['active_aiv'],dict(stages_us=stages,target_plan=plan)
