"""Reprice a frozen 950DT dispatch on a hypothetical target, without device IO."""
from copy import copy
from math import ceil
from statistics import median

from operator_model.operators.fused import templates as fusion_templates
from operator_model.operators.mhc import pre_dispatch as mhc_partition
from operator_model.profiles import attention_loop as attention_loop
from operator_model.profiles.stage_model import line, schedule
from operator_model.profiles.tile_model import plan
from operator_model.scheduling.cases import Case
from operator_model.scheduling.reference import reference_prices, _profile
from operator_model.operators.quant.repeat import price_plan
from operator_model.scheduling.tiling import quant_plan
from operator_model.operators.matmul.tiling import matmul_plan


def _compute_ratio(engine, counts, clock, ref, target, notes):
    """Work-weighted ratio of per-core peak service times, excluding clock duplication."""
    if not counts:
        return clock
    base_rates = getattr(ref, engine + '_tops')
    target_rates = getattr(target, engine + '_tops')
    base_cores = getattr(ref, engine + '_cores')
    target_cores = getattr(target, engine + '_cores')
    weighted, total = 0., 0
    for dtype, amount in counts.items():
        bits = next((b for b in ('32', '16', '8', '4') if b in dtype), None)
        a, b = base_rates.get(bits), target_rates.get(bits)
        if a and b:
            ratio = (a / base_cores) / (b / target_cores)
        else:
            ratio = clock
            notes.append(f'{engine} {dtype}: absent comparable peaks; assumed unchanged operations per core-cycle.')
        weighted += amount * ratio
        total += amount
    return weighted / total if total else clock


def _tasks(op, sh, attrs, res):
    """Independent DT tile jobs. K/sequence reductions stay within their jobs."""
    if op in ('quant', 'matmul', 'fa', 'addmm'):
        p = res.get('plan') or plan(Case(op, sh))
        if op == 'quant':
            return 0, sh[0] * ceil(sh[1]/p['slice_cols'])
        if op == 'fa':
            q, _, _, b, h = Case('fa', sh).shape
            n = b*h*ceil(q/p['q_tile_capacity'])
            return n, 2*n
        return ceil(sh[0]/p['tile'][0])*ceil(sh[1]/p['tile'][1]), 0
    if op in ('rmsnorm','mhc_sinkhorn','mhc_post','kda_recurrent'):
        if op == 'kda_recurrent':
            return 0, 1
        return 0, sh[0] if res.get('route') != 'post_channel' else res['predicted_active_aiv']
    if op == 'lightning_indexer':
        return sh[1], 2*sh[1]
    if op in ('gmm2', fusion_templates.FIRST):
        _, k, f, _ = sh
        n = sum(ceil(x/16) for x in attrs['expert_counts'])*ceil((k if op=='gmm2' else 2*f)/128)
        return n, 0 if op=='gmm2' else sum(ceil(x*f/256) for x in attrs['expert_counts'])
    t,n,d = sh
    if op == 'split_sink':
        return 0,t
    if op == 'split_post':
        slices=1
        while t<32 and ceil(d/slices)>256 and t*slices<=32:
            slices*=2
        return 0,t*slices
    return (ceil(t/16)*ceil(n*d/256),t)


def _queue_factors(op, shape, attrs, res, work, ref, target, notes):
    clock = ref.frequency_ghz / target.frequency_ghz
    cjobs,vjobs = _tasks(op, shape, attrs, res)
    def waves(n, a, b):
        return ceil(n/b)/ceil(n/a) if n else 1.
    cw = waves(cjobs, ref.cube_cores, target.cube_cores)
    vw = waves(vjobs, ref.vector_cores, target.vector_cores)
    if op in ('matmul','quant'):
        # Their target plan already prices all local tile work and wave counts.
        cw = vw = 1.
    cu = _compute_ratio('cube', work['cube_ops_by_dtype'], clock, ref, target, notes)
    vu = _compute_ratio('vector', work['vec_ops_by_dtype'], clock, ref, target, notes)
    # Finite parallel jobs and fixed local tiles. This approximate split cannot
    # identify cache misses vs. startup from an MTE busy interval.
    mw = cw if cjobs else vw
    read,write = work['logical_read_bytes'],work['logical_write_bytes']
    size = read+write if read is not None and write is not None else None
    bw0, bw1 = ref.hbm_bandwidth_gbs, target.hbm_bandwidth_gbs
    pressure = min(1., size/(1000*bw0*res['predicted_us'])) if size is not None and bw0 and bw1 else 0.
    hbm = bw0 / bw1 if pressure else 1.
    factors = dict(SIMD=vu*vw,CUBE=cu*cw,MTE1=clock*cw,FIXP=clock*cw,
                   SCALAR=clock,IDLE=clock,arrival=clock,other=clock)
    for q in ('MTE2','MTE3'):
        key = q.lower()+'_bandwidth_gbs'
        a,b = ref.details.get(key),target.details.get(key)
        local = a/b if a and b else clock*mw
        factors[q] = (1-pressure)*local + pressure*hbm
    active_v = min(vjobs,target.vector_cores)
    if op in ('rmsnorm','mhc_sinkhorn') or (op=='mhc_post' and res.get('route')=='post_token'):
        # DT partitions rows into equal-size contiguous chunks. The last
        # partial chunk can leave cores idle even when rows exceed core count.
        active_v = ceil(vjobs/ceil(vjobs/target.vector_cores))
    return factors,(min(cjobs,target.cube_cores),active_v)


def _template(op, shape, attrs, f):
    demand = fusion_templates.work(op, shape, attrs)
    candidates = []
    for template in _profile('fusion_overlap')['templates'][op]:
        ratios = {q:f.get(q, f['SCALAR'])*n/template['work'][q] for q,n in demand.items()}
        candidates.append(max(sum(us*max(ratios[q] for q in mask.split('+')) for mask,us in phases.items())
                              for phases in template['cores']))
    return median(candidates)


def _reference_time(op, shape, f, target_plan=None):
    predictor = reference_prices()
    case = Case(op,shape)
    if op == 'quant':
        # Same finite UB recurrence (including the O(log N) repeated-tile route).
        p = copy(predictor)
        coeff = predictor.quant_costs
        rates = dict(prefix='SCALAR',first_read='MTE2',read_interval='MTE2',vector='SIMD',
                     control_floor='SCALAR',write_drain_us='MTE3',suffix_us='SCALAR')
        costs = {k:({key:v*f[rates[k]] for key,v in x.items()} if hasattr(x,'items') else x*f[rates[k]]) for k,x in coeff.items()}
        p.quant_costs = costs
        p.issue_gap = [v*f['SCALAR'] for v in predictor.issue_gap]
        return price_plan(p,target_plan)
    if attention_loop.supported(case):
        coeff = predictor.attention_costs
        rates = {'first_k_latency_us':'MTE2','first_v_latency_us':'MTE2','k_initiation_interval_us':'MTE2',
                 'v_initiation_interval_us':'MTE2','cube_service_per_tile_us':'CUBE','vector_service_per_tile_us':'SIMD',
                 'mte1_service_per_tile_us':'MTE1','fix_service_per_tile_us':'FIXP'}
        costs = {k:v*f[rates.get(k,'SCALAR')] for k,v in coeff.items()}
        return attention_loop.schedule(shape[1],costs)['predicted_us']
    data = predictor.tile_schedule(case)
    names = dict(cube='CUBE',vector='SIMD',read='MTE2',write='MTE3',fix='FIXP')
    if op == 'matmul':
        model = predictor.tiles['matmul']
        costs = model['stage_costs']
        local = {k:line(costs[k],v)*f[names[k]] for k,v in target_plan['units'].items()}
        body = schedule(op,target_plan,local)
        intercept,body_gain,arrival_gain = model['total_weights']
        from math import log2
        control = (intercept + arrival_gain*log2(1+target_plan['active_aic'])
                   + model['branch_offsets'].get(target_plan['branch'], 0))
        return max(.001,body_gain*body+control*f['SCALAR'])
    stages = data['stage_services_us']
    body = schedule(op,data['plan'],{k:v*f[names[k]] for k,v in stages.items()})
    ref_body = schedule(op,data['plan'],stages)
    # Some fitted branches have a body gain >1. Preserve positivity when
    # separating the legacy residual from the explicitly scheduled body.
    gain = min(1.,data['predicted_us']/ref_body)
    return gain*body + (data['predicted_us']-gain*ref_body)*f['SCALAR']


def project(op, shape, attrs, reference_result, work, reference_chip, target_chip):
    """Return projected latency and disjoint stage times in microseconds."""
    from operator_model.workload import account_work
    attrs = attrs or {}
    ref,target = reference_chip,target_chip
    base = reference_result['predicted_us']
    if base is None:
        raise ValueError('A timed 950DT kernel is required for projection')
    notes = ['Fixed reference tile family; target occupancy is a job-count estimate.',
             'MTE/HBM split uses logical bytes and nominal bandwidth; HBM traffic is unmeasured.']
    stages = {}
    target_plan = None
    timing_diagnostics = {}
    aic=aiv=0
    def prices(part, res, w):
        nonlocal aic,aiv
        f,active = _queue_factors(part,shape,attrs,res,w,ref,target,notes)
        aic=max(aic,active[0])
        aiv=max(aiv,active[1])
        return f
    if op in ('quant','fa','matmul'):
        f=prices(op,reference_result,work)
        if op=='matmul':
            target_plan = matmul_plan(shape,target)
        elif op=='quant':
            target_plan = quant_plan(shape,target.vector_cores,target.details.get('ub_bytes') or 253952)
        if target_plan:
            aic,aiv = target_plan['active_aic'],target_plan['active_aiv']
            notes[0] = 'Target core count drives a fresh 950-policy partition; target local work and waves are priced once.'
        adjusted = None
        if op == 'quant':
            from operator_model.operators.quant.projection import project_quant
            adjusted = project_quant(shape, f, ref, target, target_plan)
        elif op == 'matmul':
            from operator_model.operators.matmul.projection import project_matmul
            adjusted = project_matmul(shape, f, ref, target, target_plan)
        elif op == 'fa':
            from operator_model.operators.attention.projection import project_fa
            adjusted = project_fa(shape, f, ref, target, reference_result)
        if adjusted is None:
            stages['scheduled_window'] = _reference_time(op, shape, f, target_plan)
        else:
            stages['scheduled_window'], timing_diagnostics = adjusted
            notes[1] = 'Timing uses pipeline stages. See timing_diagnostics for calibration scope and assumptions.'
    elif op=='addmm':
        r=reference_result
        f=prices(op,r,work)
        old=r['stage_services_us']
        new=[v*f[q] for q,v in old.items()]
        exposed=sum(old.values())-max(old.values())
        fraction=r['exposed_us']/exposed if exposed else 0.
        stages['control']=(r['control_us']+r['arrival_us'])*f['SCALAR']
        stages['queue_window']=r['plan']['waves']*max(new)+fraction*(sum(new)-max(new))
    elif op in ('rmsnorm','mhc_post','mhc_sinkhorn','kda_recurrent'):
        r=reference_result
        f=prices(op,r,work)
        stages={q:v*f.get(q,f['SCALAR']) for q,v in r['stages_us'].items()}
        if op=='rmsnorm':
            from operator_model.operators.rmsnorm.writeback import parameters as rms_parameters
            ticks=rms_parameters()['ticks_per_us']
            hidden=min(r['stages_us']['SIMD'],r['write_simd_overlap_ticks']/ticks)
            raw=r['write_union_ticks']/ticks*f['MTE3']
            stages['MTE3']=max(0.,raw-min(raw,hidden*f['SIMD'],stages['SIMD']))
    elif op in ('mhc','mhc_pre') and 'pre' in reference_result['stages_us']:
        for part,us in reference_result['stages_us'].items():
            branch,units=mhc_partition.partition(part,shape)
            key={'pre':'mhc_pre','sink':'split_sink','post':'split_post'}[part]
            child={'pre':'mhc_pre','post':'mhc_post','sink':'mhc_sinkhorn'}[part]
            w=account_work(child,shape[:2] if part=='sink' else shape)
            f=prices(key,{'predicted_us':us},w)
            qmap=dict(compute='CUBE',vector='SIMD',traffic='MTE2',control='SCALAR')
            stages[part]=sum(c['startup_us']*f['SCALAR']+c['unit_us']*units[name]*f[qmap[name]]
                             for name,c in mhc_partition.profile()['stages'][part][branch].items())
    else:
        for part,us in reference_result['stages_us'].items():
            if part in ('split_post','split_sink'):
                child='mhc_post' if part=='split_post' else 'mhc_sinkhorn'
                w=account_work(child,shape if child=='mhc_post' else shape[:2])
            elif part!=op:
                w=account_work(part,shape,attrs if op=='moe_ffn' else {})
            else:
                w=work
            f=prices(part,{'predicted_us':us},w)
            stages[part]=_template(part,shape,attrs,f)
    return dict(latency_us=sum(stages.values()),active_aiv=aiv,active_aic=aic,method='fixed_dispatch_phase_projection',
                reference_latency_us=base,stages_us=stages,target_plan=target_plan,timing_diagnostics=timing_diagnostics,notes=list(dict.fromkeys(notes)))
