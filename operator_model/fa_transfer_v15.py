"""Reprice FA fill, MMAD window, tile cadence, and drain for a target chip."""
from math import ceil

from .chips import load_chip
from .model_v2 import plan
from .model_v5 import supported
from .model_v6 import Case
from .predictor import _legacy, _profile


def _phase_rates(group, factors):
    """DT share of each wall-clock phase and its target/reference service ratio."""
    shares = []
    rates = []
    for atoms in group['dt_fractions']:
        share = sum(atoms.values())
        service = sum(weight * max(factors[q] for q in mask.split('+'))
                      for mask, weight in atoms.items())
        shares.append(share)
        rates.append(service / share)
    return shares, rates


def project_fa(shape, factors, reference, target, reference_result):
    case = Case('fa', shape)
    q, kv, depth, batch, heads = case.shape
    grid = plan(case)
    loops = grid['iterations']
    profile = _profile('results/v0.15/fa-phases.json')
    group = profile['single' if loops == 1 else 'loop']
    shares, rates = _phase_rates(group, factors)
    jobs = batch * heads * ceil(q / grid['q_tile_capacity'])
    calibrated = (reference == load_chip() and target == load_chip('Ascend950PR_957b')
                  and jobs <= min(target.cube_cores, target.vector_cores // 2))
    ref_us = reference_result['predicted_us']
    names = ('first_mmad_prefix', 'mmad_window', 'drain')
    if supported(case):
        # Only Q9 / D64 / single-head B1 has matched long-loop reference streams.
        repeats = ceil(kv / 256) - 1
        base = profile['long']['dt']
        if calibrated:
            cost = profile['long']['pr']
            issue = cost['issue']
            values = [cost['prefix'], cost['middle_fill'], repeats*issue, cost['drain']]
        else:
            # The measured issue interval already includes producer backpressure.
            # Recompute the limiting engine and aggregate external traffic floor.
            service = _legacy().model['base_v5_model']['fa_long']['costs']
            issue = max(base['issue']*factors['SCALAR'],
                        service['cube_service_per_tile_us']*factors['CUBE'],
                        service['vector_service_per_tile_us']*factors['SIMD'],
                        service['mte1_service_per_tile_us']*factors['MTE1'],
                        service['fix_service_per_tile_us']*factors['FIXP'],
                        2*256*depth*2/(1000*target.hbm_bandwidth_gbs))
            values = [base['prefix']*rates[0], base['middle_fill']*rates[1],
                      repeats*issue, base['drain']*rates[2]]
        norm = ref_us/(base['prefix']+base['middle_fill']+repeats*base['issue']+base['drain'])
        stages = dict(zip(('first_mmad_prefix','mmad_fill','steady_loop','drain'), (v*norm for v in values)))
        diag = dict(kv_tiles=1+repeats, kv_issue_tokens=256, issue_us=issue, mmads_per_core=2*ceil(kv/128))
    else:
        if calibrated:
            rates = group['pr_ratios']
        stages = dict(zip(names, (ref_us*s*r for s,r in zip(shares,rates))))
        diag = dict(kv_tiles=loops, kv_issue_tokens=grid['kv_tile_capacity'])
    diag.update(method='FA first MMAD / loop / drain', phases_us=stages,
                parameter_source='paired PR CA' if calibrated else 'DT queue exposures and target chip config')
    return sum(stages.values()), diag
