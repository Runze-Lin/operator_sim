"""No-history pricing for the standalone 4×4 FP32 Ascend Sinkhorn kernel."""
from statistics import fmean

from .chips import load_chip
from .sink_history import price as history_price


def price(shape, chip, attrs):
    """Use all reachable launch states; no observed CA time or stream history is an input."""
    cold, aic, active, initial = history_price(shape, chip, {**attrs, 'resident_aiv': 0})
    plan = dict(initial['target_plan'])
    del plan['resident_aiv'], plan['history_state']
    result = dict(target_plan=plan, schedule='sinkhorn_unknown_history',
                  parameter_source=initial['parameter_source'], kernel_api=initial['kernel_api'], stages_us={})
    if cold is None:
        return None, aic, active, {**result, 'qualification': initial['qualification']}
    register_bytes = chip.details.get('vector_register_bytes', 256)
    block_bytes = chip.details.get('ub_block_bytes', 32)
    if (register_bytes, block_bytes) != (256,32):
        return None, aic, active, {**result, 'qualification':'The measured FP32 iteration issue prices assume 256-byte Vector registers and 32-byte UB blocks'}
    if plan['register_passes'] > 1:
        # Preserve previous numeric coverage on custom low-core-count profiles
        # where the legacy T<=256 rule still applies.
        if shape[0] <= 256:
            from .mhc_sink_resource import sink
            us,aic,active,old = sink(shape,chip,attrs)
            old.update(schedule='legacy_sinkhorn_multiregister',history_timing_status='extrapolated',
                       qualification='Multiple register passes; legacy macro-stage estimate on this target')
            return us,aic,active,old
        result['qualification']='Multiple register passes per helper; default no-history issue timing awaits prospective validation'
        return None, aic, active, result

    # A=1 admits first and resident. No integer H can satisfy 0<H<A.
    states = {'first': initial}
    if active > 1:
        states['expand'] = history_price(shape, chip, {**attrs, 'resident_aiv': 1})[3]
    states['resident'] = history_price(shape, chip, {**attrs, 'resident_aiv': active})[3]
    prices = {key: state['issue_epochs_us']['end'] for key,state in states.items()}
    low, high = min(prices.values()), max(prices.values())
    stages = {key:fmean(s['stages_us'][key] for s in states.values()) for key in initial['stages_us']}
    known = chip in (load_chip(), load_chip('Ascend950PR_957b'))
    qualified = known and active > 1 and shape[0] <= 256 and 2 <= plan['iterations'] <= 32
    result.update(stages_us=stages, scenario_us=prices, modeled_range_us=[low, high],
                  point_rule='Arithmetic mean of the distinct reachable first, expand and resident schedules. Deployment probabilities unknown.',
                  relative_minimax_us=2*low*high/(low+high), modeled_minimax_risk=(high-low)/(high+low),
                  history_timing_status='estimated' if qualified else 'extrapolated',
                  qualification=('Unknown history, one Vector register pass: original blind ten-shape 950DT/PR validation; cold-only and resident-only mixes can be biased'
                                 if qualified else 'One-register schedule extrapolated beyond the original blind 950 ten-shape contract'),
                  stage_interpretation='Per-stage averages over mutually exclusive launch histories. The range describes modeled scenarios, not a guaranteed CA interval.')
    return fmean(prices.values()), aic, active, result
