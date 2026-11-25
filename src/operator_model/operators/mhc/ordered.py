"""Built-in Post token issue schedule for an observed, non-repeated launch."""
from dataclasses import replace

from operator_model.backend import _rates
from operator_model.operators.mhc.history import _prices, _token_scenario
from operator_model.operators.mhc.repeat import channel_tiles


def price_ordered_token(forecast, shape, chip, previous_route, vf=None):
    """Caller established cold/exact residency or enough coverage to prove reuse."""
    if forecast.latency_us is None or not chip.vector_tops['32']:
        return forecast
    diag = forecast.projection['timing_diagnostics']
    if diag['target_plan']['route'] != 'post_token':
        return forecast
    state = diag['context']
    if state == 'first':
        group = 'first'
    elif state in ('expand', 'reuse') and previous_route in ('post_token', 'post_channel'):
        group = state + '_after_' + previous_route.removeprefix('post_')
    else:
        return forecast
    control = _prices()['token_ordinary']['control'].get(group)
    tiles = channel_tiles(shape[2], chip)
    # The ordinary schedule identifies four full-row input issues and one output.
    # It has no per-fragment issue or handoff price for a split-width row.
    if not control or control['processes'] < 2 or not tiles or tiles['count'] != 1:
        return forecast
    scenario = _token_scenario(shape, chip, diag, group,vf)
    stages = dict(scenario.stages_us)
    timing = dict(model='post_token_ordered', context=state, ordered_scenario=group,
                  previous_route=previous_route, target_plan=diag['target_plan'],
                  shared_floor_us=diag['shared_floor_us'], stages_us=stages,
                  channel_tiles=tiles, control_source_processes=control['processes'],
                  calibration='950 ordinary-token operand issue and response stages, v0.25/v0.27')
    us = scenario.latency_us
    projection = {**forecast.projection, 'latency_us': us, 'stages_us': stages,
                  'timing_diagnostics': timing}
    return replace(forecast, latency_us=us, projection=projection,
                   status='extrapolated', reason='Observed built-in Post sequence with 950-calibrated token issue prices',
                   rates=_rates(forecast.work, us, chip))


def price_ordered_fragments(forecast, shape, chip, resident_aiv, previous_shape,vf=None):
    """Attach the estimated two-slot schedule after a successful different token Post."""
    diag=forecast.projection['timing_diagnostics']
    if diag['context']!='reuse' or diag['target_plan']['route']!='post_token' or not chip.vector_tops['32']:
        return None
    tiles=channel_tiles(shape[2],chip)
    if not tiles or tiles['count']!=2:
        return None
    from operator_model.operators.mhc.fragment_pipeline import estimate_post_fragments
    p=estimate_post_fragments(shape,chip=chip,resident_aiv=resident_aiv,previous_shape=previous_shape,vf_price=vf)
    us=p['us']
    diagnostics=dict(context='reuse',ordered_scenario='reuse_after_token',previous_route='post_token',
                     model=p['model'],target_plan=diag['target_plan'],stages_us=p['stages_us'],
                     shared_floor_us=diag['shared_floor_us'],channel_tiles=tiles,
                     qualification=p['qualification'],fragment_costs_cycles=p['fragment_costs_cycles'])
    projection={**forecast.projection,'latency_us':us,'stages_us':p['stages_us'],'timing_diagnostics':diagnostics}
    return replace(forecast,latency_us=us,projection=projection,status='extrapolated',
                   reason='Two-slot Post estimate for an observed token-to-token shape change',
                   rates=_rates(forecast.work,us,chip))
