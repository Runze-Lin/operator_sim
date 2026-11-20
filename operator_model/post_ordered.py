"""Built-in Post token issue schedule for an observed, non-repeated launch."""
from dataclasses import replace

from .backend import _rates
from .post_history import _prices, _token_scenario
from .post_repeat import channel_tiles


def price_ordered_token(forecast, shape, chip, previous_route):
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
    scenario = _token_scenario(shape, chip, diag, group)
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
