"""Optional price for two consecutive successful dispatches of the same built-in Post."""
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path

from .backend import _rates
from .post_token_pipeline import token_pipeline


@lru_cache(maxsize=1)
def _prices():
    path = Path(__file__).resolve().parents[1] / 'results/v0.25/post/prices.json'
    return json.loads(path.read_text(encoding='utf-8'))


def channel_tiles(width, chip):
    """RegBase N=4 host tiling. Both coefficient queues align to 8 FP32 values."""
    ub = chip.details.get('ub_bytes') or 253952
    # For four streams, coefficients consume 320 B. Double-buffered data uses
    # 4 B hOut, 16 B input streams and 32 B output/intermediates per column.
    max_width = ((ub - 321) // 52 // 16) * 16
    if max_width < 32:
        return None
    pieces = (width + max_width - 1) // max_width
    inner = max(32, ((width + pieces - 1) // pieces + 15) // 16 * 16)
    return dict(count=pieces, inner=inner, tail=width-(pieces-1)*inner,
                ub_bytes=ub, inherited_ub=not bool(chip.details.get('ub_bytes')))


def price_repeat(forecast, shape, chip):
    """The caller has already confirmed the full identity of the preceding stream launch."""
    if forecast.latency_us is None or not chip.vector_tops['32']:
        return forecast
    diag = forecast.projection['timing_diagnostics']
    plan = diag['target_plan']
    if diag['context'] != 'reuse' or plan['route'] != 'post_token':
        return forecast

    tiles = channel_tiles(shape[2], chip)
    if tiles is None:
        return forecast
    pipe = token_pipeline(shape, chip, plan, 'repeat', _prices(), diag['shared_floor_us'])
    us = pipe['us']
    diagnostics = dict(model='post_token_immediate_repeat', context='reuse',
                       dispatch_condition='same_successful_builtin_post', target_plan=plan,
                       shared_floor_us=diag['shared_floor_us'], stages_us=pipe['stages_us'],
                       row_work_floor_cycles=pipe['row_work_floor_cycles'],
                       gap_cycles=pipe['gap_cycles'], prices='v0.25/post/prices.json',
                       channel_tiles=tiles,
                       schedule_granularity='one_descriptor_per_operand' if tiles['count'] == 1 else 'whole_row_aggregate')
    projection = {**forecast.projection, 'latency_us': us, 'stages_us': pipe['stages_us'],
                  'timing_diagnostics': diagnostics}
    return replace(forecast, latency_us=us, status='extrapolated',
                   reason='Experimental per-token schedule for an immediate identical Post',
                   projection=projection, rates=_rates(forecast.work, us, chip))
