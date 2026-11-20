"""Inspectable BF16 full-row RMSNorm issue graph for the installed RegBase 5000 kernel.

These experimental endpoints separate a resident prefix from an immediate repeat.
Calling this module does not alter the ordinary known-history RMS calibration.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import json
from math import ceil, isfinite
from pathlib import Path
from typing import Optional

from .chips import REFERENCE_NAME, load_chip
from .rms_dispatch import price_rms
from .rms_lane_schedule import _launch_class, parameters as lane_parameters
from .rms_resource import FIRST_PAYLOAD_RELEASE, READ_PER_KIB, WRITE_PER_KIB


@dataclass(frozen=True)
class RmsGeometry:
    """Default A5 instruction/sector geometry. Sizes are bytes or register counts."""
    vector_register_bytes: int = 256
    registers_per_group: int = 2
    statistic_sector_bytes: int = 128
    sector_issue_interval_cycles: Optional[float] = None
    drain_cycles: Optional[float] = None

    def __post_init__(self):
        sizes = (self.vector_register_bytes, self.registers_per_group, self.statistic_sector_bytes)
        if any(type(value) is not int or value <= 0 for value in sizes):
            raise ValueError('RMS geometry must use positive integers')
        for value in (self.sector_issue_interval_cycles, self.drain_cycles):
            if value is not None and (type(value) not in (float, int) or not isfinite(value) or value < 0):
                raise ValueError('RMS retirement intervals must be finite nonnegative cycle counts')
        if self.vector_register_bytes % 4 or self.statistic_sector_bytes % 4:
            raise ValueError('Registers and statistic sectors must hold whole FP32 values')

    def vector_work(self, width, rows):
        fp32_lanes = self.vector_register_bytes // 4
        groups = ceil(width / (self.registers_per_group * fp32_lanes))
        tree_depth = (ceil(width / fp32_lanes) - 1).bit_length()
        return rows, rows * groups, rows * tree_depth


@lru_cache(maxsize=1)
def _reference_prices():
    path = Path(__file__).resolve().parents[1] / 'results/v0.28/rms/prices.json'
    return json.loads(path.read_text(encoding='utf-8'))['prices']


def _cycles(prices, key, *work):
    base, *service = prices[key]
    return base + sum(coef * units for coef, units in zip(service, work))


def _per_tile_dag(shape, chip, previous, repeated, diag, prices, geometry, inspect):
    tokens, width = shape
    plan = diag['target_plan']
    active = plan['active_aiv']
    stride = plan['rows_per_core']
    tile_height = plan['full_tile_rows']
    if plan['iterations'] > 2:
        # The fitted issue graph has up to two payload blocks. The original
        # multi-round model supports steady-state overlap on larger inputs.
        latency, _, _, state = price_rms(shape, chip, {'resident_aiv': previous})
        return {'us': latency, 'route': 'steady multi-UB', 'stages': None,
                'stages_us': state['stages_us']}

    def cycles(key, *work):
        return _cycles(prices, key, *work)

    read = diag['read_cycles_per_kib']
    write = diag['write_cycles_per_kib']
    vec = diag['vector_price_cycles_per_reference_cycle']
    interval = geometry.sector_issue_interval_cycles
    drain = geometry.drain_cycles
    if interval is None:
        interval = lane_parameters()['sector_interval']
    if drain is None:
        drain = lane_parameters()['drain']
    per_core = ceil(chip.vector_cores / chip.cube_cores)
    if previous == 0:
        states = ['first'] * active
    elif repeated:
        states = ['repeat'] * active
    else:
        states = [_launch_class(b, active, previous, per_core) for b in range(active)]
    counts = Counter(states)
    ranks = Counter()
    lane_rows = [min(stride, tokens - b * stride) for b in range(active)]
    # One read of the entire cooperating set per UB round. Only two rounds run here.
    round_bytes = [2 * width * sum(max(0, min(tile_height, r - j * tile_height)) for r in lane_rows)
                   for j in range(plan['iterations'])]
    events = []
    sectors = defaultdict(list)
    sector_rows = geometry.statistic_sector_bytes // 4
    last_y = 0.
    earliest_x = float('inf')
    for block, (state, rows) in enumerate(zip(states, lane_rows)):
        count = counts[state]
        skew = 0.
        if count > 1:
            rank = (ranks[state] + .5) / count - .5
            skew = cycles('gamma_spread_' + state, active - 1) * rank
        ranks[state] += 1
        gamma = cycles('gamma_' + state, active - 1) + skew
        if state == 'first':
            reference_release = chip.details.get('rms_first_payload_cycles') or FIRST_PAYLOAD_RELEASE
            gamma += reference_release - FIRST_PAYLOAD_RELEASE
        gamma_kib = 2 * width / 1024
        gamma_end = gamma + max(gamma_kib * read,
                                 cycles('gamma_response_' + state, gamma_kib * read / READ_PER_KIB))
        x = gamma + cycles('x_issue')
        earliest_x = min(earliest_x, x)
        previous_norm_end = 0.
        for tile in range(ceil(rows / tile_height)):
            row0 = block * stride + tile * tile_height
            local = min(tile_height, rows - tile * tile_height)
            kib = local * gamma_kib
            if tile:
                key = 'second_x_' + state
                if key not in prices:
                    key = 'second_x_repeat' if previous else 'second_x_first'
                x += max(cycles(key, tile_height * gamma_kib * read / READ_PER_KIB), cycles('x_issue'))
            bus = 0. if repeated else round_bytes[tile] * chip.frequency_ghz / chip.hbm_bandwidth_gbs
            x_end = x + max(kib * read, cycles('x_response_' + state, kib * read / READ_PER_KIB, bus))
            reduce = max(x_end, gamma_end) + cycles('read_to_vf')
            if tile:
                reduce = max(reduce, previous_norm_end - cycles('vector_reissue'))
            work = [term * vec for term in geometry.vector_work(width, local)]
            statistic = reduce + cycles('reduce_to_rstd_vf', *work)
            norm = statistic + cycles('rstd_to_normalize_issue', local)
            statistic_end = statistic + cycles('rstd_vf_response', local)
            norm_end = norm + cycles('normalize_response', *work)
            previous_norm_end = norm_end
            write_stat = statistic_end + cycles('issue_rstd')
            write_y = norm_end + cycles('issue_y')
            y_end = write_y + max(kib * write, cycles('y_service_' + state, kib * write / WRITE_PER_KIB))
            last_y = max(last_y, y_end)
            low = row0 // sector_rows
            high = (row0 + local - 1) // sector_rows
            # Interior sectors with one sole owner are equivalent, independently
            # of row count. Only the two boundaries can have multiple writers.
            for sector in {low, high, min(low + 1, high)}:
                sectors[sector].append((write_stat, block, tile, state))
            if inspect:
                events.append(dict(block=block, tile=tile, rows=local, row0=row0, state=state,
                    gamma=[gamma,gamma_end], x=[x,x_end], reduce_issue=reduce,
                    rstd_vf=[statistic,statistic_end], normalize=[norm,norm_end],
                    rstd_issue=write_stat, y=[write_y,y_end]))

    stat_retire = defaultdict(float)
    last_stat = 0.
    for competing_writes in sectors.values():
        retired = 0.
        for issued, block, tile, state in sorted(competing_writes):
            retired = max(issued + cycles('rstd_solo_' + state), retired + interval)
            last_stat = max(last_stat, retired)
            if inspect:
                stat_retire[block,tile] = max(stat_retire[block,tile], retired)
    for event in events:
        event['rstd_retire'] = stat_retire[event['block'],event['tile']]

    link_afterwork = 0.
    floors = [diag['local_read_port_floor_cycles'], stride * (2 * width + 4) * write / 1024]
    if repeated:
        working_set = 4 * width * tokens + 2 * width + 4 * tokens
        cache = chip.details.get('l2_bytes') or 0.
        floors.append(max(0, working_set - cache) * chip.frequency_ghz / chip.hbm_bandwidth_gbs)
        l2_bw = chip.details.get('l2_bandwidth_gbs')
        if l2_bw:
            floors.append(working_set * chip.frequency_ghz / l2_bw)
    else:
        # The last compulsory read must still produce results. The exact last
        # response is unknown, so only work common to every first tile is added.
        shortest = min(tile_height, min(lane_rows))
        work = [term * vec for term in geometry.vector_work(width, shortest)]
        after = (cycles('read_to_vf') + cycles('reduce_to_rstd_vf', *work)
                 + cycles('rstd_to_normalize_issue', shortest) + cycles('normalize_response', *work)
                 + cycles('issue_y'))
        link_afterwork = earliest_x + round_bytes[0] * chip.frequency_ghz / chip.hbm_bandwidth_gbs + after
        floors.append(min(cycles('gamma_' + state, active - 1) for state in counts)
                      + diag['shared_io_floor_cycles'])
    bounds = dict(y=last_y + drain, rstd=last_stat + drain,
                  compulsory_x_with_remaining_work=link_afterwork + drain, resource_floor=max(floors))
    return dict(us=max(bounds.values()) / (1000 * chip.frequency_ghz),
                route='RMSNorm per-UB issue graph', bounds_cycles=bounds,
                stages=events if inspect else None)


def rmsnorm_pipeline(shape, *, chip=REFERENCE_NAME, dtype=None, resident_aiv=None, immediate_repeat=False,
                     geometry=RmsGeometry(), prices=None, diagnostics=False):
    """Return a conditional dispatch in µs or three scenarios for unknown history.

    BF16 input [tokens, channels] with BF16 gamma/y and FP32 rstd. An exact
    repeat must follow successful dispatch of the same complete RMSNorm
    contract on this stream. A resident lane count alone does not assert data
    or code reuse. Diagnostics report descriptor windows in target core cycles.
    The unknown-history point equalizes relative error at the two predicted
    extremes. This is a scenario calculation, not a measured confidence band.
    An optional mapping of per-stage slopes/intercepts replaces the A5 prices.
    """
    if dtype is not None:
        from .backend import _call_attrs
        _call_attrs('rmsnorm', None, dtype)
    target = load_chip(chip)
    if type(immediate_repeat) is not bool or type(diagnostics) is not bool:
        raise ValueError('Repeat and diagnostics flags must be booleans')
    if resident_aiv is not None and (type(resident_aiv) is not int or not 0 <= resident_aiv <= target.vector_cores):
        raise ValueError('resident_aiv must be an integer between zero and the chip AIV count')
    _, _, active, diag = price_rms(shape, target, {})
    if immediate_repeat and (resident_aiv is None or resident_aiv < active):
        raise ValueError('An immediate repeat requires a fully covered resident footprint')
    stage_prices = _reference_prices()
    if prices is not None:
        if not hasattr(prices, 'items'):
            raise ValueError('Stage prices must map a known stage name to its coefficients')
        stage_prices = dict(stage_prices)
        for key, values in prices.items():
            if (key not in stage_prices or not isinstance(values, (tuple, list))
                    or len(values) != len(stage_prices[key]) or any(type(v) not in (int, float)
                        or not isfinite(v) or v < 0 for v in values)):
                raise ValueError(f'Invalid RMS stage coefficients for {key}')
            stage_prices[key] = values
    def state(previous, repeated):
        return _per_tile_dag(shape, target, previous, repeated, diag, stage_prices, geometry, diagnostics)
    common = dict(chip=target.name, active_aiv=active, plan=diag['target_plan'],
                  calibration_family='Ascend 950 key 5000 BF16 full-row',
                  extrapolated=all(target != load_chip(name) for name in ('Ascend950DT_9582','Ascend950PR_957b')))
    if resident_aiv is not None:
        return {**common, **state(resident_aiv, immediate_repeat)}
    return {**common, **_unknown_point(active, state)}


def _unknown_point(active, scenario):
    scenarios = {'first': scenario(0,False), 'repeat': scenario(active,True)}
    if active > 1:
        scenarios['one_lane_prefix'] = scenario(1,False)
    lower = min(item['us'] for item in scenarios.values())
    upper = max(item['us'] for item in scenarios.values())
    return dict(us=2*lower*upper/(lower+upper), lower_us=lower, upper_us=upper, scenarios=scenarios)


def price_registered_rms(shape, chip, attrs):
    """Public resource price, preserving known-prefix and steady multi-UB calibration."""
    legacy = price_rms(shape, chip, attrs)
    _, aic, active, diag = legacy
    if attrs.get('resident_aiv') is not None or diag['target_plan']['iterations'] > 2:
        return legacy
    prices = _reference_prices()
    geometry = RmsGeometry()
    def scenario(previous, repeat):
        return _per_tile_dag(shape, chip, previous, repeat, diag, prices, geometry, False)
    prediction = _unknown_point(active, scenario)
    point = prediction['us']
    # This point summarizes distinct physical executions. Its only additive
    # component is the aggregate; real endpoint timelines are available separately.
    return point, aic, active, {**diag, 'model': 'rmsnorm_unknown_history_issue_graph',
        'history_scenarios': prediction['scenarios'], 'history_lower_us': prediction['lower_us'],
        'history_upper_us': prediction['upper_us'],
        'residency_assumption': 'unknown_stream_history',
        'stages_us': {'unknown_history_total': point},
        'stages_semantics': 'Scenario center only. Inspect the conditional resource graph for a physical timeline.'}
