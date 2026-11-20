"""Optional RMSNorm history scenarios for a chip without per-shape CA observations."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

from operator_model.operators.rmsnorm import lane_schedule as rms_lane_schedule
from operator_model.operators.rmsnorm import writeback as rms_writeback
from operator_model.backend import _call_attrs
from operator_model.chips import REFERENCE_NAME, load_chip
from operator_model.operators.rmsnorm.dispatch import price_rms
from operator_model.operators.rmsnorm.resources import FIRST_PAYLOAD_RELEASE, FIRST_SCALAR


@dataclass(frozen=True)
class RmsHistoryBand:
    """Range of conditional model predictions in µs. CA can lie outside it.

    ``by_resident_aiv_us[0]`` describes first launch; higher indices describe
    already-resident AIV counts. H >= active_aiv has the final entry's time.
    """
    chip_name: str
    active_aiv: int
    by_resident_aiv_us: tuple[float, ...]

    @property
    def lower_us(self) -> float:
        return min(self.by_resident_aiv_us)

    @property
    def upper_us(self) -> float:
        return max(self.by_resident_aiv_us)


def _sector_finish(counts, prices, params):
    """Retire one set of competing rstd stores, with independent y completion."""
    requests = defaultdict(list)
    for (state, rows), count in counts:
        requests[state].extend([prices[rows][state]] * count)
    issues = []
    last_y = 0.0
    for state, items in requests.items():
        spread = params['spread'].get(state, 0.0)
        for rank, (base, y_delay) in enumerate(sorted(items), 1):
            issue = base + spread * (rank / (len(items) + 1) - 0.5)
            issues.append(issue)
            last_y = max(last_y, issue + y_delay)
    last_rstd = 0.0
    for issue in sorted(issues):
        last_rstd = max(issue + params['solo'], last_rstd + params['sector_interval'])
    return max(last_y, last_rstd)


def _lane_prices(rows, width, active, chip, diag, params):
    """Invariant ready/response cycles for identical logical row counts."""
    vector = rms_writeback.parameters()
    vec_price = diag['vector_price_cycles_per_reference_cycle']
    groups = rows * ((width + 127) // 128)
    read = 2 * width * (1 + rows) * diag['read_cycles_per_kib'] / 1024
    reduction = vector['reduce'][1] * groups * vec_price
    normalize = vector['normalize'][0] + vector['normalize'][1] * groups * vec_price
    y_fixed, y_per_kib = params['y_cycles']
    if chip.details.get('mte3_bandwidth_gbs'):
        y_per_kib = diag['write_cycles_per_kib']
    y_response = y_fixed + y_per_kib * 2 * width * rows / 1024
    prices = {}
    for state in ('resident', 'partition', 'core', 'sibling'):
        contention = (active - 1) * params['resident_issue_cycles_per_active'] if state == 'resident' else 0.0
        frontend = params['lead'][state] + contention
        issue = params['arrival'][state] + frontend + max(read, reduction)
        prices[state] = (issue, normalize + y_response)
    return prices


def estimate_rms_history_band(shape: Sequence[int], *, chip=REFERENCE_NAME, dtype=None) -> RmsHistoryBand:
    """Price every distinct predecessor-AIV prefix for one full-row BF16 RMSNorm.

    Input shape is [tokens, channels]. Chip accepts the same built-in SKU,
    ChipConfig, mapping or JSON file as ``estimate``. Width, dtype and UB
    constraints are the same as the RMSNorm resource backend. Invalid inputs
    raise ValueError. The minimum and maximum describe simulated states of
    this implementation; they are not confidence bounds for measured latency.
    Ordinary ``estimate`` calls do not enumerate histories.
    """
    _call_attrs('rmsnorm', None, dtype)
    target = load_chip(chip)
    if not isinstance(shape, Sequence):
        raise ValueError('Expected a [tokens, channels] sequence')
    _, _, active, diag = price_rms(shape, target, {})
    conditional = diag['conditional_us']
    first = conditional['first']
    plan = diag['target_plan']
    if plan['iterations'] > 1:
        # The current multi-UB route resolves only these three distinct states.
        values = (first,) + (conditional['expand'],) * (active - 1) + (conditional['reuse'],)
        return RmsHistoryBand(target.name, active, values)

    tokens, width = shape
    stride = plan['rows_per_core']
    clock = 1000 * target.frequency_ghz
    lanes_per_core = (target.vector_cores + target.cube_cores - 1) // target.cube_cores
    port_floor = max(diag['local_read_port_floor_cycles'], stride * (2 * width + 4) * diag['write_cycles_per_kib'] / 1024)
    shared = {}
    for state in ('expand', 'reuse'):
        release = diag['conditional_stages_us'][state]['SCALAR'] * FIRST_PAYLOAD_RELEASE / FIRST_SCALAR * clock
        shared[state] = release + diag['shared_io_floor_cycles']

    params = rms_lane_schedule.parameters()
    prices = {}
    sectors = defaultdict(list)
    for block in range(active):
        rows = min(stride, tokens - block * stride)
        if rows not in prices:
            prices[rows] = _lane_prices(rows, width, active, target, diag, params)
        first_sector = block * stride // 32
        last_sector = (block * stride + rows - 1) // 32
        # A 128 B sector spans 32 FP32 row results. All strictly interior
        # sectors are owned by only this descriptor and share the same time.
        for sector in {first_sector, last_sector, min(first_sector + 1, last_sector)}:
            sectors[sector].append((block, rows))

    values = [first]
    sector_finishes = {}
    for previous in range(1, active + 1):
        states = [rms_lane_schedule._launch_class(block, active, previous, lanes_per_core) for block in range(active)]
        finish = 0.0
        for blocks in sectors.values():
            counts = Counter((states[block], rows) for block, rows in blocks)
            # The same multiset has identical issue order statistics and store
            # retirements regardless of sector address or predecessor count.
            key = tuple(sorted(counts.items()))
            if key not in sector_finishes:
                sector_finishes[key] = _sector_finish(key, prices, params)
            finish = max(finish, sector_finishes[key])
        drain = finish + params['drain']
        shared_floor = shared['reuse' if previous == active else 'expand']
        values.append(max(drain, port_floor, shared_floor) / clock)
    return RmsHistoryBand(target.name, active, tuple(values))
