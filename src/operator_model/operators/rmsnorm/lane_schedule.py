"""Price a known RMSNorm AIV prefix for a single full-row UB tile."""
from operator_model.resources import load_parameters
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

from operator_model.operators.rmsnorm import writeback as rms_writeback
from operator_model.operators.rmsnorm.resources import FIRST_PAYLOAD_RELEASE, FIRST_SCALAR


@lru_cache(maxsize=1)
def parameters():
    """Landmark medians in 950 core cycles; no shape- or SKU-specific total times."""
    return load_parameters('rms_lanes')


def _launch_class(block, active, previous, lanes_per_core):
    # Two balanced dispatch partitions are observed on 950. A target with
    # different AIV:AIC ratio changes how many lanes share one physical core.
    first_partition_size = (active + 1) // 2
    if block < first_partition_size:
        local = block
        warm = (previous + 1) // 2
    else:
        local = block - first_partition_size
        warm = previous // 2
    if local < warm:
        return 'resident'
    if warm == 0:
        return 'partition'
    if local // lanes_per_core >= (warm + lanes_per_core - 1) // lanes_per_core:
        return 'core'
    return 'sibling'


@dataclass(frozen=True)
class _Lane:
    block: int
    state: str
    arrival: float
    frontend: float
    useful: float
    input_pipe: str
    normalize: float
    y_response: float

    @property
    def issue(self):
        return self.arrival + self.frontend + self.useful

    @property
    def y_delay(self):
        return self.normalize + self.y_response

    def issue_stages(self, offset):
        # Offset is the modeled order statistic of this lane's issue; it
        # locates the lane within the calibrated frontend response spread.
        return {'arrival': self.arrival, 'frontend_to_rstd_issue': self.frontend + offset,
                self.input_pipe: self.useful}


def schedule(shape, chip, previous, diag):
    """Compute the last rstd/y completion and its additive critical-path chain.

    Caller has already proved one UB tile/core and a positive known high-water
    mark. An rstd request can span many sectors; interior sectors touched by a
    single lane have equal timestamps. Only one such sector per lane is needed.
    """
    tokens, width = shape
    plan = diag['target_plan']
    active = plan['active_aiv']
    stride = plan['rows_per_core']
    lanes_per_core = (chip.vector_cores + chip.cube_cores - 1) // chip.cube_cores
    p = parameters()
    vector = rms_writeback.parameters()
    vector_price = diag['vector_price_cycles_per_reference_cycle']
    read_price = diag['read_cycles_per_kib']
    y_fixed, y_per_kib = p['y_cycles']
    if chip.details.get('mte3_bandwidth_gbs'):
        y_per_kib = diag['write_cycles_per_kib']
    sectors = defaultdict(lambda: defaultdict(list))
    state_counts = defaultdict(int)
    for block in range(active):
        rows = min(stride, tokens - block * stride)
        state = _launch_class(block, active, previous, lanes_per_core)
        state_counts[state] += 1
        contention = (active-1) * p['resident_issue_cycles_per_active'] if state == 'resident' else 0.
        groups = rows * ((width+127)//128)
        read = 2 * width * (1+rows) * read_price / 1024
        reduction = vector['reduce'][1] * groups * vector_price
        lane = _Lane(block, state, p['arrival'][state], p['lead'][state]+contention,
                     max(read, reduction), 'MTE2' if read >= reduction else 'SIMD_reduce',
                     vector['normalize'][0]+vector['normalize'][1]*vector_price*groups,
                     y_fixed + y_per_kib*2*width*rows/1024)
        # FP32 rstd output. Each sector contains 32 row results.
        start = block*stride // 32
        last = (block*stride + rows-1) // 32
        sectors[start][state].append(lane)
        if last != start:
            sectors[last][state].append(lane)
        if last > start+1:
            sectors[start+1][state].append(lane)

    # For a congested sector, keep the request that starts the winning
    # retirement chain; each successor adds one retirement interval.
    peak = 0.
    winner = None
    for sector, by_state in sectors.items():
        issues = []
        for state, lanes in by_state.items():
            lanes.sort(key=lambda lane: (lane.issue, lane.y_delay))
            n = len(lanes)
            for rank, lane in enumerate(lanes, 1):
                offset = p['spread'].get(state, 0.) * (rank/(n+1)-.5)
                ready = lane.issue + offset
                issues.append((ready, lane.block, lane, offset))
                y_done = ready + lane.y_delay
                if y_done > peak:
                    peak = y_done
                    winner = ('y', sector, lane, offset, 0)
        retired = 0.
        root = None
        for ready, _, lane, offset in sorted(issues, key=lambda r: (r[0], r[1])):
            if ready + p['solo'] >= retired + p['sector_interval']:
                retired = ready + p['solo']
                root = (lane, offset)
                successors = 0
            else:
                retired += p['sector_interval']
                successors += 1
        if retired > peak:
            peak = retired
            winner = ('rstd', sector, *root, successors)

    resource, sector, lane, offset, successors = winner
    stages = lane.issue_stages(offset)
    if resource == 'y':
        stages.update(SIMD_normalize=lane.normalize, MTE3_y=lane.y_response)
    else:
        stages.update(MTE3_rstd=p['solo'], sector_retirement=successors*p['sector_interval'])
    stages['drain'] = p['drain']
    cycles_per_us = 1000 * chip.frequency_ghz
    baseline = diag['conditional_stages_us']['reuse' if previous >= active else 'expand']
    release = baseline['SCALAR'] * FIRST_PAYLOAD_RELEASE / FIRST_SCALAR * cycles_per_us
    shared = diag['shared_io_floor_cycles']
    critical = resource
    window = peak + p['drain']
    if release + shared > window:
        critical = 'shared_io'
        stages = {'payload_release': release, 'shared_io': shared}
        window = release + shared
    # On a future target with a very slow port, per-core transfer volume sets
    # an independent minimum. Both ports can run while other stages execute.
    port_floors = {'MTE2': diag['local_read_port_floor_cycles'],
                   'MTE3': stride*(2*width+4)*diag['write_cycles_per_kib']/1024}
    port = max(port_floors, key=port_floors.get)
    if port_floors[port] > window:
        critical = port
        stages = {port: port_floors[port]}
    timing = {key: val/cycles_per_us for key, val in stages.items()}
    detail = dict(critical_path=critical, lane_path=resource, lane_block=lane.block,
                  lane_class=lane.state, critical_sector=sector, retirement_successors=successors,
                  scheduled_lane_finish_cycles=peak+p['drain'], issue_order_adjustment_cycles=offset,
                  shared_release_cycles=release, port_floor_cycles=port_floors,
                  lane_class_counts=dict(state_counts), represented_sectors=len(sectors))
    return timing, detail
