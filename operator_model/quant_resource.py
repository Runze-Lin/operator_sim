"""Price the target's DynamicMxQuant tiles with a two-slot resource schedule."""
from functools import lru_cache
import json
from math import log2
from pathlib import Path

from .model import ceildiv
from .quant_fragments import fragment_costs, fragment_wave
from .quant_scale import peak_scale_writers
from .quant_repeat import NEG, compose, repeat
from .target_tiling import quant_plan


@lru_cache(maxsize=1)
def _costs():
    path = Path(__file__).resolve().parents[1] / 'results/v0.17/quant.json'
    return json.loads(path.read_text())['parameters']


def resource_quant(shape, chip):
    p = _costs()
    d = chip.details
    f = chip.frequency_ghz
    clock = 1000 * f
    plan = quant_plan(shape, chip.vector_cores, d.get('ub_bytes') or 253952)
    rows, cols = plan['rows_per_core'], plan['cols_per_core']
    tr, tc = plan['tile']
    active = plan['active_aiv']
    def cycles(name):
        return d.get('quant_'+name) or p[name]
    def local_rate(name, default):
        return d[name]/f if d.get(name) else p[default]

    hbm = chip.hbm_bandwidth_gbs / f
    l2 = (d.get('l2_bandwidth_gbs') or float('inf')) / f
    shared = min(hbm, l2)
    read = local_rate('mte2_bandwidth_gbs', 'read_bytes_per_cycle')
    issue = local_rate('mte2_bandwidth_gbs', 'read_issue_bytes_per_cycle')
    write = local_rate('mte3_bandwidth_gbs', 'write_bytes_per_cycle')
    # Critical tile can be larger than mean. Aggregate bandwidth uses actual
    # tensor volume, amortized over critical tiles without padding the tails.
    equivalent_cores = shape[0]*shape[1]/(rows*cols)
    share = shared/equivalent_cores
    writers = peak_scale_writers(*shape, rows, cols)
    grain_ii = d.get('quant_fragment_grain_ii_cycles') or fragment_costs()['grain_ii_cycles']
    scale_queue = grain_ii * (writers-1)
    arrival_span = cycles('arrival_log2_cycles') * log2(active)
    frontend = (d.get('quant_vf_dispatch_cycles') or d.get('quant_fragment_frontend_cycles')
                or fragment_costs()['frontend_cycles'])
    vf_overhead = d.get('quant_fragment_vector_cycles') or fragment_costs()['vector_cycles']
    nominal = chip.vector_tops['16']
    peak = nominal*1000/f/chip.vector_cores if nominal else p['default_vector_ops_per_cycle']
    loop_work = d.get('quant_loop_work_ops') or p['vec_loop_cycles']*p['default_vector_ops_per_cycle']

    def service(h, w):
        groups = h*ceildiv(w,32)
        scale_bytes = h*2*ceildiv(w,64)
        payload = 2*h*w
        loops = ceildiv(groups,8)+ceildiv(groups,128)
        # A read may be outstanding while bandwidth is used by the other
        # cores. First-response latency belongs on the local path, not after
        # an interval in which the entire wave has transferred its payload.
        local_read = cycles('read_latency_cycles')+payload/read
        wave_tail = max(0., payload/share - arrival_span)
        read_window = max(local_read,wave_tail)
        ready_to_compute = max(frontend, read_window+cycles('vf_dependency_cycles'))
        compute = vf_overhead+loops*loop_work/peak
        data_bytes = h*w
        data_write = cycles('write_latency_cycles')+data_bytes/write
        # Data and scale have different descriptors. Small interleaved scale
        # fragments can share a granule across AIVs; contiguous full producers
        # have no such charge. scale_tail is exposed retirement after data.
        scale_write = max(p['scale_base_cycles']+scale_bytes/write,
                          data_write+p['scale_tail_cycles'])+scale_queue
        write_window = max(data_write,p['scale_issue_gap_cycles']+scale_write,
                           (data_bytes+scale_bytes)/write)
        return dict(read_bytes=payload, write_bytes=h*w+scale_bytes, scale_bytes=scale_bytes, groups=groups, loops=loops,
                    read_latency=read_window, local_read_cycles=local_read,
                    shared_read_floor_cycles=wave_tail, read_ii=payload/issue,
                    simd_ready=ready_to_compute, dependency_cycles=cycles('vf_dependency_cycles'), compute_cycles=compute,
                    vector=ready_to_compute-read_window+compute,
                    write=write_window, data_write_cycles=data_write, scale_write_cycles=scale_write,
                    scale_issue_gap_cycles=p['scale_issue_gap_cycles'])

    def matrix(h,w):
        s = service(h,w)
        ii, latency, vec, wr = (s[k] for k in ('read_ii','simd_ready','compute_cycles','write'))
        # State: free read issue port, last vector/write end, preceding vector/write end.
        return ((ii,NEG,NEG,ii,NEG),
                (latency+vec,vec,NEG,latency+vec,vec),
                (latency+vec+wr,vec+wr,wr,latency+vec+wr,vec+wr),
                (NEG,0.,NEG,NEG,NEG),(NEG,NEG,0.,NEG,NEG))
    def row(h):
        full, tail = divmod(cols,tc)
        result = repeat(matrix(h,tc),full)
        return compose(matrix(h,tail),result) if tail else result

    first = service(min(tr,rows),min(tc,cols))
    if plan['iterations']==1:
        body=first['read_latency']+first['vector']+first['write']
    else:
        full, tail = divmod(rows,tr)
        sequence = repeat(row(tr),full)
        if tail:
            sequence = compose(row(tail),sequence)
        body = sequence[2][0]
    read_bytes=2*shape[0]*shape[1]
    write_bytes=shape[0]*(shape[1]+2*ceildiv(shape[1],64))
    # At retirement an output may still be dirty in L2. Reserve room for
    # each AIV's two live input tiles before estimating the writable capacity.
    retained_fraction = d.get('quant_output_retained_fraction')
    if retained_fraction is None:
        retained_fraction = 1.
    live_inputs = min(read_bytes, active*min(2,plan['iterations'])*first['read_bytes'])
    available_l2 = max(0, (d.get('l2_bytes') or 0)-live_inputs)
    retained = retained_fraction * min(write_bytes, available_l2)
    visible_writes = write_bytes-retained
    transfer_floor = max((read_bytes+visible_writes)/hbm, (read_bytes+write_bytes)/l2)
    first_launch = cycles('launch_cycles')
    if plan['iterations'] > 1:
        first_launch = d.get('quant_prefetch_launch_cycles') or first_launch
    launch = first_launch+arrival_span
    fragment=fragment_wave(shape,plan,service,chip,loop_work,peak,p['scale_base_cycles'],p['scale_tail_cycles'],write)
    fragment_details=None
    if fragment:
        launch,body,first,fragment_details=fragment
        writers=fragment_details['peak_writers']
        scale_queue=fragment_details['critical']['extra_queue_cycles']
    # The shared link can make progress from the first issue, before the
    # modeled last core reaches its own issue point.
    # Once the final cold input arrives, at least its VF and the two
    # write descriptors must finish. Only use the smallest possible final
    # tile (including partial trailing core blocks) for this lower bound.
    last_rows = (shape[0]-1) % rows+1
    last_cols = (shape[1]-1) % cols+1
    tail = service(min((last_rows-1)%tr+1,(rows-1)%tr+1),
                   min((last_cols-1)%tc+1,(cols-1)%tc+1))
    # Upper-bounding scale contention on the final read would combine two
    # different cores; its local, contention-free write suffices here.
    tail_write = max(tail['data_write_cycles'],
                     tail['scale_issue_gap_cycles']+tail['scale_write_cycles']-scale_queue,
                     tail['write_bytes']/write)
    response_tail = tail['compute_cycles']+tail_write+cycles('vf_dependency_cycles')
    read_floor = read_bytes/shared+response_tail
    chip_floor_after_launch = max(0., max(read_floor,transfer_floor)-(0. if fragment else arrival_span))
    stages = dict(dispatch=launch/clock,
                  ub_pipeline=max(body,chip_floor_after_launch)/clock, drain=cycles('end_cycles')/clock)
    diagnostics=dict(model='target_resource_quant',target_plan=plan,stages_us=stages,
                     first_tile_cycles=first,chip_io_floor_cycles=transfer_floor,
                     assumed_offchip_read_bytes=read_bytes, assumed_offchip_write_bytes=visible_writes,
                     logical_write_bytes=write_bytes, output_retained_bytes=retained,
                     chip_floor_after_launch_cycles=chip_floor_after_launch, read_plus_tail_floor_cycles=read_floor, first_launch_cycles=first_launch,
                     shared_read_issue_spread_cycles=arrival_span,
                     effective_vec_ops_per_loop=loop_work,vec_ops_per_cycle_per_core=peak,
                     read_bytes_per_cycle_per_core=read,write_bytes_per_cycle_per_core=write,
                     shared_bytes_per_cycle=shared,scale_granule_bytes=32,scale_peak_writers=writers,
                     scale_queue_cycles=scale_queue,fragment_wave=fragment_details,calibrated_on='arch35 work and scale service from 950DT; target launch and vector phases from chip details')
    return sum(stages.values()),0,active,diagnostics
