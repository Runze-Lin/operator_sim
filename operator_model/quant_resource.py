"""Price the target's DynamicMxQuant tiles with a two-slot resource schedule."""
from functools import lru_cache
import json
from math import log2
from pathlib import Path

from .model import ceildiv
from .quant_repeat import NEG, compose, repeat
from .target_tiling import quant_plan


@lru_cache(maxsize=1)
def _costs():
    path = Path(__file__).resolve().parents[1] / 'results/v0.16/quant.json'
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

    shared = min(chip.hbm_bandwidth_gbs, d.get('l2_bandwidth_gbs') or float('inf'))/f
    read = local_rate('mte2_bandwidth_gbs', 'read_bytes_per_cycle')
    issue = local_rate('mte2_bandwidth_gbs', 'read_issue_bytes_per_cycle')
    write = local_rate('mte3_bandwidth_gbs', 'write_bytes_per_cycle')
    # Critical tile can be larger than mean. Aggregate bandwidth uses actual
    # tensor volume, amortized over critical tiles without padding the tails.
    equivalent_cores = shape[0]*shape[1]/(rows*cols)
    share = shared/equivalent_cores
    nominal = chip.vector_tops['16']
    peak = nominal*1000/f/chip.vector_cores if nominal else p['default_vector_ops_per_cycle']
    loop_work = p['vec_loop_cycles']*p['default_vector_ops_per_cycle']

    def service(h, w):
        groups = h*ceildiv(w,32)
        payload = 2*h*w
        loops = ceildiv(groups,8)+ceildiv(groups,128)
        read_service = payload/min(read, share)
        return dict(read_bytes=payload, write_bytes=h*w+groups, groups=groups, loops=loops,
                    read_latency=cycles('read_latency_cycles')+read_service,
                    read_ii=payload/min(issue,share), vector=cycles('vf_dispatch_cycles')
                    +cycles('vf_dependency_cycles')+loops*loop_work/peak,
                    write=cycles('write_latency_cycles')+(h*w+groups)/min(write,share))

    def matrix(h,w):
        s = service(h,w)
        ii, latency, vec, wr = (s[k] for k in ('read_ii','read_latency','vector','write'))
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
    write_bytes=shape[0]*(shape[1]+ceildiv(shape[1],32))
    transfer_floor=(read_bytes+write_bytes)/shared
    stages = dict(dispatch=(cycles('launch_cycles')+cycles('arrival_log2_cycles')*log2(active))/clock,
                  ub_pipeline=max(body,transfer_floor)/clock, drain=cycles('end_cycles')/clock)
    diagnostics=dict(model='target_resource_quant',target_plan=plan,stages_us=stages,
                     first_tile_cycles=first,chip_io_floor_cycles=transfer_floor,
                     assumed_offchip_read_bytes=read_bytes, assumed_offchip_write_bytes=write_bytes,
                     effective_vec_ops_per_loop=loop_work,vec_ops_per_cycle_per_core=peak,
                     read_bytes_per_cycle_per_core=read,write_bytes_per_cycle_per_core=write,
                     shared_bytes_per_cycle=shared, calibrated_on='950DT phase boundaries only')
    return sum(stages.values()),0,active,diagnostics
