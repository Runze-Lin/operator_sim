"""Absolute per-stage resource schedule for dense SBH BF16 QK/softmax/PV."""
import json
from functools import lru_cache
from math import ceil
from pathlib import Path


@lru_cache(maxsize=1)
def _costs():
    path = Path(__file__).resolve().parents[1]/'results/v0.16/fa.json'
    return json.loads(path.read_text())


def _ceildiv(a, b):
    return (a+b-1)//b


def _pad16(x):
    return _ceildiv(x,16)*16


def resource_fa(shape, chip):
    """Return critical-path microseconds, active AIC/AIV and priced stage demands."""
    if len(shape) == 3:
        shape = (*shape,1,1)
    if len(shape)!=5 or any(type(x) is not int or x<=0 for x in shape):
        raise ValueError('FA expects positive Q, KV, D, B, H')
    q, kv, d, batch, heads = shape
    if d not in (64,128):
        raise ValueError('Resource FA timing covers dense BF16 D=64 or 128')
    c = _costs()
    ghz = chip.frequency_ghz
    dt_ops = c['dt_vector_peak_ops_per_cycle']
    details = chip.details
    # Rerun the basic host policy with the target core budget. A recompiled
    # one-vector-per-cube implementation is an architectural scenario.
    sparse_grid = batch*heads*_ceildiv(q,128) < chip.cube_cores
    tq = 64 if sparse_grid else 128
    tk = 256 if (sparse_grid and kv>1024) or (d==64 and q%128==0 and kv%256==0) else 128
    outer = _ceildiv(q,tq)
    jobs = outer*batch*heads
    active = min(jobs,chip.cube_cores,chip.vector_cores)
    aiv = min(2*active,chip.vector_cores)
    waves = _ceildiv(jobs,active)
    params = c['routes']['wide' if tk==256 else 'short']
    local = details.get('fa_mte_bytes_per_cycle') or c['mte_bytes_per_cycle']
    issue = details.get('fa_issue_cycles') or c['issue_cycles_256']
    read_lat = details.get('fa_read_latency_cycles') or params['read_intercept']
    mte2 = details.get('mte2_bandwidth_gbs')
    mte3 = details.get('mte3_bandwidth_gbs')
    mte1 = (details.get('mte1_bandwidth_gbs') or c['mte1_bytes_per_cycle']*ghz)/ghz
    cube_peak = (chip.cube_tops['16'] or 0)*1000 / (chip.cube_cores*ghz)
    vec_peak = (chip.vector_tops['32'] or 0)*1000 / (chip.vector_cores*ghz)
    cube_peak = cube_peak or c['dt_cube_peak_ops_per_cycle']
    vec_rate = c['vector_ops_per_cycle'] * (vec_peak or dt_ops) / dt_ops
    read_local = min(local, mte2/ghz) if mte2 else local
    write_local = mte3/ghz if mte3 else local
    shared = min(chip.hbm_bandwidth_gbs, details.get('l2_bandwidth_gbs') or float('inf'))/ghz
    loops = _ceildiv(kv,tk)
    tail_k = (kv-1)%tk+1
    tail_q = (q-1)%tq+1
    stages = dict(launch=params['startup'],first_tile=0.,steady=0.,drain=0.)
    wave_types = []

    def wave(m, occupied, count):
        if not count:
            return
        lanes = min(2,chip.vector_cores/occupied)
        vr = ceil(m/lanes)
        read_rate = min(read_local,shared/occupied)
        write_rate = min(write_local,shared/occupied)

        def tile(n):
            fragments = _ceildiv(n,128)
            padded_ops = 2*_pad16(m)*_pad16(n)*_pad16(d)
            qk = params['qk_overhead']*fragments + padded_ops/cube_peak
            pv = params['pv_overhead']*fragments + padded_ops/cube_peak
            k_bytes = 2*n*d
            l0_bytes = 2*(2*_pad16(n)*d+_pad16(m)*(d+_pad16(n)))
            fix_bytes = 4*_pad16(m)*(_pad16(n)+d)
            demands = dict(cube=qk+pv, vector=(5*vr*n+3*vr*d)/vec_rate,
                           k_v_read=2*k_bytes/read_rate, mte1=l0_bytes/mte1,
                           fix=fix_bytes/c['fix_bytes_per_cycle'],
                           issue_control=issue*fragments/2)
            return dict(qk=qk,pv=pv,vec1=5*vr*n/vec_rate,vec2=3*vr*d/vec_rate,
                        k_bytes=k_bytes,v_bytes=k_bytes,qk_useful_ops=2*m*n*d,
                        pv_useful_ops=2*m*n*d,qk_physical_ops=padded_ops,pv_physical_ops=padded_ops,
                        vector_equiv_ops=5*m*n+3*m*d, l1_to_l0_bytes=l0_bytes,
                        fix_payload_bytes=fix_bytes, demands_cycles=demands, ii_cycles=max(demands.values()))
        first = tile(min(kv,tk))
        last = first if loops==1 or tail_k==tk else tile(tail_k)
        q_bytes = 2*m*d
        q_ready = read_lat+q_bytes/read_rate
        k_ready = params['k_lag']+read_lat+first['k_bytes']/read_rate
        v_ready = params['k_lag']+params['v_lag']+read_lat+first['v_bytes']/read_rate
        qk_start = max(q_ready,k_ready,(q_bytes+first['k_bytes'])/read_rate)+params['l0_ready']
        pv_start = max(v_ready,qk_start+first['qk']+first['vec1'])+params['handoff_control']
        fill = pv_start+first['pv']
        steady = max(loops-2,0)*first['ii_cycles']+(last['ii_cycles'] if loops>1 else 0)
        output_bytes = 2*m*d+64*m  # BF16 output plus two FP32 [m,8] statistics
        drain = last['vec2']+params['drain_control']+output_bytes/write_rate
        stages['first_tile'] += count*fill
        stages['steady'] += count*steady
        stages['drain'] += count*drain
        wave_types.append(dict(waves=count,occupied=occupied,q_rows=m,aiv_per_aic=lanes,
                               q_read_bytes=q_bytes,output_write_bytes=output_bytes,first=first,last=last,
                               k_ready_cycles=k_ready,v_ready_cycles=v_ready,qk_start_cycles=qk_start,
                               pv_start_cycles=pv_start,read_bytes_per_cycle=read_rate,
                               write_bytes_per_cycle=write_rate))

    # Flattened [batch, head, q tile] tasks. A contiguous wave of >=2 jobs
    # always contains a full Q tile when each head has full and tail tiles.
    if active==1:
        wave(tq,1,(outer-1)*batch*heads)
        wave(tail_q,1,batch*heads)
    else:
        last_jobs=(jobs-1)%active+1
        wave(min(tq,q),active,waves-1)
        wave(tail_q if last_jobs==1 else min(tq,q),last_jobs,1)
    us = {k:v/(ghz*1000) for k,v in stages.items()}
    plan = dict(active_aic=active,active_aiv=aiv,tile=[tq,tk],waves=waves,iterations=loops,
                jobs=jobs,q_tiles_per_head=outer)
    info = dict(model='target_resource_fa',target_plan=plan,stages_us=us,wave_types=wave_types,
                cube_ops_per_cycle=cube_peak,vector_equiv_ops_per_cycle=vec_rate,
                memory_scenario='cold K/V per Q tile; no assumed cross-core cache hits',
                mmads_total=2*_ceildiv(kv,128)*jobs,
                k_read_bytes_scenario=2*kv*d*jobs,v_read_bytes_scenario=2*kv*d*jobs,
                q_read_bytes=2*q*d*batch*heads,output_write_bytes=(2*d+64)*q*batch*heads,
                calibrated_on='950DT only; dense BF16 SBH D=64/128')
    return sum(us.values()),active,aiv,info
