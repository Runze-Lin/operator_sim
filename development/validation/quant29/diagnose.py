"""Forward-independent resource ablations, archived coverage, service throughput and CPU costs."""
from dataclasses import replace
from pathlib import Path
from statistics import median
from time import perf_counter_ns
import os
import re

from study import *

def run():
    reg=read(FROZEN);f=reg['predictions']['historical_b8'][DT]
    public_chip=load_chip(DT);core_sweep=[];costs={}
    for cores in (8,16,32,64,120):
        local_peak={k:(v*cores/public_chip.vector_cores if v is not None else None) for k,v in public_chip.vector_tops.items()}
        chip=replace(public_chip,name=f'same memory budget · {cores} AIV',vector_cores=cores,vector_tops=local_peak)
        shape=reg['shapes']['historical_b8'];base=estimate('quant',shape,chip=chip).latency_us
        q=scenario(shape,chip,base,mixed_bus_efficiency=f['mixed_bus_efficiency'],wire_to_configured_bandwidth=1.024)
        core_sweep.append(dict(available_aiv=cores,active_aiv=q['active_aiv'],base_us=base,
            critical_local_us=q['critical_local_us'],link_us=q['prefix_us']+q['mixed_us'],candidate_us=q['latency_us'],
            retained=q['dirty_retained_bytes'],iterations=q['plan']['iterations']))
    for sku in (DT,PR,FUT):
        profile=load_chip(sku);v=reg['predictions']['historical_b8'][sku];times=[]
        for i in range(128):
            t=perf_counter_ns()
            public=estimate('quant',shape,chip=profile)
            q=scenario(shape,profile,public.latency_us,mixed_bus_efficiency=v['mixed_bus_efficiency'],wire_to_configured_bandwidth=v['wire_to_configured_bandwidth'])
            times.append((perf_counter_ns()-t)/1e6)
            assert q['latency_us']==v['latency_us']
        costs[sku]=dict(samples=len(times),median_ms=median(times),p95_ms=sorted(times)[121],max_ms=max(times))
    # Only already finished real host logs can certify an archive. A query-only
    # shape in a workload manifest, plus an unfinished CA directory, count as neither tier.
    listed=[];unparsed=[];regs=read(ROOT/'results/v0.28/replay.json')['rows']
    known={(r['chip'],tuple(r['shape'])) for r in regs if r['op']=='quant'}
    for base in (ROOT/'runs',ROOT/'results'):
        for folder,dirs,files in os.walk(base):
            dirs[:]=[d for d in dirs if d not in ('log','log_ca','profile','simulator','dump','sources','record','ca_workspace') and not d.startswith(('PROF_','OPPROF_'))]
            p=Path(folder)/'run.log'
            if 'run.log' not in files or p.stat().st_size>32*1024**2:continue
            if ROOT/'runs/quant29' in p.parents or OLD/'raw' in p.parents:continue
            data=p.read_text(errors='replace')
            if not('DynamicMxQuant_bf16toe4m3' in data or 'case=dynamic_mx_quant' in data):continue
            shapes={(int(t),int(w)) for t,w in re.findall(r'case=dynamic_mx_quant rows=(\d+) cols=(\d+)',data)}
            if not shapes:
                unparsed.append(str(p.relative_to(ROOT)));continue
            for t,w in sorted(shapes):
                vol=t*(3*w+2*((w+63)//64))
                if vol<=112*2**20:continue
                listed.append(dict(log=str(p.relative_to(ROOT)),shape=[t,w],logical_bytes=vol,
                    registered_SKU=re.findall(r'^runtime_soc=(\S+)',data,re.M),numeric_success='SMOKE_PASS' in data,
                    official_final='All task success.' in data and 'Profiling data parse finished.' in data,
                    golden=re.findall(r'^correctness [^\r\n]+',data,re.M)))
    return dict(core_scaling=core_sweep,boundary='Vary only AIV count and chip-wide Vector peak proportionally; unchanged per-AIV work/clock, frequency and external link. Every row extrapolated.',
        timings=costs,timings_protocol='both public estimate and the candidate from scratch, same interpreter, warmed configuration; no result cache, three repeated 128-query batches.',
        audit=dict(archived_full_instr_shapes=len(known),archived_larger_host_runs=listed,unparsed_log_paths=unparsed,
                   scope='Scan results and runs host logs outside ignored simulator trees. Prior v0.28 scored archive is authoritative for complete instruction eligibility. A log without a printed shape is not evidence of novelty or a certified timing.'))

if __name__=='__main__':store(OUT/'diagnostics.json',run())
