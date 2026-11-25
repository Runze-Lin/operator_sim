"""Verify 950 SKU, golden, ELF identity and every physical retirement; retain compact receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from calibration.msopprof import CORE, sha
from calibration.mx_matmul.collect import MEMORY_CONFIG, core_windows
from calibration.mx_matmul.shards import trace_events

def receipt(path, opp):
    cost=json.loads((path/'cost.json').read_text())
    command=cost['command']
    sku=next(c.split('=')[1] for c in command if c.startswith('--soc-version='))
    log=(path/'run.log').read_text()
    if cost['exit_code']!=0 or MEMORY_CONFIG[sku] not in log or f'runtime_soc={sku}' not in log or 'SMOKE_PASS' not in log or 'Profiling data parse finished.' not in log:
        raise ValueError(f'Failed SKU or numerical qualification in {path}')
    checked,=re.findall(r'^correctness Y count=(\d+) bad=0$',log,re.M)
    kernel,=re.findall(r'Start profiling on kernel: (\S+)', log)
    base,=(path/'profile').glob('OPPROF_*')
    elf=sha(base/'dump/aicore_binary.o')
    filename = re.sub(r'_\d+(?:_mix_ai[cv])?$', '', kernel) + '.o'
    obj,=[x for x in opp.glob('*/*/'+filename) if sha(x)==elf]
    roster=CORE.findall(log)
    ticks,hashes=core_windows(base/'simulator',{n:float(t) for n,t in roster})
    assert len(ticks)==len(roster)
    lo=min(w['first_tick'] for w in ticks.values())
    hi=max(w['retire_tick'] for w in ticks.values())
    last=max(ticks,key=lambda n:ticks[n]['retire_tick'])
    events=list(trace_events(base/'simulator'/last/'trace.json'))
    names={e['pid']:e['args']['name'] for e in events if e.get('name')=='process_name'}
    def windows(pipe):
        return [(round(e['ts']*1650),round((e['ts']+e['dur'])*1650)) for e in events
                if e.get('ph')=='X' and names.get(e['pid'])==pipe and e['name'].startswith('MOV_SRC_TO_DST')]
    reads,writes=windows('MTE2'),windows('MTE3')
    op,shape=next((v,list(map(int,command[i+1:]))) for i,v in enumerate(command) if v in ('add','add_rms_norm','mul','cache_load_fp8','rope','swiglu','mhc_pre_sinkhorn','moe_gating_top_k_hash'))
    return dict(op=op,shape=shape,chip=sku,measured_us=(hi-lo)/1650,
                active_aiv=sum('.veccore' in core for core,_ in roster),
                active_aic=sum('.cubecore' in core for core,_ in roster),kernel=kernel,installed=str(obj.relative_to(opp)),elf_sha256=elf,
                probe_sha256=cost['probe_sha256'],trace_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
                checked=int(checked),ca_wall_s=cost['wall_seconds'],last_core=last,first_tick=lo,last_tick=hi,
                first_read_delay_cycles=min((a-lo for a,b in reads),default=None),
                final_store_to_retire_cycles=hi-max((b for a,b in writes),default=hi))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('directories',nargs='+',type=Path);ap.add_argument('--opp',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args()
    a.out.write_text(json.dumps([receipt(p,a.opp) for p in a.directories],indent=2)+'\n')
