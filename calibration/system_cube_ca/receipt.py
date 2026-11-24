"""Verify the complete 950 device window and inspect actual MMAD geometry."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import re
from statistics import median
from pathlib import Path

from calibration.msopprof import CORE, sha
from calibration.mx_matmul.collect import MEMORY_CONFIG, core_windows, trace_events


def read(root, sku, opp, role):
    log = (root/(sku+'.log')).read_text()
    assert ('SMOKE_PASS' in log and 'All task success.' in log and 'Profiling data parse finished.' in log
            and f'runtime_soc={sku}' in log and MEMORY_CONFIG[sku] in log)
    starts = re.findall(r'^case=(\w+) dtype=(\w+) M=(\d+) N=(\d+) K=(\d+)$', log, re.M)
    checked = re.findall(r'^checked=(\d+) bad=0$', log, re.M)
    kernels = re.findall(r'Start profiling on kernel: (\S+)', log)
    tables = re.split(r'^core_name\s+duration_time\(us\)\s+running_time\(us\)[ \t]*$',log,flags=re.M)[1:]
    assert len(starts)==len(kernels)==len(tables)==len(checked) and len(starts) in (1,4)
    base, = (root/sku/'profile').glob('OPPROF_*')
    markers = re.findall(r'\[(20\d\d-\d\d-\d\d \d\d:\d\d:\d\d)\]', log)
    receipts, seen = [], Counter()
    for (op,dt,m,n,k),name,tab,nel in zip(starts,kernels,tables,checked):
        assert int(nel)==int(m)*int(n)
        path = base if len(starts) == 1 and (base/'dump').is_dir() else base/name/str(seen[name])
        seen[name]+=1
        elf = sha(path/'dump/aicore_binary.o')
        fname = re.sub(r'_\d+(_mix_ai[cv])?$','',name)+'.o'
        obj, = [p for p in opp.glob('*/*/'+fname) if sha(p)==elf]
        rows = CORE.findall(tab)
        roster = {key:float(value) for key,value in rows}
        assert len(roster)==len(rows)
        cores, hashes = core_windows(path/'simulator',roster)
        lo = min(c['first_tick'] for c in cores.values())
        hi = max(c['retire_tick'] for c in cores.values())
        first, last, fetch, geom, active = [], [], [], Counter(), 0
        for trace in (path/'simulator').glob('*.cubecore*/trace.json'):
            events = [e for e in trace_events(trace) if e.get('ph')=='X']
            mm = [e for e in events if e.get('name')=='MMAD']
            fetch.extend(e['ts']*1650-lo for e in events if 'OUT_TO_L1' in e['name'] and 'ND2NZ' in e['name'])

            if not mm: continue
            active+=1
            first.append(min(e['ts'] for e in mm)*1650-lo)
            last.append(max(e['ts']+e['dur'] for e in mm)*1650-lo)
            for e in mm:
                d = json.loads(e['args']['detail'])
                geom[(d['m'],d['n'],d['k'],round(e['dur']*1650))]+=1
        reduce_aiv, reduce_instructions = 0, 0
        for trace in (path/'simulator').glob('*.veccore*/trace.json'):
            additions = sum(e.get('name') == 'RV_VADD' for e in trace_events(trace))
            reduce_aiv += additions > 0
            reduce_instructions += additions
        receipts.append(dict(api='aclnn'+op,sku=sku,dtype=dt,shape_MNK=list(map(int,[m,n,k])),
                gold_checked=int(nel),kernel=name,object=str(obj.relative_to(opp)),elf_sha256=elf,
                reference_role=role,ca_us=(hi-lo)/1650,launched=len(cores),
                productive_aic=active,aiv_with_vector_add=reduce_aiv,vector_add_instructions=reduce_instructions,
                first_payload_dma_cycles=round(min(fetch)) if fetch else None,
                first_mmad_spread_cycles=round(max(first)-min(first)),
                median_core_mmad_span_cycles=median(b-a for a,b in zip(first,last)),
                first_mmad_cycles=round(min(first)),last_mmad_cycles=round(max(last)),
                end_after_last_mmad_cycles=round(hi-lo-max(last)),
                mmads_m_n_k_dur_count=[[*key,cnt] for key,cnt in sorted(geom.items())],
                shards_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()))
    stamp='%Y-%m-%d %H:%M:%S'
    # Single-kernel logs may carry a single timestamp at first dispatch.
    # Use the run's newly created profiler directory and the last log write.
    profiler_start = datetime.strptime(base.name.split('_')[1], '%Y%m%d%H%M%S').timestamp()
    wall = max(0., (root/(sku+'.log')).stat().st_mtime-profiler_start)
    return dict(cases=receipts,profiling_wall_seconds=round(wall,2),log_sha256=sha(root/(sku+'.log')),
                profiler_log_span_seconds=(datetime.strptime(markers[-1],stamp)-datetime.strptime(markers[0],stamp)).total_seconds())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('opp',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--role',choices=('development','independent'),default='development'); a=p.parse_args()
    a.output.write_text(json.dumps({sku:read(a.root,sku,a.opp,a.role) for sku in MEMORY_CONFIG},indent=2)+'\n')
