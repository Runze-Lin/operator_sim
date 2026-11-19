"""Mechanism diagnosis only: verify archived DT raw traces and the v0.16 PR captures."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model.chips import load_chip
from operator_model.matmul_resource import resource_matmul
from operator_model.matmul_waves import dispatch
from operator_model.matmul_v15 import matmul_plan


def read(path):
    if path.suffix=='.db':
        db=sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)
        try:
            for core,typ,op,start,end,param in db.execute('SELECT ExecInstrCoreId,CoreTypeId,ExecInstrName,ExecInstrTickStart,ExecInstrTickEnd,ExtendParams FROM ExecutedInstructions'):
                yield (core,typ),op,start/1650,end/1650,json.loads(param or '{}')
        finally:db.close()
    else:
        for e in json.loads(path.read_bytes())['traceEvents']:
            if e['ph']=='X':
                yield e['pid'],e['name'],e['ts'],e['ts']+e['dur'],json.loads(e['args']['detail'])

def inspect(sh,chip,path,digest,ca,old_us):
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    es=list(read(path));origin=min(e[2] for e in es);end=max(e[3] for e in es)
    assert abs((end-origin)-ca)<.001
    hist=Counter(); intervals=defaultdict(list);mmads=[];fixes=[]
    for core,name,start,finish,args in es:
        if name=='MMAD':
            hist[args['m'],args['n'],args['k']]+=1;mmads.append((start,finish,args['m'],args['n']))
        if name=='FIX_L0C_TO_DST':
            intervals[str(core)].append((start,finish));fixes.append((start,finish))
    full,tail,_=dispatch(sh,matmul_plan(sh,chip));estimate,_,_,info=resource_matmul(sh,chip)
    frag=info['mmad_k'];k=sh[2];p=Counter()
    for (m,n),v in (full+tail).items():
        if k//frag:p[m,n,frag]+=v*(k//frag)
        if k%frag:p[m,n,k%frag]+=v
    # Do not interpret FIX duration as transfer service: it spans prior MMAD.
    concurrent=sum(any(b[0]<a[1] for a,b in zip(sorted(it),sorted(it)[1:])) for it in intervals.values())
    hostwnd=(max(e for s,e in fixes)-min(s for s,e in fixes))
    useful=2*sh[0]*sh[1]
    return dict(shape=sh,trace=str(path.relative_to(ROOT)),sha256=digest,source=chip.name,
                ca_us=ca,v16_frozen_us=old_us,v17_us=estimate,dispatch=info['target_plan'],
                actual_mmad_histogram={'x'.join(map(str,box)):v for box,v in sorted(hist.items())},
                shape_histogram_matches=p==hist,cores_with_overlapping_FIXP=concurrent,
                first_mmad_start_us=min(x[0] for x in mmads)-origin,last_mmad_end_us=max(x[1] for x in mmads)-origin,
                first_fix_issue_us=min(x[0] for x in fixes)-origin,last_fix_end_us=max(x[1] for x in fixes)-origin,
                ca_output_payload_over_descriptor_window_gbs=useful/hostwnd/1000,
                ca_output_bytes=useful,frozen_resource_assumptions=info['shared_traffic'])

def main():
    result=[]
    dt=load_chip();pr=load_chip('Ascend950PR_957b')
    for r in json.loads((ROOT/'results/v0.16/matmul-dt-provenance.json').read_text())['traces']:
        result.append(inspect(r['shape'],dt,ROOT/r['trace'],r['sha256'],r['reference_ticks']/1650,None))
    for r in json.loads((ROOT/'results/v0.16/sku/holdout.json').read_text()):
        if r['op']=='matmul':
            result.append(inspect(r['shape'],pr,ROOT/r['pr']['trace'],r['pr']['trace_sha256'],r['pr_ca_us'],r['pr_us']))
    (ROOT/'results/v0.17/matmul-trace-replay.json').write_text(json.dumps(dict(role='development only; historical PR measurements used to identify scheduling assumptions, not to fit timing prices',rows=result),indent=2)+'\n')
if __name__=='__main__':main()
