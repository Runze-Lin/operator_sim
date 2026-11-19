"""Extract comparable physical milestones from existing CA, one terminal AIV per launch."""
import json
from math import isclose
from pathlib import Path
import sqlite3
import sys
from collections import Counter
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.dispatch21.post_identifiability import collect
from development.validation.sku19.campaign import read,write
from campaign_collect import events
from operator_model import load_chip
from operator_model.post_resource import resource_post
OUT=ROOT/'results/post-candidate'


def summarize(entries,origin,end,r):
    first=min(v[0] for v in entries)
    pipes=Counter(); durations=Counter()
    read_ops=[];write_ops=[];loads=[];instr=Counter()
    for a,b,name,d,w,s,x in entries:
        q=d['queue_type'];instr[name]+=1
        if d['instr_type'].startswith(('RVEC','SIMD')):q='SIMD'
        pipes[q]+=1
        if b<=a or w or s or x:continue
        durations[q]+=b-a
        if q in ('MTE2','MTE3') and name.startswith('MOV_SRC_TO_DST'):
            (read_ops if q=='MTE2' else write_ops).append((a,b,d['processed_bytes']))
        if name=='RV_VLD':
            loads.append(a)
    read_ops.sort();write_ops.sort()
    assert read_ops[0][2]==96 and len(read_ops)>=3 and write_ops
    # Metadata is the first 96 B. The first operand is an actual read, even
    # when it is only the small residual coefficients on the token branch.
    issue=read_ops[1][0]
    vec=min(loads)
    if vec<issue: raise ValueError('SIMD operand load precedes first payload')
    write=write_ops[-1][0]
    if write<vec: raise ValueError('last store issued before first SIMD operand')
    ms=[first,issue,vec,write,end]
    assert origin<=first+1e-6 and all(a<=b+1e-6 for a,b in zip(ms,ms[1:]))
    phases=dict(zip(('arrival','control','read','body','retire'),[b-a for a,b in zip([origin]+ms,ms)]))
    assert isclose(sum(phases.values()),r['us'],abs_tol=.03)
    dma=lambda ops:[dict(start_us=a-origin,end_us=b-origin,bytes=n) for a,b,n in ops]
    return dict(phase_us=phases,raw_queue_instruction_counts=dict(pipes),raw_nonwait_service_us=dict(durations),
                dma_read=dma(read_ops),dma_write=dma(write_ops),simd_core_instructions={k:n for k,n in instr.items() if k in ('RV_VLD','RV_VST','RV_VMUL','RV_VMULS','RV_VADD','RV_VCAST','RV_VMADD')},
                read_bytes=sum(q[2] for q in read_ops[1:]),write_bytes=sum(q[2] for q in write_ops),
                first_vector_operand_us=vec-origin)


def extract(r):
    if 'stage' in r:
        s=r['stage'];k=max(s['windows'], key=lambda c:s['windows'][c][1]); f=ROOT/Path(s['trace']).parent/k/'trace.json'
        es=[]
        for ev in events(f):
            if ev.get('ph')=='X':
                d=json.loads(ev['args']['detail'])
                es.append((ev['ts'],ev['ts']+ev['dur'],ev['name'],d,d.get('is_wait'),d.get('is_sync'),d.get('is_duration_excluded')))
        a=min(v[0] for v in es); low,high=s['windows'][k];o=a-low
        return summarize(es,o,o+high,r)
    path=ROOT/r['trace']
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        o,end=db.execute('SELECT min(ExecInstrTickStart),max(ExecInstrTickEnd) FROM ExecutedInstructions').fetchone()
        k,t=db.execute('SELECT ExecInstrCoreId,CoreTypeId FROM ExecutedInstructions WHERE ExecInstrTickEnd=? LIMIT 1',(end,)).fetchone()
        seq=db.execute('SELECT ExecInstrTickStart,ExecInstrTickEnd,ExecInstrName,ExtendParams,ExecInstrIsWait,ExecInstrIsSynchronization,ExecInstrIsDurationExcluded FROM ExecutedInstructions WHERE ExecInstrCoreId=? AND CoreTypeId=?',(k,t))
        es=[(a/1650,b/1650,n,json.loads(d),w,s,x) for a,b,n,d,w,s,x in seq]
    return summarize(es,o/1650,end/1650,r)


def main():
    path=OUT/'phase-evidence.json'
    allrows=[]
    for r in collect():
        f=Path(r['trace']);chip=load_chip(r['chip'])
        old,_,_,diag=resource_post(r['shape'],chip,{})
        p=diag['target_plan'];row={k:r[k] for k in ('shape','chip','us','backend','state','active','trace','sha256','prior_aiv')}
        row.update(process=(str(f.parent.parent) if r['backend']=='npusim compact' else r['trace'].split('/profile/')[0]),
                   binary=r['binary'],kernel=r['kernel'],active_match=r['active']==p['active_aiv'],
                   **p,old_unknown_us=old,group='few token/wide' if r['shape'][0]<=4 and r['shape'][2]>=2048 else
                   'multiple rows/core' if p['rows_per_core']>1 else 'one row/core')
        try:row.update(extract(r))
        except (ValueError,AssertionError) as exc:row['failed']=repr(exc); print(r['trace'],r['shape'],repr(exc),flush=True)
        allrows.append(row)
    write(path,allrows)
    print('complete',len(allrows),sum('failed' in r for r in allrows),flush=True)

if __name__=='__main__':main()
