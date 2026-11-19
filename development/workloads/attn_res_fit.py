"""Fit stage clocks only on completed, golden-checked DT arch35 RESIDENT samples."""
import json
import sys
import re
from pathlib import Path
from statistics import median
from math import ceil
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/v0.19/workloads'
sys.path.insert(0,str(ROOT/'development/validation/sku17'))
from campaign_collect import events,sha

def split(row):
    h,n=row['shape'][1],row['shape'][-1]
    trace=ROOT/row['kernel']['trace']
    assert sha(trace)==row['kernel']['trace_sha256']
    es=[e for e in events(trace) if e['ph']=='X']
    origin=min(e['ts'] for e in es)
    end=max(e['ts']+e['dur'] for e in es)
    desc=[]
    for e in es:
        if e['name']!='MOV_SRC_TO_DST_ALIGNv2':continue
        d=json.loads(e['args']['detail'])
        desc.append(dict(queue=d['instr_type'],start=e['ts']-origin,end=e['ts']+e['dur']-origin,bytes=d['processed_bytes']))
    reads=sorted([e for e in desc if e['queue']=='MTE2' and e['bytes']==2*h],key=lambda e:e['start'])
    assert len(reads)==5+n and len([e for e in desc if e['queue']=='MTE3'])==4
    edges=[0.,reads[0]['start'],reads[2]['start'],reads[4]['start'],reads[-1]['start'],end-origin]
    return dict(shape=row['shape'],sha256=sha(trace),desc=desc,phase_cycles=[1650*(b-a) for a,b in zip(edges,edges[1:])])

def features(h,n):
    # Per payload AIV. Each score pass consumes one MTE2 row and three reductions;
    # the final gamma-to-output phase also includes the prefix and weighted rows.
    return [(0,1),(2,ceil(h/64)),(2,ceil(h/64)),(n,ceil(h/64)*n),(1,ceil(h/64)*(n+1))]

if __name__=='__main__':
    rows=json.loads((OUT/'attn-res-micro-ca.json').read_text())
    used=[dict(chip=r['chip'],role='full',**split(r)) for r in rows if r['qualified']]
    assert len(used)==2
    # For the real H=7168 DT run the final write has no completed window.
    # The distinct payload read issue events already bracket complete stages.
    raw=next((ROOT/'runs/workloads19/attnres_1_2_Ascend950DT_9582').glob('profile/*/device0/*/0/dump/core0.veccore0.instr_popped_log.dump'))
    reads=[]
    for line in raw.open():
        t=re.match(r'\[info\] \[(\d+)\]',line)
        if t and ' MTE2 ' in line and ' MOV_SRC_TO_DST_ALIGNv2 ' in line:
            reads.append(int(t[1]))
    assert len(reads)==8
    _,proj,w,prefix,add,bank0,bank1,gamma=reads
    used.append(dict(shape=[1,7168,8,2],chip='Ascend950DT_9582',role='partial: issued DMA bounds only',
                     trace=str(raw.relative_to(ROOT)),sha256=sha(raw),
                     phase_cycles=[None,prefix-proj,bank0-prefix,gamma-bank0,None]))
    # Strip measured variable MTE response slope at the local MTE2 service
    # rate (48.36 B/cycle from archived 950 request calibration), not latency.
    stages={}
    for j,k in enumerate(['dispatch','score_weights','prefix_add_write','bank_scores','weighted_norm_write']):
        obs=[]
        for r in used:
            if r['phase_cycles'][j] is None or (r['chip']=='Ascend950PR_957b' and j!=4):
                continue
            h,n=r['shape'][1],r['shape'][-1]; nread,vec=features(h,n)[j]
            fixed=n if k=='bank_scores' else 1
            obs.append((vec/fixed,(r['phase_cycles'][j]-2*h*nread/48.3629559884)/fixed))
        if j==0:
            a=median(y for _,y in obs); b=0.
        else:
            (x,y),(u,v)=obs
            b=max(0.,min((v-y)/(u-x),y/x,v/u))
            a=median(z-b*t for t,z in obs)
        stages[k]=dict(fixed_cycles=a,cycles_per_vector_chunk=b)
    output={'scope':'Resident one token/payload core, fused add, mix, write, materialization and output norm. DT stage timing from micro full and H7168 partial issued reads; PR full contributes only the slope of the final phase.',
            'samples':used,'stage_prices':stages}
    (OUT/'attn-res-prices.json').write_text(json.dumps(output,indent=2)+'\n')
    print(stages)
