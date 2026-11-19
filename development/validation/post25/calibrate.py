"""Fit only observable descriptor milestones; withhold every occurrence of each shape and process."""
import math,statistics,sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import read,write
from operator_model import load_chip
from operator_model.rms_resource import READ_PER_KIB,WRITE_PER_KIB
from operator_model.post_resource import resource_post
from operator_model.post_token_pipeline import token_pipeline
OUT=ROOT/'results/v0.25/post'
STATES=('first','expand','reuse','repeat')
def observations():
    samples=[]
    archive=read(ROOT/'results/post-candidate/phase-evidence.json')
    for x in archive:
        if x['route']=='post_token':
            samples.append(dict(sku=x['chip'],shape=x['shape'],state=x['state'],prior=x['prior_aiv'],active=x['active'],rows=x['rows_per_core'],us=x['us'],data=x,process=x['process'],source=x['trace'],sha=x['sha256']))
    for file in (ROOT/'results/v0.24/post/receipts').glob('*.json'):
        for r in read(file)['rows']:
            if not r['timing_qualified'] or r['plan']['route']!='post_token':continue
            samples.append(dict(sku=r['chip'],shape=r['shape'],state='repeat' if r['role']=='repeat' else r['state'],prior=r['prior_aiv'],active=r['active'],rows=r['plan']['rows_per_core'],us=r['stage']['ca_us'],data=r['ca_milestones'],process=file.name,source=r['stage']['trace'],sha=r['stage']['trace_sha256']))
    assert len({r['sha'] for r in samples})==len(samples)
    for r in samples:
        data=r.pop('data');d=r['shape'][-1];rr=data['dma_read'];ww=data['dma_write']
        assert rr[0]['bytes']==96 and len(rr)==1+4*len(ww) and all(q['bytes']==8*d for q in ww)
        # The last block may have fewer items; train the measured row schedule on its actual footprint.
        assert len(ww)<=r['rows'];rows=[]
        for i,w in enumerate(ww):
            q=rr[1+4*i:5+4*i];assert [v['bytes'] for v in q]==[64,16,2*d,8*d]
            issue=[v['start_us'] for v in q];ready=max(v['end_us'] for v in q)
            assert w['start_us']+1e-4>=ready
            rows.append(dict(read_start=issue[0],gaps=[b-a for a,b in zip(issue,issue[1:])],
                    response=[v['end_us']-v['start_us'] for v in q],ready_to_write=w['start_us']-ready,
                    write_start=w['start_us'],write_end=w['end_us'],bytes=[v['bytes'] for v in q]))
        r.update(items=rows,arrival=data['phase_us']['arrival'],control=data['phase_us']['control'],exit=max(0.,r['us']-ww[-1]['end_us']))
    return samples

def median(values):
    return float(statistics.median(values))

def fit(rows):
    obs=[]
    for r in rows:
        chip=load_chip(r['sku']);f=1000*chip.frequency_ghz;ratio=(27000/(64*1.65))/(chip.vector_tops['32']*1000/(chip.vector_cores*chip.frequency_ghz))
        for i,q in enumerate(r['items']):
            obs.append((r['state'],i==0,math.ceil(r['shape'][-1]/64)*ratio,q['ready_to_write']*f))
    kinds=sorted({(s,i) for s,i,_,_ in obs})
    x=np.array([[float(z==(s,i)) for z in kinds]+[g] for s,i,g,_ in obs]);y=np.array([v for *_,v in obs])
    curve=least_squares(lambda p:x@p-y,np.ones(x.shape[1]),bounds=(0,np.inf),loss='soft_l1',f_scale=80)
    slope=float(curve.x[-1])
    out=dict(read_cycles_per_kib=READ_PER_KIB,write_cycles_per_kib=WRITE_PER_KIB,
             ready_to_write_per_group=slope,by_state={})
    for state in STATES:
        sub=[r for r in rows if r['state']==state] or [r for r in rows if r['state']=='reuse'] or rows
        fsub=lambda r:1000*load_chip(r['sku']).frequency_ghz
        p=dict(arrival_per_log2_aiv=median([r['arrival']*fsub(r)/math.log2(r['active']) for r in sub]),
               control_to_operand=median([r['control']*fsub(r) for r in sub]),ports={})
        for first in [True,False]:
            entries=[(r,i,q) for r in sub for i,q in enumerate(r['items']) if (i==0)==first]
            if not entries:entries=[(r,i,q) for r in rows for i,q in enumerate(r['items']) if (i==0)==first]
            shape_key='first' if first else 'steady'
            p['ports'][shape_key]=dict(gaps=[median([q['gaps'][j]*fsub(r) for r,i,q in entries]) for j in range(3)],
                fixed=[max(0.,median([q['response'][j]*fsub(r)-q['bytes'][j]*READ_PER_KIB/1024 for r,i,q in entries])) for j in range(4)],
                ready_to_write_fixed=max(0.,median([q['ready_to_write']*fsub(r)-slope*math.ceil(r['shape'][-1]/64)*(27/64)/(load_chip(r['sku']).vector_tops['32']/load_chip(r['sku']).vector_cores)*(load_chip(r['sku']).frequency_ghz/1.65) for r,i,q in entries])))
        for is_first,suffix in [(True,'gap_first'),(False,'ii')]:
            for entry,pre in [('read_start','read'),('write_start','write')]:
                def one(r):
                    vals=[(b[entry]-a[entry])*fsub(r) for k,(a,b) in enumerate(zip(r['items'],r['items'][1:])) if (k==0)==is_first]
                    return median(vals) if vals else None
                vals=[one(r) for r in sub if one(r) is not None]
                if not vals:vals=[one(r) for r in rows if one(r) is not None]
                p[f'{pre}_{suffix}']=median(vals)
        p['write_fixed']=max(0.,median([(r['items'][-1]['write_end']-r['items'][-1]['write_start'])*fsub(r)-8*r['shape'][-1]*WRITE_PER_KIB/1024 for r in sub]))
        p['exit_after_write']=median([r['exit']*fsub(r)for r in sub])
        out['by_state'][state]=p
    return out

def predict(shape,chip,prior,params,trace_rows=(),previous=None):
    base,_,active,d=resource_post(shape,chip,{} if prior is None else {'resident_aiv':prior})
    plan=d['target_plan']
    if plan['route']!='post_token':return dict(us=base,baseline=base,active=active,mode='public channel fallback',plan=plan)
    s=d['context'];s='repeat' if s=='reuse' and tuple(previous or ())==tuple(shape) else s;by={k:token_pipeline(shape,chip,plan,k,params,d['shared_floor_us'],trace_rows) for k in STATES}
    if s=='unknown':
        a=min(x['us'] for x in by.values());b=max(x['us'] for x in by.values());us=2*a*b/(a+b)
        return dict(us=us,baseline=base,active=active,mode='relative minimax over states',plan=plan,range_us=[a,b])
    return dict(us=by[s]['us'],baseline=base,active=active,mode=s,plan=plan,pipeline=by[s])

def main():
    rows=observations();base=fit(rows);write(OUT/'prices.json',base);records=[]
    # Test each held-out shape using neither any window of that shape nor its acquisition process.
    for r in rows:
        key=tuple(r['shape']);processes={r['process']}
        train=[x for x in rows if tuple(x['shape'])!=key and x['process'] not in processes]
        price=fit(train);v=predict(r['shape'],load_chip(r['sku']),r['prior'],price,previous=r['shape'] if r['state']=='repeat' else None)
        records.append(dict(shape=r['shape'],state=r['state'],sku=r['sku'],source=r['source'],sha=r['sha'],rows=r['rows'],observed_rows=len(r['items']),training=len(train),processes_purged=sorted(processes),old_us=v['baseline'],candidate_us=v['us'],ca_us=r['us']))
    write(OUT/'archive.json',dict(scope='development process-and-shape purged replay only',rows=records))
    for subset in [records,[r for r in records if r['rows']>=2],[r for r in records if r['rows']>=2 and r['state']=='repeat']]:
        print('archive n',len(subset),'wape old/new',*[round(100*sum(abs(r[k]-r['ca_us']) for r in subset)/sum(r['ca_us'] for r in subset),2) for k in ('old_us','candidate_us')])
if __name__=='__main__':main()
