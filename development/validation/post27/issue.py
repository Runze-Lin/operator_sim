"""Per-stage calibration of queue issue prices using the previous successful kernel entry."""
from collections import defaultdict
from copy import deepcopy
from math import ceil
from pathlib import Path
from statistics import median
import re,sys
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import read,write
from development.validation.post25.calibrate import observations
from operator_model import load_chip
from operator_model.post_resource import resource_post
from operator_model.post_token_pipeline import token_pipeline
OUT=ROOT/'results/v0.27/post';P25=ROOT/'results/v0.25/post'
def cohort(state,prev_route):
    if state=='first':return 'first'
    return state+('_after_token' if prev_route=='post_token' else '_after_channel' if prev_route=='post_channel' else '_prior_unknown')

def windows():
    allrows=read(ROOT/'results/post-candidate/phase-evidence.json');hist={}
    for r in allrows:
        mo=re.search(r'/(\d+)_entry_kernel_\d+\.db$',r['trace'])
        if mo:hist[(r['process'],int(mo[1]))]=r
    prev_by_hash={v['sha256']:hist.get((proc,i-1),{}).get('route') for (proc,i),v in hist.items()}
    appended=[];current={}
    for v in ('v0.24','v0.25'):
        for f in sorted((ROOT/'results'/v/'post/receipts').glob('*.json')):
            group=read(f)['rows'];previous=None
            for r in group:
                if r['timing_qualified']:
                    z=r['stage'];assert not r['previous_shape'] or previous is not None
                    prev_by_hash[z['trace_sha256']]=previous
                    if v=='v0.25' and r['plan']['route']=='post_token':
                        w=dict(sku=r['chip'],shape=r['shape'],state='repeat' if r['role']=='repeat' else r['state'],prior=r['prior_aiv'],active=r['active'],rows=r['plan']['rows_per_core'],us=z['ca_us'],process='v025/'+f.name,source=z['trace'],sha=z['trace_sha256'],items=[],arrival=None,control=None)
                        c=r.get('ca_milestones')
                        if c:
                            y=c['dma_write'];x=c['dma_read'];d=r['shape'][2];n=len(y)
                            if len(x)==1+4*n and all(u['bytes']==8*d for u in y):
                                for i,y_i in enumerate(y):
                                    block=x[1+i*4:5+i*4];assert [q['bytes'] for q in block]==[64,16,2*d,8*d]
                                    w['items'].append(dict(read_start=block[0]['start_us'],write_start=y_i['start_us']))
                            w.update(arrival=c['phase_us']['arrival'],control=c['phase_us']['control'])
                        appended.append(w)
                # Check completed prior kernel identity from qualified receipt or raw profiler order.
                if r['timing_qualified']:name=r['stage']['kernel']
                else:
                    folder=ROOT/('runs/post24' if v=='v0.24' else 'runs/post25')/f.stem
                    names=re.findall(r'Start profiling on kernel: (\S+)',(folder/'run.log').read_text());name=names[r['i']]
                previous='post_token' if name.endswith('_6') else 'post_channel' if name.endswith('_5') else None
    rs=observations()+appended
    for r in rs:
        r['prev_route']=prev_by_hash.get(r['sha'])
        r['cohort']=cohort(r['state'],r['prev_route'])
    return rs

def vector_work(width,chip,prices):
    return prices['ready_to_write_per_group']*ceil(width/64)*(27/64)/(chip.vector_tops['32']/chip.vector_cores)*(chip.frequency_ghz/1.65)
def curve(obs):
    # Price max(fixed issue, compute tail + pre-calibrated per-vector-group cycles).
    if len(obs)<3 or len({n for n,y in obs})<2:return None
    a=np.asarray([v[0] for v in obs]);y=np.asarray([v[1] for v in obs])
    res=least_squares(lambda p:np.maximum(p[0],a+p[1])-y,[median(y),50.],bounds=(0,np.inf),loss='soft_l1',f_scale=70)
    return dict(minimum=float(res.x[0]),fixed_compute_tail=float(res.x[1]),n=len(obs))
def fit(rs,prices):
    observations=defaultdict(list);responses=defaultdict(list)
    for r in rs:
        if r['state']=='repeat' or not r['items']:continue
        c=load_chip(r['sku']);rate=c.frequency_ghz*1000;g=vector_work(r['shape'][2],c,prices);key=r['cohort']
        if r['control'] is not None:responses[key].append((r['control']*rate,r['arrival']*rate, r['process']))
        for term,timing in [('read','read_start'),('write','write_start')]:
            seq=[v[timing] for v in r['items']];gaps=[(b-a)*rate for a,b in zip(seq,seq[1:])]
            if gaps:observations[key,term+'_gap_first'].append((g,gaps[0]))
            if len(gaps)>=2:observations['shared',term+'_ii'].append((g,median(gaps[1:])))
    for r in rs:
        if r['state']!='repeat' or len(r['items'])<3:continue
        c=load_chip(r['sku']);rate=c.frequency_ghz*1000;g=vector_work(r['shape'][2],c,prices)
        for term,timing in [('read','read_start'),('write','write_start')]:
            seq=[v[timing] for v in r['items']];gaps=[(b-a)*rate for a,b in zip(seq[1:],seq[2:])];observations['shared',term+'_ii'].append((g,median(gaps)))
    return dict(curves={group:{kind:curve(arr) for (family,kind),arr in observations.items() if family==group} for group in sorted({k[0] for k in observations})},
                control={k:dict(median_cycles=median(z[0]for z in v),n=len(v),processes=len({z[2]for z in v})) for k,v in responses.items()})

def predict(shape,chip,h,previous_shape,params,prices):
    baseline,_,active,base=resource_post(shape,chip,{'resident_aiv':h})
    plan=base['target_plan'];state=base['context']
    if plan['route']!='post_token' or previous_shape==shape:return dict(us=baseline,base=baseline,modified=False)
    prev=None if previous_shape is None else resource_post(previous_shape,chip,{})[3]['target_plan']['route']
    return price_explicit(shape,chip,params,prices,baseline,base,state,prev)
def price_explicit(shape,c,params,prices,baseline,base,state,prev):
    key=cohort(state,prev);old=token_pipeline(shape,c,base['target_plan'],state,prices,base['shared_floor_us'])
    p=dict(prices);stage=dict(prices['by_state'][state]);p['by_state']={state:stage}
    if key in params['control'] and params['control'][key]['processes']>=2:
        stage['control_to_operand']=params['control'][key]['median_cycles']
    g=vector_work(shape[2],c,prices)
    for group in (key,'shared'):
        for kind,v in params['curves'].get(group,{}).items():
            if v:stage[kind]=max(v['minimum'],v['fixed_compute_tail']+g)
    pipe=token_pipeline(shape,c,base['target_plan'],state,p,base['shared_floor_us'])
    return dict(us=pipe['us'],base=baseline,old_pipeline_us=old['us'],modified=True,cohort=key,diag=pipe)
def run():
    prices=read(P25/'prices.json');rs=windows();nonrepeat=[r for r in rs if r['state']!='repeat'];rr=[]
    for r in nonrepeat:
        train=[t for t in rs if t['shape']!=r['shape'] and t['process']!=r['process']];params=fit(train,prices)
        c=load_chip(r['sku']);base,_,_,diag=resource_post(r['shape'],c,{'resident_aiv':r['prior']})
        out=price_explicit(r['shape'],c,params,prices,base,diag,r['state'],r['prev_route']);row=dict(sku=r['sku'],shape=r['shape'],state=r['state'],cohort=r['cohort'],process=r['process'],source=r['source'],sha256=r['sha'],previous_entry=r['prev_route'],ca_us=r['us'],public_us=base,old_pipeline_us=out['old_pipeline_us'],candidate_us=out['us'],critical_rows=len(r['items']),measured_control_us=r['control'])
        for k,prop in [('read','read_start'),('write','write_start')]:
            z=[v[prop] for v in r['items']];row[k+'_gaps_us']=[b-a for a,b in zip(z,z[1:])]
        rr.append(row)
    write(OUT/'archive.json',dict(scope='Development-only, every tested shape and acquisition process removed together. For v0.25 stage priors use original frozen values; historical evaluation therefore remains development even when new issue parameters held out.',rows=rr))
    write(OUT/'parameters.json',fit(rs,prices))
    for subset in [('all',rr)]+[(s,[r for r in rr if r['cohort']==s]) for s in sorted({r['cohort'] for r in rr})]:
        cat,a=subset
        print(cat,len(a),*[round(100*sum(abs(x[k]-x['ca_us'])for x in a)/sum(x['ca_us'] for x in a),2) for k in ('public_us','old_pipeline_us','candidate_us')])
if __name__=='__main__':run()
