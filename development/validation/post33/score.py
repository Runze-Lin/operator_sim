"""Join frozen Post prices to historical and newly arriving complete device windows."""
from collections import Counter
from datetime import datetime,timezone
from statistics import median
from time import perf_counter_ns
import json
from pathlib import Path
from prices import ROOT,OUT,read,sha
from candidate import predict
from development.validation.post32.evaluate import oracle

DT,PR='Ascend950DT_9582','Ascend950PR_957b'
def save(name,value):
    path=OUT/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def metric(rows,key):
    if not rows:return None
    y=sum(r['ca_us'] for r in rows)
    worst=max(rows,key=lambda r:abs(r[key]/r['ca_us']-1))
    return dict(windows=len(rows),shapes=len({tuple(r['shape']) for r in rows}),
        wape=100*sum(abs(r[key]-r['ca_us']) for r in rows)/y,
        maximum=100*abs(worst[key]/worst['ca_us']-1),worst_sha=worst.get('sha256'),worst_shape=worst['shape'])

def score():
    old=read('results/v0.32/post/replay.json')
    four={r['sha256']:r['candidate_shape_us'] for v in read('results/v0.32/post/channel-diagnostic.json').values() for r in v['rows']}
    params=read(OUT/'prices.json');reg=read(OUT/'freeze.json');rows=[];summary={}
    strict=read('results/v0.30/post/score.json')['bounds']
    for r in old['rows']:
        started=perf_counter_ns();p=predict(r['shape'],r['chip'],params);ms=(perf_counter_ns()-started)/1e6
        rows.append(dict(chip=r['chip'],shape=r['shape'],ca_us=r['ca_us'],kernel=r['kernel'],trace=r['trace'],sha256=r['sha256'],
            visible='shape,dtype,SKU',route=r['target_plan']['route'],v032_us=r['shape_us'],four_write_us=four[r['sha256']],
            candidate_us=p['us'],range_us=p['range_us'],policy=p['policy'],prices=p['conditional_us'],facts=p['details'],
            signed_pct=100*(p['us']/r['ca_us']-1),query_ms=ms,validation_only_actual_prior=r['scenario']))
    assert len(rows)==239 and len({r['sha256'] for r in rows})==239
    for chip in (DT,PR):
        rs=[r for r in rows if r['chip']==chip]
        paired=[]
        for control in strict[chip]['by_shape']:
            for y in control['us']:
                paired.append(next(r for r in rs if r['shape']==control['shape'] and abs(r['ca_us']-y)<1e-8))
        assert len({r['sha256'] for r in paired})==len(paired)
        summary[chip]={}
        for key,cohort in [('archive',rs),('strict_same_contract_cold_hot',paired),('channel',[r for r in rs if r['route']=='post_channel']),('token',[r for r in rs if r['route']=='post_token'])]:
            summary[chip][key]={k:metric(cohort,k) for k in ('v032_us','four_write_us','candidate_us')}
            if key in ('archive','strict_same_contract_cold_hot'):
                summary[chip][key]['oracle']={k:v for k,v in oracle(cohort).items() if k!='shapes'}
        times=sorted(r['query_ms'] for r in rs)
        summary[chip]['candidate_query_ms']=dict(median=median(times),p95=times[(95*len(times)+99)//100-1],maximum=times[-1])
        summary[chip]['model_range_contains_ca']=sum(r['range_us'][0]<=r['ca_us']<=r['range_us'][1] for r in rs)
    save('replay.json',dict(summary=summary,rows=rows,role='mechanism development, future outcomes are distinct',freeze_utc=reg['utc']))

def forward():
    f=read(OUT/'freeze.json');rows=[]
    for chip in (DT,PR):
        path=OUT/(chip+'.json');rec=read(path) if path.exists() else {}
        for p in f['rows'][chip]:
            w=next((r for r in rec.get('rows',[]) if r['i']==p['i'] and r['shape']==p['shape']),{})
            s=w.get('stage',{}) if w.get('timing_qualified') else {}
            rows.append(dict(chip=chip,ordinal=p['i'],shape=p['shape'],role=p['role'],candidate_us=p['us'],v032_us=p['v032_shape_us'],
                measured=s.get('ca_us') is not None,ca_us=s.get('ca_us'),sha256=s.get('trace_sha256'),
                signed_pct=100*(p['us']/s['ca_us']-1) if s else None,kernel=s.get('kernel'),
                outputs=w.get('all_output_elements'),output_bytes=s.get('writes_bytes'),cost=rec.get('cost'),receipt=str(path.relative_to(ROOT)) if rec else None))
    sources=[]
    for p in f['additional_prepriced']:
        path=ROOT/p['receipt'];s=None
        if path.exists():
            r=read(path)
            if p['role']=='isolated_large':
                if r.get('qualified') and r.get('shape')==p['shape']:
                    s=next((stage for stage in r.get('stages',[]) if stage.get('op')=='mhc_post' and stage.get('physical_end_all_lanes')),None)
            else:
                want='repeat' if p['role']=='candidate_channel_repeat' else 'first requested target'
                for w in r.get('rows',[]):
                    if w.get('shape')==p['shape'] and w.get('timing_qualified') and (want!='repeat' or
                          ('repeat' in (w.get('role') or '') or w.get('state')=='repeat')):
                        s=w.get('stage');break
        ca=s.get('ca_us') if isinstance(s,dict) else None
        sources.append(dict(p,ca_us=ca,sha256=s.get('trace_sha256') if s else None,
                signed_pct=100*(p['us']/ca-1) if ca else None))
    save('forward.json',dict(utc=datetime.now(timezone.utc).isoformat(),focus=rows,additional=sources,
        complete_core_windows=sum(r['measured'] for r in rows),focus_summary={ch:metric([r for r in rows if r['chip']==ch and r['measured']],'candidate_us') for ch in (DT,PR)}))

if __name__=='__main__':score();forward()
