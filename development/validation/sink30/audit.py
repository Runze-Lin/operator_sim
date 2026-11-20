"""Score all released Sinkhorn windows with geometry, execution context and strict provenance."""
import collections
from datetime import datetime,timezone
from math import isclose,ceil
from pathlib import Path
from statistics import median
import sys
import time
from register import ROOT,OUT,read,digest,DT,PR,FUT,predict
from operator_model import estimate
from operator_model.sink_history import price as conditional
from operator_model import load_chip
sys.path.insert(0,str(ROOT/'development/validation/fusion28'))
from schedule28 import estimate as pending_candidate

def rows():
    data={}
    def add(row):
        h=row['trace_sha256']
        if h in data:
            assert isclose(data[h]['ca_us'],row['ca_us'],abs_tol=.002)
            data[h]['also_in'].append(row['source'])
            if row['role'].startswith('v030 '):
                data[h]['followup_role']=row['role']
            return
        data[h]=dict(**row,also_in=[])
    def load(rel):return read(ROOT/rel)
    audit=load('results/v0.25/coverage/audit.json')
    known={r['id']:r for r in load('results/v0.28/fusion/replay.json')['rows']}
    evs={e['aggregate_sha256']:e for e in load('results/v0.26/sink/trace-evidence.json')}
    for r in audit['rows']:
        if r['op']!='mhc_sinkhorn':continue
        h,=r['trace_sha256'];e=evs[h];k=known[e['source_sha256']]
        assert r['ca_us']==e['ca_us'] and abs(k['ca_us']-r['ca_us'])<1e-7
        paths=e['source'].split('/');host=paths[1] if paths[0]=='runs' else '/'.join(paths[:2])
        add(dict(trace_sha256=h,t=r['shape'][0],iters=r['attrs']['num_iters'],sku=r['chip'],ca_us=r['ca_us'],h=k['h'],
                 state=k['state'],role='development',source=e['source'],cohort=host,provider=r['kernel'],active_measured=r['active_aiv'],context='standalone'))
    for prefix in ('v0.26/sink','v0.28/fusion'):
        folder=ROOT/'results'/prefix
        for rec in sorted((folder/'receipts').glob('*.json')):
            receipt=read(rec)
            for q in receipt.get('rows',[]):
                if not q.get('qualified'):continue
                st=q['stage'];job=next(j for j in read(folder/'register.json')['jobs'] if j['chip']==receipt['chip'] and j['group']==receipt['group'] and j['index']==q['index'])
                a=job['prior_h'];active=st['active_aiv'];state='first' if not a else ('resident' if a>=active else 'expand')
                add(dict(trace_sha256=st['trace_sha256'],t=job['shape'][0],iters=job['iters'],sku=job['chip'],ca_us=st['ca_us'],h=a,state=state,
                         role=prefix,source=str(rec.relative_to(ROOT)),cohort=prefix+'/'+receipt['group'],provider=st['kernel'],
                         active_measured=active,context='standalone',index=q['index'],original_registration=digest(folder/'register.json')))
    # Explicit API materialized with or without prior Pre in deployed workloads.
    # The actual Sink output is verified elementwise along with the rest of the chain.
    for r in load('results/v0.12/evaluation.json')['rows']:
        if r['op']!='mhc' or len(r['kernel_records'])!=3:continue
        k=r['kernel_records'][1];assert k['launch_id']==1 and k['kernel']=='entry_kernel_0'
        assert any('name=chain_sinkhorn' in s and 'bad=0' in s for s in r['correctness'])
        add(dict(trace_sha256=r['db_sha256'][1],t=r['shape'][0],iters=r['attrs']['num_iters'],sku=DT,ca_us=k['reference_us'],h=None,state='foreign Pre',
                 role='chain replay',source=r['source'],cohort='chain/'+r['source'].split('/')[1],provider=k['kernel'],
                 active_measured=k['active_aiv'],context='chain second component'))
    for r in load('results/v0.17/sku/records.json'):
        if r['op']!='mhc' or not r.get('stages'):continue
        k=r['stages'][1]
        assert k['kernel'].startswith('MhcSinkhorn_') and any('name=chain_sinkhorn' in s and 'bad=0' in s for s in r['correctness'])
        add(dict(trace_sha256=k['trace_sha256'],sku=r['chip'],t=r['shape'][0],iters=r['attrs']['num_iters'],ca_us=k['ca_us'],h=None,state='foreign Pre',
                 role='chain replay',source=k['trace'],cohort='chain/sku17',provider=k['kernel'],active_measured=k['active_aiv'],context='chain second component'))
    # Additional precision study is deduplicated from its worker and original parent.
    freeze=read(OUT/'freeze.json')
    for f in freeze['future']:
        src=ROOT/f['receipt']
        if not src.exists():continue
        rec=read(src);cohort=f['cohort'];t=f['t'];h=f['h'];i=f['iters'];sku=f['sku'];state='foreign Pre' if h is None else 'first' if h==0 else None
        if cohort.startswith('v29-'):
            if not rec['qualified']:continue
            k=next(k for k in rec['stages'] if k['kernel'].startswith('MhcSinkhorn_'))
        else:
            q=next((q for q in rec.get('rows',[]) if q['index']==f['index'] and q['qualified']),None)
            if not q:continue
            k=q['stage']
        active=k['active_aiv'];state=state or ('resident' if h>=active else 'expand')
        add(dict(trace_sha256=k['trace_sha256'],sku=sku,t=t,iters=i,ca_us=k['ca_us'],h=h,state=state,
                 role='v030 prospective' if not f['had_started'] else 'v030 overlap replay',source=f['receipt'],
                 cohort=cohort,provider=k['kernel'],active_measured=active,context='chain second component' if h is None else 'standalone'))
    return list(data.values())

def metrics(rs,key):
    scored=[r for r in rs if r.get(key) is not None]
    if not scored:return dict(windows=len(rs),priced=0,shapes=0,wape=None,max_pct=None)
    worst=max(scored,key=lambda r:abs(r[key]/r['ca_us']-1))
    return dict(windows=len(rs),priced=len(scored),shapes=len({(r['t'],r['iters'])for r in scored}),
                wape=100*sum(abs(r[key]-r['ca_us'])for r in scored)/sum(r['ca_us']for r in scored),
                max_pct=100*abs(worst[key]/worst['ca_us']-1),worst={k:worst[k]for k in ('sku','t','iters','state','source','trace_sha256','ca_us',key)})

def grade():
    rs=rows()
    for r in rs:
        c=load_chip(r['sku']);t=r['t'];i=r['iters'];attrs={'num_iters':i}
        actual=estimate('mhc_sinkhorn',(t,4),attrs,chip=c)
        reg=predict(t,i,r['sku'])
        assert (actual.latency_us is None)==(reg['us'] is None)
        if reg['us'] is not None:assert isclose(actual.latency_us,reg['us'],rel_tol=1e-11)
        d=actual.projection['timing_diagnostics'];r.update(active=d['target_plan']['active_aiv'],groups=ceil(ceil(t/c.vector_cores)/8),
            old_us=reg['prior_us'],public_us=actual.latency_us,qualification=actual.status,scenario_us=d.get('scenario_us'),envelope_us=d.get('modeled_range_us'),
            minimax_us=d.get('relative_minimax_us'),known_us=conditional((t,4),c,dict(**attrs,resident_aiv=r['h']))[0] if r['h'] is not None else None,
            future970_us=estimate('mhc_sinkhorn',[t,4],attrs,chip=FUT).latency_us)
        try:
            r['pending_robust_us']=pending_candidate([t,4],i,c)['point_us']
        except ValueError:
            r['pending_robust_us']=None
        r['active_match']=r['active']==r['active_measured']
        r['signed_pct']=100*(actual.latency_us/r['ca_us']-1)if actual.latency_us is not None else None
        r['range_contains_ca']=r['envelope_us'][0]<=r['ca_us']<=r['envelope_us'][1] if r['envelope_us'] else None
    buckets=collections.defaultdict(list)
    for r in rs:
        tags=['all','role/'+r['role'],'context/'+r['context'],'source/'+r['cohort'],'budget/'+('G1'if r['groups']==1 else 'G>1'),
              'state/'+r['state']]
        if r.get('followup_role'):tags+=['role/'+r['followup_role']]
        if r['public_us'] is not None and r['old_us'] is not None:
            tags+=['matched all','matched/'+r['context']]
        if r['role']=='development' and r['t']>64 and r['iters']==10:tags+=['legacy T>64 I10']
        if r['role']=='v0.26/sink' and r['t']<=256:tags+=['original ten-shape blind']
        if r['role']=='v0.26/sink' and r['t']>256:tags+=['original wide blind']
        if r['context']=='standalone' and r['groups']==1:tags+=['G1/'+r['state']]
        for tag in tags:buckets[r['sku'],tag].append(r)
    summary={s:{k:{x:metrics(v,x) for x in ('public_us','old_us','minimax_us','known_us','pending_robust_us')}for (chip,k),v in buckets.items() if chip==s}for s in (DT,PR)}
    pairs=read(ROOT/'results/v0.27/fusion/unknown-history.json')['strict_repeated_pairs']
    matched=collections.defaultdict(dict)
    for r in rs:
        if r['role']=='v0.26/sink' and r['t']<=256:
            matched[(r['cohort'],r['index'])][r['sku']]=r
    speed=[]
    for (cohort,index),pair in matched.items():
        d,p=pair[DT],pair[PR]
        assert (d['t'],d['iters'])==(p['t'],p['iters'])
        speed.append(dict(cohort=cohort,index=index,t=d['t'],iters=d['iters'],
                          **{k:d[k]/p[k] for k in ('ca_us','public_us','old_us','known_us')}))
    return dict(utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=digest(OUT/'freeze.json'),results=summary,rows=rs,
        original_ten_paired_dt_over_pr=speed,
        strict_same_input_lower_bound=pairs,scope='All original 54 archive primary hashes, v026 and v028 qualified full windows, plus v012/017 Pre→Sink components. Unique whole trace hash per SKU. Context, true H, device measurements never enter the unknown point.')
if __name__=='__main__':
    import json
    (OUT/'score.json').write_text(json.dumps(grade(),indent=2)+'\n')
