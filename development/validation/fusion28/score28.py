"""Report measured roles, named ablations and conditional resource schedules independently."""
from collections import defaultdict
import json
from math import isclose
from statistics import mean
from schedule28 import estimate,load_chip,OUT,ROOT
from landmarks import read,save
DT,PR,FUT='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'
MODELS=('old_unknown_us','old_known_us','old_minimax_us','point_us','mean_us','harmonic_us','known_us')

def metrics(rows,key):
    kept=[r for r in rows if r.get(key) is not None]
    if not kept:return None
    diffs=[abs(r[key]-r['ca_us']) for r in kept]
    worst=max(kept,key=lambda r:abs(r[key]/r['ca_us']-1))
    return dict(n=len(kept),geometries=len({(r['t'],r['iters']) for r in kept}),
                wape=100*sum(diffs)/sum(r['ca_us'] for r in kept),max_ape=100*abs(worst[key]/worst['ca_us']-1),
                worst=dict(t=worst['t'],iters=worst['iters'],h=worst['h'],ca_us=worst['ca_us'],pred_us=worst[key]))

def summaries(rows):
    grouped=defaultdict(list)
    for r in rows:
        groups=['all',r['role'],'old public domain' if r['t']<=256 else 'wide',
                'single register group' if r['g']==1 else 'multiple register groups',r['state']]
        if r['role']=='historical development' and r['t']>64 and r['iters']==10:groups+=['historical T>64 I10']
        for group in groups:grouped[r['sku'],group].append(r)
    out={}
    for (sku,k),rs in grouped.items():
        bands=[r['range_us'][0]<=r['ca_us']<=r['range_us'][1] for r in rs]
        out.setdefault(sku,{})[k]=dict(models={j:metrics(rs,j) for j in MODELS},within_modeled_state_range=sum(bands),total=len(rs))
    return out

def historic():
    rows=[]
    history=read(ROOT/'results/v0.27/fusion/unknown-history.json')
    for x in history['rows']:
        e=estimate([x['t'],4],x['iters'],load_chip(x['sku']))
        actual_state=x['state_for_evaluation_only'];k=e['states'][actual_state]
        rows.append(dict(id=x['id'],sku=x['sku'],t=x['t'],iters=x['iters'],source=x['source'],
              role='historical development' if x['data_role']=='development' else 'v0.26 independent now replayed',
              g=k['groups'],rows=k['rows'],active=k['active'],h=x['prior_h_for_evaluation_only'],state=actual_state,
              ca_us=x['ca_us'],old_unknown_us=x['public_us'],old_known_us=x['given_h_us'],old_minimax_us=x['robust_us'],
              point_us=e['point_us'],mean_us=e['mean_us'],harmonic_us=e['harmonic_us'],range_us=e['range_us'],
              modeled_risk_pct=e['model_uncertainty_pct'],known_us=k['us'],phase_us=k['epochs_us']))
    save(OUT/'replay.json',dict(role='All 84 older windows replayed; v0.26 wide traces used in development.',rows=rows,
        strict_repeat_evidence=history['strict_repeated_pairs'],by_sku=summaries(rows)))

def score():
    reg=read(OUT/'register.json');jobs={(x['group'],x['chip'],x['index']):x for x in reg['jobs']};rows=[];attempts=[]
    for p in sorted((OUT/'receipts').glob('*.json')):
        d=read(p);attempts.append(dict(receipt=str(p.relative_to(ROOT)),cost=d['cost'],qualified=sum(s['qualified'] for s in d['rows']),failures=[s.get('reason') for s in d['rows'] if not s['qualified']],reason=d.get('reason')))
        for s in d['rows']:
            if not s['qualified']:continue
            key=(d['group'],d['chip'],s['index']);j=jobs[key];k=j['conditional'];t=j['shape'][0];it=j['iters'];trace=s['stage']
            assert (s['shape'],s['iters'],s['h'])==(j['shape'],it,j['prior_h'])
            old_us=[v['legacy_known_us'] for v in estimate([t,4],it,load_chip(d['chip']))['states'].values()]
            mn,mx=min(old_us),max(old_us)
            check=dict(active=k['active']==trace['active_aiv'],vf_per_core=all(v==2*it+2 for v in trace['vf_calls_per_core'].values()),bytes_read_logical=64*t,bytes_written=trace['written_bytes'],golden_elements=s['all_output_elements'])
            rows.append(dict(sku=d['chip'],group=d['group'],index=s['index'],t=t,iters=it,h=s['h'],g=k['groups'],rows=k['rows'],active=k['active'],state=k['state'],role='frozen forward',
                old_unknown_us=j['old_unknown_us'],old_known_us=j['old_known_us'],old_minimax_us=2*mn*mx/(mn+mx),point_us=j['point_us'],mean_us=j['mean_us'],harmonic_us=j['harmonic_us'],
                range_us=j['range_us'],known_us=k['us'],ca_us=trace['ca_us'],id=trace['trace_sha256'],check=check,receipt=str(p.relative_to(ROOT)),
                projected_970dt_unknown_us=jobs[d['group'],FUT,s['index']]['point_us']))
    ratios=[]
    for dt in [x for x in rows if x['sku']==DT]:
        other=[x for x in rows if x['sku']==PR and (x['group'],x['index'])==(dt['group'],dt['index'])]
        if other:
            pr=other[0];ratios.append(dict(t=dt['t'],iters=dt['iters'],group=dt['group'],index=dt['index'],ca_dt_over_pr=dt['ca_us']/pr['ca_us'],
                  unknown_dt_over_pr=dt['point_us']/pr['point_us'],known_dt_over_pr=dt['known_us']/pr['known_us']))
    out=dict(registration='results/v0.28/fusion/register.json',forward_geometries_per_sku=4,expected_windows_per_sku=6,rows=rows,by_sku=summaries(rows),paired_ratios=ratios,attempts=attempts)
    tmp=OUT/'independent.tmp';save(tmp,out);tmp.replace(OUT/'independent.json')

if __name__=='__main__':historic();score()
