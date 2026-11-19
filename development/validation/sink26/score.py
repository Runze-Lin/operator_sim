"""Unchanged forecasts; keep unqualified ordinals and never hide the uncovered v0.25 domain."""
from collections import defaultdict
from evidence import ROOT,OUT,DT,PR,FUT,read,sha
from registration import REG
from replay import metrics

def score():
    from collect import atomic
    reg=read(REG);rows=[];open_cases=[];lookup={(j['group'],j['chip'],j['index']):j for j in reg['jobs']}
    for j in reg['jobs']:
        if j['chip']==FUT:continue
        group,sku,index=j['group'],j['chip'],j['index'];path=OUT/'receipts'/(group+'-'+sku+'.json')
        if not path.exists():
            open_cases.append(dict(group=group,sku=sku,index=index,status='pending'));continue
        rec=read(path);assert rec['register_sha256']==sha(REG)
        found=[v for v in rec['rows'] if v['index']==index]
        if not found or not found[0]['qualified']:
            open_cases.append(dict(group=group,sku=sku,index=index,status='failed',reason=rec.get('reason') or found[0]['reason']));continue
        rr,=found;stage=rr['stage'];k=j['known'];future=lookup[group,FUT,index]
        rows.append(dict(group=group,chip=sku,index=index,shape=j['shape'],iters=j['iters'],h=j['prior_h'],state=k['selected'],
                         ca_us=stage['ca_us'],known_us=k['latency_us'],unknown_us=j['unknown_us'],v025_us=j['v025_us'],
                         active_pred=k['active_aiv'],active_ca=stage['active_aiv'],vf_pred=k['vf_calls_per_core'],
                         vf_ca_values=sorted(set(stage['vf_calls_per_core'].values())),future_970_known_us=future['known']['latency_us'],
                         receipt=str(path.relative_to(ROOT)),raw_sha256=stage['trace_sha256']))
    groups={}
    for sku in (DT,PR):
        have=[r for r in rows if r['chip']==sku]
        combined=defaultdict(list)
        for r in have:combined[tuple(r['shape'])].append(r)
        groups[sku]={'unique_geometries_complete':len(combined),'windows_complete':len(have)}
        for title,sel in [('all finite v025', [r for r in have if r['v025_us'] is not None]),('above v025 domain',[r for r in have if r['v025_us'] is None])]:
            fields=['known_us','unknown_us']+(['v025_us'] if title=='all finite v025' else [])
            groups[sku][title]={k:metrics(sel,k) for k in fields}
        for st in ('first','expand','resident'):
            sel=[r for r in have if r['state']==st];groups[sku][st]={k:metrics(sel,k) for k in ('known_us','unknown_us')}
    pairs=[]
    for key in sorted({(r['group'],r['index']) for r in rows}):
        group,index=key;sides={r['chip']:r for r in rows if (r['group'],r['index'])==key}
        if len(sides)!=2:continue
        ratios={m:sides[DT][m]/sides[PR][m] if sides[PR][m] and sides[DT][m] else None for m in ('ca_us','known_us','unknown_us','v025_us')}
        pairs.append(dict(group=group,index=index,**ratios))
    ratio_metrics={m:metrics([r for r in pairs if r[m] is not None],m) for m in ('known_us','unknown_us','v025_us')}
    atomic(OUT/'independent.json',dict(registration_sha256=sha(REG),rows=rows,pending_or_failed=open_cases,groups=groups,
                                       paired_dt_over_pr=pairs,paired_ratio_metrics=ratio_metrics,versioned_public_default='v0.25; unchanged',
                                       role='prospective, 10 distinct v0.25-supported plus 4 extended shapes per side; T67 appears in two separately identifiable histories'))
if __name__=='__main__':score()
