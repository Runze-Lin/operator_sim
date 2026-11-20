"""Check frozen register-pass issue intervals against each qualified independent final AIV."""
from pathlib import Path
import sys
from landmarks import ROOT,OUT,read,save,summarize
from score28 import DT,PR
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'development/validation/sink26'))
from evidence import shard
def analyze():
    reg=read(OUT/'register.json');jobs={(x['group'],x['chip'],x['index']):x for x in reg['jobs']};rows=[]
    old=read(OUT/'issue-forward.json') if (OUT/'issue-forward.json').exists() else []
    lookup={r['trace_sha256']:r for r in old}
    for receipt in sorted((OUT/'receipts').glob('*.json')):
        d=read(receipt)
        for s in d['rows']:
            if not s['qualified']:continue
            st=s['stage'];cached=lookup.get(st['trace_sha256'])
            if cached:rows.append(cached);continue
            job=jobs[d['group'],d['chip'],s['index']];pred=job['conditional']
            path,dig,core,origin,_,_,ev=shard(st)
            for e in ev:e['begin']=round(e['begin']-origin);e['end']=round(e['end']-origin)
            ev.sort(key=lambda v:v['begin'])
            obs=summarize(job['shape'],job['iters'],d['chip'],ev,st['last_tick']-st['first_tick'])['milestones']
            e=pred['epochs_us'];gold=obs['pairs'];fields={
                'soft_to_first_column':(e['first_col_vf']-e['softmax_vf'],obs['first_col_issue']-obs['soft_issue']),
                'first_column_to_row':(e['first_row_vf']-e['first_col_vf'],obs['first_col_to_first_row']),
                'first_pair':(e['first_pair_period'],gold[0]),'steady_pair':(e['steady_pair_period'],obs['steady']),
                'scatter_to_write':(e['write_begin']-e['scatter_vf'],obs['write_issue']-obs['scatter_issue'])}
            fields={k:dict(pred_cycles=v*1650,ca_cycles=m,signed_cycles=v*1650-m) for k,(v,m) in fields.items() if m is not None}
            rows.append(dict(sku=d['chip'],shape=job['shape'],iters=job['iters'],h=job['prior_h'],g=pred['groups'],core=core,group=d['group'],index=s['index'],
                phases=fields,all_measured_pairs_cycles=gold,trace_sha256=st['trace_sha256'],critical_trace=path,critical_sha256=dig,
                cost_seconds=d['cost']['seconds'],ca_full_us=st['ca_us']))
    tmp=OUT/'issue-forward.tmp';save(tmp,rows);tmp.replace(OUT/'issue-forward.json')
if __name__=='__main__':
    if '--final' in sys.argv:
        import time
        while read(OUT/'state.json')['state'] not in ('finished','interrupted attempt preserved'):
            time.sleep(60)
    analyze()
