"""Score prospective resident forecasts without fitting to the held-out results."""
from pathlib import Path
import json,sys
from datetime import datetime
from attn_res_fit import split
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/v0.19/workloads'
load=lambda n:json.loads((OUT/n).read_text())
obs=load('attn-res-holdout.json')
initial=load('attn-res-ca.json')
micro=load('attn-res-micro-ca.json')
freeze=load('attn-res-holdout-freeze.json')['scenarios']
rows=[]
for m in obs:
    if not m['qualified']:
        continue
    f=next(f for f in freeze if f['chip']==m['chip'] and [f['tokens'],f['hidden'],f['slots'],f['valid']]==m['shape'])
    actual=m['kernel']['ca_us']
    actual_phases=split(m)['phase_cycles'] if m['shape'][0]==1 else None
    rows.append(dict(shape=m['shape'],chip=m['chip'],pred_us=f['latency_us'],ca_us=actual,
                     rel_error=f['latency_us']/actual-1,predicted_phases_us=f['stages_us'],
                     ca_phases_us=[v/1650 for v in actual_phases] if actual_phases else None,
                     kernel=m['kernel']['kernel'],active_aiv=m['kernel']['active_aiv'],trace_sha256=m['kernel']['trace_sha256']))
def metric(p,y):
    return dict(n=len(y),wape=sum(abs(a-b) for a,b in zip(p,y))/sum(y),
                max_relative_error=max(abs(a/b-1) for a,b in zip(p,y)))
sides={}
for c in ('Ascend950DT_9582','Ascend950PR_957b'):
    a=[r for r in rows if r['chip']==c]
    sides[c]=metric([r['pred_us'] for r in a],[r['ca_us'] for r in a])
pairs=[]
for r in rows:
    if 'DT' not in r['chip']:continue
    pr=next((s for s in rows if 'PR' in s['chip'] and s['shape']==r['shape']),None)
    if not pr:continue
    pairs.append(dict(shape=r['shape'],dt_us=r['pred_us'],dt_ca_us=r['ca_us'],pr_us=pr['pred_us'],pr_ca_us=pr['ca_us'],
                      speed=r['pred_us']/pr['pred_us'],ca_speed=r['ca_us']/pr['ca_us'],
                      future_us=next(f['latency_us'] for f in freeze if f['chip']=='Ascend970DT' and f['tokens']==r['shape'][0] and f['hidden']==r['shape'][1])))
speed=metric([r['speed'] for r in pairs],[r['ca_speed'] for r in pairs])
all_obs=initial+micro+obs
cost=dict(launches=len(all_obs),eligible=sum(x['qualified'] for x in all_obs),
          raw_mib=sum(x['cost']['raw_bytes'] for x in all_obs)/2**20, simulator_wall_s_sum=sum(x['cost']['seconds'] for x in all_obs),
          known_cpu_s_lower_bound=sum((x['cost']['cpu_seconds'] or 0) for x in all_obs),cpu_missing=sum(x['cost']['cpu_seconds'] is None for x in all_obs))
(OUT/'attn-res-evaluation.json').write_text(json.dumps(dict(qualification='resident branch only, two independent input contracts paired, conditional source work for others',
        summary=sides,speedup=speed,rows=rows,pairs=pairs,cost=cost),indent=2)+'\n')
print(sides,speed,pairs,cost)
