"""Reprice unique existing trace windows; no updates to any calibrated parameter."""
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from shadow import ROOT,predict

OUT=ROOT/'results/v0.25/cache'
audit=json.loads((ROOT/'results/v0.25/coverage/audit.json').read_bytes())
rows=[]
for sample in audit['rows']:
    op=sample['op']
    if op not in ('cast_bf16_fp32','sigmoid_fp32'):continue
    forecasts=predict(op,sample['shape'],sample['chip'])
    if not math.isclose(forecasts['uniform']['us'],sample['current_us'],abs_tol=1e-8):raise ValueError('baseline changed')
    n=math.prod(sample['shape'])
    m=n*(6 if op.startswith('cast') else 8)/2**20
    cap=forecasts['uniform']['device_L2_MiB']
    group='≤L2' if m<=cap else ('L2–2L2' if m<=2*cap else '>2L2')
    rows.append(dict(op=op,chip=sample['chip'],shape=sample['shape'],working_set_MiB=m,group=group,ca_us=sample['ca_us'],
         forecasts=forecasts,trace_sha256=sample['trace_sha256'],source=list(sample['origins']),role='development replay'))

def stats(subset):
    out={}
    for policy in ('uniform','pressure','clean_first'):
        error=[r['forecasts'][policy]['us']-r['ca_us'] for r in subset]
        out[policy]=dict(wape=100*sum(map(abs,error))/sum(r['ca_us'] for r in subset),
                       max_absolute_pct=100*max(abs(e)/r['ca_us'] for e,r in zip(error,subset)))
    return out


result=[]
for op in sorted({r['op'] for r in rows}):
    for sku in sorted({r['chip'] for r in rows}):
        for group in ('≤L2','L2–2L2','>2L2','all'):
            xs=[r for r in rows if r['op']==op and r['chip']==sku and group in ('all',r['group'])]
            if xs:result.append(dict(op=op,chip=sku,group=group,n=len(xs),**stats(xs)))
bench={}
for op,shape in [('cast_bf16_fp32',[851968,32]),('sigmoid_fp32',[1703936,12])]:
    # Warm CPU benchmark for all three scenarios including patch and config lookup; record one cold first call separately.
    t0=perf_counter_ns();predict(op,shape,'Ascend950PR_957b');warmup=(perf_counter_ns()-t0)/1e6
    values=[]
    for _ in range(101):
        start=perf_counter_ns();predict(op,shape,'Ascend950PR_957b');values.append((perf_counter_ns()-start)/1e6)
    bench[op]=dict(n=len(values),first_measured_ms=warmup,warm_median_ms=median(values),warm_p95_ms=sorted(values)[95],max_ms=max(values))
(OUT/'replay.json').write_text(json.dumps(dict(role='Existing development windows. Models predate the new paired acquisitions.',groups=result,rows=rows,shadow_query_ms=bench),ensure_ascii=False,indent=2)+'\n')
print([(r['op'],r['chip'],r['n'],round(r['uniform']['wape'],2),round(r['pressure']['wape'],2)) for r in result if r['group']=='L2–2L2'])
