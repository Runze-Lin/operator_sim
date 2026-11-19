"""Run via the isolated, installed wheel (python -I) outside the project."""
from json import loads,dumps
from math import isclose,ceil
from pathlib import Path
import statistics
from time import perf_counter_ns

root=Path(__file__).resolve().parents[3]
v=root/'results/v0.19'
def read(p):return loads(p.read_text())
start=perf_counter_ns()
import operator_model
from operator_model import estimate
import_ms=(perf_counter_ns()-start)/1e6
assert operator_model.__version__=='0.19.0' and not Path(operator_model.__file__).is_relative_to(root)
chips={'dt':'Ascend950DT_9582','pr':'Ascend950PR_957b','future':'Ascend970DT'}
seen=set();times={}
def verify(op,shape,attrs,chip,expected,mode):
    key=op,tuple(shape),str(sorted(attrs.items())),chip,mode
    if key in seen:return
    start=perf_counter_ns()
    e=estimate(op,shape,attrs,chip=chip,timing_mode=mode)
    elapsed=(perf_counter_ns()-start)/1e6
    assert (e.latency_us is None and expected is None or
            e.latency_us is not None and expected is not None and isclose(e.latency_us,expected,rel_tol=1e-12)),(key,e,expected)
    if expected is not None:assert e.status in ('estimated','extrapolated'),(key,e)
    seen.add(key);times.setdefault(op,[]).append(elapsed)

for j in read(v/'sku/freeze.json')['jobs']:
    o=j['outputs'][0]
    for s,c in chips.items():
        if s=='pr' and o['op']=='rmsnorm':continue  # independent first freeze has its own archived prices
        verify(o['op'],o['shape'],o['attrs'],c,o[s+'_us'],'resource')
for j in read(v/'rms-confirmation/freeze.json')['jobs']:
    o=j['outputs'][0]
    for s,c in chips.items():verify(o['op'],o['shape'],o['attrs'],c,o[s+'_us'],'resource')
for r in read(root/'results/v0.18/evaluation.json')['rows']:
    if r['op'] in ('rmsnorm','quant'):continue
    for s,c in chips.items():verify(r['op'],r['shape'],r['attrs'],c,r[s+'_us'],'resource' if r['op'] in ('matmul','fa') else 'auto')
for row in read(v/'prefill/release-projections.json')['rows']:
    op,t,w=row['id'].split('_')
    for s,c in chips.items():verify(op,[int(t),int(w)],{},c,row['targets'][s]['us'],'resource')
for filename in ['router-holdout-freeze.json','cast-holdout-freeze.json']:
    for row in read(v/'workloads'/filename)['jobs']:
        for c,expected in row['predictions'].items():verify(row['op'],row['shape'],{},c,expected,'resource')

for r in read(v/'workloads/attn-res-holdout-freeze.json')['scenarios']+read(v/'workloads/attn-res-extension-freeze.json')['scenarios']:
    shape=[r['tokens'],r['hidden'],r['slots'],r['valid']]
    attrs=dict(addend=True,materialize=True,write_idx=r['valid'])
    verify('attn_res_fwd',shape,attrs,r['chip'],r['latency_us'],'resource')
for chip in chips.values():
    verify('attn_res_fwd',[9,7168,8,6],{'addend':True,'materialize':True,'write_idx':6},chip,None,'resource')

def stats(xs):
    return dict(n=len(xs),median_ms=statistics.median(xs),p95_ms=sorted(xs)[ceil(.95*len(xs))-1],max_ms=max(xs))
data=dict(version=operator_model.__version__,import_ms=import_ms,overall=stats(sum(times.values(),[])),
          per_op={op:stats(xs) for op,xs in times.items()},protocol='Installed noneditable wheel outside checkout. First query per distinct op/shape/attrs/chip. Includes monolithic prefill projections and unavailable statuses. Does not charge import to per-event query.')
(v/'wheel-smoke.json').write_text(dumps(data,indent=2)+'\n')
print(data['overall']);assert data['overall']['max_ms']<100
