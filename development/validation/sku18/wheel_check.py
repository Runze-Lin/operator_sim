"""Use the isolated installed wheel, outside the checkout, on the entire current suite."""
import json
from math import isclose
from pathlib import Path
import statistics
import sys
from time import perf_counter_ns

root=Path(__file__).resolve().parents[3]
out=root/'results/v0.18'
def load(path):
    return json.loads(path.read_text())

cases=[o for j in load(out/'sku/freeze.json')['jobs'] for o in j['outputs']]
cases.extend(r for r in load(root/'results/v0.17/evaluation.json')['rows'] if r['op'] not in ('quant','rmsnorm'))
start=perf_counter_ns()
from operator_model import __version__,estimate
import operator_model
import_ms=(perf_counter_ns()-start)/1e6
assert __version__=='0.18.0' and not Path(operator_model.__file__).is_relative_to(root)
times={}
for case in cases:
    op=case['op']
    for side,chip in [('dt','Ascend950DT_9582'),('pr','Ascend950PR_957b'),('future','Ascend970DT')]:
        start=perf_counter_ns()
        e=estimate(op,case['shape'],case['attrs'],chip=chip,timing_mode='resource' if op in ('quant','matmul','fa','rmsnorm') else 'auto')
        ms=(perf_counter_ns()-start)/1e6
        times.setdefault(op,[]).append(ms)
        expected=case[side+'_us']
        assert e.status==case[side+'_status'] and (e.latency_us is None if expected is None else isclose(e.latency_us,expected,rel_tol=1e-12)),(op,chip,case['shape'])
def stats(xs):
    return dict(n=len(xs),median_ms=statistics.median(xs),p95_ms=sorted(xs)[int(.95*(len(xs)-1))],max_ms=max(xs))
data=dict(version=__version__,import_ms=import_ms,overall=stats(sum(times.values(),[])),
          per_op={op:stats(xs) for op,xs in times.items()},environment=sys.version,
          protocol='Installed wheel; first and subsequent queries in a single process on 151 contracts and all three chips; fresh configs loaded by name; import excluded.')
(out/'wheel-smoke.json').write_text(json.dumps(data,indent=2)+'\n')
print(data['overall'])
