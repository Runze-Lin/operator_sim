"""Run with the installed wheel's isolated Python, outside the repository."""
from pathlib import Path
from time import perf_counter_ns
from math import isclose
import json
import statistics
import sys

root = Path(__file__).resolve().parents[3]
out = root/'results/v0.17'
start = perf_counter_ns()
import operator_model
from operator_model import estimate
import_ms = (perf_counter_ns()-start)/1e6
assert not Path(operator_model.__file__).is_relative_to(root)
totals = {}
for group in ('core','extension'):
    freeze = json.loads((out/'sku'/f'{group}-freeze.json').read_text())
    for job in freeze['jobs']:
        for case in job['outputs']:
            op=case['op']
            for key,chip in [('dt','Ascend950DT_9582'),('pr','Ascend950PR_957b'),('future','Ascend970DT')]:
                start=perf_counter_ns()
                r=estimate(op,case['shape'],case['attrs'],chip=chip,
                           timing_mode='resource' if op in ('quant','matmul','fa') else 'auto')
                ms=(perf_counter_ns()-start)/1e6
                expected=case[key+'_us']
                match=(r.latency_us is None) if expected is None else isclose(r.latency_us,expected,rel_tol=1e-12)
                assert match and r.status==case[key+'_status'],(op,chip,case['shape'],r)
                totals.setdefault(op,[]).append(ms)
all_times=sorted(x for xs in totals.values() for x in xs)
summary=lambda xs:dict(n=len(xs),min_ms=min(xs),median_ms=statistics.median(xs),
                       p95_ms=sorted(xs)[int((len(xs)-1)*.95)],max_ms=max(xs))
data=dict(version=operator_model.__version__,import_ms=import_ms,overall=summary(all_times),
          per_op={k:summary(v) for k,v in totals.items()},protocol='Single resident interpreter, one call per frozen operator/shape and chip, includes first calls and chip/profile loading. Package import reported separately. No CANN, CA or GPU call.',
          environment=sys.version,matches_freeze_within_relative_1e_12=True)
(out/'wheel-smoke.json').write_text(json.dumps(data,indent=2)+'\n')
print(data['overall'])
