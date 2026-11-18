"""Run with an isolated installed interpreter and the path to the frozen manifest."""
from dataclasses import replace
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
from statistics import median
import sys
from time import perf_counter_ns


def main(freeze_path):
    freeze = json.loads(freeze_path.read_text())
    begin = perf_counter_ns()
    import operator_model
    from operator_model import load_chip, estimate
    import_ms = (perf_counter_ns()-begin)/1e6
    assert operator_model.__version__ == '0.16.0' and 'site-packages' in operator_model.__file__
    pr = load_chip('Ascend950PR_957b')
    scenario = replace(pr,name='anonymous-identical-config')
    checks={}
    for op in ('quant','matmul','fa'):
        rows=[r for r in freeze['predictions'] if r['op']==op and r['priority']=='primary']
        times=[]
        for turn in range(40):
            for row in rows:
                start=perf_counter_ns()
                predicted=estimate(op,row['shape'],chip=pr)
                times.append((perf_counter_ns()-start)/1e6)
                assert abs(predicted.latency_us-row['pr_us'])<1e-9
                if not turn:
                    assert predicted.work==row['work']
                    assert abs(estimate(op,row['shape'],chip=scenario).latency_us-row['pr_us'])<1e-9
                    assert abs(estimate(op,row['shape'],chip=pr,timing_mode='calibrated').latency_us-row['prior_pr_us'])<1e-9
        assert max(times)<100
        checks[op]=dict(shapes=len(rows),calls=len(times),first_ms=times[0],
                        median_ms=median(times),p95_ms=sorted(times)[ceil(.95*len(times))-1],max_ms=max(times))
    return dict(version=operator_model.__version__,import_ms=import_ms,checks=checks,
                freeze_sha256=sha256(freeze_path.read_bytes()).hexdigest(),
                scope='In-process estimate with loaded chip; imports reported separately; no shape latency cache.')


if __name__=='__main__':
    print(json.dumps(main(Path(sys.argv[1])),indent=2))
