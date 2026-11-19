"""Smoke the release installed outside the checkout, including the public long-stream diagnostics."""
from hashlib import sha256
import json
from math import ceil, isclose
from pathlib import Path
from statistics import median
import sys
from time import perf_counter_ns

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.26'


def read(path):
    return json.loads(path.read_text())


def main():
    start=perf_counter_ns()
    import operator_model as api
    import_ms=(perf_counter_ns()-start)/1e6
    assert api.__version__=='0.26.0' and sys.flags.isolated and not Path(api.__file__).is_relative_to(ROOT)
    samples={}
    def check(op,shape,attrs,sku,expected=None):
        key=(op,tuple(shape),json.dumps(attrs,sort_keys=True),sku)
        if key not in samples:
            start=perf_counter_ns()
            result=api.estimate(op,shape,attrs,chip=sku)
            ms=(perf_counter_ns()-start)/1e6
            assert ms<100 and result.latency_us is not None,(key,ms,result.reason)
            samples[key]=(ms,result.latency_us)
        if expected is not None and not isclose(samples[key][1],expected,rel_tol=1e-9):
            raise AssertionError((key,samples[key][1],expected))
    future='Ascend970DT'
    for r in read(ROOT/'results/v0.22/evaluation.json')['rows']:
        check(r['op'],r['shape'],r['attrs'],r['chip'],None if r['op']=='fa' else r['us'])
        check(r['op'],r['shape'],r['attrs'],future,r['future_us'])
    for r in read(ROOT/'results/v0.25/coverage/audit.json')['rows']:
        if r['op']=='fa':
            check('fa',r['shape'],r['attrs'],r['chip'],r['current_us'])
            check('fa',r['shape'],r['attrs'],future,r['future_970dt_us'])
    for r in read(ROOT/'results/v0.25/coverage/supplement/score.json')['rows']:
        check('fa',r['shape'],{},r['sku'],r['candidate'])
    for r in read(ROOT/'results/v0.24/kda-prefetch/supplement/comparison.json')['rows']:
        check('kda_chunk',r['shape'],{'implementation':'fla_26_9_1_v2'},r['sku'],r['candidate'][-1])
    for r in read(OUT/'memory/initial-issues-freeze.json')['cases'].values():
        check('cast_bf16_fp32',r['shape'],{},r['sku'],r['device_us'])
        e=api.estimate('cast_bf16_fp32',r['shape'],chip=r['sku'])
        assert e.projection['timing_diagnostics']['initial_read_issue_us']==r['issues_us']
    for sku in ('Ascend950DT_9582','Ascend950PR_957b',future):
        for seq in (32768,131072,262144):
            for op,shape,a in [('kda_chunk',[8,seq,12,12,128,128],{'implementation':'fla_26_9_1_v2'}),
                                ('rmsnorm',[8*seq,7168],{}),('quant',[8*seq,5120],{}),('cast_bf16_fp32',[8*seq,64],{})]:
                check(op,shape,a,sku)
    def timings(xs):
        xs=sorted(xs)
        return dict(n=len(xs),median_ms=median(xs),p95_ms=xs[ceil(.95*len(xs))-1],max_ms=xs[-1])
    times=[v[0] for v in samples.values()]
    whl,=(ROOT/'build/wheels/v026').glob('*.whl')
    result=dict(version=api.__version__,python=sys.version.split()[0],import_ms=import_ms,wheel_sha256=sha256(whl.read_bytes()).hexdigest(),
                predictions=timings(times),by_op={op:timings(v[0] for k,v in samples.items() if k[0]==op) for op in sorted({k[0] for k in samples})})
    (OUT/'wheel-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['predictions'])


if __name__=='__main__':
    main()
