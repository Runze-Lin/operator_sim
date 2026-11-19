"""Exercise the installed distribution against registered prices without a live CANN installation."""
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
from statistics import median
import sys
from time import perf_counter_ns

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.25'
DT,PR,FUTURE='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'


def read(p):
    return json.loads(p.read_text())


def timing(times):
    return dict(n=len(times),median_ms=median(times),p95_ms=sorted(times)[math.ceil(.95*len(times))-1],max_ms=max(times))


def run():
    start=perf_counter_ns()
    from operator_model import OperatorStream, __file__ as location, estimate
    import_ms=(perf_counter_ns()-start)/1e6
    assert sys.flags.isolated and not Path(location).is_relative_to(ROOT) and version('ascend-operator-model')=='0.25.0'
    seen={}
    by_route={}
    def check(op,shape,attrs,chip,expected=None):
        key=op,tuple(shape),json.dumps(attrs,sort_keys=True),chip
        if key not in seen:
            t=perf_counter_ns()
            est=estimate(op,shape,attrs,chip=chip)
            ms=(perf_counter_ns()-t)/1e6
            assert ms<100 and est.latency_us is not None,(key,ms,est.reason)
            seen[key]=est.latency_us
            by_route.setdefault(attrs.get('implementation',op),[]).append(ms)
        if expected is not None:
            assert math.isclose(seen[key],expected,rel_tol=1e-9),(key,seen[key],expected)
    for r in read(ROOT/'results/v0.22/evaluation.json')['rows']:
        check(r['op'],r['shape'],r['attrs'],r['chip'],None if r['op']=='fa' else r['us'])
        check(r['op'],r['shape'],r['attrs'],FUTURE,r['future_us'])
    for r in read(OUT/'coverage/audit.json')['rows']:
        if r['op']=='fa':
            check('fa',r['shape'],r['attrs'],r['chip'],r['current_us'])
            check('fa',r['shape'],r['attrs'],FUTURE,r['future_970dt_us'])
    for r in read(ROOT/'results/v0.24/kda-prefetch/supplement/comparison.json')['rows']:
        check('kda_chunk',r['shape'],{'implementation':'fla_26_9_1_v2'},r['sku'],r['candidate'][-1])
    for chip in (DT,PR,FUTURE):
        for seq in (32768,131072,262144):
            for op,shape,attrs in [('kda_chunk',[8,seq,12,12,128,128],{'implementation':'fla_26_9_1_v2'}),
                    ('rmsnorm',[8*seq,7168],{}),('quant',[8*seq,5120],{}),('cast_bf16_fp32',[8*seq,64],{})]:
                check(op,shape,attrs,chip)
    stream_ms=[]
    for role in ('development','independent'):
        plan=read(OUT/f'mixed-stream/{role}-freeze.json')
        for chip in (DT,PR,FUTURE):
            for group in plan['groups']:
                stream=OperatorStream(chip)
                for r in plan['rows']:
                    if (r['chip'],r['group'])!=(chip,group):
                        continue
                    cube=r['op']=='addmm'
                    attrs={'self_shape':[r['shape'][1]],'alpha':1,'beta':1} if cube else {}
                    t=perf_counter_ns()
                    est=stream.advance(r['op'],r['shape'],attrs,aic_only=cube)
                    ms=(perf_counter_ns()-t)/1e6
                    assert ms<100 and math.isclose(est.latency_us,r['preserve_us'],rel_tol=1e-9),(r,est)
                    stream_ms.append(ms)
    lower_ms=[]
    plan=read(ROOT/'results/v0.24/rms-prefix/freeze.json')
    for chip in (DT,PR,FUTURE):
        for group in plan['groups']:
            stream=OperatorStream(chip)
            stream.forget_history()
            for r in plan['rows']:
                if (r['chip'],r['group'])!=(chip,group) or r['role']=='hidden':
                    continue
                t=perf_counter_ns()
                est=stream.advance('rmsnorm',r['shape'])
                ms=(perf_counter_ns()-t)/1e6
                assert ms<100 and math.isclose(est.latency_us,r['partial_us'],rel_tol=1e-9)
                lower_ms.append(ms)
    wheel,=(ROOT/'build/wheels/v025').glob('*.whl')
    result=dict(version=version('ascend-operator-model'),python=sys.version.split()[0],import_ms=import_ms,
                wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),predictions=timing(sum(by_route.values(),[])),
                per_route={name:timing(vals) for name,vals in by_route.items()},mixed=timing(stream_ms),partial=timing(lower_ms))
    (OUT/'wheel-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:result[k] for k in ('predictions','mixed','partial')})


if __name__=='__main__':
    run()
