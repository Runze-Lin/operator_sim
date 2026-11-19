"""Check the installed wheel against immutable predictions from the release campaigns."""
from hashlib import sha256
from importlib.metadata import version
import json
from math import ceil, isclose
from pathlib import Path
from statistics import median
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/v0.24'
CHIPS = ('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT')


def read(p):
    return json.loads(p.read_bytes())


def stats(values):
    return dict(calls=len(values),median_ms=median(values),p95_ms=sorted(values)[ceil(.95*len(values))-1],max_ms=max(values))


def main():
    tick=perf_counter_ns()
    from operator_model import OperatorStream, __file__ as source, estimate
    import_ms=(perf_counter_ns()-tick)/1e6
    assert sys.flags.isolated and not Path(source).is_relative_to(ROOT) and version('ascend-operator-model')=='0.24.0'
    seen,by_op={},{}
    def check(op,shape,attrs,chip,expected=None):
        key=op,tuple(shape),json.dumps(attrs,sort_keys=True),chip
        if key not in seen:
            tick=perf_counter_ns()
            result=estimate(op,shape,attrs,chip=chip)
            ms=(perf_counter_ns()-tick)/1e6
            assert ms<100 and result.status!='unsupported',(key,result.reason,ms)
            seen[key]=result.latency_us
            by_op.setdefault(attrs.get('implementation',op),[]).append(ms)
        if expected is not None:
            assert isclose(seen[key],expected,rel_tol=1e-9),(key,seen[key],expected)
    for r in read(ROOT/'results/v0.22/evaluation.json')['rows']:
        for chip,pred in [(r['chip'],r['us']),(CHIPS[-1],r['future_us'])]:
            check(r['op'],r['shape'],r['attrs'],chip,pred)
    for r in read(OUT/'kda-prefetch/comparison.json')['rows']:
        check('kda_chunk',r['shape'],{'implementation':'fla_26_9_1_v2'},r['sku'],r['candidate'][-1])
        check('kda_chunk',r['shape'],{'implementation':'fla_26_9_1_v2'},CHIPS[-1],r['prediction_970_us'])
    for r in read(ROOT/'results/v0.23/hc-timing/supplement-register.json')['cases']+read(ROOT/'results/v0.23/hc-timing/holdout-register.json')['cases']:
        for chip,expected in r['predictions'].items():
            check('mhc_post',r['shape'],{'implementation':'vllm_hc_post'},chip,expected['device_us'] if r['shape'][-1]%4096>=64 or r['shape'][-1]%4096==0 else None)
    for chip in CHIPS:
        for seq in (32768,131072,262144):
            for op,shape,attrs in [('kda_chunk',[8,seq,12,12,128,128],{'implementation':'fla_26_9_1_v2'}),
                                   ('rmsnorm',[8*seq,7168],{}),('quant',[8*seq,5120],{}),('cast_bf16_fp32',[8*seq,64],{})]:
                check(op,shape,attrs,chip)
    partial=[]
    plan=read(OUT/'rms-prefix/freeze.json')
    for chip in CHIPS:
        for group in plan['groups']:
            stream=OperatorStream(chip)
            stream.forget_history()
            for r in plan['rows']:
                if (r['chip'],r['group'])!=(chip,group) or r['role']=='hidden':
                    continue
                tick=perf_counter_ns()
                est=stream.advance('rmsnorm',r['shape'])
                ms=(perf_counter_ns()-tick)/1e6
                assert ms<100 and isclose(est.latency_us,r['partial_us'],rel_tol=1e-9)
                partial.append(ms)
    wheel,=(ROOT/'build/wheels/v024').glob('*.whl')
    data=dict(version=version('ascend-operator-model'),python=sys.version.split()[0],wheel_sha256=sha256(wheel.read_bytes()).hexdigest(),
              python_import_ms=import_ms,predictions=stats(sum(by_op.values(),[])),per_route={op:stats(t) for op,t in by_op.items()},partial_stream=stats(partial))
    (OUT/'wheel-check.json').write_text(json.dumps(data,indent=2)+'\n')
    print(data['predictions'],data['partial_stream'])

if __name__=='__main__':
    main()
