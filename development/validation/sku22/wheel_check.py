"""Verify public source/installed parity and time distinct requests, including stateful streams."""
import json
import math
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from statistics import median
import sys
from time import perf_counter_ns

ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/v0.22'

def main():
    start=perf_counter_ns()
    from operator_model import estimate, OperatorStream, __file__ as module
    import_ms=(perf_counter_ns()-start)/1.e6
    assert sys.flags.isolated and not Path(module).resolve().is_relative_to(ROOT) and version('ascend-operator-model')=='0.22.0'
    data=json.loads((OUT/'evaluation.json').read_text());lookup={};wall={};chips=['Ascend950DT_9582','Ascend950PR_957b','Ascend970DT']
    def check(op,s,a,c,expected=None):
        key=(op,tuple(s),json.dumps(a,sort_keys=True),c)
        if key not in lookup:
            t=perf_counter_ns();v=estimate(op,s,a,chip=c);elapsed=(perf_counter_ns()-t)/1.e6
            assert v.status!='unsupported' and elapsed<100,(key,v.reason,elapsed)
            lookup[key]=v.latency_us;wall.setdefault(op,[]).append(elapsed)
        if expected is not None:assert math.isclose(lookup[key],expected,abs_tol=1.e-8),(key,lookup[key],expected)
    for r in data['rows']:
        check(r['op'],r['shape'],r['attrs'],r['chip'],r['us']);check(r['op'],r['shape'],r['attrs'],chips[-1],r['future_us'])
    rms=json.loads((ROOT/'results/rms-dispatch-audit/freeze.json').read_text());stream_times=[]
    for c in chips:
        stream=OperatorStream(c)
        for r in sorted((r for r in rms['rows'] if r['chip']==c and r['protocol']=='ordered new sequence'),key=lambda r:r['i']):
            check('rmsnorm',r['shape'],{'resident_aiv':r['prior_aiv']},c,r['candidate_known_us']);t=perf_counter_ns();y=stream.advance('rmsnorm',r['shape']);elapsed=(perf_counter_ns()-t)/1.e6
            assert elapsed<100 and math.isclose(y.latency_us,r['candidate_known_us'],abs_tol=1.e-8);stream_times.append(elapsed)
        for n in (8*32768,8*131072,8*262144):
            for op,s in [('cast_bf16_fp32',[n,32]),('cast_bf16_fp32',[n,5120]),('sigmoid_fp32',[n,12]),('quant',[n,7168]),('rmsnorm',[n,5120]),('matmul',[n,32,5120])]:check(op,s,{},c)
        check('kda_chunk',[1,64,2,2,128,128],{},c)
    def stats(g):return dict(n=len(g),median_ms=median(g),p95_ms=sorted(g)[math.ceil(.95*len(g))-1],max_ms=max(g))
    wheel,=(ROOT/'build/wheels/v022').glob('*.whl');out=dict(version=version('ascend-operator-model'),import_ms=import_ms,wheel_sha256=sha256(wheel.read_bytes()).hexdigest(),
            per_op={k:stats(v) for k,v in wall.items()},overall=stats(sum(wall.values(),[])),ordered=stats(stream_times))
    (OUT/'wheel-check.json').write_text(json.dumps(out,indent=2)+'\n');print(out['overall'],out['ordered'])

if __name__=='__main__':main()
