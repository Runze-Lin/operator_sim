"""Time ordinary estimates from an isolated installed wheel and compare frozen forecasts."""
import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from math import ceil,isclose
from pathlib import Path
from statistics import median
import sys
from time import perf_counter_ns

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.21'
CHIPS=('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT')
SIDES=dict(zip(CHIPS,('dt','pr','future')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT/'wheel-check.json',
                        help='Write a separate timing receipt when replaying a larger archive.')
    report_path=parser.parse_args().output
    start=perf_counter_ns()
    from operator_model import estimate, __version__, __file__ as source
    import_ms=(perf_counter_ns()-start)/1e6
    assert sys.flags.isolated and not Path(source).resolve().is_relative_to(ROOT)
    assert version('ascend-operator-model')==__version__=='0.21.0'
    seen={}; wall={}
    def check(op,shape,attrs,chip,frozen=None):
        key=(op,tuple(shape),json.dumps(attrs,sort_keys=True),chip)
        if key not in seen:
            start=perf_counter_ns()
            r=estimate(op,shape,attrs,chip=chip)
            ms=(perf_counter_ns()-start)/1e6
            assert ms<100 and r.status!='unsupported',(op,shape,r.reason,ms)
            seen[key]=r.latency_us
            wall.setdefault(op,[]).append(ms)
        if frozen is not None:
            assert isclose(seen[key],frozen,abs_tol=1e-7),(op,shape,chip,seen[key],frozen)
    def read(path):return json.loads(path.read_text())
    for r in read(OUT/'evaluation.json')['rows']:
        check(r['op'],r['shape'],r.get('attrs',{}),r['chip'],r['us'])
        check(r['op'],r['shape'],r.get('attrs',{}),CHIPS[2],r['future_us'])
    for folder in ('rms','dispatch','vector'):
        for j in read(OUT/folder/'freeze.json')['jobs']:
            for c in CHIPS:
                frozen=j['targets'][c]['us'] if folder=='vector' else j['outputs'][0][SIDES[c]+'_us']
                check(j['op'],j['shape'],j.get('attrs',{}),c,frozen)
    for j in read(OUT/'fusion/freeze.json')['cases']:
        check(j['op'],j['shape'],j['attrs'],j['chip'],j['latency_us'])
    for c in CHIPS:
        check('kda_chunk',[1,64,2,2,128,128],{},c)
    for tokens in (8*32768,8*131072,8*262144):
        for c in CHIPS:
            for op,shape in [('cast_bf16_fp32',[tokens,32]),('sigmoid_fp32',[tokens,12]),
                             ('quant',[tokens,7168]),('rmsnorm',[tokens,5120]),('matmul',[tokens,32,5120])]:
                check(op,shape,{},c)
    def stats(values):
        return dict(n=len(values),median_ms=median(values),p95_ms=sorted(values)[ceil(.95*len(values))-1],max_ms=max(values))
    wheel,=(ROOT/'build/wheels/v021').glob('ascend_operator_model-0.21.0-*.whl')
    out=dict(version=version('ascend-operator-model'),wheel_sha256=sha256(wheel.read_bytes()).hexdigest(),
        import_ms=import_ms,per_op={k:stats(v) for k,v in wall.items()},overall=stats(sum(wall.values(),[])))
    report_path.write_text(json.dumps(out,indent=2)+'\n')
    print(out['overall'])


if __name__=='__main__':
    main()
