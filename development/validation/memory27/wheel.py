"""Replay the installed wheel on distinct operator inputs and validate the public Kimi selector."""
from hashlib import sha256
import json
from math import ceil, isclose
from pathlib import Path
from statistics import median
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/v0.27'


def load(path):
    return json.loads((ROOT/path).read_bytes())


def metrics(times):
    ordered=sorted(times)
    return dict(n=len(times), median_ms=median(times), p95_ms=ordered[ceil(.95*len(ordered))-1],
                maximum_ms=ordered[-1])


def main():
    start=perf_counter_ns()
    import operator_model as api
    import_ms=(perf_counter_ns()-start)/1e6
    assert sys.flags.isolated and api.__version__=='0.27.0' and not Path(api.__file__).is_relative_to(ROOT)
    cases={}
    def add(op,shape,chip,attrs,us):
        key=(op,tuple(shape),chip,json.dumps(attrs,sort_keys=True))
        if key in cases and not isclose(us,cases[key],rel_tol=1e-7):
            raise ValueError('conflicting reference')
        cases[key]=us
    fused = {(r['sku'],tuple(r['shape'])):r['candidate_us']
             for r in load('results/v0.26/presink/replay.json')['rows']}
    for job in load('results/v0.27/fusion/register.json')['jobs']:
        for chip, ref in job['predictions'].items():
            fused[chip,tuple(job['shape'])] = ref['candidate_us']
            add('mhc_pre_sinkhorn',job['shape'],chip,job['attrs'],ref['candidate_us'])
    for r in load('results/v0.27/fusion/public-sink-installed.json')['rows']:
        add('mhc_sinkhorn',r['shape'],r['chip'],
            {'resident_aiv':r['resident_aiv'],'num_iters':r['iters']},r['frozen_us'])
    for r in load('results/v0.25/coverage/audit.json')['rows']:
        op={'fla_kda_v2':'kda_chunk', 'vllm_hc_post':'mhc_post'}.get(r['op'],r['op'])
        if r['current_us'] is not None:
            value = r['current_us']
            if op == 'mhc_pre_sinkhorn':
                value = fused.get((r['chip'],tuple(r['shape'])),value)
            add(op,r['shape'],r['chip'],r['attrs'],value)
        if r['future_970dt_us'] is not None:
            add(op,r['shape'],'Ascend970DT',r['attrs'],r['future_970dt_us'])
    for r in load('results/v0.22/evaluation.json')['rows']:
        if r['op'] in ('rmsnorm','mhc_post'):
            add(r['op'],r['shape'],r['chip'],r['attrs'],r['us'])
            add(r['op'],r['shape'],'Ascend970DT',r['attrs'],r['future_us'])
    for r in load('results/v0.25/coverage/supplement/score.json')['rows']:
        add('fa',r['shape'],r['sku'],{},r['candidate'])
    for r in load('results/v0.26/memory/pro-256ki-prefix-forecast.json')['cases'].items():
        chip, forecast = r
        add('cast_bf16_fp32',[2097152,64],chip,{},forecast['totals']['uniform']['us'])
    times=[]
    for (op,shape,chip,attrs),value in cases.items():
        options=json.loads(attrs)
        start=perf_counter_ns()
        pred=api.estimate(op,shape,options,chip=chip)
        times.append((perf_counter_ns()-start)/1e6)
        if pred.latency_us is None or not isclose(pred.latency_us,value,rel_tol=1e-7):
            raise ValueError((op,shape,chip,value,pred.latency_us))
    routing=[]
    for r in load('results/v0.27/kda/dispatch.json')['rows']:
        for chip, record in r['targets'].items():
            start=perf_counter_ns()
            result=api.plan_kimi_kda_prefill(r['lengths'],attention_tp=r['tp'],chip=chip)
            routing.append((perf_counter_ns()-start)/1e6)
            expected=record['plan']
            for key in ('aclnn','work_items','minimum_payload_bytes','main_kernels_us','status','full_dispatch_us'):
                assert result[key]==expected[key], (r['tag'],chip,key)
    assert max(times+routing)<100
    whl,=(ROOT/'build/wheels/v027').glob('*.whl')
    out=dict(version=api.__version__, wheel_sha256=sha256(whl.read_bytes()).hexdigest(),
             python=sys.version.split()[0],import_ms=import_ms,unique_predictions=metrics(times),
             kimi_dispatch=metrics(routing),scope='first call per distinct input in one isolated process; import and plotting excluded')
    (OUT/'wheel-check.json').write_text(json.dumps(out,indent=2)+'\n')
    print(out)


if __name__=='__main__':
    main()
