"""Reprice archived complete windows and verify the installed, CA-free v0.28 API."""
from collections import defaultdict
from hashlib import sha256
import json
from math import ceil, isclose
from pathlib import Path
import sys
from time import perf_counter_ns

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.28'
FUT='Ascend970DT'

def read(path):
    return json.loads(path.read_bytes())

def write(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def distribution(times):
    t=sorted(times)
    return dict(n=len(t),p50_ms=t[(len(t)-1)//2],p95_ms=t[ceil(.95*len(t))-1],maximum_ms=t[-1])

def metrics(rows):
    return dict(windows=len(rows),shapes=len({tuple(r['shape']) for r in rows}),
                wape=100*sum(abs(r['current_us']-r['ca_us']) for r in rows)/sum(r['ca_us'] for r in rows),
                maximum=max(100*abs(r['current_us']/r['ca_us']-1) for r in rows))

def release():
    from operator_model import __version__,estimate,estimate_post_history,rmsnorm_pipeline,estimate_matmul_splitk
    import operator_model
    if not sys.flags.isolated or Path(operator_model.__file__).is_relative_to(ROOT) or __version__!='0.28.0':
        raise ValueError('run with isolated v0.28 wheel outside the checkout')
    cache={};costs=defaultdict(list)
    def query(op,shape,attrs,chip):
        key=(op,tuple(shape),json.dumps(attrs,sort_keys=True),chip)
        if key not in cache:
            start=perf_counter_ns()
            p=estimate(op,shape,attrs,chip=chip)
            costs['estimate'].append((perf_counter_ns()-start)/1e6)
            cache[key]=p
        return cache[key]
    old=read(ROOT/'results/v0.27/replay.json')
    rows=[];groups=defaultdict(list)
    for r in old['rows']:
        op={'fla_kda_v2':'kda_chunk','vllm_hc_post':'mhc_post'}.get(r['op'],r['op'])
        attrs=r['attrs'];shape=r['shape'];chip=r['chip']
        p=query(op,shape,attrs,chip);f=query(op,shape,attrs,FUT)
        for us,prev in [(p.latency_us,r['latency_us']),(f.latency_us,r['future_970dt_extrapolated_us'])]:
            if op not in ('cast_bf16_fp32','rmsnorm','matmul') and us!=prev and (not us or not prev or not isclose(us,prev,rel_tol=1e-8)):
                raise ValueError((op,shape,chip,us,prev))
        new=dict(op=r['op'],shape=shape,attrs=attrs,chip=chip,kernel=r['kernel'],ca_us=r['ca_us'],
                 previous_us=r['latency_us'],current_us=p.latency_us,future_970dt_extrapolated_us=f.latency_us,
                 status=p.status,trace_sha256=r['trace_sha256'])
        rows.append(new)
        if new['current_us'] is not None: groups[(r['op'],chip)].append(new)
    write(OUT/'replay.json',dict(role='Archived complete device windows replayed with v0.28. Development or historical only.',
          summary=[dict(op=o,chip=c,**metrics(v)) for (o,c),v in sorted(groups.items())],rows=rows))
    lines=['# 现版对历史全窗的逐形回放','',
           '每个 shape 的设备耗时单位为 µs。相同几何但不同前序可以出现多次；这些行作为回放。新冻结的计时独立列在总误差分析。','',
           '| 算子 · 实际 kernel | 芯片 | shape · 属性 | v0.27 | 当前 | CA | 当前偏差 | 970DT 外推 |','|---|---|---|---:|---:|---:|---:|---:|']
    def fmt(x):return 'N/A' if x is None else f'{x:.3f}'
    for r in rows:
        us=r['current_us'];err='N/A' if us is None else f'{100*(us/r["ca_us"]-1):+.1f}%'
        lines.append(f'| {r["op"]} · {r["kernel"]} | {r["chip"]} | {r["shape"]} · {r["attrs"]} | {fmt(r["previous_us"])} | {fmt(us)} | {fmt(r["ca_us"])} | {err} | {fmt(r["future_970dt_extrapolated_us"])} |')
    (OUT/'shape-errors.md').write_text('\n'.join(lines)+'\n')
    candidate={(tuple(r['shape']),c):v['us'] for r in read(OUT/'matmul/streamk-freeze.json')['rows'] for c,v in r['predictions'].items()}
    for r in read(OUT/'matmul/freeze.json')['rows']+read(OUT/'matmul/long-prefill/freeze.json')['rows']+read(OUT/'matmul/wide-prefill/freeze.json')['rows']:
        for c,ref in r['predictions'].items():
            p=query('matmul',r['shape'],{},c)
            if not isclose(p.latency_us,candidate.get((tuple(r['shape']),c),ref['us']),rel_tol=1e-10):raise ValueError(('matmul',r['shape'],c))
    for r in read(OUT/'matmul/streamk-freeze.json')['rows']:
        for c,target in r['predictions'].items():
            start=perf_counter_ns();v=estimate_matmul_splitk(r['shape'],chip=c)
            costs['matmul_splitk'].append((perf_counter_ns()-start)/1e6)
            assert isclose(v['latency_us'],target['us'],rel_tol=1e-10) and v['route']==target['route']
    for c,ref in read(OUT/'cast/register.json')['prices'].items():
        assert isclose(query('cast_bf16_fp32',[5632,5120],{},c).latency_us,ref['pressure']['us'])
    for r in read(OUT/'rms/freeze.json')['rows']:
        for known in [False,True]:
            start=perf_counter_ns()
            opts=dict(resident_aiv=r['prior_aiv'],immediate_repeat=r['role']=='repeat') if known else {}
            t=rmsnorm_pipeline(r['shape'],chip=r['chip'],**opts)
            costs['rmsnorm_pipeline'].append((perf_counter_ns()-start)/1e6)
            assert isclose(t['us'],r['specific_context_us' if known else 'us'],abs_tol=1e-9)
    for c in ('Ascend950DT_9582','Ascend950PR_957b',FUT):
        for shape in ([1,4,192],[29,4,320],[157,4,1088]):
            start=perf_counter_ns();h=estimate_post_history(shape,chip=c)
            v=h.for_predecessor(h.active_aiv,previous_shape=shape).latency_us
            assert v==h.expected_us({'repeat':1.})
            costs['post_history'].append((perf_counter_ns()-start)/1e6)
    dist={k:distribution(v) for k,v in costs.items()}
    assert max(s['maximum_ms'] for s in dist.values())<100
    wheel,=(ROOT/'build/wheels/v028').glob('*.whl')
    write(OUT/'wheel-check.json',dict(version=__version__,wheel_sha256=sha256(wheel.read_bytes()).hexdigest(),
          checked_history_rows=len(rows),cost=dist,scope='wheel installed outside checkout, first distinct estimate per input; duration measured around one API call after import'))
    print(dist)

if __name__=='__main__':release()
