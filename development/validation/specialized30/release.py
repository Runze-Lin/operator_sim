"""Exercise the installed model with historical inputs, conditional contracts and new scenarios."""
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import ceil, isclose
from pathlib import Path
from statistics import median
import sys
from time import perf_counter_ns

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.30'
OLD=ROOT/'results/v0.29'
FUT='Ascend970DT'
CHIPS=('Ascend950DT_9582','Ascend950PR_957b')


def read(p):return json.loads(p.read_bytes())
def save(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def digest(p):return sha256(p.read_bytes()).hexdigest()


def same(a,b):
    if a is None or b is None:
        assert a is b,(a,b)
    else:assert isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)


def check():
    if not sys.flags.isolated: raise RuntimeError('Use python -I release.py <installed-prefix>')
    sys.path.insert(0,str(Path(sys.argv[1]).resolve()))
    import operator_model as api
    assert api.__version__=='0.30.0' and not Path(api.__file__).is_relative_to(ROOT)
    calls=defaultdict(list); cache={}; rows=[]
    def timed(name,fn,*args,**kwargs):
        start=perf_counter_ns()
        result=fn(*args,**kwargs)
        calls[name].append((perf_counter_ns()-start)/1e6)
        return result
    def public(op,shape,attrs,chip):
        key=(op,tuple(shape),json.dumps(attrs,sort_keys=True),chip)
        if key not in cache:cache[key]=timed('estimate',api.estimate,op,shape,attrs,chip=chip)
        return cache[key]
    for old in read(OLD/'replay.json')['rows']:
        name={'fla_kda_v2':'kda_chunk','vllm_hc_post':'mhc_post'}.get(old['op'],old['op'])
        shape,attrs=old['shape'],old['attrs']
        r=public(name,shape,attrs,old['chip']); future=public(name,shape,attrs,FUT)
        if old['op']!='mhc_sinkhorn':
            same(r.latency_us,old['current_us']);same(future.latency_us,old['future_970dt_extrapolated_us'])
        rows.append(dict(old,previous_us=old['current_us'],current_us=r.latency_us,
                         status=r.status,future_970dt_extrapolated_us=future.latency_us))
    for r in read(OUT/'sink/score.json')['rows']:
        attrs={'num_iters':r['iters']}
        same(public('mhc_sinkhorn',[r['t'],4],attrs,r['sku']).latency_us,r['public_us'])
    for r in read(OUT/'matmul/freeze.json')['rows']:
        q=timed('matmul_supply',api.estimate_matmul_supply,r['shape'],chip=r['chip'])
        same(q['latency_us'],r['candidate']['latency_us'])
    for r in read(OUT/'matmul-dispatch/device/freeze.json')['rows']:
        for chip,p in r['predictions'].items():
            q=public('matmul',r['shape'],{},chip)
            same(q.latency_us,p['us']);same(q.active_aic,p['public_active_aic'])
            same(timed('matmul_supply',api.estimate_matmul_supply,r['shape'],chip=chip)['latency_us'],p['supply_us'])
    for r in read(OUT/'matmul-dispatch/freeze.json')['rows']:
        same(public('matmul',r['shape'],{},r['soc']).latency_us,r['public_us'])
        same(public('matmul',r['shape'],{},FUT).latency_us,r['future_public_us'])
    for chip in (*CHIPS,FUT):
        st=api.OperatorStream(chip,track_post_history=True,experimental_post_fragments=True)
        timed('ordered_post',st.advance,'mhc_post',[121,4,1216])
        q=timed('ordered_post',st.advance,'mhc_post',[65,4,5120])
        if chip in CHIPS:
            ref,=[r for r in read(OUT/'post/score.json')['rows'] if r['chip']==chip and r['shape']==[65,4,5120] and r['previous_shape']==[121,4,1216] and r['decision']!='public ordered']
            same(q.latency_us,ref['candidate_us'])
    measurements={k:dict(n=len(v),p50_ms=median(v),p95_ms=sorted(v)[ceil(.95*len(v))-1],maximum_ms=max(v))
                  for k,v in calls.items()}
    assert all(t['maximum_ms']<100 for t in measurements.values()),measurements
    wheel,=(ROOT/'build/wheels/v030').glob('*.whl')
    save(OUT/'wheel-check.json',dict(version=api.__version__,wheel_sha256=digest(wheel),historical_rows=len(rows),
                                    unique_public_inputs=len(cache),cost_ms=measurements,
                                    timer='perf_counter_ns, full in-process API on first encounter of a query; engine may reuse parsed chip/parameter files.'))
    save(OUT/'replay.json',dict(role='Installed public model historical replay. Forward validation only where sourced separately.',rows=rows))


def snapshot():
    previous=read(OLD/'evaluation.json')
    table=previous['summary']
    # Include every new qualified full-device window once. SHA tuples also allow
    # future multi-stage records to remain distinguishable from one-stage calls.
    archived=read(OLD/'matmul/score.json')['rows']
    new=read(OUT/'matmul-dispatch/device/score.json')['rows']
    for chip in CHIPS:
        unique={tuple(r['raw_sha256']):(tuple(r['shape']),r['old_us'],r['ca_us'])
                for r in archived if r['chip']==chip}
        for r in new:
            if r['chip']==chip and r['status']=='qualified':
                unique[(r['trace'],)]=(tuple(r['shape']),r['public_us'],r['ca_us'])
        values=list(unique.values())
        errors=[abs(p-ca) for _,p,ca in values]
        wape=100*sum(errors)/sum(ca for _,_,ca in values)
        maximum=max(100*error/ca for error,(_,_,ca) in zip(errors,values))
        table['matmul'][chip]=dict(n=len(values),shapes=len({s for s,_,_ in values}),wape=wape,
                                   maximum=maximum,within_limits=wape<=25 and maximum<=50)
    # All non-Sinkhorn default prices must reproduce v0.29 in check(). Retain distinct protocol denominators.
    changed=read(OUT/'sink/score.json')
    groups=changed['results']
    for chip in CHIPS:
        p=groups[chip]['all']['public_us']
        table['mhc_sinkhorn'][chip]=dict(n=p['priced'],shapes=p['shapes'],wape=p['wape'],maximum=p['max_pct'],
                    unpriced=p['windows']-p['priced'],within_limits=p['wape']<=25 and p['max_pct']<=50)
        p=read(OUT/'post/score.json')['summary'][chip]['all']
        score=p['candidate_us']
        table.setdefault('mhc_post · experimental fragment stream',{})[chip]=dict(n=p['windows'],shapes=p['shapes'],
                   **score,within_limits=score['wape']<=25 and score['maximum']<=50)
    signatures={}; registers=[]
    sources=sorted((ROOT/'operator_model').rglob('*.py'))
    for line in (ROOT/'MANIFEST.in').read_text().splitlines():
        if line.startswith('include '):sources.append(ROOT/line.split(' ',1)[1])
    for p in sorted(OUT.rglob('*.json')):
        if p.name in ('freeze.json','channel-register.json'):
            registers.append(str(p.relative_to(ROOT)));sources.append(p)
    for p in sources:signatures[str(p.relative_to(ROOT))]=digest(p)
    save(OUT/'model.json',dict(version='0.30.0',baseline='0.29.0',hashes=signatures,registrations=registers))
    save(OUT/'evaluation.json',dict(utc=datetime.now(timezone.utc).isoformat(),model_sha256=digest(OUT/'model.json'),
        limits=previous['limits'],summary=table,legacy_source='../v0.29/evaluation.json',current_wheel='wheel-check.json',
        changed_sources={'matmul':['../v0.29/matmul/score.json','matmul-dispatch/device/score.json'],
                         'mhc_sinkhorn':'sink/score.json','mhc_post · experimental fragment stream':'post/score.json'},
        role='Archive replay, by contract and time provider. Previous qualified archive retained for unchanged defaults. Current independent, pending and host-only work listed separately.'))
    lines=['# 全量已归档设备窗与当前模型','',
           '| 算子、实现与状态合同 | 950DT 形数 / 窗数 · WAPE / 最大 | 950PR 形数 / 窗数 · WAPE / 最大 | 两端归档结果 |',
           '|---|---:|---:|---|']
    for name,group in sorted(table.items()):
        vals=[]
        for chip in CHIPS:
            p=group[chip]
            vals.append(f"{p['shapes']} / {p['n']} · {p['wape']:.2f}% / {p['maximum']:.2f}%" if p['n'] else '0 · N/A')
        passed=all(group[c]['within_limits'] is True for c in CHIPS)
        status='达标' if passed else '缺合格窗' if any(not group[c]['n'] for c in CHIPS) else '超限'
        lines.append(f"| {name} | {' | '.join(vals)} | {status} |")
    lines+=['','每格按该合同的可定价窗计算，含开发回放。总误差门槛 WAPE ≤25%、单窗最大 ≤50%。Sinkhorn 另有 DT 2 窗、PR 3 窗无公开无历史价格。Post 的双片条件路径需要调用方显式启用。V1、V2 及不同采集协议分行。970DT 只有外推。新端点只在通过数值和全核结束核验后进入其前瞻报告。','',
            '[逐形与不同协议来源](evaluation.json) · [当前 wheel 逐形回放及 970DT](replay.json) · [历史双端原始回执](../v0.29/shape-errors.md) · [独立样本与误差解释](../../docs/error-analysis.md)','']
    (OUT/'acceptance.md').write_text('\n'.join(lines))


if __name__=='__main__':
    check();snapshot()
