"""Replay an isolated installed wheel; retain callers' different timing protocols."""
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
OUT=ROOT/'results/v0.31'
OLD=ROOT/'results/v0.30'
DT,PR,FUT='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'


def read(path):
    return json.loads(path.read_bytes())


def save(path, obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def metrics(rows,field):
    priced=[r for r in rows if r[field] is not None]
    n=len(priced)
    if not n:
        return dict(n=0,shapes=0,wape=None,maximum=None,within_limits=None)
    wape=100*sum(abs(r[field]-r['ca_us']) for r in priced)/sum(r['ca_us'] for r in priced)
    maximum=100*max(abs(r[field]/r['ca_us']-1) for r in priced)
    return dict(n=n,shapes=len({tuple(r['shape']) for r in priced}),wape=wape,maximum=maximum,
                within_limits=wape<=25 and maximum<=50)


def release(prefix):
    assert sys.flags.isolated
    sys.path.insert(0,str(Path(prefix).resolve()))
    import operator_model as api
    assert api.__version__=='0.31.0' and not Path(api.__file__).is_relative_to(ROOT)
    table=read(OLD/'evaluation.json')['summary']
    rows={}
    costs=[]
    query_cache={}

    def query(op,shape,attrs,chip):
        key=(op,tuple(shape),json.dumps(attrs,sort_keys=True),chip)
        if key not in query_cache:
            begin=perf_counter_ns()
            query_cache[key]=api.estimate(op,shape,attrs,chip=chip)
            costs.append((perf_counter_ns()-begin)/1e6)
        return query_cache[key]

    def add(op,shape,chip,attrs,kernel,ca,ids,previous,source,role):
        key=(op,chip,tuple(ids))
        if key in rows:
            assert isclose(ca,rows[key]['ca_us'],abs_tol=.001) and shape==rows[key]['shape']
            return
        call={'fla_kda_v2':'kda_chunk','vllm_hc_post':'mhc_post'}.get(op,op)
        val=query(call,shape,attrs,chip)
        future=query(call,shape,attrs,FUT)
        if op!='matmul' and previous is not None:
            assert isclose(val.latency_us,previous,abs_tol=1e-8),(op,shape,chip)
        rows[key]=dict(op=op,shape=shape,attrs=attrs,chip=chip,kernel=kernel,ca_us=ca,trace_sha256=ids,
                       source=source,role=role,previous_us=previous,current_us=val.latency_us,
                       future_970dt_extrapolated_us=future.latency_us,status=val.status)
        if op=='matmul':
            plan=val.projection['target_plan']
            rows[key].update(branch=plan['branch'],mmad=plan['mmads_total'],active_aic=val.active_aic,
                             active_aiv=val.active_aiv,productive_aic=plan.get('productive_aic'))

    for r in read(OLD/'replay.json')['rows']:
        add(r['op'],r['shape'],r['chip'],r['attrs'],r['kernel'],r['ca_us'],r['trace_sha256'],
            r['current_us'],'results/v0.30/replay.json','archive replay')
    for r in read(ROOT/'results/v0.29/matmul/score.json')['rows']:
        add('matmul',r['shape'],r['chip'],{},r['kernel'],r['ca_us'],r['raw_sha256'],r['old_us'],r['source'],r['role'])
    for r in read(OLD/'matmul-dispatch/device/score.json')['rows']:
        if r['status']=='qualified':
            add('matmul',r['shape'],r['chip'],{},r['kernel'],r['ca_us'],[r['trace']],r['public_us'],
                'results/v0.30/matmul-dispatch/device/'+r['source'],'v0.30 frozen; ordinary branch')
    for r in read(OUT/'matmul/device/score.json')['rows']:
        if r['status']=='qualified':
            rec=OUT/'matmul/device'/r['receipt']; stage,=read(rec)['stages']
            add('matmul',r['shape'],r['sku'],{},stage['kernel'],r['ca_us'],[stage['trace_sha256']],r['old_us'],
                str(rec.relative_to(ROOT)),'v0.31 prospective, device')
            q=rows['matmul',r['sku'],(stage['trace_sha256'],)]
            assert isclose(q['current_us'],r['forecast_us'],abs_tol=1e-9)
    kimi=ROOT/'results/v0.29/matmul/decode'
    for j in read(kimi/'freeze.json')['targets']:
        for chip,tag in [(DT,'dt'),(PR,'pr')]:
            regular=kimi/('-'.join(map(str,j['shape']))+'-'+tag+'.json')
            for path in ([regular,kimi/'retry/receipt.json'] if j['m']==9 and chip==DT else [regular]):
                if not path.exists():continue
                r=read(path)
                if r.get('qualified'):
                    assert r['chip']==chip and r['shape']==j['shape']
                    s,=r['stages']
                    add('matmul',j['shape'],chip,{},s['kernel'],r['ca_us'],[s['trace_sha256']],
                        j['predictions'][chip]['v028']['us'],str(path.relative_to(ROOT)),'Kimi v0.29 frozen device')
    for j in read(ROOT/'results/v0.29/large/fa/freeze.json')['jobs']:
        file=ROOT/'results/v0.29/large/fa'/(j['id']+'.json')
        if file.exists() and (r:=read(file)).get('qualified'):
            s,=r['stages'];chip=r['chip']
            add('fa',j['shape'],chip,{},s['kernel'],r['ca_us'],[s['trace_sha256']],j['predictions'][chip]['us'],
                str(file.relative_to(ROOT)),'long KV, freeze before CA')
    for j in read(OUT/'post/freeze.json')['jobs']:
        for chip in (DT,PR):
            file=OUT/'post'/(j['op']+'-'+chip+'.json')
            if file.exists() and (r:=read(file)).get('qualified'):
                s,=r['stages']
                add(j['op'],j['shape'],chip,j['attrs'],s['kernel'],r['ca_us'][j['op']],[s['trace_sha256']],
                    j['predictions'][chip]['us'],str(file.relative_to(ROOT)),'v0.31 independent standalone')
    values=list(rows.values())
    comparisons={}
    for op in ('fa','matmul','vllm_hc_post'):
        by_chip={}
        for chip in (DT,PR):
            subset=[r for r in values if r['chip']==chip and r['op']==op]
            assert len(subset)>=table[op][chip]['n']
            now=metrics(subset,'current_us');before=metrics(subset,'previous_us')
            by_chip[chip]={'all':dict(current=now,previous=before)}
            if op=='matmul':
                for name,role in [('app','v0.28 application replay'),('independent','v0.31 prospective, device')]:
                    sub=[r for r in subset if r['role']==role]
                    by_chip[chip][name]=dict(current=metrics(sub,'current_us'),previous=metrics(sub,'previous_us'))
            table[op][chip]=now
        comparisons[op]=by_chip
    multi_table={}
    cases=read(OUT/'recurrent/freeze.json')['cases']
    for r in read(OUT/'recurrent/score.json')['rows']:
        job=cases[str(r['b'])]['job']
        val=query('kda_recurrent',job['shape'],job.get('attrs',{}),r['chip'])
        assert val.status=='extrapolated' and isclose(val.latency_us,r['predicted_us'],abs_tol=1e-8)
        if r['measured_ca_us'] is not None:
            assert (ROOT/r['receipt']).is_file()
            multi_table.setdefault(r['chip'],[]).append(dict(shape=job['shape'],pred=val.latency_us,ca_us=r['measured_ca_us']))
    table['kda_recurrent · dense multi-chain']={c:metrics(multi_table.get(c,[]),'pred') for c in (DT,PR)}
    post=read(OLD/'post/score.json')['rows']
    for chip in (DT,PR):
        extra=[r for r in values if r['op']=='mhc_post' and r['chip']==chip]
        if not extra:continue
        for label,column,options in [('shape','stateless_us',None),('visible stream','ordered_us',dict(track_post_history=True)),
                                     ('experimental fragment stream','candidate_us',dict(track_post_history=True,experimental_post_fragments=True))]:
            subset=[dict(shape=r['shape'],ca_us=r['ca_us'],p=r[column]) for r in post if r['chip']==chip]
            for r in extra:
                pred=(api.OperatorStream(chip,**options).advance('mhc_post',r['shape'],r['attrs']).latency_us
                      if options else r['current_us'])
                subset.append(dict(shape=r['shape'],ca_us=r['ca_us'],p=pred))
                r[label+'_us']=pred
            table['mhc_post · '+label][chip]=metrics(subset,'p')
    for j in read(OLD/'matmul-dispatch/freeze.json')['rows']:
        for c in (j['soc'],FUT):
            query('matmul',j['shape'],{},c)
    for r in read(OLD/'sink/score.json')['rows']:
        assert query('mhc_sinkhorn',[r['t'],4],{'num_iters':r['iters']},r['sku']).latency_us==r['public_us']
    for reg in [read(OLD/'matmul-dispatch/device/freeze.json'),read(OUT/'matmul/device/freeze.json')]:
        for j in reg['rows']:
            for c in (DT,PR,FUT):
                query('matmul',j['shape'],{},c)
    historical=read(OLD/'replay.json')['rows']
    assert all((r['op'],r['chip'],tuple(r['trace_sha256'])) in rows for r in historical)
    assert len(costs)==len(query_cache) and max(costs)<100,(len(costs),max(costs))
    wheel,=(ROOT/'build/wheels/v031').glob('*.whl')
    cost=dict(n=len(costs),p50_ms=median(costs),p95_ms=sorted(costs)[ceil(.95*len(costs))-1],maximum_ms=max(costs))
    save(OUT/'wheel-check.json',dict(wheel_sha256=digest(wheel),version=api.__version__,
         rows=len(rows),distinct_public_inputs=len(costs),in_process_first_query=cost))
    save(OUT/'replay.json',dict(role='Archive and explicitly labeled prospective full-device windows. Unchanged separate protocols remain in the prior per-operator data.',rows=values))
    deps=sorted((ROOT/'operator_model').rglob('*.py'))
    deps.extend(ROOT/l.split(' ',1)[1] for l in (ROOT/'MANIFEST.in').read_text().splitlines() if l.startswith('include '))
    deps.extend(OUT.rglob('freeze.json'))
    save(OUT/'model.json',dict(version=api.__version__,baseline='0.30.0',
                               hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(set(deps))}))
    save(OUT/'evaluation.json',dict(utc=datetime.now(timezone.utc).isoformat(),summary=table,
          limits=dict(wape=25,maximum=50),comparisons=comparisons,model_sha256=digest(OUT/'model.json'),
          role='All unique priced complete windows by protocol. Historical replay and frozen groups are separately labeled.',
          retained_protocols='See v0.30/evaluation.json and its named sources for archived protocols with unchanged prices.'))
    def fmt(j):
        return f"{j['shapes']} / {j['n']} · {j['wape']:.2f}% / {j['maximum']:.2f}%" if j['n'] else '0 · N/A'
    lines=['# 已归档的双端误差','','| 算子 / 输入与计时合同 | DT 形数 / 窗数 · WAPE / 最大 | PR 形数 / 窗数 · WAPE / 最大 | 同集结果 |','|---|---:|---:|---|']
    for name,group in sorted(table.items()):
        both=[group[c] for c in (DT,PR)]
        grade='缺合格窗' if any(not v['n'] for v in both) else ('达标' if all(v['within_limits'] for v in both) else '超限')
        lines.append(f"| {name} | {fmt(both[0])} | {fmt(both[1])} | {grade} |")
    lines+=['','按完整设备窗去重。门槛为每端 WAPE ≤25% 且单窗最大误差 ≤50%。历史开发回放、分芯片独立新形与运行中的大形按来源区分。没有公开价格的窗口保留资格与原因。Sinkhorn 有 2 个 DT 和 3 个 PR 多寄存器窗尚无默认价格。970DT 全为外推。','','[同一批原始时间上的版本比较](evaluation.json) · [库名、逐形、DT/PR 与 970DT](replay.json) · [全量分析](../../docs/error-analysis.md) · [各算子的大形覆盖](../v0.29/large/report.md)','']
    (OUT/'acceptance.md').write_text('\n'.join(lines))
    print(len(rows),cost,comparisons['matmul'])


if __name__=='__main__':
    release(sys.argv[1])
