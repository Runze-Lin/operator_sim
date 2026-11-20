"""Join qualified receipts to the unchanged public wheel; keep development candidates separate."""
from collections import defaultdict
from hashlib import sha256
import json
from math import isclose
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/v0.35'
BASE = ROOT/'results/v0.34'
DT, PR, FUT = 'Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT'


def read(p):
    return json.loads(p.read_bytes())


def digest(p):
    return sha256(p.read_bytes()).hexdigest()


def put(p, data):
    tmp = p.with_name(f'.{p.name}.{os.getpid()}.tmp')
    text = data if isinstance(data,str) else json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2)
    tmp.write_text(text.rstrip()+'\n')
    tmp.replace(p)


def receipts(api):
    def public(op, shape, chip, attrs):
        begin = perf_counter_ns()
        value = api.estimate(op, shape, attrs, chip=chip).latency_us
        return value, (perf_counter_ns()-begin)/1e6

    def make(label, op, chip, shape, attrs, receipt, identity, kernel, ca, frozen, role):
        us, ms = public(op,shape,chip,attrs)
        fu, future_ms = public(op,shape,FUT,attrs)
        assert isclose(us, frozen, abs_tol=1e-7) and ca>0 and len(identity)==64
        return dict(op=label, chip=chip, shape=shape, attrs=attrs, identity=[identity], kernel=kernel,
                    source=str(receipt.relative_to(ROOT)), receipt_sha256=digest(receipt), ca_us=ca, us=us,
                    extrapolated_970dt_us=fu, query_ms=ms, query_970_ms=future_ms, data_role=role)

    receipt = OUT/'hcpost/pr-full.json'
    q = read(receipt)
    assert q['qualified'] and q['identical_packed_global'] and q['global_ticks'][1]-q['global_ticks'][0]>0
    assert q['active_aiv']==len(q['all_shards'])==56 and q['output_payload_bytes']==41943040
    assert q['acquisition_sha256']==digest(ROOT/q['acquisition_receipt'])
    yield make('vllm_hc_post','mhc_post',PR,q['shape'],{'implementation':'vllm_hc_post'},receipt,
               q['global_sha256'],q['kernel'],q['ca_us'],q['public_baseline_us'],'replay')

    reg = OUT/'post/fresh-freeze.json'
    frozen = read(reg)
    for chip,tag in ((DT,'dt'),(PR,'pr')):
        p = OUT/f'post/fresh-{tag}.json'
        if not p.exists():continue
        result = read(p)
        assert result['freeze_sha256']==digest(reg) and result['chip']==chip
        if not result['complete']:continue
        assert result['cost']['started_utc']>frozen['utc'] and len(result['rows'])==len(frozen['sequence'])
        local, future = api.OperatorStream(chip), api.OperatorStream(FUT)
        for job, row in zip(frozen['rows'][chip],result['rows']):
            s = row['stage']; shape = job['shape']
            assert row['timing_qualified'] and row['shape']==shape and row['i']==job['i']
            default = make('mhc_post · shape','mhc_post',chip,shape,{},p,s['trace_sha256'],s['kernel'],
                           s['ca_us'],job['public_us'],'independent')
            yield default
            a = perf_counter_ns()
            us = local.advance('mhc_post',shape).latency_us
            b = perf_counter_ns()
            fu = future.advance('mhc_post',shape).latency_us
            # Only the shape-only public point was sealed for this exact ordered campaign.
            yield dict(default,op='mhc_post · stream',us=us,extrapolated_970dt_us=fu,query_ms=(b-a)/1e6,
                       query_970_ms=(perf_counter_ns()-b)/1e6,data_role='replay',launch_sequence=frozen['sequence'][:job['i']+1])

    reg = OUT/'quant/freeze.json'
    f = read(reg)
    for chip in (DT,PR):
        p = ROOT/f'results/v0.29/quant/{chip}.json'
        if not p.exists():continue
        v = read(p)
        assert v['sku']==chip and v['freeze_sha256']==f['source_freeze_sha256'] and v['shape']==f['shape']
        if v['qualified']:
            assert v['cost']['started_utc']>f['utc']
            yield make('quant','quant',chip,v['shape'],{},p,v['trace_sha256'],v['kernel'],v['instruction_us'],
                       f['predictions'][chip]['previous_us'],'independent')

    for r in read(OUT/'gating/score.json')['forward']:
        if r['ca_us'] is None:continue
        role = 'independent' if r['role']=='independent frozen before acquisition' else 'replay'
        yield make('moe_gating_top_k','moe_gating_top_k',r['chip'],r['shape'],{},ROOT/r['evidence'],
                   r['trace_sha256'],r['kernel'],r['ca_us'],r['public'],role)


def main(prefix):
    sys.path.insert(0,str(prefix))
    import operator_model as api
    assert sys.flags.isolated and Path(api.__file__).is_relative_to(prefix) and api.__version__=='0.34.0'
    sys.path.append(str(ROOT))
    from development.validation.release32.evaluate import summary
    from development.validation.release34.tables import metric, title
    archive = BASE/'replay.json'
    base = read(archive)['rows']
    key = lambda r:(r['op'],r['chip'],tuple(r['identity']))
    existing = {key(r):r for r in base}; extra=[]
    for row in receipts(api):
        if key(row) in existing:
            previous = existing[key(row)]
            assert previous['shape']==row['shape'] and isclose(previous['ca_us'],row['ca_us'],abs_tol=.001)
            continue
        existing[key(row)]=row;extra.append(row)
    scope = ROOT/'results/v0.33/acceptance-scope.json'
    excluded = read(scope)['excluded_error_rows']
    by = defaultdict(list)
    for row in existing.values():
        assert row['op'] not in excluded
        by[(row['op'],row['chip'])].append(row)
    lines=['| 算子与输入/计时口径 | 950DT 形/窗 · WAPE / 最大 | 950PR 形/窗 · WAPE / 最大 | 数据角色 | 误差门槛 DT / PR |',
           '|---|---:|---:|---|---|']
    groups={}
    for label in read(BASE/'evaluation.json')['summary']:
        metrics=[];statuses=[];roles=set()
        groups[label]={}
        for chip in (DT,PR):
            rr=by[(label,chip)]
            whole=summary(rr,'us');metrics.append(metric(whole))
            groups[label][chip]=dict(all=whole,**{k:summary([r for r in rr if r['data_role']==k],'us') for k in ('replay','independent')})
            roles.update(r['data_role'] for r in rr if r['us'] is not None)
            statuses.append('N/A' if not whole['n'] else '达标' if whole['maximum']<=50 and whole['wape']<=25 else '超限')
        role='待合格整窗' if not roles else '历史回放' if roles=={'replay'} else '冻结后独立' if roles=={'independent'} else '回放及冻结后新窗'
        lines.append('| '+' | '.join([title(label),*metrics,role,' / '.join(statuses)])+' |')
    put(OUT/'model.json',dict(public_version=api.__version__,wheel_sha256=read(BASE/'model.json')['wheel_sha256'],
        scope_sha256=digest(scope),evidence_release='0.35',candidate_registrations={str(p.relative_to(ROOT)):digest(p) for p in
        [OUT/'post/fresh-freeze.json',OUT/'post/freeze.json',OUT/'quant/freeze.json',OUT/'gating/freeze.json',ROOT/'results/v0.32/post/hc-pr-freeze.json']}))
    put(OUT/'supplement.json',dict(base=str(archive.relative_to(ROOT)),base_sha256=digest(archive),base_rows=len(base),
        additional_rows=extra,combined_rows=len(existing),measurement='qualified complete instruction windows only'))
    put(OUT/'evaluation.json',dict(public_version=api.__version__,summary=groups,combined_rows=len(existing),
        thresholds=dict(wape_pct=25,maximum_pct=50),excluded=excluded))
    subprocess.run([sys.executable,'-I',str(ROOT/'development/validation/coverage35/report.py'),str(prefix)],
                   cwd='/tmp',check=True)
    put(OUT/'acceptance.md','# 公开模型在 950DT 和 950PR 的 CA 误差\n\n'+'\n'.join(lines)+
        '\n\n形/窗分别是独立几何数和整窗数。各行按同一输入合同计算。表内预测来自 v0.34 公开安装版；本轮候选在专项报告中单独评分。\n\n[逐合同分角色误差和偏差](evaluation.json) · [原有逐形记录](../v0.34/replay.json) · [新增逐形与两端身份](supplement.json)')
    path=ROOT/'docs/error-analysis.md';doc=path.read_text()
    begin,end='<!-- release35-summary -->','<!-- /release35-summary -->'
    if begin in doc:
        before,tail=doc.split(begin,1);_,after=tail.split(end,1)
        updated=before+begin+'\n\n'+'\n'.join(lines)+'\n\n'+end+after
        if doc!=updated:put(path,updated)
    print('public rows',len(existing),'additional',len(extra))


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
