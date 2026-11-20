"""Per-protocol, per-chip distributions; unique shape counts never added across protocols."""
import json
from collections import Counter
from statistics import median
from math import ceil
from development.validation.release32.evaluate import summary
from development.validation.release32.summary_table import NAMES,CONTRACT

DT,PR,FUT='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'

def metric(s):
    return '0 · N/A' if not s['n'] else f"{s['shapes']}/{s['n']} · {s['wape']:.2f}% / {s['maximum']:.2f}%"

def bench(rows):
    x=sorted(r['wall_ms'] for r in rows)
    return dict(n=len(x),p50=median(x) if x else None,p95=x[ceil(.95*len(x))-1] if x else None,maximum=max(x) if x else None,over_100ms=sum(v>=100 for v in x))

def title(label):
    n,_,m=label.partition(' · ')
    if n=='attn_res_fwd' and not m:return 'Kimi AttnResFwd，含输出物化的历史合同'
    if m=='resident inference prefill':return 'Kimi AttnResFwd，UB 驻留推理 prefill'
    return NAMES.get(n,n)+('，'+CONTRACT.get(m,m) if m else ('，单状态链' if n=='kda_recurrent' else ''))

def render(root,rows,at,prices,save):
    scope=json.loads((root/'results/v0.33/acceptance-scope.json').read_bytes());excluded=scope['excluded_error_rows'];labels=sorted(set(r['op'] for r in rows)|{at},key=lambda n:(list(NAMES).index(n.partition(' · ')[0]),n))
    assert not set(excluded)&set(labels)
    overall=['| 算子与输入/计时口径 | 950DT 形/窗 · WAPE / 最大 | 950PR 形/窗 · WAPE / 最大 | 数据角色 | 误差门槛 DT / PR |','|---|---:|---:|---|---|']
    roles=['| 算子与协议 | DT 回放 | PR 回放 | DT 冻结后独立 | PR 冻结后独立 | 查询墙钟最大 ms（含 970） |','|---|---:|---:|---:|---:|---:|'];results={};thresholds=[]
    for l in labels:
        subset=[r for r in rows if r['op']==l];pall=[r for r in prices if r['op']==l];results[l]={};cells=[];passes=[]
        for c in (DT,PR):
            group=[r for r in subset if r['chip']==c];s=summary(group,'us');matched=[r for r in group if r['us'] is not None and r['previous_us'] is not None]
            byrole={}
            for role in ('replay','independent'):
                members=[r for r in group if r['data_role']==role];byrole[role]=dict(**summary(members,'us'),records=len(members),unpriced=sum(r['us'] is None for r in members))
            worst=max((r for r in group if r['us'] is not None),key=lambda r:abs(r['us']/r['ca_us']-1),default=None)
            results[l][c]=dict(full=s,prior_same_windows=summary(matched,'previous_us'),current_same_windows=summary(matched,'us'),unpriced=sum(r['us'] is None for r in group),by_role=byrole,query_ms=bench([p for p in pall if p['chip']==c]),
                worst_case=None if worst is None else {k:worst[k] for k in ('shape','attrs','source','identity','ca_us','us','kernel','data_role')})
            status='N/A' if not s['n'] else ('达标' if s['wape']<=25 and s['maximum']<=50 else '超限');passes.append(status);cells.append(metric(s))
            if status=='超限':thresholds.append([l,c,s['wape'],s['maximum']])
        classes={r['data_role'] for r in subset if r['us'] is not None};role={frozenset():'待合格整窗',frozenset({'replay'}):'历史回放',frozenset({'independent'}):'采前冻结新窗',frozenset({'replay','independent'}):'回放及冻结后新窗'}[frozenset(classes)]
        overall.append('| '+' | '.join([title(l),*cells,role,' / '.join(passes)])+' |')
        mx=bench(pall)['maximum'];rcols=[metric(results[l][c]['by_role'][role]) for role in ('replay','independent') for c in (DT,PR)];roles.append('| '+' | '.join([title(l),*rcols,'N/A' if mx is None else f'{mx:.3f}'])+' |')
    save('summary-table.md','\n'.join(overall));save('evaluation.json',dict(summary=results,threshold_pct=dict(wape=25,maximum=50),threshold_exceedances=thresholds,excluded=excluded,
        full_protocol_rows=len(rows),priced_protocol_rows=sum(r['us'] is not None for r in rows),query_ms_by_protocol={l:{c:bench([p for p in prices if p['op']==l and p['chip']==c]) for c in (DT,PR,FUT)} for l in labels}))
    # Only replace the marker-delimited current summary; older release observers skip it.
    guide = root/'docs/error-analysis.md'
    current = guide.read_text()
    start, end = '<!-- release34-summary -->', '<!-- /release34-summary -->'
    if start in current:
        before, body = current.split(start)
        _, after = body.split(end)
        refreshed = before + start + '\n\n' + '\n'.join(overall) + '\n\n' + end + after
        if refreshed != current:
            guide.write_text(refreshed)
    save('acceptance.md','\n'.join(['# v0.34 安装版逐算子误差','',*overall,'','## 同集回放与独立样本','',*roles,'',
        '形/窗分别是不同的输入维度数、完整 CA 协议窗口数。WAPE 为 Σ|预测−CA|/ΣCA，最大为最大的 |预测/CA−1|。逐端门槛为 WAPE≤25%、最大≤50%。历史与独立数据在第二表分别评估；重复物理窗在不同输入协议可各出现一次。', '',
        '所有 v0.32 的 1,979 条协议窗均重新通过当前安装公开入口，包含 mHC Post 的原始进程边界和有序前缀。新窗须有同一预测、实际芯片及形状，且冻结在设备采集之前；早于冻结已经启动的采集或其后的离线重估列回放。两条已移出验收的 KDA 合同不列入分母。970DT 仅外推。每窗保留真实库 kernel、来源及物理身份；每协议有当前及旧版同集误差、最坏形状、未定价数与查询开销。', '',
        '[全部预测与原始计时来源](replay.json) · [分组统计与最坏形状](evaluation.json) · [安装指纹](model.json) · [未缓存查询墙钟](wheel-check.json) · [有序流回放](ordered-wheel-check.json)']))
