"""Passive scorer. Ingest only qualified device receipts and compare on identical windows."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter_ns
import json,os,sys,time
from freeze import ROOT, OUT, DT, PR, FUT, digest, old
from operator_model import estimate,load_chip

def read(path):return json.loads(path.read_bytes())
def save(name,value):
    path=OUT/name;tmp=path.with_name('.'+name+'.'+str(os.getpid()));tmp.write_text(value);tmp.replace(path)
def gate(path):
    d=read(path)
    if not d.get('qualified') or not d.get('ca_us') or len(d.get('stages',[]))!=1:return None
    s,=d['stages'];shape=d.get('shape');chip=d.get('chip')
    cost=d.get('cost',{})
    if (not shape or not chip) and d.get('acquisition_receipt'):
        orig=path.parent/d['acquisition_receipt']
        if orig.is_file() and digest(orig)==d['acquisition_sha256']:
            meta=read(orig);shape=meta.get('shape');chip=meta.get('chip');cost=meta.get('cost',{})
    if not shape or chip not in (DT,PR) or not s.get('trace_sha256'):return None
    return dict(shape=shape,chip=chip,ca_us=d['ca_us'],hash=[s['trace_sha256']],source=str(path.relative_to(ROOT)),kernel=s['kernel'],observed_mmad=s['mmad'],minute=cost['seconds']/60 if cost else None)

def audit():
    freeze=read(OUT/'freeze.json');verified=all(digest(ROOT/n)==s for n,s in freeze['sha256'].items())
    if not verified:raise ValueError('Pricing source/config changed since freeze')
    independent=read(OUT/'kimi-head/freeze.json')
    if not all(digest(ROOT/n)==s for n,s in independent['dependencies'].items()):raise ValueError('Supplemental holdout changed since freeze')
    forecasts=list(freeze['rows'])+[dict(shape=independent['shape'],scenario='Kimi MLA 独立头切分 · B8 不等长 chunk',chip=sku,role='970DT extrapolation' if sku==FUT else 'prediction before device acquisition',old_us=p['old_us'],us=p['us']) for sku,p in independent['predictions'].items()]
    supplement=read(OUT/'decode/freeze.json')
    if not all(digest(ROOT/n)==h for n,h in supplement['dependencies'].items()):raise ValueError('New decode sources differ from original freeze')
    for job in supplement['jobs']:
        for sku,p in job['predictions'].items():
            forecasts.append(dict(shape=job['shape'],scenario=job['model']+' · '+job['node']+' decode',chip=sku,role='970DT extrapolation' if sku==FUT else 'prediction before device acquisition',old_us=p['old_us'],us=p['us']))
    pro = read(OUT/'pro-large/freeze.json')
    for job, description in ((pro,'每请求 4096-token 分块'),(pro['reuse_full'],'每请求完整 32Ki')):
        for sku, price in job['predictions'].items():
            previous = old(job['shape'],load_chip(sku))[0]
            role = 'prediction before device acquisition'
            if sku == FUT:
                role = '970DT extrapolation'
            elif job is not pro and sku == DT:
                role = 'running before freeze; result blinded'
            forecasts.append(dict(shape=job['shape'],scenario='DeepSeek Pro B8 · '+description,chip=sku,
                                  role=role,old_us=previous,us=price['us']))
    frozen={(tuple(r['shape']),r['chip']):r for r in forecasts};allrows={}
    for r in read(ROOT/'results/v0.31/replay.json')['rows']:
        if r['op']!='matmul':continue
        row=dict(shape=r['shape'],chip=r['chip'],ca_us=r['ca_us'],hash=r['trace_sha256'],kernel=r['kernel'],source='results/v0.31/replay.json',observed_mmad=None,minute=None);allrows[(row['chip'],tuple(row['hash']))]=row
    folders=['v0.28/matmul','v0.28/matmul/long-prefill','v0.28/matmul/wide-prefill','v0.28/matmul/multi','v0.28/matmul/narrow','v0.29/matmul/decode','v0.30/matmul-dispatch/device','v0.31/matmul/kimi-prefill','v0.31/matmul/device','v0.32/matmul/kimi-head','v0.32/matmul/decode','v0.32/matmul/pro-large']
    for sub in folders:
        folder=ROOT/'results'/sub
        for p in sorted(folder.glob('*.json')):
            if p.stem.endswith(('-initial','-strict','-coverage')):continue
            if not p.stem.endswith(('-pr','-dt','-retired','-drain')) and p.stem not in ('dt','pr'):continue
            r=gate(p)
            if r:allrows[(r['chip'],tuple(r['hash']))]=r
    inpairs=defaultdict(dict);families=defaultdict(list)
    for r in allrows.values():
        shape=tuple(r['shape']);sku=r['chip'];ch=load_chip(sku);new=estimate('matmul',shape,chip=ch,dtype='bf16');old_price=old(shape,ch)[0];d=new.projection['timing_diagnostics'];plan=new.projection['target_plan']
        f=frozen.get((shape,sku))
        if f and abs(f['us']-new.latency_us)>1e-7:raise ValueError('Public API differs from freeze')
        r.update(new_us=new.latency_us,old_us=old_price,extrapolated_970dt_us=estimate('matmul',shape,chip=FUT).latency_us,
                 pct=100*(new.latency_us/r['ca_us']-1),old_pct=100*(old_price/r['ca_us']-1),aic=new.active_aic,
                 branch=plan['branch'],model_mmad=plan['mmads_total'],role=f['role'] if f else 'mechanism development' if plan['branch']=='b_resident_l1' else 'archive replay on unchanged route')
        inpairs[shape].setdefault(sku,[]).append(r);families[sku].append(r)
    def metrics(xs):
        def calc(field):return dict(wape=100*sum(abs(r[field]-r['ca_us']) for r in xs)/sum(r['ca_us'] for r in xs),maximum=100*max(abs(r[field]/r['ca_us']-1) for r in xs))
        return dict(windows=len(xs),shapes=len(set(tuple(r['shape']) for r in xs)),old=calc('old_us'),new=calc('new_us')) if xs else {'windows':0,'shapes':0,'old':None,'new':None}
    groups={sku:{key:metrics([r for r in rows if selection(r)]) for key,selection in [('all',lambda r:True),('development',lambda r:r['role']=='mechanism development'),('blind_complete',lambda r:r['role'] in ('running before freeze; result blinded','prediction before device acquisition'))]} for sku,rows in families.items()}
    rows=sorted(allrows.values(),key=lambda r:(r['shape'],r['chip'],r['hash']))
    ratios=[]
    for shape,pair in sorted(inpairs.items()):
        if DT not in pair or PR not in pair:continue
        ratio=lambda field:median(r[field] for r in pair[PR])/median(r[field] for r in pair[DT])
        ratios.append(dict(shape=shape,old=ratio('old_us'),new=ratio('new_us'),ca=ratio('ca_us'),error_pct=100*(ratio('new_us')/ratio('ca_us')-1)))
    # Neither this replay nor the timing of the collector is a new training input.
    holdouts=[]
    for f in forecasts:
        if f['chip']==FUT:continue
        qualified=[r for r in rows if r['shape']==f['shape'] and r['chip']==f['chip']]
        holdouts.append(dict(scenario=f['scenario'],shape=f['shape'],sku=f['chip'],role=f['role'],old=f['old_us'],new=f['us'],future=frozen[tuple(f['shape']),FUT]['us'],ca=median(r['ca_us'] for r in qualified) if qualified else None))
    now=datetime.now(timezone.utc).isoformat()
    out=dict(utc=now,source_freeze_sha256=digest(OUT/'freeze.json'),supplemental_freeze_sha256=digest(OUT/'kimi-head/freeze.json'),groups=groups,holdouts=holdouts,ratios=ratios,rows=rows)
    save('score.json',json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    ls=['# 大 Matmul：相同 CA 窗的双端比较','','| 集合 | 950DT 形数 / 窗数 · 旧→新 WAPE、最大 | 950PR 形数 / 窗数 · 旧→新 WAPE、最大 |','|---|---|---|']
    for key,title in [('all','全部合格窗，含开发回放'),('development','用于 B 常驻建模的开发回放'),('blind_complete','冻结前未看结果，已回收')]:
        def fmt(x):return f"{x['shapes']}/{x['windows']} · {x['old']['wape']:.2f}→{x['new']['wape']:.2f}%，{x['old']['maximum']:.2f}→{x['new']['maximum']:.2f}%" if x['windows'] else '0 / 0 · N/A'
        ls.append(f"| {title} | {fmt(groups[DT][key])} | {fmt(groups[PR][key])} |")
    ls+=['','| 每卡 [M,N,K] | 来源 | DT：旧 / 新 / CA µs | PR：旧 / 新 / CA µs | PR/DT：旧 / 新 / CA | 970DT µs（外推） |','|---|---|---:|---:|---:|---:|']
    important=set(tuple(r['shape']) for r in rows if r['branch']=='b_resident_l1' or r['shape'][1:]==[2304,1536])|set(shape for shape,_ in frozen)
    for shape in sorted(important):
        cells=[];oldnew=[];act=[];future=estimate('matmul',shape,chip=FUT).latency_us
        for sku in (DT,PR):
            measured=inpairs.get(shape,{}).get(sku,[]);new=estimate('matmul',shape,chip=sku).latency_us;oldus=old(shape,load_chip(sku))[0];oldnew.append((oldus,new))
            ca=median(r['ca_us'] for r in measured) if measured else None;act.append(ca)
            cells.append(f"{oldus:.3f} / {new:.3f} / "+(f"[{ca:.3f}](../../../{measured[0]['source']})" if ca is not None else '—'))
        role='冻结盲测，见下表' if (shape,DT) in frozen else 'B 常驻开发回放' if any(r['branch']=='b_resident_l1' for r in inpairs[shape].get(DT,[])) else 'Kimi 普通多页排程'
        ls.append(f"| {list(shape)} | {role} | {cells[0]} | {cells[1]} | {oldnew[1][0]/oldnew[0][0]:.3f} / {oldnew[1][1]/oldnew[0][1]:.3f} / "+(f"{act[1]/act[0]:.3f}" if all(act) else '—')+f" | {future:.3f} |")
    ls+=['','| 前瞻留出及外部分支 | 芯片 | 冻结时 | CA µs |','|---|---|---|---:|']
    labels={'running before freeze; result blinded':'采集中，结果未看','prediction before device acquisition':'采集前冻结','already received before v0.32 freeze; replay only':'已有回执，仅回放'}
    for r in holdouts:ls.append(f"| {r['scenario']} {r['shape']} | {r['sku'][6:11]} | {labels[r['role']]} | "+(f"{r['ca']:.3f}" if r['ca'] else '—')+' |')
    rejected=ROOT/'results/v0.29/matmul/decode/144-2304-1536-pr.json'
    if rejected.exists() and not read(rejected).get('qualified'):
        d=read(rejected);cost=d['cost'];ls+=['',f"Kimi [144,2304,1536] 的 950PR 接口已启动，但达到 {cost['seconds']/60:.1f} 分钟上限（退出码 {cost['exit_code']}）；该端没有正式设备 µs，不计 WAPE。[失败原回执](../../../results/v0.29/matmul/decode/144-2304-1536-pr.json)。"]
    ls+=['','— 表示没有数值和完整设备窗均通过的回执；不插补，970DT 没有 CA。Kimi 的宽输出 prefill 和 2304 通道 decode 均不能把 B 放入 L1，保留普通排程。每个完整设备窗按芯片和 trace SHA 去重；同形多次实测在上面的逐形表取中位数。独立窗与开发窗均列入总览，但只有单列盲测用于检验新规则。','', '[公式、来源与算例](method.md) · [每一实际窗口、接口 kernel、MMAD 与比值](score.json) · [冻结时间、模型指纹与三端预测](freeze.json) · [不同 N/K 的前瞻条件](kimi-head/freeze.json) · [单槽状态](kimi-head/state.json) · [与直接复用 DT 比](paired.md) · [三模型 B1/9/11/16 与 MTP 同表](decode.md) · [Pro 整批与分块设备实验](pro-large/README.md) · [Flash 每行增量成本](scaling.md)','','最后更新：'+now]
    save('report.md','\n'.join(ls)+'\n')
    from paired import render
    render()
    from roster import build
    build(out)
    return all(r['ca'] is not None for r in holdouts)
def watch():
    """Wake only for a changed endpoint or queue state; each scorer reads the current register."""
    import subprocess
    previous = None
    results = ROOT/'results'
    folders = [results/v for v in ('v0.28/matmul/long-prefill','v0.28/matmul/wide-prefill',
               'v0.30/matmul-dispatch/device','v0.31/matmul/kimi-prefill','v0.32/matmul/kimi-head',
               'v0.32/matmul/decode','v0.32/matmul/pro-large')]
    while True:
        files = [f for folder in folders for f in folder.glob('*.json') if
                 f.stem in ('dt','pr','state','freeze') or f.stem.endswith(('-dt','-pr','-retired','-drain'))]
        current = tuple(sorted((str(p),p.stat().st_mtime_ns) for p in files))
        if current != previous:
            ret = subprocess.run([sys.executable, __file__], cwd=ROOT)
            if ret.returncode == 0:
                previous = current
        time.sleep(60)


if __name__=='__main__':
    if '--watch' in sys.argv:
        watch()
    else:
        audit()
