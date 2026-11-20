"""Publish one row per qualified Pro CA endpoint when the acquisition writes its receipt."""
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT/'results/v0.32/matmul/pro-large'
MID = HERE/'freeze.json'
OLD = ROOT/'results/v0.28/matmul/wide-prefill'
LOCKS = ROOT/'runs/matmul32/pro-large'
DT, PR, FUT = 'Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    return str(path.relative_to(ROOT))


def atomic(path, data):
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(data)
    temp.replace(path)


def observation(frozen, sku, receipt, data_role):
    shape=frozen['shape']
    predictions=frozen['predictions']
    pred=predictions[sku]
    common=dict(model='DeepSeek-V4-Pro',node='indexer.weights_proj',shape=shape,chip=sku,
                new_us=pred['us'],old_us=pred['us'],reused_dt_us=predictions[DT]['us'],
                extrapolated_970dt_us=predictions[FUT]['us'],aic=pred['active_aic'],aiv=pred['active_aiv'],
                branch=pred['branch'],model_mmad=pred['mmads'],waves=pred['waves'],role=data_role,
                source=rel(receipt),baseline_basis='The frozen coarse_tile route dispatches to v0.31 unchanged.')
    if not receipt.exists():
        return dict(**common,status='pending',qualified=False,ca_us=None)
    r=read(receipt)
    common.update(receipt_sha256=sha(receipt),cost=r.get('cost'))
    if not (r.get('qualified') and r.get('ca_us') and r['shape']==shape and r['chip']==sku):
        return dict(**common,status='not qualified',qualified=False,ca_us=None,reason=r.get('reason','wrong receipt identity'))
    expected=[f'correctness count={shape[0]*shape[1]} bad=0']
    stages=r.get('stages',[])
    if len(stages)!=1 or r.get('correctness')!=expected or stages[0]['binary_sha256']!=read(MID)['installed_matmul_elf']:
        return dict(**common,status='not qualified',qualified=False,ca_us=None,reason='numerical coverage or ELF differs')
    s,=stages
    pct=100*(pred['us']/r['ca_us']-1)
    return dict(**common,status='measured',qualified=True,ca_us=r['ca_us'],pct=pct,old_pct=pct,
                absolute_pct=abs(pct),hash=[s['trace_sha256']],kernel=s['kernel'],active_aic=s['active_aic'],
                active_aiv=s['active_aiv'],observed_mmad=s['mmad'],correctness=r['correctness'],
                stage_trace=s['trace'],minute=r.get('cost',{}).get('seconds',0)/60)


def inputs():
    return [*[(HERE/(t+'.json')) for t in ('dt','pr')],
            *[(OLD/('DeepSeek-V4-Pro-'+t+'.json')) for t in ('dt','pr')],HERE/'state.json',OLD/'state.json']


def refresh():
    f=read(MID)
    full=f['reuse_full']
    paths=inputs()
    rows=[]
    for idx,r in enumerate([f,full]):
        for i,sku in enumerate((DT,PR)):
            role=('prospective frozen before acquisition' if idx==0 or sku==PR
                  else 'original v0.28 registration; v0.32 frozen after start, before receipt')
            rows.append(observation(r,sku,paths[idx*2+i],role))
    pairs=[]
    for geometry in [f['shape'],full['shape']]:
        d,p=[x for x in rows if x['shape']==geometry]
        pairs.append(dict(shape=geometry,predicted_dt_over_pr=d['new_us']/p['new_us'],
                          measured_dt_over_pr=d['ca_us']/p['ca_us'] if d['qualified'] and p['qualified'] else None))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=sha(MID),windows=rows,
                measured_rows=[r for r in rows if r['qualified']],paired=pairs,
                state={rel(p):read(p) for p in paths[-2:] if p.exists()})
    atomic(HERE/'score.json',json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    lines=['# DeepSeek-V4-Pro 索引投影双端回执','',
        '| [M,N,K] | 芯片 | 预测 µs | CA µs | 有符号误差 | 970DT 外推 µs | 当前资格 |',
        '|---|---|---:|---:|---:|---:|---|']
    for r in rows:
        ref=f'{r["ca_us"]:.3f}' if r['qualified'] else '—'
        err=f'{r["pct"]:+.2f}%' if r['qualified'] else '—'
        name={'pending':'待收','not qualified':'未通过','measured':'完整 CA'}[r['status']]
        source=ROOT/r['source']
        if source.exists():
            name=f'[{name}]({os.path.relpath(source,HERE)})'
        lines.append(f'| {r["shape"]} | {r["chip"]} | {r["new_us"]:.3f} | {ref} | {err} | {r["extrapolated_970dt_us"]:.3f} | {name} |')
    lines+=['',f'本次刷新 {result["utc"][:19]} UTC，四个预订端点中已有 {len(result["measured_rows"])} 个完整设备窗。',
        '有符号误差等于预测值除以实测值减一。两端都有完整设备窗时，JSON 才会填入同形 DT/PR 耗时比。',
        '全形 DT 已在 v0.28 开跑，v0.32 在结果返回前登记了同路线预报。分块形状两端与全形 PR 在 v0.32 登记时尚未开跑。',
        '[单算子形状与资源推导](README.md) · [聚合 JSON](score.json) · [采集进度](state.json)','']
    atomic(HERE/'live.md','\n'.join(lines))
    return all(r['status']!='pending' for r in rows)


def watch():
    LOCKS.mkdir(parents=True,exist_ok=True)
    with (LOCKS/'publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=None
        while True:
            current=tuple(p.stat().st_mtime_ns if p.exists() else None for p in inputs())
            if current!=previous:
                if refresh():
                    return
                previous=current
            time.sleep(5)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['once','watch'])
    {'once':refresh,'watch':watch}[parser.parse_args().command]()
