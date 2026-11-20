"""Score immutable forecasts when the original shared CA queue has written complete qualified receipts."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from freeze import OUT,ROOT,OLD,read,digest
sys.path.insert(0,str(ROOT/'development/validation/matmul29'))
from full_load_score import outcome


def replace(name,value):
    p=OUT/name;tmp=p.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(value);tmp.replace(p)


def score():
    reg=read(OUT/'freeze.json')
    v29={(tuple(r['shape']),r['chip']):r for r in read(OLD/'freeze.json')['rows']}
    rows=[]
    for r in reg['rows']:
        o=outcome(v29[(tuple(r['shape']),r['chip'])]);ca=o.get('ca_us');p=r['candidate'];z=p.get('latency_us')
        rows.append(dict(shape=r['shape'],chip=r['chip'],role=r['role'],us=z,v029_us=r['v029_us'],public_us=r['older_public_us'],
            ca_us=ca,signed_error_pct=(100*(z/ca-1) if z and ca else None),ablation=r['ablation_us'],
            candidate_roster_mmad_match=o.get('candidate_roster_mmad_match'),status=o['status'],is_terminal=o['terminal'],
            receipt=o.get('receipt'),receipt_sha256=o.get('sha256'),trace_sha256=o.get('trace_sha256'),
            launch_actual=dict(aic=o.get('active_aic'),aiv=o.get('active_aiv'),mmad=o.get('mmad'),elf=o.get('elf_sha256')),
            numeric=o.get('correctness'),schedule_ms=r['query_ms'],ca_wall_s=o.get('cost_seconds'),raw_bytes=o.get('peak_bytes')))
    summary={}
    fields=('public_us','v029_us','us')
    for role in ('development','running unseen','strictly before acquisition','receipt pre-existed unread'):
        complete=[r for r in rows if r['role']==role and r['ca_us']]
        n=len(complete)
        summary[role]=dict(n=n,wape={f:100*sum(abs(r[f]-r['ca_us']) for r in complete)/sum(r['ca_us'] for r in complete) if n else None for f in fields},
                           max_abs_rel={f:max((abs(100*(r[f]/r['ca_us']-1)) for r in complete),default=None) for f in fields})
    pair=[]
    for m in sorted({r['shape'][0] for r in rows}):
        dt,pr=[next(r for r in rows if r['shape'][0]==m and r['chip'].startswith(c)) for c in ('Ascend950DT','Ascend950PR')]
        actual=pr['ca_us']/dt['ca_us'] if pr['ca_us'] and dt['ca_us'] else None
        pair.append(dict(m=m,dt_role=dt['role'],pr_role=pr['role'],pr_over_dt=actual,models={f:pr[f]/dt[f] for f in fields},
                         model_ratio_error_pct={f:100*(pr[f]/dt[f]/actual-1) if actual else None for f in fields}))
    state='finished' if all(r['is_terminal'] for r in rows) else 'waiting original queue'
    metadata=dict(utc=datetime.now(timezone.utc).isoformat(),watch_pid=os.getpid(),phase=state,freeze_sha256=digest(OUT/'freeze.json'),summary=summary,pairs=pair,rows=rows)
    replace('score.json',json.dumps(metadata,indent=2)+'\n')
    fmt=lambda x:'—' if x is None else f'{x:.3f}'
    labels={'development':'开发回放','running unseen':'冻结时正在采集','strictly before acquisition':'采前冻结','receipt pre-existed unread':'已采而未读','extrapolation only':'外推'}
    lines=['# 双端 CA 与跨芯片比值','',f"更新时间 {metadata['utc']}，窗口收到后按冻结值评分。时间单位为 µs。",'',
           '| [M,128,512] 中的 M | 芯片 | 数据角色 | 原公共入口 | v0.29 | v0.30 | CA | v0.30 偏差 % | CA 墙钟 min |','|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['shape'][0]:,} | {r['chip'].replace('Ascend','').split('_')[0]} | {labels[r['role']]} | {fmt(r['public_us'])} | {fmt(r['v029_us'])} | {fmt(r['us'])} | {fmt(r['ca_us'])} | {fmt(r['signed_error_pct'])} | {fmt(r['ca_wall_s']/60) if r['ca_wall_s'] else '—'} |")
    lines+=['','| 数据角色 | 合格端点 | 公共 WAPE% | v0.29 WAPE% | v0.30 WAPE% | v0.30 最大误差% |','|---|---:|---:|---:|---:|---:|']
    for k,g in summary.items():
        w=g['wape'];lines.append(f"| {labels[k]} | {g['n']} | {fmt(w['public_us'])} | {fmt(w['v029_us'])} | {fmt(w['us'])} | {fmt(g['max_abs_rel']['us'])} |")
    lines+=['','| M | PR/DT 实测耗时比 | 公共预测 | v0.29 | v0.30 | v0.30 比值偏差 % |','|---:|---:|---:|---:|---:|---:|']
    for p in pair:
        w=p['models'];lines.append(f"| {p['m']:,} | {fmt(p['pr_over_dt'])} | {fmt(w['public_us'])} | {fmt(w['v029_us'])} | {fmt(w['us'])} | {fmt(p['model_ratio_error_pct']['us'])} |")
    lines+=['','CA 缺失、未合格和外推分别在 JSON 的 status 中列出。所有输入按芯片、形状和数据角色保留，不以机制不符删除有效耗时。[逐端来源、对照和状态](score.json) · [建模说明](report.md)','']
    replace('comparison.md','\n'.join(lines))
    return state=='finished'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--detach',action='store_true');a=parser.parse_args()
    if a.detach:
        with (OUT/'watch.log').open('a') as f:
            p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--watch'],cwd=ROOT,start_new_session=True,
                               stdin=subprocess.DEVNULL,stdout=f,stderr=f);print(p.pid)
    elif a.watch:
        with (OUT/'.score.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            while not score():time.sleep(90)
    else:score()
