"""Refresh comparisons on receipt arrival without launching simulations."""
from pathlib import Path
import fcntl
import json
import os
import sys
import time

import check
from check import FREEZE, OLD, RAW, RES, SKUS, read, score


def monitor_save(path,data):
    tmp=path.with_suffix(f'.monitor-{os.getpid()}.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

# A receipt can arrive while the serial collector is regenerating its own score.
check.save=monitor_save


def render():
    score();rows=read(RES/'score.json')['rows'];f=lambda v:'N/A' if v is None else f'{v:.3f}'
    lines=['# RecurrentKda：冻结后的多链验证','',
           '| B / 链数 | 平台 | AIV 轮数 | 预测 µs | 重付启动基线 µs | CA µs | 预测误差 | 基线误差 |',
           '|---|---|---:|---:|---:|---:|---:|---:|']; pairs={}
    for r in rows:
        t=r['measured_ca_us'];p=r['predicted_us'];base=r['cold_repeat_baseline_us'];pairs.setdefault(r['b'],{})[r['chip']]=r
        num=lambda u:'N/A' if t is None else f'{100*(u/t-1):+.2f}%'
        lines.append(f"| {r['b']} / {r['chains']} | {r['chip']} | {r['rounds']} | {p:.3f} | {base:.3f} | {f(t)} | {num(p)} | {num(base)} |")
    lines+=['','加速比按 `T_DT/T_PR` 定义。绝对误差较小表示更接近同形 CA；970 始终只做外推。','']
    for b,v in pairs.items():
        a,z=v[SKUS[0]],v[SKUS[1]];ca=None if a['measured_ca_us'] is None or z['measured_ca_us'] is None else a['measured_ca_us']/z['measured_ca_us']
        lines.append(f"- B={b}：DT/PR 预测 {a['predicted_us']/z['predicted_us']:.3f}×，双端 CA 比 {f(ca)}。")
    lines+=['','B6 两端可用时按每个 AIV 上 beta 发起的实际次数区分首次和后续链。']
    for tag in ('dt','pr'):
        rec=RES/f'control-{tag}.json';v=read(rec)['verdict'] if rec.is_file() else {}
        if v.get('qualified'):
            intervals=sorted(n for xs in v['repeat_beta_intervals_ticks'].values() for n in xs);r=pairs[6][SKUS[0 if tag=='dt' else 1]];q=read(FREEZE)['cases']['6']['predictions'][r['chip']]
            lines.append(f"- {tag.upper()} 实测：链数→核数 {v['chain_histogram']}，同核相邻 beta 的间隔 {intervals[0]/1650:.3f}–{intervals[-1]/1650:.3f} µs（{len(intervals)} 对）"+f"；模型每条后续链 {q['per_chain_body_us']:.3f} µs。" if intervals else f"- {tag.upper()} 无重复 beta 发起证据。")
        elif v:lines.append(f"- {tag.upper()}：不合格，{v.get('reason')}。")
    lines+=['','[核验口径](method.md) · [原始冻结](freeze.json) · [运行预算与状态](state.json) · [逐端数值/收据](score.json)','']
    target=RES/'report.md';tmp=target.with_suffix('.tmp');tmp.write_text('\n'.join(lines));tmp.replace(target)

def watch():
    RAW.mkdir(exist_ok=True,parents=True)
    with (RAW/'monitor.lock').open('a') as fp:
        fcntl.flock(fp,fcntl.LOCK_EX|fcntl.LOCK_NB)
        paths=[RES/f'control-{t}.json' for t in ('dt','pr')]+[OLD/f'recurrent-{t}.json' for t in ('dt','pr')]; prev=None;limit=time.monotonic()+72*3600
        while time.monotonic()<limit:
            state=tuple(p.stat().st_mtime_ns if p.exists() else None for p in paths)
            if state!=prev:render();prev=state
            if all(x is not None for x in state):return
            time.sleep(60)

if __name__=='__main__':(watch if sys.argv[1:] == ['watch'] else render)()
