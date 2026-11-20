"""Read the pre-existing campaign receipts and compare each completed whole-kernel window."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
import time
from pathlib import Path

from full_load_freeze import CA, OUT, RAW, ROOT, digest, read


def save(name,text):
    p=OUT/name
    temp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    temp.write_text(text);temp.replace(p)


def outcome(row):
    name=row['acquisition']
    if not name:
        return dict(status='no CA, extrapolation only',terminal=True)
    path=ROOT/name
    orig=read(path) if path.exists() else None
    separate=path.with_name(path.stem+'-drain.json')
    if not orig:
        current=RAW/path.relative_to(CA).with_suffix('')
        return dict(status='acquiring' if current.exists() else 'queued',terminal=False)
    receipt=orig
    if separate.exists():
        receipt=read(separate)
        assert receipt['acquisition_sha256']==digest(path)
        used=separate
    else:
        used=path
    base=dict(receipt=str(used.relative_to(ROOT)),sha256=digest(used),original=str(path.relative_to(ROOT)),
              cost_seconds=orig['cost']['seconds'],peak_bytes=orig['cost']['peak_bytes'])
    if not receipt.get('qualified'):
        # The owning supervisor can fill a supplemental END+drain receipt after the initial parse.
        status=read(CA/('long-prefill/state.json' if path.parent!=CA else 'prefill-status.json'))
        advancing=(status.get('phase')=='finished' or status.get('case') not in (None,path.stem))
        return dict(**base,status='unqualified' if advancing else 'receipt available, awaiting supplemental verification',
                    reason=receipt.get('reason'),terminal=advancing)
    assert orig['shape']==row['shape'] and orig['chip']==row['chip']
    assert receipt['correctness']==[f"correctness count={row['shape'][0]*row['shape'][1]} bad=0"]
    stage,=receipt['stages']
    assert stage['trace_sha256'] and stage['binary_sha256'] and len(stage['windows'])==stage['active_aic']+stage['active_aiv']
    assert 'matmul' in stage['kernel'].lower() and receipt['ca_us']==stage['ca_us']>0
    if 'terminal' in receipt:
        assert set(receipt['terminal'])==set(stage['windows'])
        assert all(v['first_tick']<=v['physical_end']<=v['retire_tick'] for v in receipt['terminal'].values())
    cand=row['candidate']
    match=(stage['kernel'].endswith('_131074') and cand['active_aic']==stage['active_aic'] and stage['active_aiv']==0
           and stage['mmad']==cand['mmad_total'])
    ca=receipt['ca_us']
    return dict(**base,status='whole kernel qualified',terminal=True,ca_us=ca,kernel=stage['kernel'],candidate_roster_mmad_match=match,
                active_aic=stage['active_aic'],active_aiv=stage['active_aiv'],mmad=stage['mmad'],
                trace=stage['trace'],trace_sha256=stage['trace_sha256'],elf_sha256=stage['binary_sha256'],
                correctness=receipt['correctness'],basis=receipt.get('basis','Original strict all-core endpoint verifier'),
                signed_error_pct=100*(cand['latency_us']/ca-1),old_signed_error_pct=100*(row['legacy_us']/ca-1))


def score():
    reg=read(OUT/'freeze.json')
    # Stored numeric forecasts remain authoritative if a later checkout edits a source.
    source_drift=[path for path,h in reg['sha256'].items() if not (ROOT/path).exists() or digest(ROOT/path)!=h]
    rows=[]
    for item in reg['rows']:
        r=outcome(item)
        rows.append(dict(shape=item['shape'],chip=item['chip'],role=item['role'],candidate_us=item['candidate']['latency_us'],
                         all_output_writeback_us=item['candidate'].get('all_output_writeback_us'),legacy_us=item['legacy_us'],**r))
    groups={}
    for name in ['development','in flight, blinded at freeze','registered before acquisition','completed but unseen at freeze']:
        completed=[r for r in rows if r['role']==name and r.get('ca_us')]
        def err(k):
            den=sum(r['ca_us'] for r in completed)
            return 100*sum(abs(r[k]-r['ca_us']) for r in completed)/den if den else None
        groups[name]=dict(completed=len(completed),wape_pct=err('candidate_us'),legacy_wape_pct=err('legacy_us'),
                          max_abs_pct=max((abs(r['signed_error_pct']) for r in completed),default=None))
    finished=all(r['terminal'] for r in rows)
    result=dict(utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),phase='finished' if finished else 'waiting for original CA',
                freeze_sha256=digest(OUT/'freeze.json'),groups=groups,rows=rows,source_drift=source_drift)
    save('score.json',json.dumps(result,indent=2)+'\n')
    md=['# 分组比较','',f"最后核对 {result['utc']}。设备时间为 µs，采集墙钟为分钟。",'',
        '| 每卡 M | 芯片 | 角色 | 旧版 | 候选 | CA | 候选偏差 | 离线分钟 |',
        '|---:|---|---|---:|---:|---:|---:|---:|']
    labels={'development':'开发回放','in flight, blinded at freeze':'冻结时正在采集','registered before acquisition':'新采前冻结','completed but unseen at freeze':'已采未读','extrapolation only':'外推'}
    statuses={'queued':'队列等待','acquiring':'采集中','unqualified':'采集未合格','receipt available, awaiting supplemental verification':'等待逐核终点复核','no CA, extrapolation only':'无 CA'}
    for r in rows:
        f=lambda v,precision:'—' if v is None else f'{v:.{precision}f}'
        ca=f(r.get('ca_us'),3) if r.get('ca_us') else statuses[r['status']]
        md.append(f"| {r['shape'][0]:,} | {r['chip'].replace('Ascend','').split('_')[0]} | {labels[r['role']]} | {r['legacy_us']:.3f} | {f(r['candidate_us'],3)} | {ca} | {f(r.get('signed_error_pct'),2)} | {f(None if 'cost_seconds' not in r else r['cost_seconds']/60,1)} |")
    md+=['','| 组别 | 合格端点 | 旧 WAPE | 候选 WAPE | 候选最大绝对相对误差 |','|---|---:|---:|---:|---:|']
    for role,g in groups.items():
        md.append(f"| {labels[role]} | {g['completed']} | {f(g['legacy_wape_pct'],2)} | {f(g['wape_pct'],2)} | {f(g['max_abs_pct'],2)} |")
    md+=['','WAPE = Σ|预测−CA| / ΣCA，表内误差单位为 %。970DT 的 M=32768 未通过多波准入。候选工作量、源文件指纹、每端 ELF、核数及实际 MMAD 可由机器可读结果核对。[冻结合同](freeze.json) · [最新逐端结果](score.json) · [方法及算例](report.md)','']
    save('score.md','\n'.join(md))
    return finished


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    if args.watch:
        with (OUT/'.watch.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            while not score():time.sleep(90)
    else:
        score()
