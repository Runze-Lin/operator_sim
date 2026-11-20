"""Score only complete registered full windows, retaining cross-family and UB ambiguity."""
import argparse
import json
import os
from pathlib import Path
import time
from stage_candidate import DEV,OUT,read,sha
def write(path,text):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text);tmp.replace(path)

def score():
    frozen=read(OUT/'stage-freeze.json');jobs=read(OUT/'freeze.json')['jobs'];rawflag={(x['job'],x['chip']):x['started'] for x in frozen['roles_before_large_CA']}
    assert sha(OUT/'freeze.json')==frozen['parent_sha256']
    rows=[]
    for j in jobs:
        for target in j['outputs']:
            op=target['op'];t=target['shape'][0]
            if op not in ('mhc_pre','mhc_sinkhorn'):continue
            for sku in DEV:
                rec=OUT/(j['id']+'-'+sku+'.json')
                receipt=read(rec) if rec.exists() else {}
                if receipt:assert receipt['registration_sha256']==frozen['parent_sha256']
                reference=receipt.get('ca_us',{}).get(op) if receipt.get('qualified') else None
                diag=('qualified' if reference is not None else 'unqualified' if receipt else
                      'started' if (OUT/'raw'/(j['id']+'-'+sku)).exists() else 'queued')
                if op=='mhc_pre':
                    case=next(v for v in frozen['pre'] if v['chip']==sku and v['t']==t);candidates={'M-K':case['candidate']['us']}
                    selected='M-K'
                else:
                    case=next(v for v in frozen['sink'] if v['chip']==sku and v['t']==t)
                    h=case['conditional']['scenarios'];candidates={}
                    for state,sc in h.items():
                        if sc['explicit_one_ub_us'] is not None:candidates[state]=sc['explicit_one_ub_us']
                        else:
                            for z in sc['block_scenarios']:candidates[f"{state}-cap{z['cap']}"]=z['us']
                    selected='first' if j['op']==op and h['first']['explicit_one_ub_us'] is not None else None
                model=target['model'][sku]['us']
                gaps={k:100*(v/reference-1) if v is not None and reference else None for k,v in candidates.items()}
                rows.append(dict(case=j['id'],op=op,chip=sku,shape=target['shape'],status=diag,reason=receipt.get('reason'),
                                 source=rec.name if rec.exists() else None,started_before_stage_freeze=rawflag[j['id'],sku],
                                 ca_us=reference,public_us=model,public_signed_pct=(100*(model/reference-1) if reference and model else None),
                                 candidates_us=candidates,signed_pct=gaps,comparable_point=selected,
                                 comparable_signed_pct=gaps.get(selected),qualified=reference is not None))
    write(OUT/'stage-score.json',json.dumps(dict(freeze_sha256=sha(OUT/'stage-freeze.json'),rows=rows),indent=2)+'\n')
    lines=['# 预登记大形阶段模型的实测进度','',
      '| 实际调用 | 路线 | 芯片 | 公开 / 已选候选 / CA (µs) | 候选误差 | 状态 |',
      '|---|---|---|---:|---:|---|']
    def fmt(v):return '—' if v is None else f'{v:.3f}'
    for r in rows:
        v=r['candidates_us'].get(r['comparable_point']);percent=r['comparable_signed_pct']
        stage='实测' if r['qualified'] else r['status']
        lines.append(f"| {r['case']} | {r['op']} | {r['chip']} | {fmt(r['public_us'])} / {fmt(v)} / {fmt(r['ca_us'])} | {fmt(percent)+'%' if percent is not None else '—'} | {stage} |")
    lines.extend(['','独立单发且单 UB 的 Sinkhorn 可以按冷启动定价。三段链的 Sinkhorn 无可验证的同族前序；跨 UB 的阶段价格保留两种块容量假设。表内只给有明确比较资格的单点，所有场景价格及各自有符号偏差完整列在 JSON。970DT 始终作为架构外推。','',
        '[预注册方法](stage-report.md) · [逐假设数值与资格](stage-score.json)'])
    write(OUT/'stage-live.md','\n'.join(lines)+'\n')
    return all(r['status'] in ('qualified','unqualified') for r in rows)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    if args.watch:os.nice(18)
    while not score() and args.watch:
        time.sleep(90)
