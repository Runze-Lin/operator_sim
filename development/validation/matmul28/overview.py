"""Consolidate full all-core measurements from the five registered BF16 Matmul campaigns."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.28/matmul'
DT,PR,FUT='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'
GROUPS={'.':'report.md','wide-prefill':'report.md','long-prefill':'report.md','narrow':'report.md','multi':'report.md'}

def read(p):
    return json.loads(p.read_bytes())

def store(p,s):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(s);temp.replace(p)

def price():
    import operator_model as api
    from operator_model import backend,matmul_splitk,matmul_resource
    if not sys.flags.isolated or api.__version__!='0.28.0':raise RuntimeError('use the isolated release wheel')
    rows=[];seen=set()
    for name,report in GROUPS.items():
        reg=read(OUT/name/'freeze.json')
        for j in reg['rows']:
            sh=j['shape']; m,n,k=sh
            assert tuple(sh) not in seen
            seen.add(tuple(sh))
            b=j.get('requests',j.get('runtime_example',{}).get('batch',8))
            q=m//b
            assert b*q==m
            preds={}
            for c in (DT,PR,FUT):
                p=api.estimate('matmul',sh,chip=c,dtype='bf16')
                preds[c]=dict(us=p.latency_us,aic=p.active_aic,aiv=p.active_aiv,work=p.work)
            rows.append(dict(group=name,model=j['model'],node=j['node'],shape=sh,requests=b,tokens_per_request=q,
                        phase=j.get('lane','prefill' if 'prefill' in name else 'decode'),predictions=preds,detail=str(Path(name)/report)))
    modules={v.__name__:sha256(Path(v.__file__).read_bytes()).hexdigest() for v in (backend,matmul_splitk,matmul_resource)}
    with (OUT/'overview-prices.json').open('x') as f:
        json.dump(dict(utc=datetime.now(timezone.utc).isoformat(),modules_sha256=modules,rows=rows,
            role='Installed public model, mixed known and pending workload shapes. Earlier independent registrations determine each experimental data role.'),f,indent=2);f.write('\n')

def receipt(j,tag):
    group=j['group'];chip=DT if tag=='dt' else PR
    if group=='wide-prefill': slug=j['model']+'-'+tag
    elif group=='long-prefill':slug=str(j['tokens_per_request'])+'-'+tag
    else:slug='-'.join(map(str,j['shape']))+'-'+tag
    path=OUT/group/(slug+'.json')
    if not path.exists():return dict(status='pending',ca_us=None)
    d=read(path)
    if group=='.' and not d['qualified']:
        extra=OUT/(slug+'-drain.json')
        if extra.exists() and read(extra).get('qualified'):
            d.update(read(extra));path=extra
    assert d.get('chip',chip)==chip and d.get('shape',j['shape'])==j['shape']
    stages=d.get('stages') or ([d['observed']] if d.get('observed') else [])
    return dict(status='qualified' if d.get('qualified') else 'unqualified',ca_us=d.get('ca_us') if d.get('qualified') else None,
                receipt=str(path.relative_to(OUT)),kernel=[s['kernel'] for s in stages],
                wall_s=d.get('cost',{}).get('seconds'),reason=d.get('reason'))

def render():
    frozen=read(OUT/'overview-prices.json'); rows=[];completed=0;accepted=0
    for j in frozen['rows']:
        targets={}
        for chip,tag in [(DT,'dt'),(PR,'pr')]:
            data=receipt(j,tag); p=j['predictions'][chip]
            completed+=data['status']!='pending'
            accepted+=data['status']=='qualified'
            targets[chip]=dict(**p,**data,signed_pct=100*(p['us']/data['ca_us']-1) if data['ca_us'] else None)
        rows.append(dict(**{k:v for k,v in j.items() if k!='predictions'},targets=targets,extrapolated_970dt_us=j['predictions'][FUT]['us']))
    body=['# 三个模型的 Matmul 实测总表','',
          f'共 {len(rows)} 个不同的每卡 [M,N,K]，950DT/PR 共 {len(rows)*2} 项。已返回 {completed} 项，其中 {accepted} 项通过全数值及全核指令窗核验。所有输入与权重均为连续 BF16 ND，N、K 取目标投影在 TP8 下的真实维度。本表使用 v0.28 已封存的公开路由。独立性按各项原登记判断，总表同时包含开发回放与新形。','',
          '| 场景 · 模型 | 投影 | B×token | [M,N,K] | DT 公开预测 / CA (误差) µs | PR 公开预测 / CA (误差) µs | 970DT 外推 µs |',
          '|---|---|---:|---|---:|---:|---:|']
    for r in rows:
        vals=[]
        for c in (DT,PR):
            d=r['targets'][c]
            val='待回收' if d['status']=='pending' else ('未合格' if d['ca_us'] is None else f'{d["ca_us"]:.3f} ({d["signed_pct"]:+.1f}%)')
            if 'receipt' in d:val=f'[{val}]({d["receipt"]})'
            vals.append(f'{d["us"]:.3f} / {val}')
        body.append(f'| {r["phase"]} · {r["model"]} | [{r["node"]}]({r["detail"]}) | {r["requests"]}×{r["tokens_per_request"]} | {r["shape"]} | {vals[0]} | {vals[1]} | {r["extrapolated_970dt_us"]:.3f} |')
    body+=['','M 为此次提交的总行数。B=1 的 MTP8 校验和 B=9 的普通 decode 可能得到相同的 M=9；B8×128Ki 和 B8×256Ki 分别为 1,048,576 和 2,097,152 行。实际部署中的量化、权重转置或其他融合会影响 kernel 选择。[单核分段与真实 MMAD](streamk-analysis.md) · [当前 WAPE 与直接复用 DT](public-report.md) · [重选输出块的候选及独立验证](multi/report.md) · [逐项数据与资格](overview.json)','']
    store(OUT/'overview.json',json.dumps(dict(rows=rows,completed=completed,qualified=accepted,pending=2*len(rows)-completed),ensure_ascii=False,indent=2)+'\n')
    store(OUT/'README.md','\n'.join(body))
    return completed==2*len(rows)

if __name__=='__main__':
    if sys.argv[1]=='price':price()
    elif sys.argv[1]=='render':render()
    elif sys.argv[1]=='watch':
        while not render():time.sleep(60)
