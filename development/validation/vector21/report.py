"""Publish same-set metrics using versioned predictions and qualified CA only."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.vector21.campaign import assert_resource_revision, S,C,DT,PR,CHIPS,OUT,OLD,read,write
from operator_model import estimate

V20=OUT.parents[1]/'v0.20'


def development():
    path=OUT/'development.json'
    if path.exists():
        return
    assert_resource_revision()
    rows=[]
    def row(op,shape,chip,old,ca,source):
        x=estimate(op,shape,chip=chip)
        assert x.latency_us>0
        rows.append(dict(op=op,shape=shape,chip=chip,old_us=old,us=x.latency_us,ca_us=ca,
                         projected_970_us=estimate(op,shape,chip=CHIPS['future']).latency_us,
                         source=str(source.relative_to(ROOT)),role='development replay'))
    for r in read(V20/'newop/evaluation.json')['rows']:
        for chip in (DT,PR):
            d=r['devices'][chip]
            row(S,r['shape'],chip,d['model_us'],d['ca_us'],V20/'newop/holdout.json')
    dev=read(V20/'newop/development.json')
    old_program='''import json,sys
from operator_model import estimate,__version__
assert __version__=="0.20.0"
print(json.dumps([estimate("sigmoid_fp32",r["shape"],chip=r["sku"]).latency_us for r in json.load(sys.stdin)]))'''
    labels=json.loads(subprocess.check_output([OLD,'-I','-c',old_program],cwd='/tmp',
                                              text=True,input=json.dumps(dev)))
    for r,old in zip(dev,labels):
        row(S,r['shape'],r['sku'],old,r['kernel']['ca_us'],V20/'newop/development.json')
    for r in read(V20/'large/evaluation.json')['cases']:
        for chip in (DT,PR):
            if 'ca_us' in r['devices'].get(chip,{}):
                row(C,r['shape'],chip,r['current'][chip]['us'],r['devices'][chip]['ca_us'],
                    V20/'large'/('cast_'+'_'.join(map(str,r['shape']))+'_'+chip+'.json'))
    for r in read(V20/'large/prior-cast-replay.json')['rows']:
        for side,chip in [('dt',DT),('pr',PR)]:
            row(C,r['shape'],chip,r[side]['current'],r[side]['ca'],V20/'large/prior-cast-replay.json')
    write(path,rows)


def metrics(rows,key):
    seq=[(r[key],r['ca_us']) for r in rows if r[key] is not None]
    if not seq:
        return dict(n=0,wape_pct=None,max_pct=None)
    return dict(n=len(seq),wape_pct=100*sum(abs(p-t) for p,t in seq)/sum(t for p,t in seq),
                max_pct=100*max(abs(p/t-1) for p,t in seq))


def publish():
    f=read(OUT/'freeze.json')
    jobs={j['id']:j for j in f['jobs']}
    rows=read(OUT/'development.json')
    completed=read(OUT/'holdout.json') if (OUT/'holdout.json').exists() else []
    for r in completed:
        j=jobs[r['id']]
        if r['qualified']:
            t=j['targets'][r['chip']]
            rows.append(dict(op=r['op'],shape=r['shape'],chip=r['chip'],old_us=t['old_us'],us=t['us'],
                             ca_us=r['ca_us'],projected_970_us=j['targets'][CHIPS['future']]['us'],
                             wall_s=r['cost']['seconds'],role='post-freeze independent',
                             source='results/v0.21/vector/holdout.json'))
    groups={};speed={}
    for op in SHAPES:
        for role in ('development replay','post-freeze independent'):
            subset=[r for r in rows if r['op']==op and r['role']==role]
            for chip in (DT,PR):
                sel=[r for r in subset if r['chip']==chip]
                groups[op,role,chip]={k:metrics(sel,k) for k in ('old_us','us')}
            pair=[]
            for d in subset:
                if d['chip']!=DT: continue
                p=next((p for p in subset if p['shape']==d['shape'] and p['chip']==PR),None)
                if p:
                    pair.append({k:d[k]/p[k] if d[k] is not None and p[k] is not None else None
                                 for k in ('old_us','us','ca_us')})
            speed[op,role]={k:metrics(pair,k) for k in ('old_us','us')}
    write(OUT/'evaluation.json',dict(summary=[dict(op=o,role=r,chip=c,**v) for (o,r,c),v in groups.items()],
        dt_over_pr_summary=[dict(op=o,role=r,**v) for (o,r),v in speed.items()],
        completed=len(completed),qualified=sum(x['qualified'] for x in completed),registered=40,rows=rows,
        excluded=[dict(id=x['id'],chip=x['chip'],reason=x['reason']) for x in completed if not x['qualified']]))
    fmt=lambda v:'—' if v is None else f'{v:.2f}'
    lines=['# 连续 Vector 流水的成对验证','','|算子|数据角色|SKU|样例数|v0.20 WAPE / max|v0.21 WAPE / max|',
           '|---|---|---|---:|---:|---:|']
    for (op,role,chip),v in groups.items():
        cols=[' / '.join(fmt(v[k][m]) for m in ('wape_pct','max_pct')) for k in ('old_us','us')]
        lines.append(f"|{op}|{role}|{chip}|{v['us']['n']}|{'|'.join(cols)}|")
    lines+=['',f'误差单位为百分比。新留出完成 {len(completed)} / 40 次。分母只包括通过时间窗和全元素校验的记录。每个 SKU 和算子分别统计。Sigmoid DT 的三个早期开发点用原版 v0.20 wheel 回放，逐组两版比较使用完全相同的 CA 窗口。','',
            '|算子|数据角色|每卡 shape|SKU|v0.20 / µs|v0.21 / µs|CA / µs|v0.21 误差|970DT 外推 / µs|',
            '|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('|{}|{}|{}|{}|{}|{}|{}|{:+.1f}%|{}|'.format(r['op'],r['role'],r['shape'],r['chip'],
            fmt(r['old_us']),fmt(r['us']),fmt(r['ca_us']),100*(r['us']/r['ca_us']-1),fmt(r['projected_970_us'])))
    lines+=['','[冻结模型与预测](freeze.json) · [机器可读结果与 DT/PR 加速比](evaluation.json) · [参数来源](method.md) · [独立长 shape 流水](independent-figures.md)']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')

SHAPES=(S,C)
if __name__=='__main__':
    import time
    development()
    stamp=None
    while True:
        holdout=OUT/'holdout.json'
        current=holdout.stat().st_mtime_ns if holdout.exists() else -1
        if current!=stamp:
            publish()
            if '--follow' in sys.argv:
                subprocess.run([sys.executable,str(Path(__file__).with_name('independent_figures.py'))],check=False)
            stamp=current
        if '--follow' not in sys.argv or (holdout.exists() and len(read(holdout))==40):
            break
        time.sleep(20)
