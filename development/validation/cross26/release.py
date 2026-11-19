"""Snapshot qualified rows and failed numerical checks without averaging their populations."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.26'
OLD=ROOT/'results/v0.24/kda-prefetch/supplement'
FA=ROOT/'results/v0.25/coverage/supplement/score.json'
SKUS=('Ascend950DT_9582','Ascend950PR_957b')


def read(path):
    return json.loads(path.read_bytes())


def save(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def metrics(rows):
    if not rows:
        return None
    return dict(n=len(rows),wape=100*sum(abs(x-y) for x,y in rows)/sum(y for _,y in rows),
                maximum=100*max(abs(x/y-1) for x,y in rows))


def kda_rows():
    old=read(OLD/'comparison.json')
    rows=list(old['rows'])
    failed=[]
    for marker,folder in [(OLD/'register.json',OLD),(OUT/'kda/register.json',OUT/'kda'),(OUT/'kda/aligned-register.json',OUT/'kda')]:
        for c in read(marker)['cases']:
            src=folder/(c['id']+'.json')
            if not src.exists():continue
            d=read(src)
            if not d['qualified']:
                failed.append(dict(id=c['id'],shape=c['shape'],sku=c['sku'],numerical=d['output_groups'],timing_us=None,
                                   source=str(src.relative_to(ROOT)),cost_s=d['cost']['seconds']))
            elif folder!=OLD:
                stages=[s for s in d['stages'] if s['family']]
                q=c['predictions'][c['sku']]
                rows.append(dict(id=c['id'],shape=c['shape'],sku=c['sku'],source=str(src.relative_to(ROOT)),
                                actual=[s['ca_us'] for s in stages]+[d['main_stages_us']],candidate=q['stages_us']+[q['us']],
                                v023=q['stages_old']+[q['v023_us']],extrapolated_970_us=c['predictions']['Ascend970DT']['us'],
                                cost_s=d['cost']['seconds']))
    assert len({(r['sku'],tuple(r['shape'])) for r in rows})==len(rows)
    per_sku={sku:{group:{mode:metrics([(r[mode][i],r['actual'][i]) for r in rows if r['sku']==sku])
                 for mode in ('candidate','v023')} for i,group in enumerate(('prepare','fwdh','finalize','total'))} for sku in SKUS}
    return dict(rows=rows,failed=failed,by_sku=per_sku)


def shape_table(data):
    rows=data['rows']
    tbl=['# KDA 三段 V2 的完整设备窗','',
         '每次调用使用发布版 FLA 26.9.1 的三个顺序 kernel；每段从全核第一条指令到末核退休，合计扣除段间宿主间隔。四组数值、整核退休、内存后端和 ELF 均合格才纳入耗时。下表的预测在各个样本采集前生成；后两批是在看到较早结果后补充的几何。','',
         '| 输入 T / H | 芯片 | Prepare / FwdH / Finalize 模型 µs | 相应 CA µs | 三段模型 / CA µs | 偏差 | 970DT 仅外推 µs |','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        _,t,h,*_=r['shape']
        a,b=r['candidate'],r['actual']
        src=Path(r['source'])
        import os
        link=os.path.relpath(ROOT/src,OUT/'kda')
        fmt=lambda v:' / '.join(f'{x:.3f}' for x in v[:3])
        tbl.append(f"| [{t} / {h}]({link}) | {r['sku'][6:11]} | {fmt(a)} | {fmt(b)} | {a[3]:.3f} / {b[3]:.3f} | {100*(a[3]/b[3]-1):+.2f}% | {r['extrapolated_970_us']:.3f} |")
    tbl+=['','### 第四块的状态','',
          'T=193、每卡两个头，尾块只有一行。950PR 上第三块的状态写在 11.752 µs 退休；第四块 W 的读早在 10.444 µs 已发起，但第四块 H 读要等到 11.761 µs。模型对应时刻为 12.200、11.229、12.200 µs。依赖次序得到验证，前三块积累的时间差仍会影响最后一行。','',
          '![四个状态更新的资源模型与实际末核](pr-four-chunks.png)','','[950DT 同形图](dt-four-chunks.png) · [逐块时间、来源与 ELF](pr-four-chunks.json)','',
          '### 保留未通过的数值检查','',
          'T=257 与 T=248 都在两种 SKU 上得到相同的设备数组。T=257 的 Aqk 在 [head=0,row=243,col=44] 偏离双精度发布算法参考 5.2452×10⁻⁶，允许 3.1325×10⁻⁶。对 T=248，Aqk 全通过，而 Akk 在 [0,242,40] 偏离 9.5367×10⁻⁶，允许 6.2915×10⁻⁶。attention 和最终状态在四次调用中均通过。原验收合同对这四个 shape×SKU 的耗时均记 N/A。双端分别确认了导出的每个 BF16 元素和同步状态。','',
          '[T=257 实际坐标](failed-257.json) · [T=248 实际坐标](failed-248.json) · [按正式设备内核的中间精度复现](numeric-audit.md) · [冻结、回执与全量评估](../evaluation.json)']
    (OUT/'kda/README.md').write_text('\n'.join(tbl)+'\n')


def main():
    k=kda_rows();shape_table(k)
    active=dict(fa=read(FA),post=read(OUT/'post/score.json'),sink=read(OUT/'sink/independent.json'),
                fused_replay=read(OUT/'presink/replay.json')['groups'])
    fused=list((OUT/'presink').glob('fused-*.json'))
    active['fused_receipts']={f.stem:read(f) for f in fused}
    active['kda_numeric_contract']={
        'role':'Separate mixed-precision consistency check. Preserve the registered FP64 reference outcome.',
        'original_failures':k['failed'], 'controls':str((OUT/'kda/numeric-audit-control.json').relative_to(ROOT)),
        'forward_registration':str((OUT/'kda/numeric-audit-freeze.json').relative_to(ROOT)),
        'official_prospective':{p.name:{'qualified':read(p)['qualified'],'counts':read(p)['output_groups']}
                                for p in sorted((OUT/'kda').glob('numeric26-*-official.json'))}}
    active['warm_schedule_status']=str((OUT/'post/warm-schedule.md').relative_to(ROOT))
    previous=ROOT/'results/v0.25/coverage/audit.json'
    refs=[previous,ROOT/'results/v0.22/evaluation.json']
    save(OUT/'evaluation.json',dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),kda=k,**active,
         continuing_full_shapes=[{'case':s,'receipt':str(p.relative_to(ROOT)) if p.exists() else None} for s in ('pro-b8-256ki-dt','pro-b8-256ki-pr') for p in [ROOT/'results/v0.23/pro-prefill'/(s+'.json')]],
         retained_whole_suite=[dict(source=str(p.relative_to(ROOT)),sha256=sha256(p.read_bytes()).hexdigest()) for p in refs]))
    pins=[p for p in (OUT/'kda/register.json',OUT/'kda/aligned-register.json',OUT/'post/registration.json',OUT/'sink/register.json',OUT/'presink/freeze.json',
          OUT/'memory/initial-issues-freeze.json',OUT/'memory/pro-256ki-prefix-forecast.json',ROOT/'results/v0.25/post/prices.json',
          ROOT/'operator_model/post_repeat.py',ROOT/'operator_model/stream.py',ROOT/'operator_model/backend.py',ROOT/'operator_model/chips.py',
          *sorted((ROOT/'configs/chips').glob('*.json')))]
    save(OUT/'model.json',dict(version='0.26.0',baseline='v0.25',source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in pins},
         scope='Cast initial MTE2 pair in default diagnostics; experimental identical token-Post stream timing. Sinkhorn and fused PreSinkhorn mechanism candidates are scored separately. KDA V2 uses the previously frozen pipeline.'))


if __name__=='__main__':
    main()
