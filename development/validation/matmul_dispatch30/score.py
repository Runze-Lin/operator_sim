"""Score only completed full-device probes against the pre-CA prices."""
from math import isclose
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.matmul_dispatch30.device import OUT,REGISTER,ca,save,CHIPS


def metrics(rows,prediction,actual='ca_us'):
    data=[r for r in rows if r.get(prediction) is not None and r.get(actual) is not None]
    if not data:
        return dict(n=0,wape=None,maximum=None)
    delta=[abs(r[prediction]-r[actual]) for r in data]
    return dict(n=len(data),wape=100*sum(delta)/sum(r[actual] for r in data),
                maximum=100*max(d/r[actual] for d,r in zip(delta,data)))


def report():
    reg=ca.read(REGISTER)
    rows=[]; paired=[]
    def fmt(n):return '—' if n is None else f'{n:.3f}'
    lines=['# 同一 Flash 投影的独立设备验证','',
           '每形在 950DT、950PR 各启动一个完整 aclnnMatmul。默认预测和分核在新设备采集前冻结。所有时间为设备 µs，采集墙钟为秒。B1×9 和 B9×1 共用 M=9，汇总只计一次。','',
           '| [M,128,512] | 芯片 | 公共预测 | CA | 偏差 % | AIC 预 / 实 | MMAD 实 | 墙钟 s |',
           '|---:|---|---:|---:|---:|---:|---:|---:|']
    for job in reg['rows']:
        m=job['shape'][0]; pair={}
        for chip,tag in CHIPS:
            label=f'{m}-{tag}'
            paths=[OUT/(label+'-retired.json'),OUT/(label+'.json')]
            rec=next((p for p in paths if p.exists() and ca.read(p).get('qualified')),None)
            frozen=job['predictions'][chip]
            r=dict(shape=job['shape'],chip=chip,status='unmeasured',public_us=frozen['us'],split_us=frozen['placement_us'],
                   supply_us=frozen['supply_us'],direct_dt_us=job['predictions'][ca.DT]['us'],
                   predicted_active_aic=frozen['public_active_aic'],future_extrapolated_us=job['predictions'][ca.FUT]['us'],
                   future_supply_extrapolated_us=job['predictions'][ca.FUT]['supply_us'],ca_us=None)
            if rec:
                doc=ca.read(rec);st,=doc['stages']
                if doc['registration_sha256']!=ca.digest(REGISTER) or doc['shape']!=job['shape'] or doc['chip']!=chip:
                    raise ValueError('Qualified receipt from different contract')
                if not isclose(doc['model_us'],frozen['us']):raise ValueError('Unfrozen baseline')
                r.update(status='qualified',ca_us=doc['ca_us'],signed_pct=100*(frozen['us']/doc['ca_us']-1),active_aic=st['active_aic'],
                         active_aiv=st['active_aiv'],mmad=st['mmad'],wall_s=doc['cost']['seconds'],bytes=doc['cost']['peak_bytes'],
                         source=rec.name,correctness=doc['correctness'],kernel=st['kernel'],elf=st['binary_sha256'],trace=st['trace_sha256'])
            elif paths[-1].exists():
                r.update(status='unqualified',source=paths[-1].name,reason=ca.read(paths[-1]).get('reason'))
            rows.append(r);pair[chip]=r
            dev=f"[{fmt(r['ca_us'])}]({rec.name})" if rec else '—'
            lines.append(f"| {m:,} | {tag.upper()} | {fmt(r['public_us'])} | {dev} | {fmt(r.get('signed_pct'))} | {r['predicted_active_aic']} / {r.get('active_aic','—')} | {r.get('mmad','—')} | {fmt(r.get('wall_s'))} |")
        dt,pr=pair[ca.DT],pair[ca.PR]
        if dt['ca_us'] and pr['ca_us']:
            paired.append(dict(shape=job['shape'],actual=pr['ca_us']/dt['ca_us'],public=pr['public_us']/dt['public_us'],
                               reuse_dt=1.0))
    summary={}
    for tag,group in [('decode',[r for r in rows if r['shape'][0]!=524288]),('prefill',[r for r in rows if r['shape'][0]==524288])]:
        summary[tag]={chip:{price:metrics([r for r in group if r['chip']==chip],price) for price in ('public_us','direct_dt_us','supply_us')} for chip,_ in CHIPS}
    lines+=['','| 已返回的小 M 窗口 | 窗数 | 公共 WAPE / 最大 % | 直接复用 DT 的 WAPE / 最大 % |',
            '|---|---:|---:|---:|']
    for chip,tag in CHIPS:
        g=summary['decode'][chip]; p=g['public_us']; direct=g['direct_dt_us']
        lines.append(f"| {tag.upper()} | {p['n']} | {fmt(p['wape'])} / {fmt(p['maximum'])} | {fmt(direct['wape'])} / {fmt(direct['maximum'])} |")
    lines+=['','| 已经双端配对的 M | CA 时间比 PR/DT | 公开预测比 | 直接复用 DT |','|---:|---:|---:|---:|']
    for r in paired:lines.append(f"| {r['shape'][0]} | {fmt(r['actual'])} | {fmt(r['public'])} | 1.000 |")
    lines+=['','共用权重 128 KiB，输入与输出的行数按单次提交计算。签名 key 2 的普通 Cube 分支允许在保持 MMAD 数相同的情况下改变每条指令的有效行数和执行周期。对分批需要结合实测发起数、尾块与核数判断。完整数字核验、ELF、核名册与原始 CPU 秒可从每行回执查看。部分初始收据缺少归档文件名；补充收据仅恢复文件名并在原 trace 上重做原有的 END、异步尾退休与全输出检查。','',
            '[全部预报与 970DT 外推](freeze.json) · [逐端与分组指标](score.json) · [队列状态](state.json)','']
    save(OUT/'score.json',dict(utc=ca.now(),register_sha256=ca.digest(REGISTER),rows=rows,summary=summary,ratios_pr_over_dt=paired,
                                    ratio_metrics={k:metrics(paired,k,'actual') for k in ('public','reuse_dt')}))
    tmp=OUT/'report.md.tmp';tmp.write_text('\n'.join(lines));tmp.replace(OUT/'report.md')


if __name__=='__main__':report()
