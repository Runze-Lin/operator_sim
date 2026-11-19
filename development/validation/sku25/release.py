"""Publish per-shape evidence and generate the same release summary in the docs."""
from hashlib import sha256
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.25'
DT,PR='Ascend950DT_9582','Ascend950PR_957b'


def read(p):
    return json.loads(p.read_bytes())


def write(p,data):
    p.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def stamp(p):
    return dict(path=str(p.relative_to(ROOT)),sha256=sha256(p.read_bytes()).hexdigest())


def main():
    mixed=read(OUT/'mixed-stream/comparison.json')
    post=read(OUT/'post/evaluation.json')
    audit=read(OUT/'coverage/audit.json')
    kda_path=ROOT/'results/v0.24/kda-prefetch/supplement/comparison.json'
    kda=read(kda_path)
    perf=read(OUT/'wheel-check.json')
    freezes=[OUT/p for p in ['coverage/fa-freeze.json','mixed-stream/independent-freeze.json','mixed-stream/development-freeze.json',
                             'post/freeze.json','kda-v1/registration.json']]
    freezes+=list(sorted(OUT.glob('mixed-stream/recovery-*-freeze.json')))
    freezes+=[p for p in [OUT/'post/repeat-policy-freeze.json',OUT/'cache/freeze.json',OUT/'coverage/supplement/register.json'] if p.is_file()]
    freezes+=[ROOT/'results/v0.23/pro-prefill/freeze.json',ROOT/'results/v0.24/kda-prefetch/supplement/register.json']
    write(OUT/'model.json',dict(version='0.25.0',freezes=[stamp(p) for p in freezes],sources=[stamp(p) for p in sorted((ROOT/'operator_model').glob('*.py'))],
                                 chips=[stamp(p) for p in sorted((ROOT/'configs/chips').glob('Ascend*.json'))]))
    evidence=[]
    def add(op,role,chip,shape,pred,ca,future,receipt,**fields):
        evidence.append(dict(op=op,role=role,chip=chip,shape=shape,model_us=pred,ca_us=ca,
                             signed_pct=100*(pred/ca-1) if pred is not None and ca is not None else None,
                             future_970_us=future,receipt=str(receipt.relative_to(ROOT)),**fields))
    fa=read(OUT/'coverage/supplement/score.json')
    for r in fa['rows']:
        add('fa','冻结后独立 '+r['cohort'],r['sku'],r['shape'],r['candidate'],r['ca'],r['future_us'],ROOT/r['receipt'],v024_us=r['old'])
    for r in mixed['rows']:
        add('rmsnorm','独立，纯 AIC 后，'+r['state'],r['chip'],r['shape'],r['preserve_us'],r['ca_us'],r['future_us'],OUT/'mixed-stream/comparison.json',v024_us=r['public_us'])
    for r in post['rows']:
        # The future_us field in this campaign refers to the same experimental schedule.
        add('mhc_post','独立，逐 token 候选，'+r['mode'],r['chip'],r['shape'],r['candidate_us'],r['ca_us'],r['future_us'],OUT/'post/evaluation.json',v024_us=r['public_us'],production=False)
    for r in kda['rows']:
        add('kda_chunk V2','冻结后三段总计',r['sku'],r['shape'],r['candidate'][-1],r['actual'][-1],r['extrapolated_970_us'],kda_path)
    for p in sorted((ROOT/'results/v0.23/pro-prefill').glob('pro-b8-*-blocks.json')):
        r=read(p)
        proof=p.with_name(p.name.replace('-blocks',''))
        assert read(proof)['qualified']
        add('cast_bf16_fp32','Pro 整批独立 CA',r['chip'],r['shape'],r['model_us'],r['ca_us'],r['projection_970dt_us'],proof)
    write(OUT/'evaluation.json',dict(rows=evidence,historical_audit=stamp(OUT/'coverage/audit.json'),source_freezes=[stamp(p) for p in freezes]))
    lines=['# 冻结预测与实测逐形对照','','预测、CA、970DT 情景均为设备微秒。正误差表示高估。Post 逐 token 路线处于验证中。','',
           '| 算子 | 角色 | 平台 | 每卡 shape | 模型 / CA µs | 误差 | 970DT 外推 µs |','|---|---|---|---|---:|---:|---:|']
    for r in evidence:
        lines.append(f"| {r['op']} | {r['role']} | {r['chip'][6:11]} | {r['shape']} | {r['model_us']:.3f} / {r['ca_us']:.3f} | {r['signed_pct']:+.2f}% | {r['future_970_us']:.3f} |")
    lines+=['','[逐窗收据、数据角色与预测版本](evaluation.json) · [其他路线的全部历史验证](coverage/report.md) · [总误差分析](../../docs/error-analysis.md)']
    (OUT/'shape-errors.md').write_text('\n'.join(lines)+'\n')
    p=ROOT/'docs/error-analysis.md'
    text=p.read_text().splitlines()
    def row(name,cells,after=None):
        label=f'| {name} |'
        line=label+' '+' | '.join(cells)+' |'
        if any(s.startswith(label) for s in text):
            i=next(i for i,s in enumerate(text) if s.startswith(label));text[i]=line
        else:
            i=next(i for i,s in enumerate(text) if s.startswith('| '+after+' |'));text.insert(i+1,line)
    metrics=[next(r['model'] for r in mixed['metrics'] if r['chip']==c and r['state']=='all') for c in (DT,PR)]
    row('RMSNorm（已知纯 AIC 间隔）',[f"{m['n']} · {m['wape']:.2%} / {m['max']:.2%}" for m in metrics]+['新冻结，同一流跨列偏置 Addmm，完整前序及物理退出',
        ' / '.join('本组达标' if m['wape']<=.2 and m['max']<=.4 and m['n']>=10 else ('样本待补' if m['n']<10 else '超限') for m in metrics)],'RMSNorm（可见部分前序）')
    def a_metric(c,role):
        return next(m['current'] for m in audit['summary'] if m['op']=='fa' and m['sku']==c and m['role']==role)
    row('FA',[f"{m['shapes']} · {m['wape']:.2f}% / {m['max_ape']:.2f}%" for m in [a_metric(DT,'development replay v0.25'),a_metric(PR,'development replay v0.25')]]+['当前资源模型，全部历史回放，40 / 41 窗','回放达标 / 回放达标'])
    cells=[]
    verdict=[]
    for chip in (DT,PR):
        group=fa['timing_by_sku'][chip]
        n=group['shapes']
        cur=group['metrics']['candidate']
        cells.append(f"{n} · {cur['wape_pct']:.2f}% / {cur['max_ape_pct']:.2f}%")
        verdict.append('样本待补' if n<10 else '本组达标' if cur['wape_pct']<=20 and cur['max_ape_pct']<=40 else '超限')
    row('FA（新形）',cells+['两批前向独立，同一冻结资源模型',' / '.join(verdict)],'FA')
    ks=[kda['metrics'][c]['total']['candidate'] for c in (DT,PR)]
    row('KDA chunk，FLA 26.9.1 三段 V2',[f"{m['n']} · {m['wape']:.2f}% / {m['maximum']:.2f}%" for m in ks]+['两批前向独立，同一冻结模型',' / '.join('本组达标' if m['n']==10 and m['wape']<=20 and m['maximum']<=40 else '见累计进度' for m in ks)])
    for label,verdict in [('Cast BF16→FP32','达标 / 容量组超限'),('Kimi KDA beta Sigmoid','达标 / 容量组超限'),('mHC Sinkhorn','多 token 组超限 / 达标'),('mHC PreSinkhorn','达标 / 多 token 组超限')]:
        i=next(i for i,s in enumerate(text) if s.startswith('| '+label+' |'))
        cols=text[i].split('|');cols[-2]=' '+verdict+' ';text[i]='|'.join(cols)
    p.write_text('\n'.join(text)+'\n')
    p=ROOT/'README.md';text=p.read_text().splitlines()
    idx=next(i for i,s in enumerate(text) if s.startswith('隔离安装的 v0.'))
    q=perf['predictions']
    text[idx]=f"隔离安装的 v0.25 wheel 在三种芯片配置上核对 {q['n']:,} 个常规输入，单次查询 P95 为 {q['p95_ms']:.3f} ms、最大 {q['max_ms']:.3f} ms。跨纯 Cube 的 {perf['mixed']['n']} 次有序派发最大 {perf['mixed']['max_ms']:.3f} ms，部分历史的 {perf['partial']['n']} 次调用最大 {perf['partial']['max_ms']:.3f} ms。设备延迟按 µs 返回。[安装版开销](results/v0.25/wheel-check.json) · [逐算子误差](docs/error-analysis.md)"
    p.write_text('\n'.join(text)+'\n')


if __name__=='__main__':
    main()
