"""Plot registered new long shapes from terminal-core trace shards, no new CA."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.vector21.campaign import OUT, S, DT, read
from development.validation.vector21.figures import render
from development.validation.sigmoid20.figure import data as sigmoid_trace, merge
from campaign_collect import events


def cast_trace(row):
    stage=row['stage']
    core=max(stage['windows'],key=lambda c:stage['windows'][c][1])
    shard=ROOT/Path(stage['trace']).parent/core/'trace.json'
    first=float('inf')
    last=-first
    dma={'MTE2':[],'MTE3':[]}
    vec=[]
    for e in events(shard):
        if e.get('ph')!='X':
            continue
        a,w=e['ts'],e['dur']
        first=min(first,a);last=max(last,a+w)
        if e['name']=='RV_VCVT_F2F' and w>0:
            vec.append([a,a+w])
        if e['name']=='MOV_SRC_TO_DST_ALIGNv2':
            d=json.loads(e['args']['detail']);q=d['queue_type']
            if q in dma and not d.get('is_duration_excluded'):
                dma[q].append(dict(start=a,length=w,bytes=d['processed_bytes']))
    lo,hi=stage['windows'][core]
    assert abs(last-first-(hi-lo))<.03
    origin=first-lo
    for seq in dma.values():
        seq.sort(key=lambda e:e['start'])
        for e in seq:
            e['start']-=origin
    return dict(terminal=core,ca_us=row['ca_us'],trace_sha256=stage['trace_sha256'],
                dma=dma,conversion_union=[[a-origin,b-a] for a,b in merge(vec)])


def publish():
    rpath=OUT/'holdout.json'
    jobs={j['id']:j for j in read(OUT/'freeze.json')['jobs']}
    lines=['# 新长 prefill 的冻结预测与独立 CA','','各图上栏取 v0.21 的冻结方法，下栏使用该次通过逐核时间窗与全部输出校验的终止核。横轴从全核最早首条指令开始。']
    for row in read(rpath) if rpath.exists() else []:
        if row['shape'][0]<393216 or not row['qualified']:
            continue
        tag='dt' if row['chip']==DT else 'pr'
        op=row['op'];slug='holdout-'+row['id']+'-'+tag
        out=OUT/(slug+'.json');png=ROOT/'docs/examples/figures'/('v021-'+slug+'.png')
        reg=jobs[row['id']]['targets'][row['chip']]
        if not out.exists() or not png.exists():
            try:
                source=(sigmoid_trace(dict(kernel=row['stage'],sku=row['chip'],shape=row['shape']))
                        if op==S else cast_trace(row))
                render(op,row['shape'],row['chip'],source,slug,reg)
            except AssertionError:
                # A missing terminal descriptor disqualifies tile alignment;
                # it need not invalidate a separately qualified full window.
                lines.append(f"\n{row['shape']} {row['chip']} 的终止核描述符与模型分块暂不能逐个配对。")
                continue
        d=read(out)
        assert abs(reg['us']-d['model_us'])<1e-7
        lines+=['',f"## {op} · {row['shape']} · {row['chip']}",'',
                f"冻结模型 {reg['us']:.3f} µs；完整 CA {row['ca_us']:.3f} µs；相对误差 {100*(reg['us']/row['ca_us']-1):+.1f}%。",'',
                f"![模型与独立 CA 逐块同轴流水](../../../docs/examples/figures/{png.name})",'',
                f"[完整时间戳、分块与 trace 哈希]({out.name})"]
    if len(lines)==3:
        lines+=['','首个完整新长 shape 正在采集。']
    (OUT/'independent-figures.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
    publish()
