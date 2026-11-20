"""Report numerical and physical large-Post evidence as the serial experiments return."""
import argparse
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.31/post'
DT,PR,FUT='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'


def read(p):
    return json.loads(p.read_bytes())


def render():
    reg=read(OUT/'freeze.json')
    status=read(OUT/'state.json')
    rows=[]
    for j in reg['jobs']:
        for c in (DT,PR):
            label=j['op']+'-'+c; file=OUT/(label+'.json')
            p=j['predictions'][c]
            r=dict(op=j['op'],shape=j['shape'],chip=c,public_us=p['us'],known_first_us=p.get('known_first',{}).get('us'),
                   future970_extrapolated_us=j['predictions'][FUT]['us'],ca_us=None,
                   stage=status['phase'] if label==status.get('endpoint') else 'pending')
            if file.exists():
                d=read(file)
                r.update(stage='qualified' if d['qualified'] else 'unqualified',receipt=file.name,
                         reason=d.get('reason'),wall_min=d['cost']['seconds']/60,raw_gib=d['cost']['peak_bytes']/2**30)
                if d['qualified']:
                    s,=d['stages']
                    assert s['physical_end_all_lanes'] and d['golden']==j['expected_numerics']
                    r.update(ca_us=d['ca_us'][j['op']],kernel=s['kernel'],active_aiv=s['active_aiv'],sha=s['trace_sha256'])
                    r['signed_pct']=100*(p['us']/r['ca_us']-1)
            rows.append(r)
    def display(x):return '—' if x is None else f'{x:.3f}'
    lines=['# Flash 的 1,024 个调度 token，mHC Post 双库对照','',
           '每请求本步交 128 个 token、单卡 B=8 时，总计 1,024 行；实际 Flash 的隐藏宽为 5,120，残差通道数为 4。每次独立冷启动，全部 20,971,520 个 BF16 输出核对 golden。设备窗须跨全部参与核、逐核物理 END，校验库 .text 和官方 duration。两个库具有各自的数据布局、算子实现和宿主校验，分别计分。','',
           '| 输入与实现 | 芯片 | 默认预测 / CA µs | 已知首次预测 µs | 默认误差 | 970DT 外推 µs | 证据 |',
           '|---|---|---:|---:|---:|---:|---|']
    language={'acquiring':'采集中','qualifying':'校验中','pending':'排队','unqualified':'未合格'}
    for r in rows:
        phase=f"[合格全窗]({r['receipt']})" if r['stage']=='qualified' else language.get(r['stage'],r['stage'])
        err='—' if r['ca_us'] is None else f"{r['signed_pct']:+.2f}%"
        lines.append(f"| {r['op']} {r['shape']} | {r['chip'][6:11]} | {display(r['public_us'])} / {display(r['ca_us'])} | {display(r['known_first_us'])} | {err} | {display(r['future970_extrapolated_us'])} | {phase} |")
    lines+=['','内置接口为 aclnnMhcPost，vLLM 扩展为 aclnnHcPost。每个正式端点一次，预算 3 小时、24 GiB。每端真实墙钟、原始盘和分核将随回执写入 JSON。已知首次列用于梳理驻留状态，缺少前序的调用仍按默认模型评分。970DT 的预测没有设备测量。','','[采前预测和口径](freeze.json) · [逐窗与成本](score.json) · [算例与方法](../../../docs/examples/mhc_post.md)','']
    (OUT/'report.md').write_text('\n'.join(lines))
    (OUT/'score.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
    return tuple((j['stage'],j.get('receipt')) for j in rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');arg=parser.parse_args()
    previous=None
    while True:
        stamps=[(p.name,p.stat().st_mtime_ns) for p in OUT.glob('*.json') if p.name not in ('score.json','freeze.json')]
        if previous!=stamps:
            previous=stamps; status=render()
        if not arg.watch or all(x[0] in ('qualified','unqualified') for x in status):break
        time.sleep(30)
