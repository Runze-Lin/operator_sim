"""Update the human-readable table from verified receipts."""
from campaign import OUT,RAW,ROOT,CHIPS,read,stamp
import json
import os

def report():
    reg=read(OUT/'freeze.json');state=read(OUT/'state.json') if (OUT/'state.json').is_file() else {}
    matrix=[]
    lines=['# 大形 Cube、MoE 与 KDA 双端验证','',
           f"登记于 {reg['utc']}，当前状态 {state.get('phase','等待启动')}，端点 {state.get('endpoint','—')}。预测与 CA 时间单位均为 µs。",'',
           '| 算子 | 单卡 shape | DT 预测 / CA | PR 预测 / CA | 970DT 外推 | 双端状态 |','|---|---|---:|---:|---:|---|']
    f=lambda x:'—' if x is None else f'{x:,.2f}'
    for j in reg['jobs']:
        for i,o in enumerate(j['ops']):
            rec=[]
            for chip,tag in zip(CHIPS[:2],('dt','pr')):
                p=OUT/f"{j['id']}-{tag}.json";r=read(p) if p.is_file() else {}
                v=j['predictions'][o][chip];sc=v.get('independent_state_scenario',{}).get('scenario_us')
                row=dict(op=o,shape=j['shape'],chip=chip,prediction_us=v['us'],state_scenario_us=sc,ca_us=None if not r else r['operators'][i]['ca_us'],
                         status=('已合格' if r['verdict']['qualified'] else '不合格') if r else ('采集中' if (RAW/f"{j['id']}-{tag}"/'run.log').exists() else '已排队'),
                         signed_error_pct=None if not r else r['operators'][i]['signed_error_pct'],receipt=None if not r else str(p.relative_to(ROOT)),
                         wall_seconds=None if not r else r['cost']['seconds'],peak_bytes=None if not r else r['cost']['peak_bytes'])
                rec.append(row);matrix.append(row)
            future=j['predictions'][o][CHIPS[2]]
            def val(r):return '—（情景 '+f(r['state_scenario_us'])+'）' if r['state_scenario_us'] is not None else f(r['prediction_us'])
            a,b=rec
            ff=f(future['us']) if future['us'] else '情景 '+f(future.get('independent_state_scenario',{}).get('scenario_us'))
            lines.append(f"| {o} | {j['shape']} | {val(a)} / {f(a['ca_us'])} | {val(b)} / {f(b['ca_us'])} | {ff} | {a['status']} / {b['status']} |")
    lines+=['| KDA 融合 V1 | 公开 packed 950 路由 | — / — | — / — | — | 原库同步未合格 |',
            '','8 个端点是 4 组新几何的双端采集。MoE 的两个阶段及组合共用同一真实完整 run。双端都合格后才有实测耗时比。[冻结数值与旧实测](freeze.json) · [实时逐端矩阵及墙钟](matrix.json) · [采集状态](state.json)','']
    lines.append((OUT/'method.md').read_text())
    for name,data in [('report.md','\n'.join(lines)),('matrix.json',json.dumps(dict(utc=stamp(),state=state,rows=matrix),indent=2)+'\n')]:
        p=OUT/name;tmp=p.with_suffix('.'+str(os.getpid())+'.tmp');tmp.write_text(data);tmp.replace(p)

if __name__=='__main__':report()
