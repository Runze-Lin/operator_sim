"""Generate the standalone quantification and error report from immutable forecasts."""
from study import DT,PR,FUT,OUT,ROOT,FROZEN,OLD,read


def render(d):
    reg=read(FROZEN)
    path=OUT/'report.md'
    lines=['# DynamicMXQuant · 大输入共享搬运','',
        '|输入与计时口径|芯片|形状|公开 / 旧资源 / 新情景 (µs)|实测 (µs)|公开 / 新情景有符号误差|','|---|---|---|---:|---:|---:|']
    for v in d['measurements']:
        us=v['us'];field='待测' if us is None else f'{us:.3f}'
        err='N/A' if us is None else f"{100*(v['old_us']/us-1):+.2f}% / {100*(v['candidate_us']/us-1):+.2f}%"
        lines.append(f"|{v['role']} · {v['metric']}|{v['chip'][6:11]}|{v['shape']}|{v['old_us']:.3f} / {v['resource_us']:.3f} / {v['candidate_us']:.3f}|{field}|{err}|")
    path.write_text('\n'.join(lines)+'\n')
