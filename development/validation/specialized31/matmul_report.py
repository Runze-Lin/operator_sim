"""Summarize matched Matmul shapes without mixing a prospective test with development."""
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
from release import OUT,DT,PR,metrics,read,save


def render():
    rows=[r for r in read(OUT/'replay.json')['rows'] if r['op']=='matmul']
    groups={'全部已归档':rows,'既有应用形':[r for r in rows if r['role']=='v0.28 application replay'],
            'B11 新设备留出':[r for r in rows if r['role']=='v0.31 prospective, device']}
    lines=['# 公共 Matmul 的同集对照','',
           '每一行的旧版和新版使用完全相同的设备时间。第一组包含历史机制开发，独立 B11 设备时间另列。统计单位为 %。','',
           '| 数据角色 | 芯片 | 不同形 / 窗 | v0.30 WAPE / 最大 | v0.31 WAPE / 最大 |',
           '|---|---|---:|---:|---:|']
    def fmt(x):
        return f"{x['wape']:.2f} / {x['maximum']:.2f}" if x['n'] else 'N/A'
    pairs={}
    for name,group in groups.items():
        matched=defaultdict(lambda:defaultdict(list))
        for c in (DT,PR):
            ss=[v for v in group if v['chip']==c]
            old,new=metrics(ss,'previous_us'),metrics(ss,'current_us')
            lines.append(f"| {name} | {c[6:11]} | {new['shapes']} / {new['n']} | {fmt(old)} | {fmt(new)} |")
            for v in ss:matched[tuple(v['shape'])][c].append(v)
        pairs[name]=[]
        for shape,loc in sorted(matched.items()):
            if len(loc)!=2:continue
            def ratio(k):
                return median(x[k] for x in loc[PR])/median(x[k] for x in loc[DT])
            pairs[name].append(dict(shape=list(shape),ca_us=ratio('ca_us'),current=ratio('current_us'),
                                    previous=ratio('previous_us'),copy_dt=1.))
    lines+=['','按每端同一 shape 的完整窗取中位数，再计算 PR/DT 耗时比。直接复用 DT 给 PR 的比值恒为 1。比值 WAPE 对各个实测比加权；本表中的比值没有时间单位。','',
            '| 数据角色 | 两端同形数 | v0.30 比值 WAPE / 最大 | v0.31 比值 WAPE / 最大 | 直接复用 DT |',
            '|---|---:|---:|---:|---:|']
    for name,g in pairs.items():
        cols=[fmt(metrics(g,k)) for k in ('previous','current','copy_dt')]
        lines.append(f"| {name} | {len(g)} | {' | '.join(cols)} |")
    lines+=['','重选输出块解决了 Pro 窄输出、长 K 上的约两倍高估。普通大 M 的 B 常驻、冷读与回写仍是不同问题。大形 v0.30 条件估计与默认时延分列。新形的工作区探针已知，设备耗时和真实 MMAD 在冻结后采集。970DT 沿用实现规则后的核数、分块、吞吐和带宽重新计算，均为外推。','','[全量逐形和旧预测](../replay.json) · [逐形跨芯片比值](ratios.json) · [独立 CA、分核与逐页指令](device/report.md) · [大形预填充](../../v0.30/matmul/comparison.md)','']
    save(OUT/'matmul/ratios.json',dict(definition='PR/DT, measured ratios dimensionless; see ca_ratio field',sets={
        name:[{('ca_ratio' if k=='ca_us' else k):v for k,v in row.items()} for row in grp] for name,grp in pairs.items()}))
    (OUT/'matmul/report.md').write_text('\n'.join(lines))


if __name__=='__main__':render()
