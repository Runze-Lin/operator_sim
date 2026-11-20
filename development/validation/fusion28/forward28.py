"""Publish rows with the pre-registered errors and true whole-device windows."""
import json
from math import ceil
from pathlib import Path
from statistics import mean
from landmarks import OUT,read
from schedule28 import load_chip
from score28 import DT,PR,FUT

def forward():
    reg=read(OUT/'register.json');d=read(OUT/'independent.json')
    all_jobs={(j['group'],j['index'],j['chip']):j for j in reg['jobs']}
    text=['# 冻结后的四个新几何','',
        '登记 [register.json](register.json) 在全部正式采集前封存。未知前序点由 shape、迭代次数与目标资源独立给出。读取真实时序后使用已登记的 H 对照同一模型，前序状态列用于解释。每端四种几何，共六个调用窗。970DT 均为外推。','',
        '| SKU | 已完成窗 / 几何 | 无前序 WAPE / 最大 | 等几何权重 WAPE | 已知 H WAPE / 最大 | 原公开无 H WAPE / 最大（可定价窗） | CA 运行墙钟 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def percent(metric):
        if not metric:return 'N/A'
        return f"{metric['wape']:.2f}% / {metric['max_ape']:.2f}%"
    for sku in (DT,PR):
        rs=[r for r in d['rows'] if r['sku']==sku];short=sku[6:11]
        if not rs:continue
        grouped={}
        for r in rs:grouped.setdefault((r['t'],r['iters']),[]).append(r)
        balanced=100*sum(mean(abs(x['point_us']-x['ca_us']) for x in v) for v in grouped.values())/sum(mean(x['ca_us'] for x in v) for v in grouped.values())
        ms=d['by_sku'][sku]['all']['models'];duration=sum(a['cost']['seconds'] for a in d['attempts'] if sku in a['receipt'])
        old=ms['old_unknown_us'];text.append(f"| {short} | {len(rs)} / {len(grouped)} | {percent(ms['point_us'])} | {balanced:.2f}% | {percent(ms['known_us'])} | {percent(old)}（{old['n'] if old else 0}） | {duration:.1f} s |")
    text.extend(['','| SKU | 调用/序号 | T / I | R / G | H / 状态 | v0.27 无 H µs | 冻结无 H（范围）µs | 带 H µs | 全 CA µs | 无 H 误差 | 970DT 无 H µs |',
                 '|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|'])
    rows=sorted(d['rows'],key=lambda r:(r['sku'],r['group'],r['index']))
    for r in rows:
        old='N/A' if r['old_unknown_us'] is None else f"{r['old_unknown_us']:.3f}";lo,hi=r['range_us'];err=100*(r['point_us']/r['ca_us']-1)
        text.append(f"| {r['sku'][6:11]} | {r['group']} / {r['index']} | {r['t']} / {r['iters']} | {r['rows']} / {r['g']} | {r['h']} / {r['state']} | {old} | {r['point_us']:.3f}（{lo:.3f}–{hi:.3f}）| {r['known_us']:.3f} | {r['ca_us']:.3f} | {err:+.2f}% | {r['projected_970dt_unknown_us']:.3f} |")
    text+=['','相同几何的高水位按原宿主实际发射顺序构造，独立定义是本轮拟合未见其输入及计时。等几何权重先对同一几何多次执行的绝对误差和 CA 时间分别取均值，再跨几何求和相除。所有已计分窗保留原始 SHA、数据完整性以及实际核数、VF 数、输出字节检查。[逐窗资源检查与双芯片设备时间比](independent.json) · [队列状态](state.json)']
    for sku in (DT,PR):
        rec=next((x for x in rows if x['sku']==sku and x['t']==1153 and x['index']==2),None)
        if rec:
            fn=f'independent-{sku[6:11].lower()}-t1153-i6'
            if (OUT/(fn+'.png')).is_file():text+=['',f'![{sku[6:11]} 新形的冻结模型与最晚结束 AIV 流水]({fn}.png)',f'[同轴 µs、原始 trace、VF 口径和全窗回执]({fn}.json)']
    if all(sum(r['sku']==s for r in rows)==6 for s in (DT,PR)):
        text+=['','## 两端计时能否分辨快慢','', '| T / I | 调用 | CA DT/PR | 未知 H DT/PR | 已知 H DT/PR |','|---|---|---:|---:|---:|']
        for x in d['paired_ratios']:text.append(f"| {x['t']} / {x['iters']} | {x['group']}/{x['index']} | {x['ca_dt_over_pr']:.4f} | {x['unknown_dt_over_pr']:.4f} | {x['known_dt_over_pr']:.4f} |")
        text+=['','比值大于 1 表示 PR 较快。接近 1 的差距可能小于各自的控制与 DMA 等待残差。试验使用同一 950 代架构、无硬件名分支的资源重定价。新四形覆盖小工作量和中间寄存器组数，尚缺十形独立新合同。']
    p=OUT/'forward.tmp';p.write_text('\n'.join(text)+'\n');p.replace(OUT/'forward.md')
if __name__=='__main__':forward()
