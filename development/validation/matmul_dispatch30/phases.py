"""Attribute signed time-ratio residuals to observed intervals, using existing small full windows."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.matmul_dispatch30.device import CHIPS,OUT,ca,save
from operator_model import estimate


def collect():
    small=ca.read(OUT/'score.json')['rows']
    rows=[]
    for r in small:
        if r['shape'][0]>144 or r['status']!='qualified':continue
        doc=ca.read(OUT/r['source'])
        stage,=doc['stages']
        # A per-core shard uses its own clock origin; the vendor aggregate is aligned to the whole launch.
        global_path=ROOT/stage['trace']
        source=ca.read(global_path)
        timed=[e for e in source['traceEvents'] if e['ph']=='X']
        first=min(e['ts'] for e in timed); last=max(e['ts']+e['dur'] for e in timed)
        core=max(stage['windows'],key=lambda c:stage['windows'][c][1])
        payload=[e for e in timed if e['pid']==core and e['name']=='MOV_OUT_TO_L1_MULTI_ND2NZ']
        mmads=[e for e in timed if e['pid']==core and e['name']=='MMAD']
        if not payload or not mmads:continue
        knots=[first,min(e['ts'] for e in payload),min(e['ts'] for e in mmads),
               max(e['ts']+e['dur'] for e in mmads),last]
        assert all(b>=a for a,b in zip(knots,knots[1:])) and abs(last-first-r['ca_us'])<2/1650
        prediction=estimate('matmul',r['shape'],chip=r['chip']); diag=prediction.projection
        if diag['target_plan'].get('waves')!=1:continue
        s=diag['stages_us']
        if any(s.get(key,0) for key in ('shared_link_floor','additional_output_waves')):continue
        budgets=[s['launch'],s['read_fill']+s['l0_lead'],s['page_intervals']+s['last_mmad'],s['write_tail']]
        assert abs(sum(budgets)-r['public_us'])<1e-8
        rows.append(dict(shape=r['shape'],chip=r['chip'],core=core,mmad_on_core=len(mmads),measured_us=r['ca_us'],
                         predicted_us=r['public_us'],ca_intervals=[b-a for a,b in zip(knots,knots[1:])],
                         model_intervals=budgets,source=r['source'],trace=r['trace']))
    save(OUT/'phases.json',dict(intervals=['first instruction to first operand request','first operand request to first MMAD issue',
             'first MMAD issue to final MMAD retirement','final MMAD retirement to final device event'],rows=rows))
    lines=['# 相同 Flash 权重上哪些阶段决定两端差别','',
      '坐标从全片首条指令开始。每个芯片取最后完成核的首笔实质 GM→L1 读、首个 MMAD 发起、末个 MMAD 退休与全片最后事件，按相邻界标给出四段不重叠的墙钟区间。模型侧对应启动、输入填充与 L0 提前量、乘加区间及写回尾段。实测列是设备事件界标；阶段内存在多种并发指令。时间单位 µs，每格为模型 / CA。','',
      '| M | 目标 | 末核 | 到首笔操作数读 | 读到首个 MMAD | 首个发起到末个 MMAD 退休 | 退休至全片结束 |',
      '|---:|---|---|---:|---:|---:|---:|']
    for r in rows:
        tag='DT' if r['chip']==ca.DT else 'PR'
        cells=[f'{x:.3f} / {y:.3f}' for x,y in zip(r['model_intervals'],r['ca_intervals'])]
        lines.append(f"| {r['shape'][0]} | {tag} | {r['core']} | {' | '.join(cells)} |")
    lines+=['','不同 SKU 在相同活动核数和指令数下也会有不同的设备窗。上表只归因到出现偏差的时间段；宿主采集噪声、内存初态及具体微架构成因还需要重复或受控实验。独立输入的预测沿用采前封存值。','',
            '[逐事件来源、时间差](phases.json) · [同形独立预测、实测、误差和真实 ELF](report.md)','']
    (OUT/'phases.md').write_text('\n'.join(lines))


if __name__=='__main__':collect()
