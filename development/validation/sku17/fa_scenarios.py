"""Compare the same dense FA dispatch under DT and proposed 970DT resources."""
import json
from dataclasses import replace
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate,load_chip

CASES=[
 ('单头校验，32k',(9,32768,64,1,1)),('单头校验，128k',(9,131072,64,1,1)),
 ('单头校验，256k',(9,262144,64,1,1)),('单头校验，512k',(9,524288,64,1,1)),('校验 Q17，B4 H8',(17,32768,64,4,8)),('校验 Q9，B4 H8',(9,32768,64,4,8)),
 ('并发校验，B16 H8',(9,32768,64,16,8)),
 ('并发校验，B32 H8',(9,32768,64,32,8)),('并发校验，B256 H8',(9,32768,64,256,8)),
 ('prefill，B1 H8',(512,32768,64,1,8)),('并发校验，B128 H8',(9,32768,64,128,8)),('prefill，B4 H8',(512,32768,64,4,8)),
 ('prefill，B16 H8',(512,32768,64,16,8)),('单波向量工作，32头',(32,8192,64,1,32))]

def capture(shape,chip):
    r=estimate('fa',shape,chip=chip,timing_mode='resource')
    p=r.projection;diag=p['timing_diagnostics'];w=max(diag['wave_types'],key=lambda w:w['waves']*w['first']['ii_cycles'])
    return dict(us=r.latency_us,cores=[r.active_aic,r.active_aiv],plan=p['target_plan'],
                stages=p['stages_us'],bottleneck=w['first']['steady_bottleneck'],
                kv_read_bw_fraction=w['steady_kv_read_fraction_of_hbm_peak'],
                full_tile_cycles=w['first']['demands_cycles'],shape=shape)

def main():
    dt=load_chip();future=load_chip('Ascend970DT')
    pooled=replace(future,details={**future.details,'fa_aiv_per_aic':2})
    rows=[dict(name=n,dt=capture(sh,dt),scenario=capture(sh,future),pooled=capture(sh,pooled)) for n,sh in CASES]
    out=ROOT/'results/v0.17';(out/'fa-scenarios.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
    names=dict(cube='Cube',vector='Vector',issue_control='K/V 发起节拍',k_v_read='K/V 读取',mte1='MTE1',fix='FIX')
    lines=['# FA 在 950DT 与 970DT 情景上的瓶颈','','单卡 BF16、非因果 SBH、D64，Q/K/V 等头；使用同一套资源估时。按每个 shape 贡献稳态周期最多的一类 wave，选取其满 KV 块的最长资源需求。首发和排空仍计入总时间。表中两端均为模型预测，970DT 全部为外推。B/H 已是本卡值。','','| 场景 | [Q,KV,D,B,H] | DT → 970 核 (C,V) | KV 块长 DT → 970 | 波数 DT → 970 | DT → 970 稳态瓶颈 | DT / 970 µs | DT/970 | 970 池化假设 µs |','|---|---|---|---:|---:|---|---:|---:|---:|']
    for r in rows:
        a,b=r['dt'],r['scenario']
        lines.append(f"| {r['name']} | {a['shape']} | {a['cores']} → {b['cores']} | {a['plan']['tile'][1]} → {b['plan']['tile'][1]} | {a['plan']['waves']} → {b['plan']['waves']} | {names[a['bottleneck']]} → {names[b['bottleneck']]} | {a['us']:.3f} / {b['us']:.3f} | {a['us']/b['us']:.3f}× | {r['pooled']['us']:.3f} |")
    lines+=['','默认 970 情景给每个繁忙 AIC 配一个 AIV；池化情景允许从空闲 AIV 借至最多两个。在高并发且 AIC、AIV 均满时，两种假设通常接近。单头 Q9 的 KV 长度只增加串行块数，核数增加很难缩短它；相同 shape 若库新增 split-KV 则应显式改变任务图。每波核数、每个瓶颈候选的周期、K/V 字节占情景 HBM 峰值比例见[逐场景数据](fa-scenarios.json)。该带宽利用率按模型冷读需求计算，不是物理计数器。']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,3.9))
    for ax,(q,b,h) in zip(axes,[(17,4,8),(9,16,8)]):
        row=next(r for r in rows if tuple(r['dt']['shape'])==(q,32768,64,b,h))
        for ix,(key,label) in enumerate([('dt','950DT'),('scenario','970DT scenario')]):
            costs=row[key]['full_tile_cycles']
            ax.barh([i+(ix-.5)*.34 for i in range(len(costs))],[v/row[key]['plan']['tile'][1] for v in costs.values()],.34,label=label,
                    color=['#447493','#cc743d'][ix])
        ax.set_yticks(range(len(costs)),list(costs));ax.invert_yaxis()
        ax.set_xlabel('Steady-state service (target cycles / KV row)')
        ax.set_title(f'Q={q}, B={b}, H={h}, KV=32768')
    axes[0].legend();fig.tight_layout();fig.savefig(out/'fa-bottlenecks.png',dpi=130);plt.close(fig)
    lines.extend(['','![FA 稳态资源竞争](fa-bottlenecks.png)','','两图将每个完整 KV 块的服务周期除以该芯片的 KV 块长以便比较。每个芯片最长一项决定稳态发起间隔。Q17、B4、H8 时，DT 选 128 个 KV 行且每 AIC 配两个 AIV，970 选 256 个 KV 行且每 AIC 配一个 AIV，两端的分块和绑定均发生变化。启动和尾部另计；970 曲线均为情景外推。Q=9 表示验证 8 个推测 token 加 1 个主模型 token。表中的 prefill 是单个 512 行块；生成 1k 输出的整网时间还需累加各校验步，并按实际接受长度及批量调度更新 KV。'])
    (out/'fa-scenarios.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
    main()
