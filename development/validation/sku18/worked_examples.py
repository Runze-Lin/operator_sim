"""Render two development-trace examples offline using the v0.17 paired archive."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate
from development.validation.sku18.rms_evidence import digest, spans
from development.validation.sku18 import rms_examples

SHAPES = {'quant':[5,320], 'rmsnorm':[64,2048]}
CHIPS = {'dt':'Ascend950DT_9582','pr':'Ascend950PR_957b','future':'Ascend970DT'}
NAMES = {'quant':'DynamicMXQuant','rmsnorm':'RMSNorm'}
QUEUES = ['SCALAR','MTE2','SIMD','MTE3']
COLORS = ['#66768c','#3a75af','#138377','#ce882d']


def merge(intervals):
    result = []
    for a,b in sorted(intervals):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1],b)
        else:
            result.append([a,b])
    return result


def critical(record):
    path = ROOT/record['trace'][0]
    assert digest(path)==record['trace_sha256'][0],path
    origin,start,end,rows = spans(path)
    series = defaultdict(list)
    for queue,a,b,_ in rows:
        if queue.startswith(('RVEC','VEC','SIMD')):
            queue='SIMD'
        if queue in QUEUES:
            series[queue].append((a-origin,b-origin))
    assert abs(end-origin-record['ca_us']) < .003
    return ({q:merge(series[q]) for q in QUEUES},start-origin,end-origin)


def figure(op,s,results,pair):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(9.0,6.6),layout='constrained')
    grid=fig.add_gridspec(3,1,height_ratios=[1.2,1,1])
    axis=fig.add_subplot(grid[0,0])
    for delta,key,label,color in [(-.23,'prior','v0.17','#94adc1'),(0.,'model','v0.18 resource','#377a9b'),(.23,'measured','CA measured','#d3a663')]:
        xs=[];ys=[]
        for n,alias in enumerate(CHIPS):
            val=results[alias][key]
            if val is not None:
                xs.append(n+delta);ys.append(val)
        bars=axis.bar(xs,ys,.20,label=label,color=color)
        axis.bar_label(bars,labels=[f'{n:.2f}' for n in ys],padding=2,fontsize=8)
    axis.set_xticks([0,1,2],['950DT','950PR','970DT scenario'])
    axis.set_ylim(0,max(max(t['prior'],t['model'],t['measured'] or 0) for t in results.values())*1.25)
    axis.set_ylabel('Device time (µs)')
    axis.set_title(f'{NAMES[op]} {s}   |   Development replay')
    axis.legend(loc='upper right',ncols=3,fontsize=8)
    traces={a:critical(pair[a]) for a in ('dt','pr')}
    limit=max(v[2] for v in traces.values())*1.025
    for i,alias in enumerate(('dt','pr')):
        ax=fig.add_subplot(grid[i+1,0])
        series,start,end=traces[alias]
        for row,q in enumerate(QUEUES):
            ax.broken_barh([(a,b-a) for a,b in series[q]],(row-.28,.56),facecolors=COLORS[row])
        ax.set_yticks(range(len(QUEUES)),QUEUES);ax.set_ylim(3.65,-.65)
        ax.axvline(end,color='#32485a',lw=.9,ls='--')
        ax.set_xlim(0,limit);ax.grid(axis='x',alpha=.17)
        ax.set_title(f'{CHIPS[alias].split("_")[0][6:]}  |  last-finishing AIV from {start:.3f} to {end:.3f} µs',loc='left',fontsize=10)
        if alias=='pr':ax.set_xlabel('Time from earliest device instruction across all cores (µs)')
    dst=ROOT/'docs/examples/figures'/f'{op}.png'
    fig.savefig(dst,dpi=165);plt.close(fig)
    return traces


def rms_single_tile_prices(shape):
    """Expand each single-tile dependency and its startup-state price in the chosen example."""
    from operator_model import load_chip
    from operator_model.rms_v12 import parameters
    from operator_model.rms_v11 import pressure
    from operator_model.wide_v10 import parameters as startup_parameters
    from math import ceil
    p=parameters(); overhead=startup_parameters()['phases']['rmsnorm']
    lines=['### 两笔写是怎样重叠的','',
           '以统计量写的发起时刻为 0。令 Δ 为依赖归一化工作完成后主输出写的发起时刻，s₁、s₂ 分别为两笔写的服务周期。主输出的完成时刻为 max(s₁,Δ+s₂)。各相交区间合并计一次。以下三端均只有一个 UB tile。','',
           '| 平台 | s₁ = 固定周期 + 邻近写者项 | Δ = 固定周期 + 归一化组工作 | s₂ = 固定周期 + 写 KiB × 斜率 | 写并集 / 隐藏 / 暴露（周期） |',
           '|---|---|---|---|---|']
    states=['','对于同一形状，启动状态改变关键核的到达、控制、投入读和向量发射时间。以下给出未知 H 时实际进入总预测的两个端点及权重。']
    t,d=shape
    for alias,name in CHIPS.items():
        chip=load_chip(name)
        xi=estimate('rmsnorm',shape,chip=chip,timing_mode='resource')
        diag=xi.projection['timing_diagnostics'];plan=diag['target_plan']
        n,=plan['ub_tile_rows'];a=plan['active_aiv']
        near=pressure(a,n,p['window_bytes']);vec=diag['vector_price_cycles_per_reference_cycle']
        z=ceil(d/128)*n;y=2*d*n/1024
        start=p['normalize'][0]+p['normalize'][1]*z*vec
        s1=p['rstd'][0]+p['rstd'][1]*near;s2=p['y'][0]+p['y'][1]*y
        union=diag['store_union_cycles'];exposed=diag['stages_us']['MTE3']*1000*chip.frequency_ghz
        lines.append(f'| {name} | {p["rstd"][0]:.2f}+{p["rstd"][1]:.2f}×{near:g}={s1:.2f} | '
                     f'{p["normalize"][0]:.2f}+{p["normalize"][1]:.4f}×{z}×{vec:.5f}={start:.2f} | '
                     f'{p["y"][0]:.2f}+{y:.2f}×{p["y"][1]:.4f}={s2:.2f} | '
                     f'{union:.2f} / {union-exposed:.2f} / {exposed:.2f} |')
        times=diag['conditional_us'];lo=min(times,key=times.get);hi=max(times,key=times.get)
        w=times[hi]/(times[lo]+times[hi])
        states.append(f'- {name} 的 {lo} 与 {hi} 各为 {times[lo]:.5f}、{times[hi]:.5f} µs；'
                      f'权重为 {w:.5f}、{1-w:.5f}。例如 SCALAR 对应 DT 截距 {overhead["SCALAR"][lo]:.5f}、'
                      f'{overhead["SCALAR"][hi]:.5f} µs，按此权重混合并乘 1.65/{chip.frequency_ghz:g}，'
                      f'得 {diag["stages_us"]["SCALAR"]:.5f} µs。')
    lines.extend(states[:2]+['']+states[2:])
    return '\n'.join(lines)


def render(op,s,pair):
    shape,attrs=pair['shape'],pair['attrs']
    api={'quant':'aclnnDynamicMxQuant','rmsnorm':'aclnnRmsNorm'}[op]
    output={}
    work=None
    for alias,name in CHIPS.items():
        new=estimate(op,shape,attrs,chip=name,dtype='bf16',timing_mode='resource')
        assert new.latency_us is not None and new.projection['reference_latency_us'] is None
        work=new.work
        ca=pair[alias]['ca_us'] if alias!='future' else None
        output[alias]=dict(prior=pair[alias+'_us'],model=new.latency_us,measured=ca,
                           cores=new.active_aiv,observed=pair[alias].get('active_aiv') if ca else None)
    traces=figure(op,s,output,pair)
    if op=='quant':
        from development.validation.sku18.quant_examples import explain
    else:
        explain=rms_examples.explain
    text=explain(op,shape,attrs)
    if op=='rmsnorm':
        text+='\n\n'+rms_single_tile_prices(shape)
    lines=[f'# {NAMES[op]} 算例：本卡 {shape[0]} 行 × {shape[1]} 列','',
           f'**开发回放。** 本例复用已归档的两个 950 SKU 的合格 CA 窗口，定位 v0.17 的误差。970DT 的两列模型时间均为架构情景外推。新的独立 shape 单列于[总误差报告](../error-analysis.md)。','',
           f'公开调用为 CANN 9.2.1 ops-nn 的 `{api}`，输入 BF16。公共接口统计 {work["vec_ops"]:,} 次有用 Vector 标量等价运算、0 次 Cube 运算。逻辑输入与输出分别为 {work["logical_read_bytes"]:,} 和 {work["logical_write_bytes"]:,} B。' +
           ('向量计数包含绝对值和分组最大值比较；scale 编码未展开。' if op=='quant' else '导出每行 FP32 rstd，归约与乘法按 FP32 计数。'),'',
           '```python','from operator_model import estimate','',
           "for chip in ('Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT'):",
           f"    r = estimate('{op}', {shape}, dtype='bf16', chip=chip, timing_mode='resource')",
           "    print(r.latency_us, r.active_aiv, r.projection['stages_us'])",'```','',
           '## 同一输入的设备窗口','',
           '| 平台 | v0.17 µs | v0.18 µs | CA µs | v0.17 / v0.18 有符号误差 | v0.18 / CA 活跃 AIV |',
           '|---|---:|---:|---:|---:|---:|']
    for alias,t in output.items():
        old,new,ca=t['prior'],t['model'],t['measured']
        err=f'{(old/ca-1)*100:+.2f}% / {(new/ca-1)*100:+.2f}%' if ca else 'N/A'
        measured=f'{ca:.5f}' if ca else 'N/A'
        lines.append(f"| {CHIPS[alias]} | {old:.5f} | {new:.5f} | {measured} | {err} | {t['cores']} / {t['observed'] or 'N/A'} |")
    lines.extend(['',('v0.17、v0.18 都显式使用资源路径。' if op=='quant' else
                     'v0.17 使用当时的公共默认 auto（DT 校准，PR/970 为 DT 阶段投影）；v0.18 三端都显式使用 resource，表中默认未传 resident_aiv。'),
                  '误差以 (预测−实测)/实测 计算。','',
                  f'![{NAMES[op]} 两个 950 实测临界核的读、算、写队列与三端模型时间](figures/{op}.png)','',
                  '下方两幅时间线各选该次 kernel 最晚结束的 AIV，横轴以全核最早指令为零。每条色带合并同一队列有有效持续时间的指令，跳过 duration-excluded 记录。两条色带在同一时刻出现表示流水重叠。虚线是全核窗口终点。','',
                  '| 实测平台 | 实际 kernel / trace SHA 前 12 位 | 尾核起点 µs | SCALAR / MTE2 / SIMD / MTE3 在途并集，各 µs | 输出校验 |','|---|---|---:|---|---|'])
    for alias in ('dt','pr'):
        rec=pair[alias];intervals,start,_=traces[alias]
        spans_=' / '.join(f'{sum(b-a for a,b in intervals[q]):.4f}' for q in QUEUES)
        checks=rec.get('correctness') or []
        if isinstance(checks,str):checks=[checks]
        lines.append(f'| {CHIPS[alias]} | `{rec["kernel"]}` / `{rec["trace_sha256"][0][:12]}` | {start:.4f} | {spans_} | '+'；'.join(f'`{c}`' for c in checks)+' |')
    if op=='quant':
        lines+=['','DT 与 PR 各用 10 个 AIV，行内分成 256 列和 64 列两片。两次实测最晚完成的都是 64 列尾片，分别位于 scale 偏移 48 和 38，只读 128 B、写 2 B scale。模型选中了偏移 30 的 256 列跨段写者，因此仍高估 PR 12.99%。图中画的是实测尾核；下方公式从模型选中的核推导。']
    if op=='rmsnorm':
        lines+=['','[64,2048] 在 PR 分给 32 核，每核两行。CA 的 MTE3 在途 1.3576 µs，扣除 SIMD 重叠 0.3891 µs 后，净暴露 0.9685 µs。v0.17 把该项估为 4.6404 µs，新版为 1.1011 µs；总时间仍低估 20.2%，主要差在临界核启动错峰与 SCALAR。']
        lines+=['','DT 的 compact DB 使用运行时入口 `entry_kernel_1388`；PR 官方 CA 导出包内专有 kernel 名。两个平台都已检查 y 与 rstd。']
    lines.extend(['','## 从输入到预测','',text,'',
                  f'[v0.17 双 SKU 测量清单和冻结预测](../../results/v0.17/sku/pairs.json) · [v0.18 参数、机制与误差来源](../../results/v0.18/{"quant" if op=="quant" else "rms"}-method.md) · [复现程序](../../development/validation/sku18/worked_examples.py)'])
    (ROOT/'docs/examples'/f'{op}.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--op',choices=SHAPES)
    args=parser.parse_args()
    archive=json.loads((ROOT/'results/v0.17/sku/pairs.json').read_text())
    for op,s in SHAPES.items():
        if args.op and args.op!=op:continue
        pair=next(r for r in archive if r['op']==op and r['shape']==s)
        render(op,s,pair)
