"""Render frozen model page intervals and measured CA instructions on a common axis."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import sys

from freeze import OUT, ROOT, DT, PR, digest
from extract import summarize
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

FIG=ROOT/'docs/examples/figures/matmul-tile-pages-pr.png'
PALETTE={'MTE2':'#137988','MTE1':'#5275ad','MMAD':'#7354a6','FIXP':'#ac4929'}


def draw_panel(ax,tiles,finish,*,measured,launch,core,maxrows,limit):
    ink='#192e3b';sub='#425866'
    ticks=[];labels=[];nt=0
    for ti,tile in enumerate(tiles):
        rect='×'.join(map(str,tile['rect']))
        for pg,page in enumerate(tile['pages']):
            y=-nt;ticks.append(y);labels.append(f'{rect} · K page {pg+1}');nt+=1
            for j,engine in enumerate(['MTE2','MTE1','MMAD']):
                pos=y+.22-j*.22
                ranges=page[engine] if measured else [page[engine]]
                for a,b in ranges:
                    ax.barh(pos,b-a,left=a,height=.14,color=PALETTE[engine],linewidth=0,
                            alpha=.52 if not measured and (pg or ti) and j<2 else 1)
                if measured and engine=='MMAD':
                    ax.vlines([a for a,_ in ranges],pos-.105,pos+.105,color='#492a77',linewidth=.7)
        y=-nt;nt+=1;ticks.append(y);labels.append(f'{rect} · write')
        write=tile['write']
        ax.barh(y,write[1]-write[0],left=write[0],height=.22,color=PALETTE['FIXP'],alpha=.41 if measured else 1)
        if not measured:
            ax.barh(y,write[2]-write[1],left=write[1],height=.22,color=PALETTE['FIXP'],alpha=.36)
        if ti+1<len(tiles):ax.axhline(y-.5,color='#dbe3e8',lw=.85)
    if not measured:
        t=launch;ax.barh(0,t,height=.41,color='#e5ecef')
        ax.text(t/2,0,'Launch',ha='center',va='center',fontsize=9,color=sub)
    ax.set_ylim(-maxrows+.45,.55);ax.set_xlim(0,limit+.55)
    ax.set_yticks(ticks,labels,fontsize=9.5)
    caption='Resource model · critical waves' if not measured else f'CA trace · {core} (last to finish)'
    ax.set_title(caption,loc='left',fontsize=13,weight='bold',color=ink,pad=10)
    ax.text(1,1.057,f'{finish:.3f} µs',transform=ax.transAxes,ha='right',va='bottom',fontsize=13,weight='bold',color=ink)
    ax.axvline(finish,color='#536776',ls=(0,(3,3)),lw=1.1)
    ax.xaxis.set_major_locator(MultipleLocator(1 if limit<18 else 2))
    ax.grid(axis='x',color='#dbe3e8',lw=.72);ax.set_axisbelow(True)
    ax.tick_params(axis='y',length=0,pad=10,labelcolor=ink);ax.tick_params(axis='x',colors=sub)
    for sp in ax.spines.values():sp.set_visible(False)


def render(frozen,data,image):
    model=frozen['resource_model'];d=model['diagnostics'];pl=d['target_plan']
    ref=[(model['model_tiles'],model['latency_us']),(data['ca_tiles'],data['ca_us'])]
    sizes=lambda tiles:sum(len(t['pages'])+1 for t in tiles)
    maxrows=max(sizes(t) for t,_ in ref)
    fig,axes=plt.subplots(2,1,figsize=(11.3,7.8+.25*max(0,maxrows-6)),sharex=True)
    fig.subplots_adjust(left=.207,right=.962,top=.81,bottom=.14,hspace=.4)
    ink='#192e3b';sub='#425866'
    fig.text(.207,.957,'950PR MatMulV3  |  1280 × 2048 × 512',fontsize=21,weight='bold',color=ink)
    fig.text(.207,.919,f"BF16 ND   ·   {pl['active_aic']} modeled / {data['core_count']} measured AICs   ·   "
            f"{d['mmads_total']} modeled / {data['mmad']} measured MMADs",fontsize=11.5,color=sub)
    fig.legend([Patch(color=x) for x in PALETTE.values()],['MTE2 / GM → L1','MTE1 / L1 → L0','Cube / MMAD','FIXP'],
        ncol=4,frameon=False,loc='upper left',bbox_to_anchor=(.198,.905),labelcolor=ink)
    limit=max(x for _,x in ref)
    for idx,(ax,(tiles,finish)) in enumerate(zip(axes,ref)):
        draw_panel(ax,tiles,finish,measured=bool(idx),launch=model['launch_us'],core=data['core'].split('.')[0],
                   maxrows=maxrows,limit=limit)
    axes[-1].set_xlabel('Device time from first instruction (µs)',fontsize=11,color=ink,labelpad=10)
    fig.text(.207,.043,'Model: faded read/load intervals follow the page issue spacing. CA: ticks mark individual MMAD issues.',fontsize=9.4,color=sub)
    fig.text(.207,.018,'Pale orange: model write drain (top); full FIXP descriptor, including dependency waits (bottom).',fontsize=9.4,color=sub)
    fig.savefig(image,dpi=160,facecolor='white')
    plt.close(fig)


def write(name,obj):
    path=OUT/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def publish():
    reg=json.loads((OUT/'freeze.json').read_text())
    receipt=json.loads((OUT/'observation.json').read_text());assert receipt['qualified']
    obs=receipt['observation'];stage=obs['stages'][0];source=ROOT/stage['trace']
    assert receipt['freeze_sha256']==digest(OUT/'freeze.json')
    try:
        case=summarize(source,reg['shape'],stage['trace_sha256'],stage['ca_us'])
    except (AssertionError,KeyError,ValueError) as exc:
        write('page-audit.json',dict(complete_page_mapping=False,ca_us=stage['ca_us'],kernel=stage['kernel'],
            active_aic=stage['active_aic'],active_aiv=stage['active_aiv'],measured_mmad=stage['mmad'],
            trace_sha256=stage['trace_sha256'],reason=repr(exc)))
        raise
    assert case['mmad']==stage['mmad'] and case['core_count']==stage['active_aic']
    case.update(chip=PR,shape=reg['shape'],kernel=stage['kernel'],api=obs['api'],
        fingerprint=stage['binary_sha256'],golden=obs['correctness'],trace=stage['trace'],
        trace_sha256=stage['trace_sha256'],freeze_sha256=digest(OUT/'freeze.json'),
        measured_window='max end on any core minus earliest issue from any core; official all-core makespan',
        scope='First and later GM/MTE1 model page starts reconstructed from steady issue intervals. Actual page correspondence checked from A/B global addresses, L1 range, MMAD fragment dimensions, flags and output destination.',
        page_decoder_sha256=digest(Path(__file__).with_name('extract.py')))
    write('page-audit.json',case)
    render(reg,case,FIG)
    m=reg['resource_model'];diag=m['diagnostics'];cost=obs['cost'];dt=json.loads(DT.read_text())
    hist='、'.join(f'{cnt} 个 {a}×{b}' for a,b,cnt in case['tile_histogram'])
    plan=diag['target_plan'];shape=reg['shape']
    pages=sorted({k for items in case['per_core_tiles'].values() for v in items for k in v['pages_k']})
    core_hist=Counter(len(v) for v in case['per_core_tiles'].values());percore='、'.join(f'{n} 核各 {k} 块' for k,n in sorted(core_hist.items()))
    first_model=m['model_tiles'][0]['pages'][0]['MMAD'][0]
    first_ca=case['ca_tiles'][0]['pages'][0]['MMAD'][0][0]
    dt_ratio=dt['ca_us']/case['ca_us'];pr_ratio=dt['model_us']/m['latency_us']
    stage_table=['| 所在关键路径 | 预测 µs | CA µs | 预测−CA µs |','|---|---:|---:|---:|']
    if [t['rect'] for t in case['ca_tiles']]==[t['rect'] for t in m['model_tiles']]:
        for idx,(mt,ct) in enumerate(zip(m['model_tiles'],case['ca_tiles'])):
            pair=(mt['pages'][0]['MTE2'][0],min(start for start,_ in ct['pages'][0]['MTE2']))
            milestones=[('首笔 GM→L1 发起',*pair)]
            for pg,(mp,cp) in enumerate(zip(mt['pages'],ct['pages'])):
                milestones.append((f'第 {pg+1} 页首条 MMAD 发起',mp['MMAD'][0],min(start for start,_ in cp['MMAD'])))
            milestones.append(('FIXP 阶段结束',mt['write'][-1],ct['write'][-1]))
            for caption,a,b in milestones:stage_table.append(f'| 第 {idx+1} 块 · {caption} | {a:.3f} | {b:.3f} | {a-b:+.3f} |')
    else:
        stage_table=['模型关键路径和实测末核处理的矩形序列不同。两栏以各自的真实矩形标注；逐页时刻见 JSON。']
    comparison='\n'.join(stage_table)
    text=f"""# 950PR Matmul 两波流水与逐页 CA

BF16 ND 上的 `aclnnMatmul`，轴顺序 [M,N,K] = {shape}。同一份可执行文件用于 950DT 和 950PR。PR 预测 **{m['latency_us']:.3f} µs**，完整 CA **{case['ca_us']:.3f} µs**，有符号误差 **{m['latency_us']/case['ca_us']-1:+.2%}**。模型与 CA 均从各自全核窗口的首条指令起算。采集前已冻结全部模型页、活动 AIC 与 MMAD 数。[冻结记录](freeze.json) · [数值及全核窗口](observation.json)

![950PR 同一横坐标上的模型与实际 CA 按输出矩形、K 页对照](../../../docs/examples/figures/matmul-tile-pages-pr.png)

|项目|模型|完整 CA|
|---|---:|---:|
|AIC / AIV|{plan['active_aic']} / {plan['active_aiv']}|{stage['active_aic']} / {stage['active_aiv']}|
|输出块|{plan['effective_output_tiles']}|{case['output_tiles']}|
|MMAD 总数|{diag['mmads_total']}|{case['mmad']}|
|有用 Cube 运算|{diag['logical_cube_ops']:,}|{case['cube_ops']:,}|
|图中首个 MMAD 发起 µs|{first_model:.3f}|{first_ca:.3f}|

模型把 40 个 256×256 母块映射到 28 个 AIC。第一轮 28 个整块，余下 12 个沿 M 各一分为二，第二轮是 24 个 128×256 矩形，各块分两页 K=256。每页用四个 K=64 的 MMAD。在确认的 PR trace 中，写出的几何直方图为 **{hist}**，每核分工为 **{percore}**，读描述符覆盖的 K 页深度为 {pages}。下栏按最后退休的 **{case['core']}** 的实际输出块标注。模型尾部取资源关键路径，下栏是实测末核；“首次 MMAD”比较的范围也据此确定。

{comparison}

模型的 MTE2 窗口表示两操作数到达本页预算，CA 首笔取同一页较早的 DMA 发起；FIXP 的总窗终点计入实际依赖等待。判断页间差异时还可以对比上表中相邻两行 MMAD 首发之差。

从原始 `FIX_L0C_TO_DST` 目的地址还原每个输出矩形的行、列起点，并验证整个 M×N 输出恰覆盖一次。每一页的 A/B 全局地址和实际字节按该矩形的两条边分别核对；L0A 与 L0B 的搬运大小随 M、N 各自变化，非整 16 边按寄存矩阵块粒度补齐。按矩阵 C 初始化和末条 unit flag 划分 MMAD 累加链，逐链验证累计 K、操作数和页内 MMAD 数；图中每个紫色竖刻度来自实际指令。图上 FIXP 浅色实际条包含依赖等待，不能将整条直接解释为纯传输。模型浅色读条是用页发起间隔重建的预算。[全部核的矩形与末核逐指令区间](page-audit.json) · [非整 16 边在真实矩形上的解码对照](decoder-controls.json)

950DT 同形窗口为 {dt['model_us']:.3f} / {dt['ca_us']:.3f} µs（模型 / CA），32 核、32 个 256×256 整块和 32 个 128×128 尾块，实测 512 MMAD。DT/PR 耗时比实测 {dt_ratio:.4f}，预测 {pr_ratio:.4f}，相对偏差 {pr_ratio/dt_ratio-1:+.2%}。970DT 的 {reg['future_us']:.3f} µs 为假定实现下的外推。[同形 DT 图](../../../docs/examples/figures/matmul-tile-pages.png)

实际库 kernel 为 `{stage['kernel']}`。宿主校验全部 {shape[0]*shape[1]:,} 个输出；每核指令跨度也与官方 duration 表核对。一次 PR CA 墙钟 {cost['seconds']/60:.1f} min，保留 {cost['raw_bytes']/2**20:.1f} MiB 原始文件。上述数据是该 shape 在 PR 的单次实测，不能单独估计其他形状的 WAPE。[原始 trace 指纹与计费](observation.json) · [三组解码器回归](decoder-controls.json) · [独立解码器](../../../development/validation/matmul_pages_pr/extract.py)

可从已提交的逐页 JSON 重画，不需要 CANN 或原始 trace。

```bash
python development/validation/matmul_pages_pr/publish.py --replot
```
"""
    (OUT/'report.md').write_text(text)


if __name__=='__main__':
    if sys.argv[1:]==['--replot']:
        render(json.loads((OUT/'freeze.json').read_text()),json.loads((OUT/'page-audit.json').read_text()),FIG)
    elif not sys.argv[1:]:
        publish()
    else:
        raise SystemExit('usage: publish.py [--replot]')
