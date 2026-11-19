"""Publish current totals, a same-contract comparison and the per-case independent results."""
from pathlib import Path
import json
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate
from development.validation.sku17 import publish as prev

V=ROOT/'results/v0.18'
SKU=V/'sku'
BASE='3f3f64d3e5e67ac312c5370c364de53a0b169213'
TARGETS=('quant','rmsnorm')
CHIPS={'dt':'Ascend950DT_9582','pr':'Ascend950PR_957b','future':'Ascend970DT'}
NAMES={'quant':'DynamicMXQuant','rmsnorm':'RMSNorm'}
load=prev.load
save=prev.save
table=prev.render_table


def snapshot(path):
    return subprocess.check_output(['git','show',f'{BASE}:{path}'],cwd=ROOT,text=True)


def fmt(v):
    return 'N/A' if v is None else f'{v:.3f}'


def percentages(m):
    return 'N/A' if not m['n'] else f"{m['wape']:.2%} / {m['max_ape']:.2%}"


def enrich(data,archive):
    row={k:v for k,v in data.items() if k not in ('pr','dt')}
    for side in ('dt','pr'):
        row[side+'_observation']=prev.trace_summary(data[side])
    for side,chip in CHIPS.items():
        e=estimate(row['op'],row['shape'],row['attrs'],chip=chip,timing_mode='resource')
        assert e.latency_us and e.status=='extrapolated'
        if archive:
            row['old_'+side+'_us']=data[side+'_us']
        else:
            assert abs(e.latency_us/data[side+'_us']-1)<1e-12
        row.update({side+'_us':e.latency_us,side+'_aiv':e.active_aiv,side+'_aic':e.active_aic,
                    side+'_status':e.status})
    row.update(evaluation_release='v0.18',pair_tier=('direct' if not archive else 'archived'),
               role='development replay' if archive else 'frozen independent DT and PR',
               api=data['pr']['api'],library='ops-nn',future_role='extrapolated; no target CA')
    row['auto_dt_us']=estimate(row['op'],row['shape'],row['attrs'],chip=CHIPS['dt']).latency_us
    row['auto_speedup']=row['auto_dt_us']/row['pr_us']
    row['speedup_ca']=row['dt_ca_us']/row['pr_ca_us']
    row['speedup_pred']=row['dt_us']/row['pr_us']
    row['speedup_old']=row['old_dt_us']/row['old_pr_us']
    row['speedup_error']=row['speedup_pred']/row['speedup_ca']-1
    for unused in ('speedup_relative_error','speedup_reuse_dt'):
        row.pop(unused,None)
    return row


def stats(rows):
    d=prev.summarize(rows)
    d['old_speed']=prev.metric(rows,'speedup_old','speedup_ca')
    return d


def detailed(rows,op):
    g=[r for r in rows if r['op']==op]
    labels={}
    lines=[]
    for r in g:
        kernels=[]
        for s in ('dt','pr'):
            name=r[s+'_observation']['kernel']
            kernels.append(labels.setdefault(name,f'K{len(labels)+1}'))
        times=lambda k:' / '.join(fmt(r[t]) for t in ('old_'+k+'_us',k+'_us',k+'_ca_us'))
        err=lambda k:f"{r[k+'_us']/r[k+'_ca_us']-1:+.1%}"
        cores=lambda k:f"{r[k+'_aiv']}→{r[k+'_observation']['active_aiv']}"
        lines.append(['×'.join(map(str,r['shape'])),'/'.join(kernels),times('dt'),err('dt'),times('pr'),err('pr'),
                      ' / '.join(fmt(r[k]) for k in ('speedup_old','speedup_pred','speedup_ca')),
                      fmt(r['future_us']),cores('dt')+' / '+cores('pr')])
    headers=('shape','DT/PR kernel','DT 旧/新/CA µs','新误差','PR 旧/新/CA µs','新误差','DT/PR 旧/新/CA','970 外推 µs','AIV 预测→CA DT / PR')
    return [f'### {NAMES[op]}','',f'ops-nn · `{g[0]["api"]}`。BF16，本卡 [行,通道]，设备窗口取全核最早指令到最晚指令。','',
            *table(headers,lines),'',*[' · '.join((label,name)) for name,label in labels.items()],'']


def chart(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for axes,op in zip(axs,TARGETS):
        g=sorted([r for r in rows if r['op']==op],key=lambda r:r['shape'][0]*r['shape'][1])
        names=['×'.join(map(str,r['shape'])) for r in g]
        for ax,(title,old,new,ref) in zip(axes,[('950DT','old_dt_us','dt_us','dt_ca_us'),('950PR','old_pr_us','pr_us','pr_ca_us'),('DT/PR','speedup_old','speedup_pred','speedup_ca')]):
            for k,label,marker in [(ref,'CA','o'),(old,'v0.17','x'),(new,'v0.18','s')]:
                ax.plot(range(len(g)),[r[k] for r in g],marker=marker,label=label,linewidth=1.1,markersize=4)
            ax.set_title(f'{NAMES[op]} · {title}')
            ax.set_xticks(range(len(g)),names,rotation=60,ha='right',fontsize=8)
            ax.set_ylabel('Speedup (×)' if title=='DT/PR' else 'Device time (µs)')
            ax.grid(axis='y',alpha=.25)
        axes[0].legend(fontsize=8)
    fig.savefig(V/'comparison.png',dpi=140)
    plt.close(fig)


def main():
    hist=load(ROOT/'results/v0.17/evaluation.json')
    archive=load(ROOT/'results/v0.17/sku/pairs.json')
    old_doc=snapshot('docs/error-analysis.md')
    current=json.loads(snapshot('results/current/evaluation.json'))
    wheel=load(V/'wheel-smoke.json')
    pairs=load(SKU/'pairs.json')
    assert len(pairs)==20 and len({(r['op'],tuple(r['shape'])) for r in pairs})==20
    prior_shapes={(r['op'],tuple(r['shape'])) for r in archive}
    assert all((r['op'],tuple(r['shape'])) not in prior_shapes for r in pairs)
    new=[enrich(r,False) for r in pairs]
    dev=[enrich(r,True) for r in archive if r['op'] in TARGETS]
    rows=[r for r in hist['rows'] if r['op'] not in TARGETS]+new
    summary=dict(hist['summary'])
    comparisons=[]
    for op in TARGETS:
        g=[r for r in new if r['op']==op];assert len(g)==10
        s=summary[op]=stats(g)
        comparisons.append([NAMES[op],percentages(s['old_dt']),percentages(s['dt']),percentages(s['old_pr']),percentages(s['pr']),percentages(s['old_speed']),percentages(s['speed'])])
    cost=load(SKU/'all-costs.json')
    totals=load(SKU/'campaign-completion.json')
    summary_rows=[]
    for h in current['operator_summary']:
        op=h['op'];s=summary[op];v=s['future_range']
        if op in TARGETS:
            role='DT 阶段定价，v0.18 双端冻结留出'
        elif s['pr']['n']:
            role=('DT 资源定价' if op in ('matmul','fa') else 'DT 阶段校准')+'，沿用 v0.17 留出'
        else:
            role='仅结构，CA 同步失败'
        summary_rows.append([h['name'],role,s['dt']['n'],percentages(s['dt']),s['pr']['n'],percentages(s['pr']),
                             percentages(s['speed']),'–'.join(fmt(x) for x in v) if v else 'N/A'])
    page=['# 算子耗时预测与 CA 校准误差','',*table(('算子','计时资格与数据角色','DT N','DT WAPE / 最大','PR N','PR WAPE / 最大','DT/PR 倍速 WAPE / 最大','970DT 外推范围 µs'),summary_rows),'',
          'WAPE = Σ|预测−观测| / Σ观测；最大值是单例绝对相对误差。逐例倍速为同一 shape 的 DT 时间 / PR 时间，大于 1 表示 PR 更快。970DT 全列均为沿用 950 实现策略及部分每核成本的外推。',
          '','两条更新路线的汇总只使用新留出。另 13 条可计时路线的实现和验证集沿用 v0.17；其中部分配对使用历史 DT compact trace 和 PR 官方 trace，采集方式有别。Chunk KDA 的设备时间继续为 N/A。','',
          old_doc.split('## 芯片资源及模拟条件')[1].split('## 在相同合同')[0].strip(),'',
          '## 冻结后同集比较','','每个单元格为 WAPE / 最大绝对相对误差。Quant 两版都用 resource。RMSNorm 旧版使用当时可用的 auto，新版在两端使用 resource；沿用旧 DT auto 的应用仍可复现历史校准值。','',
          *table(('算子','DT v0.17','DT v0.18','PR v0.17','PR v0.18','倍速 v0.17','倍速 v0.18'),comparisons),'',
          '![新留出预测与 CA](../results/v0.18/comparison.png)','','图中每个点都是同一 shape 的完整 CA 窗口。横轴按元素数排序；分核、尾块和核的首发偏斜都可使相邻点的时间不单调。','']
    for op in TARGETS:
        s=summary[op]
        g=[r for r in new if r['op']==op]
        worst=max(g,key=lambda r:abs(r['pr_us']/r['pr_ca_us']-1))
        matches=lambda side:sum(r[side+'_aiv']==r[side+'_observation']['active_aiv'] for r in g)
        page.extend(detailed(new,op))
        page.extend([f'新留出 DT / PR 的活动 AIV 数分别吻合 {matches("dt")}/10、{matches("pr")}/10。PR 最大误差为 {worst["shape"]}，v0.17、新模型和 CA 分别为 {worst["old_pr_us"]:.3f}、{worst["pr_us"]:.3f}、{worst["pr_ca_us"]:.3f} µs。十例合计时延的 DT/PR 比值为 CA {s["speed"]["observed"]:.3f}×、模型 {s["speed"]["predicted"]:.3f}×。','',
                     f'[队列、参数与原始偏差来源](../results/v0.18/{"quant" if op=="quant" else "rms"}-method.md) · [工作量到设备时间的算例](examples/{op}.md)',''])
    audit=load(SKU/'trace-evidence.json')['rows']
    evidence={(r['op'],tuple(r['shape']),r['chip']):r for r in audit}
    assert len(evidence)==40
    auto_rows=[]
    for op in TARGETS:
        group=[r for r in new if r['op']==op]
        auto_rows.append([NAMES[op],percentages(prev.metric(group,'auto_dt_us','dt_ca_us')),
                          percentages(prev.metric(group,'dt_us','dt_ca_us')),
                          percentages(prev.metric(group,'auto_speedup','speedup_ca')),
                          percentages(prev.metric(group,'speedup_pred','speedup_ca'))])
    page+=['## 系统调用默认模式时的实际差别','','950DT 的 auto 保留原来的 DT 校准入口，950PR 的 auto 对这两条路线等同新版 resource。默认调用会同时使用两种模型。下表在同一新留出上对照，列中均为 WAPE / 最大绝对相对误差。','',
           *table(('算子','DT auto','DT resource','两端 auto 的倍速','两端 resource 的倍速'),auto_rows),'']
    page+=['## 新留出的流水与分组误差','','末核指该次 CA 最后发完的 AIV。MTE3 忙时取其有效指令的时间并集，减去与 SIMD 的交集后得到暴露写窗口。DMA 在途可包含等待；下表单位为 µs。','']
    for op in TARGETS:
        g=[r for r in new if r['op']==op]
        strata=[]
        for key in sorted({r['regime'] for r in g}):
            part=[r for r in g if r['regime']==key]
            z=stats(part)
            strata.append([key,len(part),percentages(z['old_dt']),percentages(z['dt']),percentages(z['old_pr']),percentages(z['pr']),percentages(z['old_speed']),percentages(z['speed'])])
        page+=['### '+NAMES[op],'',*table(('预登记组','N','DT 旧','DT 新','PR 旧','PR 新','倍速旧','倍速新'),strata),'']
        phases=[];cold_dt=[];cold_pr=[]
        for r in g:
            source=evidence[op,tuple(r['shape']),CHIPS['pr']]
            e=estimate(op,r['shape'],chip=CHIPS['pr'],timing_mode='resource')
            d=e.projection['timing_diagnostics']
            union=source['queue_payload_union_us'].get('MTE3',0)
            overlap=source['mte3_simd_overlap_us']
            reads=source['queue_payload_union_us'].get('MTE2',0)
            streams=[x for x in source['critical_core_dma_streams'] if x['queue']=='MTE3']
            descriptors=sum(x['count'] for x in streams)
            if op=='rmsnorm':
                cold_dt.append(estimate(op,r['shape'],{'resident_aiv':0},chip=CHIPS['dt'],timing_mode='resource').latency_us)
                cold_pr.append(d['conditional_us']['first'])
                phases.append([str(r['shape']),d['target_plan']['rows_per_core'],d['target_plan']['ub_tile_rows'],descriptors,
                               fmt(union),fmt(overlap),fmt(union-overlap),fmt(d['stages_us']['MTE3']),
                               ' / '.join(fmt(d['conditional_us'][k]) for k in ('first','expand','reuse'))])
            else:
                frag=d['fragment_wave']
                phases.append([str(r['shape']),'按核写粒度排队' if frag else '原 UB 流水',d['target_plan']['iterations'],
                               d['scale_peak_writers'],fmt(d['scale_queue_cycles']/1650),descriptors,fmt(reads),fmt(union),fmt(overlap)])
        if op=='rmsnorm':
            header=('shape','PR 最忙核行数','UB 各轮行数','末核写描述符数','MTE3 CA','重叠 CA','暴露 CA','暴露预测','首次 / 扩核 / 热态预测')
            wape=lambda vals,k:sum(abs(v-r[k]) for v,r in zip(vals,g))/sum(r[k] for r in g)
            note=f'这批 CA 的采集条件为隔离首次调用。若系统调度器显式传入 resident_aiv=0，DT 和 PR WAPE 分别为 {wape(cold_dt,"dt_ca_us"):.2%}、{wape(cold_pr,"pr_ca_us"):.2%}。主表保留未提供该信息时的预测；未知态和首次态回答不同的系统问题。'
        else:
            header=('shape','模型路径','最忙核轮数','峰值写者数','模型排队 µs','末核写描述符数','MTE2 CA','MTE3 CA','与 SIMD 重叠 CA')
            note='峰值写者数按地址计算，不能直接推断关键核一定等待所有这些核。模型排队列是一次关键请求的增量，CA 忙时列是全窗口指令并集，两者口径不同。仍走原 UB 流水的样例可检验通用修正是否越过证据覆盖的范围。'
        page += [*table(header,phases),'',note,'']
        if op=='quant':
            changed=[r for r in g if abs(r['pr_us']-r['old_pr_us'])>1e-9]
            z=stats(changed)
            page+= [f'PR 的 {len(changed)} 个短写输入实际改变了预测，这组 PR WAPE 从 {z["old_pr"]["wape"]:.2%} 到 {z["pr"]["wape"]:.2%}。其余 {len(g)-len(changed)} 例仍走原 UB 流水。DT [3,1920] 从高估变为低估，[7,4480] 的估计则往相反方向移动；这两点拉高了 DT 整组 WAPE。当前发起顺序和排队间隔仍是按 DT 指令归纳的有效值，不能解释成硬件仲裁顺序的测量。','']
        else:
            page+= ['未知驻留的单值按三态极端误差平衡，包含热态的较短控制时长。新 CA 每个都是独立进程的首次调用；即使分核正确，这种比较仍可能表现为系统性低估。提供状态后的结果检验的是同一模型的条件分支。','']
    page+=['[两端逐核 DMA 指令、忙时、地址对应程序计数器与 trace 校验值](../results/v0.18/sku/trace-evidence.json)','']
    page+=['## 开发回放及误差来源','','RMSNorm [64,2048] 在 PR 上由 32 核各处理两行。旧版虽给出正确核数，却放大了 DT 上每核一行、64 个写者的写回成本。旧版、新版和 CA 的暴露写回分别是 4.640、1.101 和 0.968 µs。新版未知态的剩余低估主要在 SCALAR 和核到达；已知首次 H=0 的预测为 4.654 µs。','',
           'Quant [5,320] 的 PR 最晚核写 2 B scale，与 4 个生产者共享该区段（含自身）；全片最拥挤区段有 7 个写者。旧版对最忙核叠加最坏竞争。新版按请求就绪次序定价，仍预测了不同的末核，因此残余误差不能全部归为写回。32 B 区段及请求返回间隔是模型假设，实际物理仲裁器尚未确定。','',
           '下列旧样本参与了本轮诊断。表中的 CA 沿用原归档，倍速受采集方式与调用状态影响。']
    for op in TARGETS:page.extend(detailed(dev,op))
    timings=wheel['overall']
    cost_text=(f'隔离安装 wheel 后的 {timings["n"]} 次预测，中位 {timings["median_ms"]:.3f} ms、P95 {timings["p95_ms"]:.3f} ms、最大 {timings["max_ms"]:.3f} ms，导入另计 {wheel["import_ms"]:.2f} ms。')
    page+=['## 仿真成本与计时资格','',cost_text,'',
           f'本次 CA 共 {len(cost)} 次物理尝试，实际并行历时 {totals["wall_s"]/60:.2f} 分钟，原始数据 {totals["raw_bytes"]/2**30:.2f} GiB，其中 {sum(bool(c["eligible"]) for c in cost)} 次通过完整窗口与数值检查。两端沿用同一 host 参数；每条记录包含 SKU、实际 kernel、数值检查、二进制摘要与逐核窗口。'+
           ' [新留出双端证据](../results/v0.18/sku/pairs.json) · [预算与采集成本](../results/v0.18/sku/campaign-completion.json) · [预测开销](../results/v0.18/wheel-smoke.json)','',
           '## 其余路线逐例与库身份','','下列路线的 v0.17 预测及 CA 至今适用。D 为两端新采同宿主，A 为同后端复用历史 DT，M 为跨后端配对。旧资料标注的“本轮”指 v0.17 验证。','']
    previous_details=old_doc.split('## 本轮逐例与库身份',1)[1].split('## 950DT 参考校准路线',1)[0]
    for block in re.split(r'(?=^### )',previous_details,flags=re.M):
        if block.startswith('### ') and block.splitlines()[0] not in ('### dynamicMXquant','### RMSNorm'):
            page.append(block.strip()+'\n')
    historical=current['operator_summary']
    page+=['## 950DT 参考校准路线的历史摘要','','此表记录保留的 DT 校准入口。新增的资源模式和旧校准入口使用不同的分核、状态与服务规则；其独立集分别列示。','',
           *table(('算子','开发 N','历史独立 N','DT 历史独立 WAPE / 最大'),[
               [h['name'],h['development'],h['independent'],f"{h['wape']:.2%} / {h['max_ape']:.2%}" if h['independent'] else 'N/A']
               for h in historical]),'']
    page+=['## 950DT 参考校准路线'+old_doc.split('## 950DT 参考校准路线',1)[1]]
    (ROOT/'docs/error-analysis.md').write_text('\n'.join(page).rstrip()+'\n')
    changes=['# v0.18 两条 Vector 路线','','Quant 重新估算同粒度 scale 写入的暴露时间，RMSNorm 在目标芯片上按实际核数重新分配行，并计算读、归约、写回的暴露窗口。参数来源及局限分别见 [Quant](quant-method.md) 和 [RMSNorm](rms-method.md)。','',
             '每条路线冻结后各用 10 个新 shape，同宿主测两端。各格为 WAPE / 最大绝对相对误差。','',*table(('算子','DT 旧','DT 新','PR 旧','PR 新','倍速旧','倍速新'),comparisons),'',cost_text,'',
             f'CA 共 {len(cost)} 次尝试，历时 {totals["wall_s"]/60:.2f} 分钟，产生 {totals["raw_bytes"]/2**30:.2f} GiB；原始 shape、数据角色及采集成本见[审计](sku/campaign-completion.json)。970DT 只有配置驱动的外推。[全量误差与逐例分析](../../docs/error-analysis.md)']
    q=summary['quant'];rms=summary['rmsnorm']
    changes+=['',f'Quant 在 10 个新 shape 上，PR WAPE {q["old_pr"]["wape"]:.2%} → {q["pr"]["wape"]:.2%}，DT 为 {q["old_dt"]["wape"]:.2%} → {q["dt"]["wape"]:.2%}。同组直接把新版 DT resource 估计当作 PR 耗时的 WAPE 为 {q["reuse_dt"]["wape"]:.2%}，仍略优于显式 PR 路线。RMSNorm 的默认未知驻留态 PR WAPE 为 {rms["pr"]["wape"]:.2%}，详见总误差分析中与隔离首次条件的同集对照。']
    if len(cost)>40:
        changes+=['',f'原计划为 40 次双端启动；有 {len(cost)-40} 次追加物理尝试。原始超限记录与通过核验的重试分别计费，预测保持冻结。']
    (V/'changes.md').write_text('\n'.join(changes)+'\n')
    payload=dict(version='v0.18',previous_version='v0.17',summary=summary,rows=rows,development_replay=dev,wheel=wheel,
                 cost=totals,reference_calibration_release='v0.12',unchanged_evidence_release='v0.17')
    save(V/'evaluation.json',payload)
    current.update(version='v0.18',previous_version='v0.17',release_summary=summary,release_rows=rows,pr_summary={op:s['pr'] for op,s in summary.items()},
                   qualification='Current explicit resource for quant, rmsnorm, matmul, fa. Changed routes: fresh matched DT/PR; other routes: unchanged v0.17 evidence.',collection_cost=totals,target_query_wall_ms=wheel,
                   reference_query_wall_scope='rows and operator_summary are the 356 historical DT calibrated predictions. query_wall is legacy; target_query_wall_ms measures v0.18. release_rows and release_summary include new v0.18 holdouts for quant and rmsnorm and retained v0.17 results for other routes.',
                   paired_sku=current['paired_sku']+new,development_replay=dev)
    save(ROOT/'results/current/evaluation.json',current)
    chart(new)


if __name__=='__main__':
    main()
