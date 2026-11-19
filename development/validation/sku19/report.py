"""Publish matched holdouts, workload-scale validation and retained routes."""
from pathlib import Path
import json
import math
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import CHIPS, OUT, digest, read, write
from development.validation.sku17.publish import metric, render_table as table, summarize, trace_summary
from operator_model import estimate

BASE='29b0e467916da8239abb591e5db10144ed4c2fa6'
V=OUT.parent
OPS=('rmsnorm','quant')
LABEL={'quant':'DynamicMXQuant','rmsnorm':'RMSNorm','moe_gating_top_k':'MoE sigmoid Top-K','cast_bf16_fp32':'Cast BF16→FP32','attn_res_fwd':'Kimi AttnResFwd'}


def snapshot(path):
    return subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT,text=True)


def num(v):
    return 'N/A' if v is None else f'{v:.3f}'


def pct(x):
    return 'N/A' if x is None else f'{x:.2%}'


def metrics(m):
    return f"{pct(m['wape'])} / {pct(m['max_ape'])}" if m['n'] else 'N/A'


def shape(r):
    return '×'.join(str(x) for x in r['shape'])


def directions(rows,pred):
    return sum((r[pred]-1)*(r['speedup_ca']-1)>0 or r[pred]==r['speedup_ca']==1 for r in rows)


def single(r):
    out={k:v for k,v in r.items() if k not in ('dt','pr')}
    for side in ('dt','pr'):
        out[side+'_observation']=trace_summary(r[side])
    out.update(pair_tier='direct',api=r['dt']['api'],library='ops-nn',future_role='970DT extrapolation',
               evaluation_release='v0.19',speedup_error=r['speedup_pred']/r['speedup_ca']-1)
    return out


def summary(rows):
    s=summarize(rows)
    s['old_speed']=metric(rows,'speedup_old','speedup_ca')
    return s


def describe(rows):
    names={}
    tab=[]
    for r in rows:
        def times(side):
            return ' / '.join(num(r.get(k)) for k in ('old_'+side+'_us',side+'_us',side+'_ca_us'))
        def err(side):
            return pct(r[side+'_us']/r[side+'_ca_us']-1)
        def k(side):
            name=r[side+'_observation']['kernel']
            return names.setdefault(name,'K'+str(len(names)+1))
        cores=' / '.join(f"{r[x+'_aiv']}→{r[x+'_observation']['active_aiv']}" for x in ('dt','pr'))
        tab.append([shape(r),k('dt')+'/'+k('pr'),times('dt'),err('dt'),times('pr'),err('pr'),
                    ' / '.join(num(r.get(x)) for x in ('speedup_old','speedup_pred','speedup_ca')),num(r['future_us']),cores])
    return [*table(['本卡 shape','DT / PR kernel','DT 旧 / 新 / CA µs','DT 误差','PR 旧 / 新 / CA µs','PR 误差',
                   'DT/PR 旧 / 新 / CA','970DT 外推 µs','AIV 预测→CA DT / PR'],tab),'',
            *[alias+' = '+name for name,alias in names.items()],'']


def plot(groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,7.4),layout='constrained')
    for axs,(op,rs) in zip(axes,groups.items()):
        rows=sorted(rs,key=lambda r:math.prod(r['shape']))
        for i,(a,side) in enumerate(zip(axs,('dt','pr','speed'))):
            keys=('old_'+side+'_us',side+'_us',side+'_ca_us') if side!='speed' else ('speedup_old','speedup_pred','speedup_ca')
            for k,label,sty in zip(keys,['v0.18','v0.19','CA'],['x--','s-','o-']):
                a.plot(range(len(rows)),[r[k] for r in rows],sty,label=label,markersize=4)
            a.set_xticks(range(len(rows)),[shape(r) for r in rows],rotation=60,ha='right',fontsize=8)
            a.set_title(LABEL[op]+' / '+('DT:PR' if i==2 else '950'+side.upper()))
            a.set_ylabel('DT time / PR time' if i==2 else 'Device time (µs)')
            if i==2:a.axhline(1,color='grey',lw=.8)
            a.grid(axis='y',alpha=.25)
        axs[0].legend()
    fig.savefig(V/'comparison.png',dpi=138)
    plt.close(fig)


def workload_observation(record):
    k=record.get('kernel',{})
    receipt={name:k.get(name) for name in ('kernel','ca_us','active_aic','active_aiv','trace','trace_sha256','binary_sha256','total_events')}
    receipt['correctness']=record.get('correctness')
    return receipt


def workload_rows():
    out=[]
    for op,id_ in [('moe_gating_top_k','router'),('cast_bf16_fp32','cast')]:
        f=V/'workloads'/(id_+'-qualified.json')
        reg=V/'workloads'/(id_+'-holdout-freeze.json')
        if not f.exists() or not reg.exists():continue
        records=read(f)
        for j in read(reg)['jobs']:
            grouped={x['sku']:x for x in records if x['shape']==j['shape'] and x['qualified']}
            r=dict(op=op,id=j['id'],attrs={},shape=j['shape'],pair_tier='direct',evaluation_release='v0.19',
                   role='frozen independent',future_role='970DT extrapolation',speedup_old=None,old_dt_us=None,old_pr_us=None)
            for side,chip in CHIPS.items():
                e=estimate(op,j['shape'],chip=chip,timing_mode='resource')
                r[side+'_us']=j['predictions'][chip]
                assert e.latency_us and math.isclose(e.latency_us,r[side+'_us'],rel_tol=1e-12),(op,side,j['shape'])
                r[side+'_status']=e.status;r[side+'_aiv']=e.active_aiv;r[side+'_aic']=e.active_aic
                if side=='future':continue
                obs=workload_observation(grouped.get(chip,{}))
                r[side+'_observation']=obs
                r[side+'_ca_us']=obs['ca_us']
            r['speedup_ca']=r['dt_ca_us']/r['pr_ca_us'] if r['dt_ca_us'] and r['pr_ca_us'] else None
            r['speedup_pred']=r['dt_us']/r['pr_us']
            out.append(r)
    for tag in ('attn-res-holdout','attn-res-extension'):
        attn=V/'workloads'/('attn-res-extension-holdout.json' if tag.endswith('extension') else tag+'.json')
        if not attn.exists():continue
        scenarios=read(V/'workloads'/(tag+'-freeze.json'))['scenarios']
        observed={(tuple(x['shape']),x['chip']):x for x in read(attn) if x['qualified']}
        keys=tuple(dict.fromkeys((x['tokens'],x['hidden'],x['slots'],x['valid']) for x in scenarios))
        for key in keys:
            attrs=dict(addend=True,write_idx=key[-1],materialize=True)
            r=dict(id='attnres_'+'_'.join(map(str,key)),op='attn_res_fwd',shape=list(key),attrs=attrs,pair_tier='direct',
                   role='supplemental frozen holdout' if tag.endswith('extension') else 'first frozen resident holdout',old_dt_us=None,old_pr_us=None,speedup_old=None)
            for side,chip in CHIPS.items():
                frozen=next(x for x in scenarios if (x['tokens'],x['hidden'],x['slots'],x['valid'])==key and x['chip']==chip)
                e=estimate('attn_res_fwd',key,attrs,chip=chip)
                r[side+'_us']=frozen['latency_us'];r[side+'_aiv']=e.active_aiv;r[side+'_aic']=e.active_aic;r[side+'_status']=e.status
                assert math.isclose(r[side+'_us'],e.latency_us,rel_tol=1e-12)
                if side!='future':
                    obs=workload_observation(observed.get((key,chip),{}))
                    r[side+'_ca_us']=obs['ca_us']
                    r[side+'_observation']=obs
            r['speedup_pred']=r['dt_us']/r['pr_us']
            r['speedup_ca']=r['dt_ca_us']/r['pr_ca_us'] if r['dt_ca_us'] and r['pr_ca_us'] else None
            out.append(r)
    routes={'moe_gating_top_k':('fp32','aclnnMoeGatingTopK','ops-transformer'),
            'cast_bf16_fp32':('bf16','aclnnCast','ops-math'),
            'attn_res_fwd':('bf16','aclnnAttnResFwd','vllm-ascend custom arch35')}
    for r in out:
        r['dtype'],r['api'],r['library']=routes[r['op']]
        r.update(evaluation_release='v0.19',future_role='970DT extrapolation')
    return out


def stage_examples(op, rows):
    """Use the last-finishing core on the real trace; do not sum overlapping pipes."""
    evidence = {r['trace_sha256']: r for r in read(OUT/'trace-evidence.json')['rows']}
    largest = max(rows, key=lambda r: math.prod(r['shape']))
    worst = max(rows, key=lambda r: abs(r['pr_us']/r['pr_ca_us']-1))
    comparisons = []
    for case in {r['id']: r for r in (worst, largest)}.values():
        for side in ('dt', 'pr'):
            sha = case[side+'_observation']['trace_sha256']
            if isinstance(sha, list): sha = sha[0]
            observed = evidence.get(sha)
            if not observed: continue
            forecast = estimate(op, case['shape'], chip=CHIPS[side], timing_mode='resource')
            plan = forecast.projection['target_plan']
            stages = forecast.projection['stages_us']
            queues = observed['queue_payload_union_us']
            comparisons.append([shape(case), side.upper(), plan['active_aiv'], plan['iterations'],
                                num(case[side+'_us'])+' / '+num(case[side+'_ca_us']),
                                '，'.join(k+' '+num(x) for k,x in stages.items() if x>=.05),
                                ' / '.join(num(queues.get(k,0)) for k in ('MTE2','SIMD','MTE3')),
                                num(observed['mte3_simd_overlap_us'])])
    return ['新留出中 PR 相对偏差最大的输入及元素最多的输入如下。模型列是全片关键路径上的暴露阶段，CA 列是终止核上有效指令区间的并集，可彼此重叠。两者的差不能直接解释为某个端口的独占延迟。单位 µs。','',
            *table(['shape','芯片','预测 AIV','预测忙核轮数','模型 / CA','模型暴露阶段',
                    'CA MTE2 / SIMD / MTE3','CA 写与 SIMD 重叠'],comparisons),'',
            '[逐描述符发起、响应与终止核](../results/v0.19/sku/trace-evidence.json)','']


def prefill_section(chunks):
    data=read(V/'prefill/forecasts.json')
    rows={r['id']:r for r in data['v019']['rows']}
    for ident,row in rows.items():
        op,ts,ds=ident.split('_')
        for side,chip in CHIPS.items():
            row['targets'][side]['us']=estimate(op,(int(ts),int(ds)),chip=chip,timing_mode='resource').latency_us
    examples=[]
    for scene in data['scenarios']:
        p=rows[scene['id']]['targets']
        if scene['kernel_rows']!=scene['total_input_tokens'] and scene['name']!='Flash-32k-b8':continue
        waves='/'.join(str(p[s]['plan']['iterations']) for s in CHIPS)
        times=[num(p[s]['us']) for s in CHIPS]
        examples.append([scene['name'],LABEL[scene['op']],scene['id'].split('_',1)[1].replace('_','×'),str(scene['chunks']),waves,*times])
    page=['## 大 prefill 与每核多轮','','下表输入按 rank-local 总 token 预算分块。例：8 个 32k 请求共有 262144 行，预算为 2048 时每次执行 [2048,5120]、共 128 个 chunk。表中时间是单次 kernel 的模型值。把 128 次相加需额外假设同一启动上下文；逐层开销还取决于每层的实际调用次数与依赖。', '',
          *table(['场景','算子','单次 shape','调用数','每忙核 UB 轮数 DT/PR/970','DT 预测 µs','PR 预测 µs','970 外推 µs'],examples),'',
          '下表给出真正单次执行了全部 2048 行的两端 CA。使用发布模型与相同输入的实测逐端比较；PR RMSNorm 在补测结束后按第二次冻结模型重放，同下方独立留出分开统计。','']
    if chunks:page+=describe(chunks)
    paired_ids={r['id'] for r in chunks}
    partial=[]
    for job in read(OUT/'freeze.json')['jobs']:
        if job['role']!='prefill_chunk' or job['id'] in paired_ids:continue
        obs={r['chip']:r['ca_us'] for r in read(OUT/'final-records.json') if r['case_id']==job['id']}
        modeled=job['outputs'][0]
        partial.append([LABEL[job['op']],shape(job),*(num(modeled[s+'_us'])+' / '+num(obs.get(CHIPS[s])) for s in ('dt','pr')),num(modeled['future_us'])])
    if partial:
        page += table(['部分完成','shape','DT 预测 / CA µs','PR 预测 / CA µs','970DT 外推 µs'],partial)+['']
    page+=['在 PR [2048,5120] 上，最终 RMSNorm 预测 31.655 µs，CA 17.820 µs（+77.64%）。晚结束核的本地阶段合计 14.544 µs；共享约束再加 17.111 µs。当前全片冷读 20,981,760 B、写 20,979,712 B，按 1,400 GB/s 服务，再加 2776 周期的首次读发起。如果全部输出都能在末指令前驻留 L2，仅让冷输入走 HBM，这一条件下的时间为 max(14.544,2776/1650+20981760/1400000)=16.669 µs。这个情景展示误差对写回边界的敏感性，CA 指令记录没有给出实际 HBM 字节，模型仍按冷读写输出点估计。整批约 2.69 GB 的输出超过该 PR 的 112 MiB L2，不能把同一驻留比例移到 262144 行。','',
           '![模型的长序列趋势与完整 CA 观测](../results/v0.19/prefill-scaling.png)','']
    for op in OPS:
        p=V/'prefill'/(op+'_262144_5120.json')
        if not p.exists():page.append(f'- {LABEL[op]} 整批 DT 实验尚无合格结果。');continue
        d=read(p);why=d['cost']['stop_reason'] or d.get('reason','完成')
        page.append(f"- {LABEL[op]} 整批 DT [262144,5120]：{num(d['reference_us'])} µs，{why}，墙钟 {d['cost']['seconds']:.1f} s，最大临时盘 {d['cost']['peak_bytes']/2**20:.1f} MiB。")
    return page+['','[全核记录、实际每轮行数与整批实验成本](../results/v0.19/prefill/capture-analysis.md)','','每核轮数由 UB 容量与分配到该核的行列决定。固定通道、核数、内存供给及 kernel 分支后，足够长的同形 tile 序列按稳态发起间隔增长，首尾占比下降。缓存可容纳量、分核阶梯或 tiling 分支变化都会改变斜率。整批缺少完整 CA 时，线性区间只能作为模型假设。 [场景、预算及冻结预测](../results/v0.19/prefill/forecasts.json)','']


def report():
    hist=read(ROOT/'results/v0.18/evaluation.json')
    prevdoc=snapshot('docs/error-analysis.md')
    current=json.loads(snapshot('results/current/evaluation.json'))
    pairs=list(map(single,read(OUT/'final-pairs.json')))
    groups={op:[r for r in pairs if r['op']==op and r['sample_role']=='holdout'] for op in OPS}
    first_rms=groups['rmsnorm']
    confirmation=V/'rms-confirmation'
    groups['rmsnorm']=[single(r) for r in read(confirmation/'pairs.json')]
    assert all(len(rs)==10 for rs in groups.values())
    chunks=[r for r in pairs if r['sample_role']=='prefill_chunk']
    for r in chunks:
        if r['op']=='rmsnorm':
            r['first_freeze_pr_us']=r['pr_us']
            r['pr_us']=estimate('rmsnorm',r['shape'],chip=CHIPS['pr'],timing_mode='resource').latency_us
            r['speedup_pred']=r['dt_us']/r['pr_us']
            r['speedup_error']=r['speedup_pred']/r['speedup_ca']-1
            r['sample_role']='supplemental replay after second freeze'

    results=[r for r in hist['rows'] if r['op'] not in OPS]+sum(groups.values(),[])
    sums=dict(hist['summary'])
    for op,rs in groups.items():sums[op]=summary(rs)
    extra=workload_rows();results.extend(extra)
    for op in dict.fromkeys(r['op'] for r in extra):sums[op]=summary([r for r in extra if r['op']==op])
    initial=prevdoc.split('WAPE =',1)[0].strip().splitlines()
    for op,name in [('rmsnorm','RMSNorm'),('quant','dynamicMXquant')]+[(o,LABEL[o]) for o in sums if o not in hist['summary']]:
        s=sums[op];arr=s['future_range']
        role='v0.19 冻结留出，DT/PR 阶段定价' if op in ('quant','attn_res_fwd','rmsnorm') else 'v0.19 冻结留出，DT 阶段校准'
        cells=[name,role,s['dt']['n'],metrics(s['dt']),s['pr']['n'],metrics(s['pr']),metrics(s['speed']),('–'.join(num(x) for x in arr) if arr else 'N/A')]
        line=table(['x']*len(cells),[cells])[-1]
        i=next((i for i,row in enumerate(initial) if row.startswith('| '+name+' |')),None)
        if i is None:initial.append(line)
        else:initial[i]=line
    page=initial+['','WAPE = Σ|预测−CA| / ΣCA。最大误差为单例最大绝对相对误差。每例加速比为 DT 耗时 / PR 耗时，超过 1 表示 PR 更快。970DT 全部为独立芯片配置驱动的外推。其余路线沿用 v0.17 的合格数据，Chunk KDA 设备时间仍缺有效同步窗口。','',
                 '## 芯片资源及模拟条件','',prevdoc[prevdoc.index('| 配置项 |'):prevdoc.index('## 冻结后同集比较')].strip(),'']
    comparisons=[]
    for op,rs in groups.items():
        s=sums[op]
        comparisons.append([LABEL[op],metrics(s['old_dt']),metrics(s['dt']),metrics(s['old_pr']),metrics(s['pr']),metrics(s['old_speed']),metrics(s['speed']),pct(s['reuse_dt']['wape'])])
    header=['算子','DT v0.18','DT v0.19','PR v0.18','PR v0.19','倍速 v0.18','倍速 v0.19','复用新 DT 预测 PR 的 WAPE']
    page+=['## 冻结后同集比较','','两版均显式调用 resource、不传 resident_aiv，在每个 SKU 单独启动。每格为 WAPE / 最大误差。最后一列将本版 DT 微秒数原样用于 PR。','',*table(header,comparisons),'','![双端耗时与加速比](../results/v0.19/comparison.png)','']
    page+=['RMSNorm 在首轮冻结的十例上仍有快慢方向偏差。随后仅用 v0.17/v0.18 的旧流水估计 PR 首次 SCALAR 和有效读发起两项周期，在第二次冻结后另取十个新输入。表头 RMSNorm 来自第二次冻结，Quant 来自第一次冻结。两版预测均先于对应参考时间。','']
    for op,rs in groups.items():
        s=sums[op];agg=s['speed']
        page+=['### '+LABEL[op]+' 的留出样例','',
               f"10 例合计的 DT/PR 为 CA {agg['observed']:.3f}×，v0.18 {sum(r['old_dt_us'] for r in rs)/sum(r['old_pr_us'] for r in rs):.3f}×，v0.19 {agg['predicted']:.3f}×。逐例方向命中 {directions(rs,'speedup_old')}/10 → {directions(rs,'speedup_pred')}/10。活动核预测匹配 {sum(r[t+'_aiv']==r[t+'_observation']['active_aiv'] for t in ('dt','pr') for r in rs)}/20。实际调用 {rs[0]['api']}，安装包 ops-nn。", '',
               *describe(rs)]
        automatic_dt=[estimate(op,r['shape'],chip=CHIPS['dt']).latency_us for r in rs]
        auto_wape=sum(abs(t-r['dt_ca_us']) for t,r in zip(automatic_dt,rs))/sum(r['dt_ca_us'] for r in rs)
        page += [f"若系统对两端均调用 auto，DT 在历史域内使用原参考模型，DT WAPE 为 {auto_wape:.2%}；PR 仍走上述资源路线。跨芯片研究应给两端相同的 timing_mode。",'']
        page+=stage_examples(op,rs)
        if op=='rmsnorm':
            aggregate_ok=(agg['predicted']-1)*(agg['observed']-1)>0
            worst_delta=max(rs,key=lambda r:abs((r['dt_us']-r['pr_us'])-(r['dt_ca_us']-r['pr_ca_us'])))
            residual=(worst_delta['dt_us']-worst_delta['pr_us'])-(worst_delta['dt_ca_us']-worst_delta['pr_ca_us'])
            wrong=[shape(r) for r in rs if (r['speedup_pred']-1)*(r['speedup_ca']-1)<=0]
            page+=[f"第二组的十个调用串行求和后，预测与 CA 的芯片排序{'一致' if aggregate_ok else '相反'}。对合计 DT−PR 差值影响最大的形状是 {shape(worst_delta)}，其差值残差为 {residual:+.3f} µs。逐例判错的形状为 {'、'.join(wrong) or '无'}。", '']
            first={r['id']:r['targets'] for r in read(confirmation/'first-schedule-replay.json')['rows']}
            earlier=[{**r,'dt_us':first[r['id']]['dt']['us'],'pr_us':first[r['id']]['pr']['us'],
                      'speedup_pred':first[r['id']]['dt']['us']/first[r['id']]['pr']['us']} for r in rs]
            pre=summary(earlier)
            page+=[f"在同一组第二次冻结留出上，用第一次冻结源码重放，PR WAPE 为 {pct(pre['pr']['wape'])}，倍速 WAPE 为 {pct(pre['speed']['wape'])}，方向 {directions(earlier,'speedup_pred')}/10。只替换两项启动价格后，变为 {pct(s['pr']['wape'])}、{pct(s['speed']['wape'])} 和 {directions(rs,'speedup_pred')}/10。DT 定价完全相同。 [首版重放](../results/v0.19/rms-confirmation/first-schedule-replay.json)",'']
            page+=['初调后的首组十例在 DT / PR / 速度上的 WAPE 为 4.44% / 8.17% / 5.80%，方向仍为 3/10。以下是这个问题的诊断，数据角色为第一次冻结。首组中七例模型给出接近 1 的倍速（0.996–1.000），CA 却在 1.016–1.078。例如 [54,1984] 两端都是 54 核、同一 SIMD 工作量，终止核的 MTE3 在途并集 DT 3.266 µs、PR 2.521 µs。采用相同写者价格未能描述所有 SKU 局部波动。对 [507,7168]，初版 PR 本核路径为 8.393 µs，全片冷读写约束又拉长 3.802 µs，预测 12.195 µs、实测 9.806 µs。仅靠指令 trace 无法辨别真实 HBM 写入比例。','',
                    '第二次冻结使用七组更早的同分块、同 binary、孤立首次双端流水。PR 的独占 SCALAR 周期在七组上均低于 DT，中位差 −353 周期；整片首次有效输入发起中位差 −194 周期。配置分别给出 2580.5、2776 周期。已知热态/扩核保留原参数，970DT 继承 DT。下表保留首轮的原预测以供复核。 [校准来源及按 shape 留一检验](../results/v0.19/rms-followup/crossval.json) · [第二次冻结预测](../results/v0.19/rms-confirmation/freeze.json)','',
                    *describe(first_rms)]
        else:
            page+=['新版在这组留出上 10/10 个相对方向命中。[1023,7168] 的 DT 与 PR 均在 CA 的 0.5% 内；[641,5120] 的 PR 仍低估 14.06%。DT 整体 WAPE 上升 0.78 个百分点，主要是 32–128 行分核边界区；最大相对误差则由 11.04% 降至 9.78%。PR 直接复制 v0.18 DT 的 WAPE 是 14.03%，直接复制 v0.19 DT 是 15.21%。','']
        split=[]
        for group in ('small','boundary','large'):
            g=[r for r in rs if r['regime']==group];x=summary(g)
            split.append([group,len(g),*(pct(x[k]['wape']) for k in ['old_dt','dt','old_pr','pr','old_speed','speed','reuse_dt'])])
        page+=table(['行数组','N','DT 旧','DT 新','PR 旧','PR 新','倍速旧','倍速新','复制 DT 到 PR'],split)
        page+=['','small 为少于 32 行，boundary 为 32–128 行，其余为 large。每组均使用该组 CA 的 WAPE 分母。','']
    page+=prefill_section(chunks)
    audits=read(OUT/'marker-audit.json')
    page+=['## 指令记录的完整性','',
           '表头 WAPE 衡量模型对官方全核时间窗的误差。正确性逐元素检查、全核窗口资格、逐指令机制完整性分别核验。即使导出成功，少数核也可能缺尾标记或某些搬出描述符；这些形状可以复核 profiler 给出的时长，不能用缺失的指令反推物理写带宽。下列敏感性分析进一步要求每个核都记录了末标记和搬出指令。这一检查仍不能证明每个字节都经过 HBM。','']
    qa=[]
    for op,group in groups.items():
        valid=[];flags=[]
        for r in group:
            a=[x for x in audits if x.get('id')==r['id']]
            if len(a)==2 and all('missing' in x and not any(x['missing'].values()) for x in a):valid.append(r)
            else:flags.append(shape(r))
        ss=summary(valid)
        qa.append([LABEL[op],len(valid),pct(ss['dt']['wape']),pct(ss['pr']['wape']),pct(ss['speed']['wape']),', '.join(flags) or '无'])
    page+=table(['算子','双端标记完整例数','DT WAPE','PR WAPE','倍速 WAPE','缺标记或缺回写的 shape'],qa)+['','[逐端、逐核标记核对](../results/v0.19/sku/marker-audit.json) · [完整窗口恢复与对照](../results/v0.19/sku/window-recovery.json) · [大 shape 采集分析](../results/v0.19/prefill/capture-analysis.md)','']
    page+=['## 开发回放与误差来源','',
      'RMSNorm 在旧 10 例上，合计 DT/PR 从 0.919× 变为 1.013×，CA 为 1.033×。只改首次启动为 0.956×，再换每核流水为 0.996×，从首个有效读起计全片带宽才达到 1.013×。每核行数、共享统计量的写者和发起时间分散决定了芯片侧差别。默认给出独立首次调用的点估计。对旧 DT compact 序列，省略驻留信息时，11 个首次、19 个扩核、23 个复用样本的 WAPE 分别为 5.39%、22.07%、47.78%。传入已知高水位时复用组为 10.80%。这三个分组给出了系统仿真中启动上下文的代价。 [阶段与每个开发样例](../results/v0.19/rms-replay.json)','',
      'Quant 在旧 10 例上，PR WAPE 由 12.98% 降至 7.40%，复用新版 DT 为 14.75%。一条 PR trace 的两个 52,736 B 读仅隔 695 周期发起，旧版按整片带宽均摊把它拉到约 3480 周期；小 shape 上首个有效读的起点又存在两个 SKU 间可测的差值。新版根据发起、响应、VF 就绪、脏输出驻留分别估计。输出驻留比例及 L2 有效占用仍是假设，970 缺少对应 CA。 [Quant 原理和误差归因](../results/v0.19/quant-method.md) · [开发样例](../results/v0.19/quant-replay.json)','']
    for op in dict.fromkeys(r['op'] for r in extra):
        g=[r for r in extra if r['op']==op]
        good=[r for r in g if r['speedup_ca']]
        scope={'moe_gating_top_k':'CANN ops-transformer 的 aclnnMoeGatingTopK。2 个 DT 开发样本定价，10 个新 shape 双端冻结验证。',
               'cast_bf16_fp32':'CANN ops-math 的 aclnnCast。1 个合格 DT 样本定价，10 个新 shape 双端冻结验证。PR [96,7168] 输出位型检查通过，但写描述符只覆盖 6/56 核，官方窗口与全输出写回的机制证据分列。同一冻结输入的第二次 PR CA 写回 56/56 核，3.508 µs（首次 2.875）。仅换这一个观测的敏感性为 PR WAPE 7.36%、倍速 WAPE 5.14%。[逐端 DMA 审计与重复开销](../results/v0.19/workloads/cast-evaluation.json)',
               'attn_res_fwd':'上游 vLLM-Ascend 自定义 aclnnAttnResFwd，驻留分支使用两端小宽度阶段证据定价。先冻结验证 2 对，再选 8 对新 shape 独立于校准数据，模型及价格相同。10 个 shape 在两端的全部 4 路数值均通过，快慢方向为 0/10；固定 1 倍的速度 WAPE 为 3.59%，模型为 4.24%。PR 上原样复用 DT 耗时的 WAPE 为 31.90%，该资源模型为 33.10%。宽通道深 bank 的尾段服务高估。重读分支返回工作量，整体耗时为空。[来源与阶段误差](../results/v0.19/workloads/attn-res-extension-evaluation.json)'}
        page+=['### '+LABEL[op]+' 的新路线','',scope[op],'',*describe(good)]
        if len(good)<len(g):
            page+=['未取得完整配对的例子（µs）。','',*table(['shape','DT 预测/CA','PR 预测/CA','970 外推'],
                   [[shape(r),num(r['dt_us'])+' / '+num(r['dt_ca_us']),num(r['pr_us'])+' / '+num(r['pr_ca_us']),num(r['future_us'])] for r in g if not r['speedup_ca']]),'']
    page+=['## 采集与预测成本','',cost_paragraph(),'',*cost_breakdown(),wheel_paragraph(),'',
           '两个主算子使用同一版 msopprof 模拟器，关闭重复的原始指令文本导出，保留完整跨核事件窗。数值、SKU、模拟内存、kernel 身份、所有活动核及其时长均通过后才纳入统计。此前沙箱阻止 profiler 的本机 socket，10 个基础设施尝试（含中止）单列；已退出但无流水的 host 不能充当设备参考。[物理尝试](../results/v0.19/sku/costs.json) · [早期采集成本](../results/v0.19/sku/sandbox-costs.json) · [逐核窗口](../results/v0.19/sku/final-records.json)','',
           '## 其余路线逐例与库身份'+prevdoc.split('## 其余路线逐例与库身份',1)[1]]
    (ROOT/'docs/error-analysis.md').write_text('\n'.join(page).rstrip()+'\n')
    write(V/'evaluation.json',dict(version='v0.19',previous_version='v0.18',rows=results,summary=sums,prefill_chunk=chunks,
                                   cost=read(OUT/'completion.json'),rms_confirmation_cost=read(confirmation/'completion.json'),first_rms_independent=first_rms,conditional='separate warm-state development replay in rms-replay.json'))
    current.update(version='v0.19',previous_version='v0.18',release_rows=results,release_summary=sums,paired_sku=current['paired_sku']+sum(groups.values(),[])+extra,
                   rms_first_holdout=first_rms, rms_confirmation_cost=read(confirmation/'completion.json'), pr_summary={o:x['pr'] for o,x in sums.items()},target_query_wall_ms=read(V/'wheel-smoke.json'),
                   reference_query_wall_scope='rows/operator_summary contain the archived DT reference route. release_rows/summary contain latest matched holdouts, with sku-specific roles.',
                   collection_cost=read(OUT/'completion.json'),development_replay={'rmsnorm':'results/v0.19/rms-replay.json','quant':'results/v0.19/quant-replay.json'},
                   qualification='Two changed Vector resource schedules and new workload routes; other routes retain v0.17 evidence.')
    write(ROOT/'results/current/evaluation.json',current)
    confirm_cost=read(confirmation/'completion.json')
    status_notes=(f"RMSNorm 二次冻结后独立形状 {confirm_cost['paired']} 对，CA 首末启动跨度 {confirm_cost['wall_s']/60:.2f} 分钟。"
                  "2048 行补充实测的共享 HBM 约束仍使 PR RMSNorm 高估约 78%。"
                  "两条 262144 行 DT 尝试各限 15 分钟，仅能核对启动及分核。Kimi 融合残差在大 H / 深 bank 的尾段还有高估。")
    changes=['# v0.19','', 'RMSNorm 按启动状态、目标核数和 rstd 写竞争调度。首批 10 对上方向 3/10，补充价格完全来自更早的 7 对同分块旧 trace，重新冻结和选取 10 个独立输入。Quant 在本地读发起、读响应、VF 和片外退休上分别计价。新增模型契约及覆盖范围见[工作负载假跑](../../docs/workload-coverage.md)。','',*table(header,comparisons),'',cost_paragraph(),'',status_notes,'',wheel_paragraph(),'','[逐例结果、限制与 970DT 外推](../../docs/error-analysis.md) · [建模方法](../../docs/modeling-strategy.md)']
    (V/'changes.md').write_text('\n'.join(changes)+'\n')
    plot(groups)
    from large_plot import plot as plot_large
    plot_large(sum(groups.values(),[])+chunks)


def cost_paragraph():
    d=read(OUT/'completion.json')
    return f"首轮 10+10 个主留出及两个补充 chunk（含一次同合同补采）共启动 {d['attempts']} 次 CA，取得合格设备窗口 {d['qualified_windows']} 个，首个启动至最后结束 {d['wall_s']/60:.2f} 分钟，产物 {d['raw_bytes']/2**30:.2f} GiB。第二次 RMSNorm 独立验证另计。整批长任务、新算子和基础设施尝试在成本表单列。"


def cost_breakdown():
    costs=read(OUT/'completion.json')
    sandbox=read(OUT/'sandbox-costs.json')
    pilots=[read(V/'prefill'/(op+'_262144_5120.json'))['cost'] for op in OPS]
    cast=read(V/'workloads/cast-evaluation.json')['cost']
    router=read(V/'workloads/router-evaluation.json')['costs']
    attn=read(V/'workloads/attn-res-extension-evaluation.json')['cost']
    conf=read(V/'rms-confirmation/completion.json')
    data=[('主留出与 2048 行补测',costs['attempts'],costs['qualified_windows'],costs['simulator_wall_s_sum'],costs['raw_bytes'],costs['cpu_s_accounted_lower_bound']),
          ('RMSNorm 第二次冻结独立验证',conf['attempts'],2*conf['paired'],conf['simulator_wall_s_sum'],conf['raw_bytes'],conf['cpu_s_accounted_lower_bound']),
          ('受限 socket 的首次启动',len(sandbox),0,sum(r.get('seconds',0) or 0 for r in sandbox),sum(r.get('raw_bytes',0) or 0 for r in sandbox),sum(r.get('cpu_seconds',0) or 0 for r in sandbox)),
          ('262144 行整批试跑',len(pilots),0,sum(r['seconds'] for r in pilots),sum(r['raw_bytes'] for r in pilots),None),
          ('MoE 路由（含开发）',router['physical_attempts'],router['qualified']+router['recovered_complete_windows'],router['simulator_wall_sum_s'],router['retained_bytes'],router['cpu_s_lower_bound']),
          ('Cast（含开发与写回复测）',cast['attempts'],cast['qualified'],cast['simulator_wall_s_sum'],cast['raw_bytes'],cast['cpu_s_lower_bound']),
          ('AttnResFwd（含开发及补充形状）',attn['launches'],attn['eligible'],attn['simulator_wall_s_sum'],attn['raw_mib']*2**20,attn['known_cpu_s_lower_bound'])]
    total=('总计（含试跑、失败、校准和复核）',*(sum(row[i] or 0 for row in data) for i in range(1,6)))
    data.append(total)
    return [*table(['范围','物理尝试','合格窗口','逐次墙钟之和 min','留存 GiB','已记 CPU h（下界）'],
            [[label,attempt,ok,f'{wall/60:.1f}',f'{raw/2**30:.2f}',num(cpu/3600 if cpu else None)] for label,attempt,ok,wall,raw,cpu in data]),'',
            '同一进程已有内层导出并通过恢复核查时只计一次。并行尝试的墙钟求和包含重叠，仅表示采集工作量。CPU 缺失的终止任务不计入下界。','']


def wheel_paragraph():
    d=read(V/'wheel-smoke.json');w=d['overall']
    return f"隔离安装的 v0.19 wheel 共 {w['n']} 次查询，中位 {w['median_ms']:.3f} ms、P95 {w['p95_ms']:.3f} ms、最大 {w['max_ms']:.3f} ms。导入另计 {d['import_ms']:.2f} ms。"

if __name__=='__main__':report()
