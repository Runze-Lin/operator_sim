"""Publish default-route errors using the frozen quartet and the complete, qualified PR holdout."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from statistics import median

ROOT=Path(__file__).resolve().parents[3]
R=ROOT/'results';V=R/'v0.16';S=V/'sku'; DOC=ROOT/'docs'
METHODS=('pr_us','legacy_resource_us','dt_us','prior_pr_us')
OPS=('quant','matmul','fa')
NAMES=dict(quant='DynamicMXQuant',matmul='Matmul',fa='FA')
def load(p):return json.loads(p.read_text())
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def relocate(s,old):
    return re.sub(r'\]\(([^() ]+)\)',lambda m:m[0] if m[1].startswith(('https:', 'http:', '#')) else ']('+os.path.relpath(old/m[1],DOC)+')',s)
def metrics(rows):
    out={}
    for k in METHODS:
        err=[r[k]-r['pr_ca_us'] for r in rows];ape=[abs(x)/r['pr_ca_us'] for x,r in zip(err,rows)]
        out[k]=dict(wape=sum(map(abs,err))/sum(r['pr_ca_us'] for r in rows),max_ape=max(ape),
            mean_signed_us=sum(err)/len(err),worst_id=rows[ape.index(max(ape))]['id'])
    out.update(n=len(rows),cores_matched=sum((r['pr_aic'],r['pr_aiv'])==(r['active_aic'],r['active_aiv']) for r in rows),
        mmads_matched=sum(r['projection']['timing_diagnostics'].get('mmads_total',0)==r['pr']['selected_timeline_counts']['MMAD'] for r in rows),
        paired_dt=sum(r['dt_ca_us'] is not None for r in rows),dt_max_ms=max(r['dt_query_ms'] for r in rows),pr_max_ms=max(r['pr_query_ms'] for r in rows))
    return out
def value(r):
    return f"{100*r['wape']:.2f}% / {100*r['max_ape']:.2f}%" if r['wape'] is not None else 'N/A'
def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,5.4),layout='constrained')
    for ax,op in zip(axes,OPS):
        rs=[r for r in rows if r['op']==op]
        for i,(k,label,marker,color) in enumerate(zip(METHODS,['v0.16 resource','v0.15 anonymous','DT reuse','v0.15 PR calibrated'],['o','x','^','s'],['#00766f','#a75816','#6d5890','#687380'])):
            ax.scatter([100*(r[k]/r['pr_ca_us']-1) for r in rs],[y+(i-1.5)*.14 for y in range(len(rs))],label=label,marker=marker,c=color,s=28)
        ax.axvline(0,color='#3e4146',lw=.8);ax.set_yticks(range(len(rs)),['×'.join(map(str,r['shape'])) for r in rs],fontsize=8)
        ax.grid(axis='x',alpha=.2);ax.set_title(NAMES[op]);ax.set_xlabel('Signed error vs PR CA (%)')
    axes[0].legend(fontsize=8,loc='best');fig.savefig(V/'comparison.png',dpi=150);plt.close(fig)

def diagnose(r):
    op=r['op'];ca=r['pr_ca_us'];st=r['projection']['stages_us'];ph=r['pr']['phase_landmarks'];core=max(r['pr']['core_windows'],key=lambda x:x['end_us'])
    s=f"{NAMES[op]} `{r['shape']}`：预测 {r['pr_us']:.3f}、PR CA {ca:.3f} µs（{100*(r['pr_us']/ca-1):+.2f}%），复用 DT 为 {r['dt_us']:.3f} µs。"
    if op=='matmul':
        p=ph[core['core']];first=p['MOV_OUT_TO_L1_MULTI_ND2NZ']['first_start_us'];end=p['MMAD']['last_end_us']
        s+=f"尾核发首条 GM→L1 数据读在 {first:.3f} µs，模型启动项 {st['launch']:.3f} µs；最后 MMAD 至退出剩 {ca-end:.3f} µs。模型全部输出波写尾项合计 {st['write_tail']:.3f} µs，多波时不能与末波尾段直接相减。"
        if r['shape']==[1024,2048,32]:
            import hashlib
            path=ROOT/r['pr']['trace'];data=path.read_bytes();assert hashlib.sha256(data).hexdigest()==r['pr']['trace_sha256']
            events=[e for e in json.loads(data)['traceEvents'] if e['ph']=='X'];origin=min(e['ts'] for e in events)
            mm=[(e,json.loads(e['args']['detail'])) for e in events if e['name']=='MMAD']
            hist=Counter((x['m'],x['n'],x['k']) for e,x in mm)
            last_full=max(e['ts']+e['dur']-origin for e,x in mm if x['m']==256)
            first_tail=min(e['ts']-origin for e,x in mm if x['m']<256)
            s+=f" 预测和 CA 都用 {r['active_aic']} 个 AIC，但模型按 32 个 256×256 输出块算两波；实际为 {hist[256,256,32]} 次 [256,256,32]、{hist[86,128,32]} 次 [86,128,32] 和 {hist[84,128,32]} 次 [84,128,32]。说明末尾 4 个整块进一步拆分，24 核各算一个小尾块。大块最后一条 MMAD 在 {last_full:.3f} µs 完成，第一条尾块 MMAD 早在 {first_tail:.3f} µs 就已发起；串行收取第二波完整首读与写尾和实测不符。"

    elif op=='fa':
        mm=[v['MMAD'] for v in ph.values() if 'MMAD' in v]
        first=min(v['first_start_us'] for v in mm);mm_end=max(v['last_end_us'] for v in mm)
        s+=f"首次 MMAD 在 {first:.3f} µs，最后 MMAD 完成后还有 {ca-mm_end:.3f} µs；预测稳态 {st['steady']:.3f} µs、排空 {st['drain']:.3f} µs。首次 MMAD 包含读和 L0 等待，不可直接和 launch 相减归因为启动项。"
    else:
        s+=f"预测调度、UB 流水、排空分别 {st['dispatch']:.3f}、{st['ub_pipeline']:.3f}、{st['drain']:.3f} µs。"
    if r['dt_ca_us'] is not None:
        s+=f" 配对 DT CA {r['dt_ca_us']:.3f} µs，实测跨芯片差 {ca-r['dt_ca_us']:+.3f} µs，而资源预测相对 DT 基线的差为 {r['pr_us']-r['dt_us']:+.3f} µs。"
    else:s+=' 没有该 shape 的 DT CA 配对；无法从两个预测的差值推断设备实际加速。'
    return s

def publish():
    rs=load(S/'holdout.json');counts=Counter(r['op'] for r in rs);assert counts==Counter(dict.fromkeys(OPS,10))
    att=load(S/'attempts.json');cost=load(S/'campaign-cost.json');freeze=load(S/'freeze.json')
    assert freeze['utc']<min(a['started_utc'] for a in att)
    assert {r['id'] for r in rs}=={a['id'] for a in att if a['qualified']}
    frozen={r['id']:r for r in freeze['predictions']}
    for r in rs:
        assert all(r[k]==frozen[r['id']][k] for k in METHODS)
        assert r['pr_us']==r['auto_pr_us'] and abs(r['prior_pr_us']-r['calibrated_pr_us'])<1e-8
        assert all('bad=0' in q for q in r['pr']['correctness'])
    groups={op:[r for r in rs if r['op']==op] for op in OPS}
    assert all(len({tuple(r['shape']) for r in g})==10 for g in groups.values())
    metrics_by={op:metrics(g) for op,g in groups.items()}
    archive=load(R/'v0.15/evaluation.json')
    current=load(R/'current/evaluation.json');hist=[r for r in current['paired_sku'] if r.get('evaluation_release')!='v0.16']
    assert len(hist)==55
    for r in hist:
        r.setdefault('role_at_collection',r['role']);r['role']='historical_pr_pre_v016'
    for r in rs:
        r.update(target_us=r['pr_us'],target_aic=r['pr_aic'],target_aiv=r['pr_aiv'],dt_estimate_us=r['dt_us'],evaluation_release='v0.16')
    serial=[{**{k:r[k] for k in ('id','op','shape','role','pr_ca_us','dt_ca_us',*METHODS,'kernel','pr_aic','pr_aiv','active_aic','active_aiv','pr_query_ms','dt_query_ms')},
        'api':r['contract']['api'],'family':r['contract']['family'],'correctness':r['pr']['correctness'],'selected_timeline_counts':r['pr']['selected_timeline_counts'],
        'mmad_predicted':r['projection']['timing_diagnostics'].get('mmads_total',0),'trace_sha256':r['pr']['trace_sha256'],
        'observed_events':r['pr']['events'],'trace':r['pr']['trace'],'logical_read_bytes':r['work']['logical_read_bytes'],'logical_write_bytes':r['work']['logical_write_bytes'],
        'ca_wall_seconds':r['pr_cost']['seconds'],'signed_relative_error':{k:r[k]/r['pr_ca_us']-1 for k in METHODS}} for r in rs]
    summaries=archive['dt_summary']
    qualification='30 fresh PR shapes; archived DT auto calibration remains unchanged. Target phase costs use DT evidence only; 950 family dispatch structure is not PR-blind.'
    payload=dict(version='v0.16',previous_version='v0.15',pr_summary=metrics_by,dt_summary=summaries,qualification=qualification,
        qualification_records='sku/holdout.json',freeze='sku/freeze.json',freeze_sha256=hashlib.sha256((S/'freeze.json').read_bytes()).hexdigest(),rows=serial,collection_cost=cost)
    doc=['# 算子耗时预测与 CA 校准误差','',
         '| 算子 | DT 开发 / 独立 | DT 独立 WAPE / 最大 | PR 本轮独立 | PR 独立 WAPE / 最大 | 查询最大 ms（DT / PR） | 当前默认计时及校准状态 |',
         '|---|---:|---:|---:|---:|---:|---|']
    for s in summaries:
        p=metrics_by.get(s['op']);dt=s['query_max_ms'];t='N/A' if dt is None else f'{dt:.3f} / '+(f"{p['pr_max_ms']:.3f}" if p else 'N/A')
        state='DT 校准；PR 仅资源外推、已验证' if p else 'DT 校准；PR 未验证' if s['independent'] else '结构；不具备耗时资格'
        doc.append(f"| {s['name']} | {s['development']} / {s['independent']} | {value(s)} | {10 if p else 0} | {value(p['pr_us']) if p else 'N/A'} | {t} | {state} |")
    doc+=['','WAPE＝Σ|预测−CA|/ΣCA；最大误差是同组最大的 |预测−CA|/CA。表中 DT 是原已校准的 `auto` 默认路线和历史测时；PR 是本轮冻结后第一次采集的 30 个 shape、同一进程内完成单次查询的墙钟。N/A 不代表零误差。每个算子只展示一个当前默认预测，四种方法的结果仅在下文用于对照。','',
          '## 两款芯片的配置与 CA 差异','']
    old=(R/'v0.15/error-analysis.md').read_text();chips=old.split('## 两款芯片的配置与 CA 差异\n',1)[1].split('## 新 PR 留出集',1)[0]
    chips=chips.replace('预测使用参考阶段成本和假设的带宽敏感度','资源模式缺省用 DT 分阶段估算的每核有效率；有实测可显式覆盖')
    doc+=[relocate(chips,R/'v0.15').strip(),'','## 四种预测，同一批 PR CA','',
         '三类新资源模型（`timing_mode="resource"`，本轮与 PR `auto` 等值）只使用当前输入和目标配置运行。校准读、指令、控制与队列时间的来源仅是 DT；不同于 v0.15“匿名 PR”——后者隐藏名称，但仍以 DT 时间为锚点或使用旧阶段缩放。“复用 DT”直接使用 DT 预测，不是 DT CA。“v0.15 PR 校准”允许已知目标的历史时序，与新版显式 `calibrated` 逐点核对。**四列均在新 PR 采集前冻结**；资源法不会按误差切换赢家。','',
         '| 算子 | v0.16 仅资源 | v0.15 匿名 PR | 直接复用 DT | v0.15 已校准 PR | AIC/AIV 同时命中 | MMAD 总数命中 | 实测 DT 配对 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for op,p in metrics_by.items():doc.append('| '+' | '.join([NAMES[op],*[value(p[k]) for k in METHODS],f"{p['cores_matched']}/10",f"{p['mmads_matched']}/10" if op!='quant' else '不适用（Vector）',f"{p['paired_dt']}/10"])+' |')
    doc+=['','相对三种对照的同集 WAPE 变化（百分点；负数表示误差下降）：']
    for op,p in metrics_by.items():
        delta='、'.join(f"{100*(p['pr_us']['wape']-p[k]['wape']):+.2f}" for k in METHODS[1:])
        if op==OPS[0]:doc.append('')
        doc.append(f"- {NAMES[op]}：相对匿名 PR、复用 DT、已校准 PR 分别 {delta}。")
    doc+=['','![30 个独立 shape：四种冻结估时的有符号误差](../results/v0.16/comparison.png)','',
         '独立性范围是**新 PR shape 的时延**。Matmul 的库分块、Quant 的 UB 行列分配和 FA 的库路线来自既有实现知识，且曾在旧 PR 点上检验，不能声称从未看过该架构。阶段价格的 DT 证据：[Quant](../results/v0.16/quant-dt-evidence.json)、[Matmul](../results/v0.16/matmul-dt-provenance.json)、[FA](../results/v0.16/fa-dt-evidence.json)。PR 外存模拟使用 LPDDR5 子模型，不等同商品规格的持续带宽实测。没有独立的目标 L2 命中、每核 MTE 吞吐、读首延迟或跨 CV 同步测量时，继承 DT 阶段量和实现假设；960/970/980 的准确性尚未验证。','',
         '### 哪些阶段仍有误差','']
    for op,p in metrics_by.items():
        r=next(r for r in groups[op] if r['id']==p['pr_us']['worst_id']);doc.extend([diagnose(r),''])
        if op=='matmul' and r['shape']==[1536,2048,16]:
            doc+=['该 shape 的 28 核各两次 MMAD：48 次 [224,256,16] 加 8 次 [192,256,16]，K 没有再切。FIX 写回坐标覆盖 M×N 恰好一次。因此不能归因为 K 切分错误。末核第二个输出块和前一块的流水不能用两个独立完整波表示。[每核 MMAD、FIX 与覆盖](../results/v0.16/sku/1536-multiwave.json)。','']
            doc+=[diagnose(next(x for x in groups['matmul'] if x['shape']==[1024,2048,32])),'']
    qrows=load(V/'quant-pr-phases.json')['rows']
    assert len(qrows)==10 and all(q['observed']['ub_loops']==1 for q in qrows)
    doc+=['Matmul 按冻结时的波数分开看：'+ '；'.join(f"{w} 波 {len(g)} 例，WAPE {metrics(g)['pr_us']['wape']:.2%}" for w in (1,2) for g in [[r for r in groups['matmul'] if r['projection']['target_plan']['waves']==w]])+'。没有剔除两波样例，它们均计入上面的总误差。','',
          'Quant 合格的 10 个末核 trace 均只跑了一轮 UB；两个多 UB 主样例触及单次原始数据上限，被有序替补取代。本轮因此没有独立验证多 UB 的稳态发起间隔。','']
    q=next(x for x in qrows if x['id']==metrics_by['quant']['pr_us']['worst_id']);a,b=q['predicted'],q['observed']
    doc+= [f"Quant 同一最差例的末核为 `{q['critical_core']}`：首块输入模型/实测 {a['first_input_bytes']}/{b['first_input_bytes']} B，UB 轮数 {a['ub_loops']}/{b['ub_loops']}；发首个有效数据读之前 {a['dispatch']:.3f}/{b['dispatch']:.3f} µs，首读持续 {a['read_latency']:.3f}/{b['read_latency']:.3f} µs，向量阶段 {a['vector']:.3f}/{b['vector']:.3f} µs，首块写窗 {a['write']:.3f}/{b['write']:.3f} µs。各阶段会重叠，不能把差值机械相加。[10 个样例的关键核阶段](../results/v0.16/quant-pr-phases.json)。",'']
    long=load(V/'fa-pr-32k.json')
    doc+= [f"**长上下文实测**（`{long['shape']}`）：首 MMAD 预测 {long['predicted_first_mmad']:.4f}、实测 {long['actual_first_mmad']:.4f} µs；KV 稳态预测 {long['predicted_full_tile_ii']:.5f} µs/块，而 K 和 V 的第 8–119 块实测发起间隔分别 {long['k_steady_issue_us']:.5f}、{long['v_steady_issue_us']:.5f}。MMAD 均为 {long['actual_mmads']} 次；逐块偏差在长循环中累积。总预测 {long['total_predicted']:.3f}、PR CA {long['total_ca']:.3f} µs。[逐阶段来源](../results/v0.16/fa-pr-32k.json)。此处 KV 是 32,767，不是对 128k–512k 的 PR 实测。",'',
            'DT 13/17/10 个不同 shape 分别为 Quant/Matmul/FA 的阶段参数来源。旧 PR 12/22/21 个 shape 上的时延仅作冻结后的开发资料回放，不加入本轮独立集。[逐算子 DT 校准/其他历史组、旧 PR 回放与资源敏感性](../results/v0.16/resource-replay.md)。','']
    doc+=['相同核数不保证相同排队、同样发射间隔或读写重叠。以上区间和端到端结果可证实误差位置，不能从一个模拟总窗反推物理 HBM 利用率。对长 FA，读不到中短 trace 的循环稳态时，相关预测不能当实测。','',
          '### 逐 shape：库身份、正确性、真实指令与分核','',
          '统一口径：PR CA＝全核最早有效指令到最后有效指令的完整窗口（µs），不是主机调用墙钟。各行输入 dtype、轴和属性、原始正确性文本、trace/二进制 SHA、每核起止时刻保存在[原始验收记录](../results/v0.16/sku/holdout.json)。编号代表安装包实际启动的 kernel，不是开源同名实现。','']
    kernels={k:f'K{i+1}' for i,k in enumerate(sorted({r['kernel'] for r in rs}))}
    for op,g in groups.items():
        c=g[0]['contract'];doc+= [f"**{NAMES[op]}** — {c['family']} / `{c['api']}`，{c['dtype'].upper()}，轴 `{','.join(c['axes'])}`。"]
        doc+=['']
        for kernel in sorted({r['kernel'] for r in g}):doc.append(f"- {kernels[kernel]}：`{kernel}`")
        doc+=['','| shape | kernel | PR CA | 仅资源（误差） | 匿名 PR | 复用 DT | 校准 PR | (C,V) 预→实 | MMAD 预/实 | golden |','|---|---|---:|---:|---:|---:|---:|---|---:|---|']
        for r in g:
            pred=r['projection']['timing_diagnostics'].get('mmads_total',0);gold='；'.join(r['pr']['correctness']);gold=gold.replace('correctness ','').replace('comparison=exact_bytes','逐字节')
            doc.append(f"| `{r['shape']}` | {kernels[r['kernel']]} | {r['pr_ca_us']:.3f} | {r['pr_us']:.3f}（{100*(r['pr_us']/r['pr_ca_us']-1):+.2f}%） | {r['legacy_resource_us']:.3f} | {r['dt_us']:.3f} | {r['prior_pr_us']:.3f} | ({r['pr_aic']},{r['pr_aiv']})→({r['active_aic']},{r['active_aiv']}) | {pred}/{r['pr']['selected_timeline_counts']['MMAD']} | {gold} |")
        doc+=['']
    doc+=['Quant 同时比对所有量化输出和 scale 字节；Matmul 比对完整输出，FA 比对全部 attention 输出（不验证辅助 softmax 统计量）。MMAD 检查使用独立 CA 计数；Quant 的 0/0 不表示没有 Vector 指令，也不是全部 ISA 被验证。','',
         '## CA 成本与系统查询开销','',
         f"新 PR CA {cost['attempted']} 次，合格 {cost['qualified']} 次，预登记 30+6 个候选，仅按资格规则选取；没有新 DT CA。{cost['workers']} 个并发 worker，总墙钟 {cost['wall_seconds']:.1f} s；单次之和 {cost['sum_simulator_wall_seconds']:.1f} s、已知 CPU 时间 {cost['simulator_cpu_seconds']/3600:.2f} 小时（{cost['unknown_cpu_rows']} 次缺 CPU 计数，下界）、原始数据 {cost['raw_bytes']/2**30:.2f} GiB。失败/超时也计入成本，不参与误差分母。",'',
         '| 算子 | 合格单次 CA 秒（中位 / 最大） | 冻结时单次 PR 查询 ms（中位 / 最大） |','|---|---:|---:|']
    rejected=[a for a in att if not a['qualified']]
    if rejected:
        why='；'.join(f"{a['id']}（{a['reason']}，{a['seconds']:.1f} s）" for a in rejected)
        substitutes='、'.join(r['id'] for r in rs if r['priority']!='primary')
        doc[-2:-2]=[f"未通过的主采集：{why}。替补 {substitutes} 按冻结顺序进入。超限大 shape 没有完整 CA 结果，不能从替补上的误差推断其准确性。",'']
    for op,g in groups.items():
        w=[r['pr_cost']['seconds'] for r in g];q=[r['pr_query_ms'] for r in g];doc.append(f'| {NAMES[op]} | {median(w):.1f} / {max(w):.1f} | {median(q):.3f} / {max(q):.3f} |')
    wheel=load(V/'wheel-check.json')
    wc=wheel['checks']; n=sum(t['calls'] for t in wc.values())
    doc+=['',f"安装 wheel 后在源码目录外调用 `estimate`，{n} 次中最慢 {max(t['max_ms'] for t in wc.values()):.3f} ms；三个算子各 10 种 shape、400 次。首次导入 {wheel['import_ms']:.3f} ms 单列，均满足 100 ms 查询目标。[wheel 检查](../results/v0.16/wheel-check.json)。"]
    doc+=['','[冻结的 36 个预测及哈希](../results/v0.16/sku/freeze.json) · [所有采集尝试、成本与拒绝原因](../results/v0.16/sku/attempts.json) · [本轮评估 JSON](../results/v0.16/evaluation.json) · [迭代变化](../results/v0.16/changes.md)','',
         relocate('## 950DT 已校准路线的同集对照'+old.split('## 950DT 已校准路线的同集对照',1)[1],R/'v0.15')]
    # The 356 historical DT rows describe the default DT route, not a fresh validation of resource(DT).
    current.update(version='v0.16',previous_version='v0.15',pr_summary=metrics_by,paired_sku=hist+rs,qualification=qualification,
        target_query_wall_ms={op:dict(source='v0.16 preregistered, in-process resource',maximum=p['pr_max_ms']) for op,p in metrics_by.items()},
        reference_query_wall_scope='query_wall contains archived calibrated-route measurements from v0.15; current resource query walls in target_query_wall_ms',
        collection_cost={'reference_archive':'results/v0.12/evaluation.json','campaign':cost,'new_pr_success_seconds':sum(x['seconds'] for x in att if x['qualified']),
                         'new_pr_rejected_seconds':sum(x['seconds'] for x in att if not x['qualified'])})
    plot(rs);dump(V/'evaluation.json',payload);dump(R/'current/evaluation.json',current)
    (DOC/'error-analysis.md').write_text('\n'.join(doc).rstrip()+'\n')
if __name__=='__main__':publish()
