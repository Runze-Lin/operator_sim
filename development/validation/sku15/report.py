"""Publish the v0.15 accuracy report from frozen predictions and qualified CA windows."""
import json
import math
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from development.validation.report_v13 import relocate
from development.validation.sku14.metrics import comparison
from operator_model import estimate, load_chip

OUT = ROOT / 'results/v0.15'
OLD = ROOT / 'results/v0.14'
DOCS = ROOT/'docs'
PR = 'Ascend950PR_957b'
OPS = ('fa','matmul')
NAMES = {'fa':'FA','matmul':'Matmul','quant':'DynamicMXQuant'}


def load(p):
    return json.loads(p.read_text())


def dump(p, obj):
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def case_key(r):
    shape = tuple(r['shape'])
    return r['op'], shape+((1,1) if r['op']=='fa' and len(shape)==3 else ())


def err(m):
    return f"{m['wape']:.2%} / {m['max_ape']:.2%}"


def match(r):
    return (r['pr_aic'],r['pr_aiv']) == (r['active_aic'],r['active_aiv'])


def figure(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.4),layout='constrained')
    for ax,op in zip(axes,OPS):
        group=sorted((r for r in rows if r['op']==op),key=lambda r:r['pr_ca_us'])
        ax.axvline(0,color='#2b3f4b',lw=.9)
        for key,label,marker,color,offset in [('dt_us','Reuse DT','o','#a25e36',-.15),
                  ('prior_pr_us','v0.14 PR','+','#4d6493',0),('pr_us','v0.15 PR','x','#087869',.15)]:
            ax.scatter([100*(r[key]/r['pr_ca_us']-1) for r in group],
                       [j+offset for j in range(len(group))],label=label,color=color,marker=marker,s=34)
        ax.set_yticks(range(len(group)), ['×'.join(map(str,r['shape'])) for r in group],fontsize=8)
        ax.set_title(NAMES[op]);ax.set_xlabel('Signed error vs 950PR CA (%)')
        ax.grid(axis='x',alpha=.18)
        ax.spines[['top','right','left']].set_visible(False)
    axes[0].legend(loc='best',fontsize=8)
    fig.savefig(OUT/'comparison.png',dpi=160)
    plt.close(fig)


def collect_rows():
    older=ROOT/'results/v0.13/sku'
    old=load(older/'pairs.json')+load(older/'holdout.json')+load(OLD/'sku/holdout.json')
    rows=load(OUT/'sku/holdout.json')
    if Counter(r['op'] for r in rows)!={'fa':10,'matmul':10} or not any(r['op']=='fa' and r['shape'][1]==32768 for r in rows):
        raise ValueError('Requires ten qualified new shapes per operator, including FA 32k')
    seen={case_key(r) for r in old}
    frozen={r['id']:r for r in load(OUT/'sku/freeze.json')['predictions']}
    for r in rows:
        key=case_key(r)
        if key in seen: raise ValueError(f'PR shape reused {key}')
        seen.add(key)
        prior=frozen[r['id']]
        if key!=case_key(prior): raise ValueError('frozen case shape mismatch')
        for field in ('dt_us','pr_us','prior_pr_us'):
            if r[field]!=prior[field]: raise ValueError('prediction changed')
        pr=estimate(*key,chip=PR)
        dt=estimate(*key)
        if not math.isclose(pr.latency_us,r['pr_us'],rel_tol=1e-10) or not math.isclose(dt.latency_us,r['dt_us'],rel_tol=1e-10):
            raise ValueError('deployed model differs from freeze')
    for r in old:
        op,shape=case_key(r)
        original_holdout=r.get('role')=='independent_pr'
        r['role']='independent_pr' if original_holdout and op=='quant' else 'development'
        public=estimate(op,shape,chip=PR)
        r.update(target_us=public.latency_us,dt_estimate_us=estimate(op,shape).latency_us,
                 target_aic=public.active_aic,target_aiv=public.active_aiv)
    return old,rows


def publish():
    old,rows=collect_rows()
    # The same physical resource vector but no matching SKU calibration.
    anonymous=replace(load_chip(PR),name='anonymous-PR-resources')
    ablation=[]
    for r in rows:
        masked=estimate(r['op'],r['shape'],chip=anonymous)
        if masked.latency_us is None or (masked.active_aic,masked.active_aiv)!=(r['pr_aic'],r['pr_aiv']):
            raise ValueError('Withholding target phase calibration unexpectedly changed the dispatch')
        ablation.append(dict(id=r['id'],op=r['op'],shape=r['shape'],pr_ca_us=r['pr_ca_us'],
                             pr_us=masked.latency_us,dt_us=r['dt_us'],identified_pr_us=r['pr_us']))
    blind={op:comparison([r for r in ablation if r['op']==op])['pr_us'] for op in OPS}
    dump(OUT/'resource-only.json',dict(protocol='Post-freeze numerical ablation on identical PR hardware; no target-SKU timing profile. No refit.',summary=blind,rows=ablation))
    stats={op:comparison([r for r in rows if r['op']==op]) for op in OPS}
    wheel=load(OUT/'wheel-check.json')['checks']
    current=load(ROOT/'results/v0.12/evaluation.json')
    quant=load(OLD/'evaluation.json')['pr_summary']['quant']
    official={**stats,'quant':quant}
    lines=['# 算子耗时预测与 CA 校准误差','',
           '| 算子 | DT 开发 / 独立 | DT 独立 WAPE / 最大误差 | PR 开发 / 独立 | PR 独立 WAPE / 最大误差 | 最慢查询 ms（DT / PR） | 校准状态 |',
           '|---|---:|---:|---:|---:|---:|---|']
    for s in current['operator_summary']:
        op=s['op'];d=s['development'];n=s['independent']
        pdev=sum(r['op']==op and r['role']=='development' for r in old)
        p=official.get(op)
        status='两端 CA' if p else 'DT 有 CA，PR 待测' if n else '仅结构'
        dt_time=wheel['Ascend950DT_9582']['per_op'].get(op)
        pr_time=wheel[PR]['per_op'].get(op)
        times=f"{dt_time['max_ms']:.3f} / {pr_time['max_ms']:.3f}" if dt_time and pr_time else 'N/A'
        s['query_max_ms']=dt_time['max_ms'] if dt_time else None
        lines.append(f"| {s['name']} | {d} / {n} | {err(s) if n else 'N/A'} | {pdev} / {p['n'] if p else 0} | "
                     f"{err(p['pr_us']) if p else 'N/A'} | {times} | {status} |")
    lines.extend(['','WAPE 是同组绝对误差之和除以 CA 总耗时，最大误差是最差单例的绝对相对误差。'
        'FA 和 Matmul 旧 PR 点已转入开发，这两项的独立误差只来自本轮冻结之后首次采集的各 10 个 shape。'
        'Quant 沿用 v0.14 的 10 个独立点，DT 沿用已验证的参考路线。查询时间为安装包在同一进程内的实测墙钟。','',
        relocate((OLD/'chips.md').read_text(),OLD,DOCS).strip(),'','## 新 PR 留出集与两套基线','',
        '三种预测均在采集前冻结，使用相同的单卡 BF16 输入和同一个 PR CA 分母。“复用 DT”取 DT 模型估时。'
        'Matmul 使用 ops-nn 的 `aclnnMatmul`，FA 使用 ops-transformer 的 `aclnnFlashAttentionScoreV2`。'
        '逐点记录保存实际安装包启动的 kernel、主程序、数值检查和全核时间窗。','',
        '| 算子 | 新 shape | v0.15 WAPE / 最大 | v0.14 WAPE / 最大 | 复用 DT WAPE / 最大 | 核数命中 |',
        '|---|---:|---:|---:|---:|---:|'])
    for op in OPS:
        s=stats[op]; hits=sum(match(r) for r in rows if r['op']==op)
        lines.append(f"| {NAMES[op]} | 10 | {err(s['pr_us'])} | {err(s['prior_pr_us'])} | {err(s['dt_us'])} | {hits}/10 |")
    lines+=['','### 用新芯片时，多少精度依赖 PR 的校准数据','',
            '另外做了一次事后检验：完整保留 950PR 的核数、精度算力、时钟、带宽及缓冲配置，仅隐去 SKU 身份，'
            '让已冻结的程序执行未知芯片的阶段迁移。这使两种预测的分块与核数完全一致，差别来自以 DT 为参考的阶段成本及其校准方式。'
            '这组对照复用了同一批 20 个 CA，没有增加采集，也没有用它选参数。','',
            '| 算子 | 只给资源配置 WAPE / 最大 | 启用已知 PR 校准 WAPE / 最大 |','|---|---:|---:|']
    for op in OPS:
        lines.append(f"| {NAMES[op]} | {err(blind[op])} | {err(stats[op]['pr_us'])} |")
    lines+=['','PR 与 DT 仍属于同代架构；两边均已用于开发库策略，这个模拟未知 SKU 的对照不能建立未来代际的误差界。'
            '[逐点资源配置对照](../results/v0.15/resource-only.json) · [单参数敏感性实验](../results/v0.15/scenarios.json)']
    lines+=['','![相同 PR 实测下三种预测的逐点误差](../results/v0.15/comparison.png)','',
            relocate((OUT/'diagnosis.md').read_text(),OUT,DOCS).strip(),'','### 逐 shape 与库 kernel','',
            '| 算子 / 每卡 shape | DT CA µs | PR CA µs | v0.15 PR µs | v0.14 PR µs | 复用 DT µs | v0.15 有符号误差 | 活跃 AIC/AIV 预测 → CA | PR kernel |',
            '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for r in rows:
        dt='N/A' if r.get('dt_ca_us') is None else f"{r['dt_ca_us']:.3f}"
        lines.append(f"| {NAMES[r['op']]} `{r['shape']}` | {dt} | {r['pr_ca_us']:.3f} | {r['pr_us']:.3f} | "
                     f"{r['prior_pr_us']:.3f} | {r['dt_us']:.3f} | {r['pr_us']/r['pr_ca_us']-1:+.2%} | "
                     f"{r['pr_aic']}/{r['pr_aiv']} → {r['active_aic']}/{r['active_aiv']} | `{r['kernel']}` |")
    lines.extend(['','### 旧 PR trace 的开发回放','',
                  '| 算子 / shape | PR CA µs | 当前预测 µs | 误差 | AIC/AIV 预测 → CA |','|---|---:|---:|---:|---|'])
    for r in old:
        if r['op'] not in OPS: continue
        ca=r['pr']['ca_us'];us=r['target_us']
        lines.append(f"| {NAMES[r['op']]} `{r['shape']}` | {ca:.3f} | {us:.3f} | {us/ca-1:+.2%} | "
                     f"{r['target_aic']}/{r['target_aiv']} → {r['pr']['active_aic']}/{r['pr']['active_aiv']} |")
    lines+=['','### DynamicMXQuant 在 PR 的逐例校验','',
            '| 每卡 shape | PR CA µs | v0.15 µs | 有符号误差 | 角色 |','|---|---:|---:|---:|---|']
    for r in old:
        if r['op']=='quant':
            t=r['pr']['ca_us'];u=r['target_us']
            lines.append(f"| `{r['shape']}` | {t:.3f} | {u:.3f} | {u/t-1:+.2%} | "
                         f"{'独立（v0.14）' if r['role']=='independent_pr' else '开发'} |")
    lines+=['','Quant 使用 `aclnnDynamicMxQuant` 的 BF16 → E4M3FN、block32 路线，十个留出点的 AIV 数均命中，'
            '最大偏差来自低并发 [4,256] 的读写窗。[逐例关键路径](../results/v0.14/quant-independent.md)','',
            relocate((OUT/'cost.md').read_text(),OUT,DOCS).strip(),'',
            '[冻结预测](../results/v0.15/sku/freeze.json) · [逐点 CA 核验记录](../results/v0.15/sku/holdout.json) · '
            '[当前评估 JSON](../results/v0.15/evaluation.json) · [本轮变更](../results/v0.15/changes.md)','']
    previous=(OLD/'error-analysis.md').read_text()
    lines.append(relocate(previous[previous.index('## 950DT 已校准路线的同集对照'):],OLD,DOCS))
    content='\n'.join(lines)
    figure(rows)
    (DOCS/'error-analysis.md').write_text(content)
    (OUT/'error-analysis.md').write_text(relocate(content,DOCS,OUT))
    summaries=current['operator_summary']
    dump(OUT/'evaluation.json',dict(version='v0.15',pr_summary=official,dt_summary=summaries,
         qualification='sku/holdout.json',freeze='sku/freeze.json',
         rows=[{k:r.get(k) for k in ('id','op','shape','role','pr_us','prior_pr_us','dt_us','pr_ca_us','dt_ca_us','pr_aic','pr_aiv','active_aic','active_aiv','kernel')} for r in rows]))
    for s in summaries:
        for g in s.get('groups',{}).values():
            g.update(previous_wape=g['wape'],previous_max_ape=g['max_ape'])
    index=load(ROOT/'results/v0.13/evaluation.json')['rows']
    for r,ref in zip(current['rows'],index,strict=True):
        if (r['op'],r['shape'])!=(ref['op'],ref['shape']):raise ValueError('DT references differ')
        r.update(previous_us=r['predicted_us'], query_max_ms=wheel['Ascend950DT_9582']['archived_row_ms'][ref['id']])
    for r in rows:
        r.update(target_us=r['pr_us'],dt_estimate_us=r['dt_us'],target_aic=r['pr_aic'],target_aiv=r['pr_aiv'],
                 pr_to_dt_ratio=r['pr_ca_us']/r['dt_ca_us'] if r.get('dt_ca_us') else None)
    current.update(version='v0.15',previous_version='v0.14',reference_calibration_release='v0.12',
                   paired_sku=old+rows,pr_summary=official,query_wall={sku:data['per_op'] for sku,data in wheel.items()},
                   collection_cost={'reference_archive':'results/v0.12/evaluation.json',
                       'new_pr_success_seconds':sum(a['seconds'] for a in load(OUT/'sku/attempts.json') if a['qualified']),
                       'new_pr_rejected_seconds':sum(a['seconds'] for a in load(OUT/'sku/attempts.json') if not a['qualified']),
                       'control_seconds':load(OUT/'sku/preflight.json')['wall_seconds'],
                       'campaign':load(OUT/'sku/campaign-cost.json')})
    dump(ROOT/'results/current/evaluation.json',current)


if __name__=='__main__':
    publish()
