"""Replay every recorded BF16 Matmul whole window through the current ordinary and auto APIs."""
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from time import perf_counter
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from operator_model import estimate,load_chip
from operator_model.matmul_resource import resource_matmul
from development.validation.matmul28.streamk_audit import audit as refresh,read,write,metrics,OUT,DT,PR,FUT
from development.validation.matmul28.overview import receipt as registered_receipt

def run():
    refresh()
    rows=[];seen=set()
    def add(sku,shape,ca,ids,source,role,kernel,**extra):
        if set(ids)&seen:return
        seen.update(ids)
        old=resource_matmul(shape,load_chip(sku))[0];t=perf_counter();e=estimate('matmul',shape,chip=sku);ms=(perf_counter()-t)*1000
        res=estimate('matmul',shape,chip=sku,timing_mode='resource');assert res.latency_us==e.latency_us
        assert abs(sum(e.projection['stages_us'].values())-e.latency_us)<1e-7
        other=estimate('matmul',shape,chip=DT);future=estimate('matmul',shape,chip=FUT)
        rows.append(dict(sku=sku,shape=shape,ca_us=ca,old_us=old,public_us=e.latency_us,direct_dt_us=other.latency_us,extrapolated_970dt_us=future.latency_us,
            branch=e.projection['timing_diagnostics']['streamk_selection']['route'],aic=e.active_aic,aiv=e.active_aiv,
            status=e.status,cube_ops=e.work['cube_ops'],logical_vec_ops=e.work['vec_ops'],implementation_work=e.work.get('implementation_work'),
            query_ms=ms,source=source,role=role,kernel=kernel,raw_ids=ids,**extra))
    arch=ROOT/'results/v0.25/coverage/audit.json'
    for x in read(arch)['rows']:
        if x['op']=='matmul':
            add(x['chip'],x['shape'],x['ca_us'],x['trace_sha256'],x['origins'],'historical archive replay',x['kernel'],original_roles=x['origins'])
    for x in read(OUT/'streamk-score.json')['rows']:
        if x.get('status')=='qualified':
            add(x['sku'],x['shape'],x['ca_us'],[x['trace_sha256']],x.get('drain_receipt',x['receipt']),
                'v0.28 application replay',x['kernel'],original_splitk_role=x['role'],lane=x['lane'],observed_cores=[x['aic'],x['aiv']])
    for item in read(OUT/'overview-prices.json')['rows']:
        if item['group']=='.':continue  # The original 16 geometries were audited above.
        for sku,tag in ((DT,'dt'),(PR,'pr')):
            summary=registered_receipt(item,tag)
            if summary['status']!='qualified':continue
            path=OUT/summary['receipt']
            evidence=read(path)
            stage,=evidence.get('stages') or [evidence['observed']]
            add(sku,item['shape'],summary['ca_us'],[stage['trace_sha256']],str(path.relative_to(ROOT)),
                'v0.28 application replay',stage['kernel'],registration=item['group']+'/freeze.json',
                lane=item['phase'],observed_cores=[stage['active_aic'],stage['active_aiv']])
    gs={};ratios={}
    for sel in ('historical archive replay','v0.28 application replay','all','one_output_split_k','v0.28 ordinary'):
        group=[x for x in rows if sel=='all' or sel==x['role'] or sel==x['branch'] or sel=='v0.28 ordinary' and x['role']=='v0.28 application replay' and x['branch']=='ordinary']
        gs[sel]={}
        for sku in (DT,PR):
            subset=[x for x in group if x['sku']==sku]
            gs[sel][sku]={key:metrics(subset,key) for key in ('old_us','public_us','direct_dt_us')}
        pairs=defaultdict(lambda:defaultdict(list))
        for r in group:pairs[tuple(r['shape'])][r['sku']].append(r)
        matched=[]
        for shape,p in pairs.items():
            if set(p)!={DT,PR}:continue
            a,b=p[DT][0],p[PR][0];m=median(x['ca_us']for x in p[DT])/median(x['ca_us']for x in p[PR])
            matched.append(dict(shape=shape,ca_us=m,old_us=a['old_us']/b['old_us'],public_us=a['public_us']/b['public_us'],n_dt=len(p[DT]),n_pr=len(p[PR])))
        ratios[sel]=dict(rows=matched,old=metrics(matched,'old_us'),current=metrics(matched,'public_us'),direction_matches=sum((r['ca_us']>1)==(r['public_us']>1) for r in matched))
    out=dict(role='Deployment replay, no scores presented here are a new prospective study.',rows=rows,groups=gs,paired_ratios=ratios)
    write(OUT/'public-score.json',json.dumps(out,indent=2,allow_nan=False)+'\n');render(out)

def render(d):
    lines=['# 当前常规 Matmul 查询的双端误差','',
          '| 数据 | SKU | 窗数 | 原普通 WAPE / 最大 | 当前 auto 与 resource WAPE / 最大 | 直接使用 DT 时间 WAPE / 最大 |',
          '|---|---|---:|---:|---:|---:|']
    names={'historical archive replay':'v0.25 汇总的历史全窗','v0.28 application replay':'本轮应用形回放','all':'去重合计','one_output_split_k':'单输出切 K','v0.28 ordinary':'本轮其余输出几何'}
    def fmt(a):return f"{a['wape']:.2f}% / {a['max_ape']:.2f}%" if a['n'] else 'N/A'
    for g,v in d['groups'].items():
        for sku,c in v.items():lines.append(f"| {names[g]} | {sku[6:11]} | {c['public_us']['n']} | {fmt(c['old_us'])} | {fmt(c['public_us'])} | {fmt(c['direct_dt_us'])} |")
    lines+=['','以旧普通资源公式为基线。本表的直接 DT 基线指同一 shape 的当前 DT 预测直接用作 PR 时间。纳入原归档及新整窗，按原始 trace SHA 去重。公共入口采纳了已看过的新形证据，因此这些数据在本表均按已有机制核验或开发回放处理。原本真正前瞻的端点保留其历史来源，未增加样本数。[逐窗、散列与数据角色](public-score.json) · [单矩形机制的原冻结](streamk-analysis.md)','',
           '| 数据 | DT、PR 同形配对数 | 原预测比 WAPE / 最大 | 当前预测比 WAPE / 最大 | 当前快慢方向相符 |','|---|---:|---:|---:|---:|']
    for g,v in d['paired_ratios'].items():
        lines.append(f"| {names[g]} | {v['current']['n']} | {fmt(v['old'])} | {fmt(v['current'])} | {v['direction_matches']} |")
    lines+=['','每个可配对 shape 在各端取完整窗中位数，再计算 DT/PR 时间比。直接 DT 时间给 PR 对应比值 1。原始测量不完全同时，接近 1 的符号可能受短核的控制或等待抖动影响。','',
           '| [M,N,K] | SKU | 当前分支 | 旧 / 当前 / CA µs | 误差 | 实测 AIC/AIV | 970DT 外推 µs |','|---|---|---|---:|---:|---:|---:|']
    for r in d['rows']:
        if r['role']!='v0.28 application replay':continue
        lines.append(f"| {r['shape']} | {r['sku'][6:11]} | {r['branch']} | {r['old_us']:.3f} / {r['public_us']:.3f} / {r['ca_us']:.3f} | {100*(r['public_us']/r['ca_us']-1):+.2f}% | {'/'.join(map(str,r['observed_cores']))} | {r['extrapolated_970dt_us']:.3f} |")
    ex=next(r for r in d['rows'] if r['shape']==[16,64,7168] and r['sku']==DT)
    work=ex['implementation_work'];lines+=['','## 使用与外推','',
       '调用 estimate("matmul", [M,N,K], chip=...) 时，auto 与 resource 经过同一物理选择。逻辑 cube_ops=2MNK、vec_ops=0，以及逻辑输入/输出流量仍表示调用者的矩阵乘。work["implementation_work"] 和 timing_diagnostics["internal_work"] 另外报告 FP32 部分和读、写、归约加法以及成本模型中扣除基线后的额外字节。普通无切 K 和 Addmm 分支保留原语义。','',
       f"例如 DT [16,64,7168] 的有用 Cube 工作为 {ex['cube_ops']:,} ops，32 个部分和分别带来 {work['fp32_partial_write_bytes']:,} 字节写入、{work['fp32_partial_read_bytes']:,} 字节读取，归约约 {work['fp32_reduction_adds']:,} 次 FP32 加法。归约计价项 {work['priced_incremental_fixup_bytes']:,} 字节是模型的增量，并非读写总和。",'',
       '970DT 依其 120 AIC、120 AIV、频率与带宽重算。当前 DAV_3510 风格的阈值在 120 AIC 时要求 ceil(K/256)×256 ≥15360。K7168 因此在 970 场景走普通路径。若今后 K 越过门槛，执行仍依赖后续库是否实现 1:1 混合模板。两种 950 模型的核数变化已可对比。970 的耗时和分支均为外推。','',
       f"当前表共 {len(d['rows'])} 个唯一完整窗，进程内普通调用墙钟中位 {median(x['query_ms'] for x in d['rows']):.3f} ms、最大 {max(x['query_ms'] for x in d['rows']):.3f} ms。设备时间及原始采集墙钟分别保留在逐窗表和采集回执。",'',
       '旧普通分块可能把 [144,64,7168] 分类为九个输出 tile，而实际 4098 每个 AIC 都处理完整 144×64 矩形，各自只有一段 K。DT 每核实际 MMAD 深度 64+64+64+32。对 [9,1280,5120] 则确有五个 N=256 输出块，每块的 K 由多个 AIC 分担。开源 host 为不同 K 切数再次重选 baseM/baseN 并检查 L0、L1、核与带宽；这是一个待验证的独立机制，当前普通回退的误差照录。[重选块候选与八个回放窗](multi/report.md) · [同轴时间图](multi/dt-144-retile.png)','']
    write(OUT/'public-report.md','\n'.join(lines))
if __name__=='__main__':run()
