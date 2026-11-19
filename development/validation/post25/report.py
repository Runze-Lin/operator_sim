"""Score frozen independent launches while keeping archive, unknown history and extrapolation separate."""
import json, math, statistics, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.post24.report import fmt,stats
from development.validation.post24.model import predict as shadow24
from development.validation.post25.calibrate import OUT,predict
from development.validation.sku19.campaign import read,write,now
from operator_model import load_chip

def main():
    reg=read(OUT/'freeze.json');a,b=reg['measured'];new=reg['extrapolated'][0];prices=read(OUT/'prices.json')
    future={(r['group'],r['i']):r['us'] for r in reg['rows'] if r['chip']==new}
    oldparams=read(ROOT/'results/v0.24/post/freeze.json')['parameters']
    files=sorted((OUT/'receipts').glob('*.json'));rows=[];cost=[];errors=[];by={}
    if any(not (OUT/'physical'/f.name).exists() for f in files):
        from development.validation.post25.physical import main as verify_physical
        verify_physical()
    for f in files:
        obj=read(f);cost.append(dict(sku=obj['chip'],group=obj['group'],**obj['cost']))
        if obj.get('reason'):errors.append(dict(group=obj['group'],sku=obj['chip'],reason=obj['reason']))
        for r in obj['rows']:
            if not r['timing_qualified']:
                errors.append(dict(sku=r['chip'],group=r['group'],i=r['i'],reason=r.get('reason')));continue
            s=r['stage'];u=s['ca_us'];p=r.get('pipeline',{})
            row=dict(chip=r['chip'],group=r['group'],i=r['i'],shape=r['shape'],prior_aiv=r['prior_aiv'],role=r['role'],known_state=r['state'],mode=r['mode'],active=r['active'],ca_aiv=s['active_aiv'],rows_per_core=r['plan']['rows_per_core'],ca_us=u,candidate_us=r['us'],public_us=r['old_known_us'],old_shadow_us=shadow24(r['shape'],load_chip(r['chip']),r['prior_aiv'],r['previous_shape'],oldparams)['us'],candidate_unknown_us=r['candidate_unknown_us'],public_unknown_us=r['old_unknown_us'],future_us=future[r['group'],r['i']],kernel=s['kernel'],matched_kernel=s['kernel'].endswith('_6') if r['plan']['route']=='post_token' else s['kernel'].endswith('_5'),trace=s['trace'],hash=s['trace_sha256'],elements_checked=r['all_output_elements'],query_ms=r['predict_ms'],stages=p.get('stages_us'))
            by[r['group'],r['i'],r['chip']]=row;rows.append(row)
    cols=('candidate_us','public_us','old_shadow_us','candidate_unknown_us','public_unknown_us')
    summary={};speed=[]
    for sku in (a,b):
        rr=[r for r in rows if r['chip']==sku];subsets={'all':rr}
        subsets.update({st:[r for r in rr if r['mode']==st] for st in ('first','expand','reuse','repeat')})
        subsets['one-row']=[r for r in rr if r['rows_per_core']==1];subsets['multi-row']=[r for r in rr if r['rows_per_core']>1]
        summary[sku]={str(k):{c:stats(r,c) for c in cols} for k,r in subsets.items()}
    for r in rows:
        key=r['group'],r['i'];p=by.get((*key,b))
        if r['chip']!=a or not p:continue
        speed.append(dict(shape=r['shape'],group=r['group'],i=r['i'],role=r['role'],ca_us=r['ca_us']/p['ca_us'],**{k:r[k]/p[k] for k in cols}))
    ratios={k:stats(speed,k) for k in cols}
    archive=read(OUT/'archive.json')['rows']
    archival={k:{key:stats(v,key) for key in ('old_us','candidate_us')} for k,v in [('all',archive)]+[(st,[r for r in archive if r['state']==st]) for st in ('first','expand','reuse','repeat')]}
    costs=[]
    for r in reg['rows']:
        chip=load_chip(r['chip'])
        for attempt in range(8):
            t=time.perf_counter_ns();c=predict(r['shape'],chip,r['prior_aiv'],prices,previous=r['previous_shape']);costs.append((time.perf_counter_ns()-t)/1e6)
            assert abs(c['us']-r['us'])<1e-9
    costs.sort();bench=dict(n=len(costs),p50_ms=statistics.median(costs),p95_ms=costs[math.ceil(.95*len(costs))-1],max_ms=max(costs),first_frozen_max_ms=max(r['predict_ms'] for r in reg['rows']))
    write(OUT/'evaluation.json',dict(utc=now(),shape_query='BF16 with hRes, N=4',unique_shapes={s:len({tuple(r['shape']) for r in rows if r['chip']==s}) for s in (a,b)},qualified=len(rows),receipts=len(files),rows=rows,summary=summary,ratios_dt_over_pr=speed,ratio_errors=ratios,unqualified=errors,archive_only=archival,cost=cost,query=bench))
    md=['# 内置 mHC Post 的 token 循环流水','', '| 芯片及已知前序 | 独立窗 / 不同 shape | 逐行候选 WAPE / 最大 | 当前公开 v0.24 | 上轮 v0.24 影子 | 不传前序的逐行候选 | 不传前序的公开模型 |','|---|---:|---:|---:|---:|---:|---:|']
    for sku in (a,b):
        for k,s in summary[sku].items():
            n=len({tuple(r['shape']) for r in rows if r['chip']==sku and (k=='all' or r['mode']==k or (k=='one-row' and r['rows_per_core']==1) or (k=='multi-row' and r['rows_per_core']>1))})
            if s['candidate_us']:md.append('| '+sku+' / '+k+' | '+f"{s['candidate_us']['n']} / {n}"+' | '+' | '.join(fmt(s[c]) for c in cols)+' |')
    md+=['',f'两端各冻结 10 个不同形状，每个形状相邻调用两次。当前有 {len(files)}/8 个进程回执、{len(rows)}/40 个合格全窗。每个进程使用新设备流，实际 ELF 来自 CANN 9.2.1 的 ops_transformer/mhc_post_apt，宿主接口为 aclnnMhcPost。核数、物理 END、官方逐核时长、逐核 END_LABEL、有效写回总字节和全部输出元素每次独立核验。候选尚在影子评估中。', '',f"注册前共扫描 {read(OUT/'novelty.json')['manifests']} 份历史结果，所有新形的结果 shape 字段均无命中。测试 T=57…449，D=192…5120，D=5120 对应 DeepSeek V4 Flash 的隐藏宽度。[完整输入与原始冻结价格](freeze.json) · [历史查重](novelty.json)",'']
    if rows:
        worst=max(rows,key=lambda r:abs(r['candidate_us']/r['ca_us']-1))
        md+= [f"已完成样本中，最大偏差为 {worst['chip']} 的 {worst['shape']}（{worst['mode']}），预测 {worst['candidate_us']:.3f} µs，CA {worst['ca_us']:.3f} µs，有符号偏差 {100*(worst['candidate_us']/worst['ca_us']-1):+.2f}%。",'']
    md+=['## 通用机制与参数','',r'给定同流上一条算子的已知驻留核数 H，以及是否紧邻相同调用，分别选择首次、扩核、普通复用或原样重复。A 是计划活动 AIV 数，R 是最长核分配的 token 行数。只有按 token 分核且 N=4 才使用这一候选。每行从 GM 读四组输入，其字节数为 $B=(64,16,2D,8D)$，写出 $8D$ 字节。已有完整逐指令导出可以将四组读和单笔写按 GM 地址步进对应到相同 token。', '',
        r'以周期为单位。首行读发起为 $r_0=a\log_2 A+c$，$a$ 是末核加入随活动核数的成本，$c$ 是末核加入到有效操作数发起。行内第 j 笔读在 $r_i+o_j$ 发起，$o_j$ 为该端口相邻发起间隔的累加。数据全部就绪于 $r_i+L_i$，$L_i=\max_j(o_j+h_j+B_j k_2/1024)$；$h_j$ 是该描述符的固定响应，$k_2$ 是目标 MTE2 每 KiB 服务周期。', '',
        r'首笔写 $w_0=r_0+L_0+b_0+g\lceil D/64\rceil v$。$b$ 是输入就绪到写发起扣除每组算术后的固定交接，$g$ 从真实 FP32 数据操作数就绪到该行写发起的 DB 时刻鲁棒拟合，当前每 64 通道约 '+f"{prices['ready_to_write_per_group']:.2f}"+r' 周期。$v$ 为参考与目标每 AIV 每周期 FP32 算力比。首轮/后续轮分别使用各自端口固定响应与交接。首至次轮的读、写发起间隔独立于稳态间隔。', '',
        r'之后的行使用 $r_i=r_1+(i-1)I_r$，$w_i=\max(w_1+(i-1)I_w,r_i+L_s+b_s+g\lceil D/64\rceil v)$。发起间隔同时受向量与 MTE 服务下界约束。最后排空一笔 $h_w+8Dk_3/1024$ 的写响应和少量可观测的退出尾项。公开 v0.24 的全芯片 L2/HBM 字节下界仍独立取最大值。若缺 MTE 端口带宽则沿用 950 的每 KiB 周期；含 HBM 的实际共享峰值、核数和频率来自目标芯片文件。该闭式计算与 token 行数无关。','',
        '固定响应、间隔、首轮/稳态之间的差异均来自真实描述符记录，延迟扣除了对应字节服务。已知的进程/形状用整组剔除复核。D=5120 远超多轮拟合常见宽度；资源或 kernel 行为出现变化时只能据独立测量判定。紧邻重复代表可见的流关系，相同输入可由新物理分配承载，无法据此单独证明芯片命中缓存。970DT 没有 CA，全部数值为外推。','',
        f"目前 {sum(r['matched_kernel'] for r in rows)}/{len(rows)} 个合格窗观察到的安装包 kernel 分支与预测路线一致，{sum(r['active']==r['ca_aiv'] for r in rows)}/{len(rows)} 个的实测活动核数与模型相符。每行对应的完整 kernel 名和 trace 身份在逐形 JSON。",'', '## 开发档案回放','', '| 状态 | 窗口 | 公开模型 WAPE /最大 | 候选 WAPE / 最大 |','|---|---:|---:|---:|']
    for k,v in archival.items():md.append(f"| {k} | {v['old_us']['n']} | {fmt(v['old_us'])} | {fmt(v['candidate_us'])} |")
    md+=['','这 31 窗已经参与建模，评分时剔除了所评全形状的两端窗口以及本次被测进程。首次、普通热态与原样重复分开考察。相同样本用途与冻结后的新批次在表中独立列出。[阶段校准代码](../../../development/validation/post25/calibrate.py) · [逐段价格](prices.json) · [开发窗口及对照](archive.json)','',
        '## 逐形实测与 970DT 外推','', '| 芯片 | T×4×D | 调用状态 | 核数 模型/CA | 最长核行数 | CA µs | 候选 µs | 有符号偏差 | 公开 µs | 上轮影子 µs | 970DT µs |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        t,n,d=r['shape'];md.append(f"| {r['chip']} | {t}×{n}×{d} | {r['mode']} | {r['active']}/{r['ca_aiv']} | {r['rows_per_core']} | {r['ca_us']:.3f} | {r['candidate_us']:.3f} | {100*(r['candidate_us']/r['ca_us']-1):+.1f}% | {r['public_us']:.3f} | {r['old_shadow_us']:.3f} | {r['future_us']:.3f} |")
    md+=['','## 跨芯片及成本','',f"两端同一流序号的加速比为 DT 耗时 / PR 耗时。目前 {len(speed)} 对。加速比自身的 WAPE / 最大误差分别为逐行候选 {fmt(ratios['candidate_us'])}、公开模型 {fmt(ratios['public_us'])}。逐组含身份、每核和原始 trace 信息见 [评估数据](evaluation.json) 和 [合格回执](receipts/)。", '',
        f"预算为每组每 SKU 90 分钟、12 GiB，同时只运行一个 CA。已回执累计 CA 墙钟 {sum(r['seconds'] for r in cost)/60:.1f} 分钟，峰值单进程原始落盘 {max((r['peak_bytes'] for r in cost),default=0)/2**30:.2f} GiB。{bench['n']} 次查询热进程、参数及芯片已载入，墙钟 p50/p95/最大 {bench['p50_ms']:.3f}/{bench['p95_ms']:.3f}/{bench['max_ms']:.3f} ms；最慢的初次注册查询 {bench['first_frozen_max_ms']:.3f} ms。", '']
    for slug,cap in [('two-token-repeat','两轮的独立形状'),('four-token-repeat','四轮的独立稳态形状'),('multi-token-repeat','更多轮次的独立压力形状')]:
        if (OUT/(slug+'.png')).is_file():md+=['![同一组 token 上模型与下 CA 的同轴对照]('+slug+'.png)', '',cap+'，GM 写地址与四组输入步进交叉核实。相同横坐标单位 µs，逐笔交接与真实 SIMD 活动的粒度已在图内标明。[绘图数据]('+slug+'.json)','']
    pair=OUT/'two-token-repeat.json'
    if pair.exists():
        ex=read(pair);c=ex['measured'];m=ex['model'];sg=m['stages_us'];t,n,d=ex['shape']
        md+=['## 复算一个新形状','',f"取 {ex['sku']}、[T,N,D]=[{t},{n},{d}]、BF16、同流立即重复。预测分核为 {ex['active']} 个 AIV，每个最长核处理 {m['rows_per_core']} 个 token。每行读 64、16、{2*d}、{8*d} B，写 {8*d} B。上方图用输出地址确定实际末核 {c['core']} 上的两个 token，预测则使用最长核上的第 1、2 行。",'', '| 模型阶段 | µs |','|---|---:|']
        for key,value in sg.items():md.append(f'| {key} | {value:.4f} |')
        md+=['',f"合计 {m['us']:.4f} µs。独立 CA 全核设备窗 {c['ca_us']:.4f} µs。下面的读发起与写回区间均以全核最早指令为零，首行写仍在执行时下一行即可发起读。", '', '| 实际 token | 行首读发起 模型/CA µs | 末个操作数就绪 模型/CA µs | 写发起 模型/CA µs | 写完成 模型/CA µs |','|---:|---:|---:|---:|---:|']
        for x,y in zip(m['timeline'],c['items']):
            md.append(f"| {y['token']} | {x['reads_us'][0][0]:.3f}/{y['reads_us'][0][0]:.3f} | {max(z[1] for z in x['reads_us']):.3f}/{max(z[1] for z in y['reads_us']):.3f} | {x['write_us'][0]:.3f}/{y['write_us'][0]:.3f} | {x['write_us'][1]:.3f}/{y['write_us'][1]:.3f} |")
        md+=['','[全部独立窗口的逐轮间隔、阶段误差和不匹配事项](milestones.md)','']
    algebra=read(OUT/'schedule-check.json') if (OUT/'schedule-check.json').exists() else None
    if algebra:
        md+=['## 超长输入的查询成本','',f"将闭式排程与逐行递推在 {algebra['frozen_token_cases']} 个冻结配置的 {algebra['observed_rows']} 行逐项比较，最大发起时间偏差 {algebra['maximum_unrolled_drift_us']:.2g} µs。用 B=8、S=32k/128k/256k、D=5120 的内置接口做了 9 次纯模型计时，最长核达到 {max(v['rows_per_aiv'] for v in algebra['large_shape_cost']):,} 行，最大查询墙钟 {max(v['call_time_ms'] for v in algebra['large_shape_cost']):.3f} ms。这里的真实整网绑定和其他 HcPost 分支有单独条目；本批没有这九个整段 shape 的 CA。[逐芯片外推与查询计时](schedule-check.json) · [实际 vLLM HcPost 分支](../../../docs/examples/hc_post_vllm.md)",'']
    phys=OUT/'physical-summary.json'
    if phys.exists():
        ph=read(phys);md+=['## 去除合成终止标签的全物理窗口','', f"用同一批日志另按完整 ELF 入口到全部真实指令结束重新核验，目前 {ph['windows']} 个物理窗口。实物理 END、实际官方活动核、输出体积及独立数值金标均须完整。该口径排除所有核的工具合成 END_LABEL，并作为单独对照。", '', '| 芯片 | 不同 shape | 候选 WAPE / 最大 | 已公开 WAPE / 最大 |', '|---|---:|---:|---:|']
        for chip,s in ph['summary'].items():md.append(f"| {chip} | {ph['shapes'][chip]} | {fmt(s['candidate_us'])} | {fmt(s['public_us'])} |")
        md+=['','[逐核窗口与失败原因](physical/) · [单一口径总表](physical-summary.json)','']
    if (OUT/'repeat-policy-freeze.json').exists():
        from development.validation.post25.repeat_policy_score import main as policy_score
        policy_score();md+=['[仅在精确重复时切换的预登记部署策略及单独五形留出](repeat-policy.md)','']
    if errors:md+=['## 无合格窗口的调用','']+[str(x) for x in errors]+['']
    (OUT/'report.md').write_text('\n'.join(md).rstrip()+'\n')
if __name__=='__main__':main()
