"""Score prospectively fixed Post estimates; keep unresolved windows and the unknown-state protocol separate."""
from collections import Counter,defaultdict
from math import ceil
from pathlib import Path
import json,statistics,sys,time
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.post24.model import predict
from operator_model import load_chip
from development.validation.sku19.campaign import read,write,now
OUT=ROOT/'results/v0.24/post'
def stats(rows,key,truth='ca_us'):
    if not rows:return None
    err=[abs(r[key]-r[truth]) for r in rows]; return dict(n=len(rows),wape=100*sum(err)/sum(r[truth] for r in rows),maximum=100*max(d/r[truth] for d,r in zip(err,rows)))
def fmt(m):return f"{m['wape']:.2f}% / {m['maximum']:.2f}%" if m else 'N/A'
def classify(r):return 'immediate repeat' if r['role']=='repeat' else 'delayed revisit' if r['role']=='revisit' else r['state']
def main():
    src=read(OUT/'freeze.json');seqs=src['groups'];future={(x['group'],x['i']):x['us'] for x in src['rows'] if x['chip']=='Ascend970DT'}
    rows=[];costs=[];errors=[];receipts=sorted((OUT/'receipts').glob('*.json'));summary={}
    for f in receipts:
        obj=read(f); history=0
        costs.append(dict(chip=obj['chip'],group=obj['group'],qualified=obj['qualified'],**obj['cost']))
        if 'reason' in obj:errors.append(dict(receipt=f.name,reason=obj['reason']))
        for r in obj['rows']:
            if not r['timing_qualified']:
                errors.append(dict(receipt=f.name,i=r['i'],shape=r['shape'],reason=r.get('reason')));history=None;continue
            s=r['stage'];c=classify(r);obs=r.get('ca_milestones',{})
            row=dict(chip=r['chip'],group=r['group'],i=r['i'],shape=r['shape'],protocol=c,role=r['role'],mode=r['mode'],
                     prior_aiv=r['prior_aiv'],observed_prior_aiv=history,prior_known_matches=history==r['prior_aiv'], route=r['plan']['route'], fragment_width=r['plan']['fragment_width'], rows_per_core=r['plan']['rows_per_core'],
                     predicted_aiv=r['active'],measured_aiv=s['active_aiv'],ca_us=s['ca_us'],candidate_us=r['us'],
                     baseline_us=r['old_known_us'],unknown_us=r['candidate_unknown_us'],future_us=future[r['group'],r['i']],
                     kernel=s['kernel'],kernel_sha=s['binary_sha256'],trace_sha=s['trace_sha256'],trace=s['trace'],golden_items=r['all_output_elements'],observed_phase_us=obs.get('phase_us'),predicted_stages_us=r['stages'],timing_qualified=True)
            rows.append(row)
            if history is not None:history=max(history,s['active_aiv'])
    cover=[];speed=[];groups=defaultdict(list)
    for r in rows:groups[r['chip'],tuple(r['shape'])].append(r)
    for (sku,shape),jobs in groups.items():
        j={r['role']:r for r in jobs}
        if 'entry' not in j or 'repeat' not in j:continue
        a,b=sorted([j['entry']['ca_us'],j['repeat']['ca_us']])
        if j['entry']['unknown_us']!=j['repeat']['unknown_us']:raise ValueError('Shape-only saw history')
        cover.append(dict(chip=sku,shape=shape,min_us=a,max_us=b,unknown_minimax_relative_pct=100*(b-a)/(a+b),best_shape_only_us=2*a*b/(a+b)))
    by={(r['chip'],r['group'],r['i']):r for r in rows};dt,pr=src['measured']
    for _,g,i in [k for k in by if k[0]==dt]:
        if (pr,g,i) not in by:continue
        x,y=by[dt,g,i],by[pr,g,i]
        speed.append(dict(group=g,i=i,shape=x['shape'],protocol=x['protocol'],ca_us=x['ca_us']/y['ca_us'],candidate_us=x['candidate_us']/y['candidate_us'],baseline_us=x['baseline_us']/y['baseline_us'],unknown_us=x['unknown_us']/y['unknown_us']))
    for sku in src['measured']:
        subset=[r for r in rows if r['chip']==sku]
        summary[sku]={group:{k:stats(xs,k) for k in ('candidate_us','baseline_us','unknown_us')} for group,xs in [('all',subset)]+[(g,[r for r in subset if r['protocol']==g]) for g in ('first','expand','reuse','immediate repeat','delayed revisit')]}
    tests=[]
    for r in src['rows']:
        chip=load_chip(r['chip'])
        for _ in range(4):
            t0=time.perf_counter_ns();v=predict(r['shape'],chip,r['prior_aiv'],r['previous_shape'],src['parameters']);tests.append((time.perf_counter_ns()-t0)/1e6)
            assert abs(v['us']-r['us'])<1e-10
    tests.sort();bench=dict(n=len(tests),median_ms=statistics.median(tests),p95_ms=tests[ceil(.95*len(tests))-1],max_ms=max(tests),first_frozen_max_ms=max(r['predict_ms'] for r in src['rows']))
    speed_summary={k:stats(speed,k) for k in ('candidate_us','baseline_us','unknown_us')}
    write(OUT/'evaluation.json',dict(utc=now(),receipts=len(receipts),windows=len(rows),errors=errors,summary=summary,rows=rows,costs=costs,same_shape_lower_bounds=cover,dt_over_pr=speed,ratio_summary=speed_summary,online_benchmark=bench))
    md=['# 内置 mHC Post 的有序调用验证','', '| 芯片 / 本次协议 | 完整窗 | 候选 WAPE / 最大 | v0.23 WAPE / 最大 | 无前序 WAPE / 最大 |','|---|---:|---:|---:|---:|']
    for sku in src['measured']:
        for g,out in summary[sku].items():
            if out['candidate_us']:md.append(f"| {sku} / {g} | {out['candidate_us']['n']} | {fmt(out['candidate_us'])} | {fmt(out['baseline_us'])} | {fmt(out['unknown_us'])} |")
    md+=['',f"冻结了每颗芯片 10 个新 shape、4 个独立进程、22 次调用。目前有 {len(receipts)}/8 个进程回执，{len(rows)}/44 个完整设备窗。每个进程使用同一 aclrtStream，逐次计算单独的完整 BF16 输出金标。每端 10 组相邻重复；另有 2 次隔形回访。固定输入包含 T、四条残差流、宽度 D、dtype、芯片和可选前序。所有 970DT 数值均为外推。", '',
         '这里的候选是冻结后的影子预测，尚未接入公开后端。v0.23 列是系统实际可调用的同流预测，无前序列也是现有公开行为。旧档案只用于阶段校准与选择要测试的假设，误差表只计注册之后的数据。遍查 1,329 份历史 JSON 结果清单时，仅 [1,4,192] 的形状在旧 v0.8 整链备用清单出现，既无对应运行目录亦无已发布实测；本次是首次可核验的独立 Post 实测。[形状查重](novelty.json)', '',
         '新输入覆盖 1 至 257 个 token 和 192 至 7168 个通道。最大的 [257,4,832] 输出有 855,296 个 BF16 元素，按 4 条残差流的完整输入、系数和输出合计约 3.69 MiB；该点检验最长核多轮循环。5120 和 7168 来自 Flash、Pro 的真实隐藏宽度；实际 vLLM 路由可能选择另一套 HcPost。本批仅测库内置 Post，未取得整段 8×32k 至 8×256k 的 Post 全量窗口。', '',
         '## 从调用上下文得到阶段与价格','',
         '当前流首次调用 H=0。超出同流历史最大使用核数 H 的后继为扩核，其他为可复用；成功派发后 H 更新为 max(H,A)。若紧邻调用的完整逻辑形状相同，且为通道分核路线，候选使用重复时观测到的发起、响应和交接成本。宿主每次重新构建张量；相同数值及调度合同不能证明物理地址和缓存命中相同。参数校准分开保留 90 个旧设备窗以及两端各一个旧同形重复，后者只有一个校准 shape。', '',
         '实际 9.2.1 安装包调用 aclnnMhcPost，属于 ops_transformer/mhc_post_apt，BF16 结果为 FP32 累加后舍入。固定参考源码 9.1 的 arch35 AIV 分核与 UB 常驻机制用于解释数据复用；正式表以这次实际 ELF .text、内核名、核数和输出量为准。', '',
         r'每个末核拆成五段：全核首条至末核加入、末核首条至第一笔有效操作数 DMA 发起、第一笔发起至首条 SIMD 操作数 load、首条 SIMD load 至最后写描述符发起、最后写发起至末核退出。对于读窗口，$R=\max_j(I_j+(c_j+B_j k/1024)/F)+h/F$，$I_j$ 是从首笔开始累积的发起间隔（µs），$B_j$ 是端口描述符字节数，$k$ 是目标 MTE2 每 KiB 的服务周期，$c_j$ 是扣除端口服务后的固定响应周期，$h$ 是最后输入就绪到首个 SIMD 操作数的交接周期，$F$ 是目标频率（周期/µs）。通道路线先对齐 BF16 片宽再按一行/四行发起；token 路线实际使用四笔。', '',
         '其余里程碑按共同的频率、每核向量吞吐、tile 片宽、每核轮数和活动核数定价。首发 token 路线有 4 个档案窗口直接测量相应分段；通道重复路线的固定段按两个旧端点该段的观测周期标定，沿用旧档案的服务斜率。若是其他状态，候选复用公开 v0.23 的资源阶段。所有状态同时应用共享 L2/HBM 物理下界。未提供历史时保留原公开的相对误差最小最大中心；此路径每对的两次预测必须相同。', '',
         '[校准参数、单位对应项与冻结逐次预测](freeze.json) · [采集身份及预算](collection.json) · [对实际 v0.23.0 公共流的 132 次零差异复核](baseline-parity.json)', '',
         '## 每形状设备时间','', '| SKU | 协议 | [T,4,D] | 前序 H | 预测 / 实测 AIV | 候选 / CA µs | 差异 | v0.23 / 无历史 µs | 970DT 外推 µs | 内核入口 |','|---|---|---|---:|---:|---:|---:|---:|---:|---|']
    total_groups=[]
    for sku in src['measured']:
        items=[r for r in rows if r['chip']==sku]
        observed_modes={}
        for mode in ('repeat operand schedule','first operand schedule','public resource stages'):
            xx=[r for r in items if r['mode']==mode]
            if xx:
                observed_modes[mode]=dict(count=len(xx),unique_shapes=len({tuple(r['shape']) for r in xx}),candidate=stats(xx,'candidate_us'),baseline=stats(xx,'baseline_us'))
        total_groups.append((sku,observed_modes))
    decision=['### 当前候选的可推广边界', '', '| SKU / 事前注册路径 | 不同 shape / 窗口 | 影子 WAPE / 最大 | 公开 v0.23 WAPE / 最大 |','|---|---:|---:|---:|']
    for sku,by_mode in total_groups:
        for mode,v in by_mode.items():
            decision.append(f"| {sku} / {mode} | {v['unique_shapes']} / {v['count']} | {fmt(v['candidate'])} | {fmt(v['baseline'])} |")
    decision+=['', '通道路线的相同输入紧邻重复有专门的 DMA 证据；实际调度需要同时知道前一调用的实现、形状与成功派发状态。token 冷发在本轮每端仅一个新实测入口。扩核、隔形回访、token 多轮热态，以及用户仅给 shape 的调用均沿用公开预测。保留冻结候选作为独立对照；局部状态的收益不能推及其他状态或通用 shape-only 默认值。','']
    md[md.index('## 从调用上下文得到阶段与价格'):md.index('## 从调用上下文得到阶段与价格')]=decision
    insertion=md.index('## 每形状设备时间')
    explanations=['### 分段参数与单位', '', '| 路线 / 前序 | 首行读次数 | 相邻发起间隔周期 | 固定响应周期 | 交接周期 | 标定窗 |','|---|---:|---|---|---:|---:|']
    for route,states in src['parameters'].items():
        for state,v in states.items():
            port=v['port']
            series=lambda values:' / '.join(f'{a:.2f}' for a in values)
            explanations.append(f"| {route} / {state} | {len(port['response_fixed_cycles'])} | {series(port['gap_cycles'])} | {series(port['response_fixed_cycles'])} | {port['handoff_cycles']:.2f} | {v['calibration_windows']} |")
    explanations+=['', '用目标芯片的本地 MTE2 带宽换算每 KiB 的服务周期。缺少该配置时使用 950 旧独立端口价 18.396 周期/KiB。上表的发起间隔和输入就绪交接取对应归一化 trace 周期的中位数，固定响应先扣除实际描述符字节的服务周期。', '', r'令 $s_f=1.65/f$，$s_v=(27/64)/(P_{32}/C_v)$，其中 $f$ 是目标 GHz，$P_{32}$ 是整芯片 FP32 向量 TOP/s，$C_v$ 是物理 AIV 数。下表截距以参考芯片微秒计，乘 $s_f$。到达阶段另加斜率乘 $s_f\,\log_2(A)/6$；主体阶段两项斜率分别乘 $s_v u\lceil K/64\rceil/32$ 与 $s_f(u-1)$。$A$ 是当前实际拟派发核数，$K$ 是单片宽度，$u$ 是最长核的轮数。控制和退出两段只有固定项。普通阶段斜率源于旧训练窗对应里程碑的非负稳健回归。', '', '| 路线 / 前序 | 阶段 | 固定项 µs | 按列出的工作量单位计的斜率 µs |','|---|---|---:|---|']
    for route,states in src['parameters'].items():
        for state,v in states.items():
            for phase,t in v['phase'].items():
                explanations.append(f"| {route} / {state} | {phase} | {t['base_us']:.5f} | {' / '.join(f'{x:.5f}' for x in t['slopes']) or '—'} |")
    explanations+=['', '| 本轮配置 | AIV | GHz | 整芯片 FP32 Vector TOP/s | HBM GB/s |','|---|---:|---:|---:|---:|']
    for name in (*src['measured'],'Ascend970DT'):
        chip=load_chip(name);explanations.append(f"| {name} | {chip.vector_cores} | {chip.frequency_ghz:g} | {chip.vector_tops['32']:g} | {chip.hbm_bandwidth_gbs:g} |")
    explanations+=['', '输入配置的三种芯片均未指定 MTE2 实测速率，当前回退沿用上述独立服务价。970DT 使用 120 个 AIV 与 1.8 GHz、更换后的 FP32 Vector 峰值，UB 容量按参考配置继承；本轮没有其 CA 数据。[970DT 参数及来源](../../../configs/chips/Ascend970DT.json)', '','[宿主单流调用顺序](../../../smoke/vector_v8.cpp) · [完整输出金标](../../../smoke/extension_indexer_mhc.cpp)', '']
    md[insertion:insertion]=explanations
    for r in sorted(rows,key=lambda z:(z['chip'],seqs.index(z['group']),z['i'])):
        md.append(f"| {r['chip'][6:11]} | {r['protocol']} | {r['shape']} | {r['prior_aiv']} | {r['predicted_aiv']}/{r['measured_aiv']} | {r['candidate_us']:.3f} / {r['ca_us']:.3f} | {(r['candidate_us']/r['ca_us']-1)*100:+.1f}% | {r['baseline_us']:.3f} / {r['unknown_us']:.3f} | {r['future_us']:.3f} | MhcPost …_{r['kernel'].split('_')[-1]} |")
    md+=['','## 决定误差的阶段','']
    phase_review={}
    for sku in src['measured']:
        same=[r for r in rows if r['chip']==sku]
        detailed=[r for r in same if r['mode']!='public resource stages' and r['observed_phase_us']]
        if detailed:
            segments=('arrival','control','read','body','retire')
            signed={q:statistics.median(r['predicted_stages_us'][q]-r['observed_phase_us'][q] for r in detailed) for q in segments}
            absolute={q:statistics.median(abs(r['predicted_stages_us'][q]-r['observed_phase_us'][q]) for r in detailed) for q in segments}
            worst=max(detailed,key=lambda r:abs(r['candidate_us']/r['ca_us']-1))
            key=max(segments,key=lambda q:abs(worst['predicted_stages_us'][q]-worst['observed_phase_us'][q]))
            phase_review[sku]=dict(windows=len(detailed),median_signed_us=signed,median_absolute_us=absolute,largest_relative_row=dict(shape=worst['shape'],protocol=worst['protocol'],largest_stage=key,predicted_us=worst['predicted_stages_us'][key],observed_us=worst['observed_phase_us'][key]))
            md.append(f"{sku} 的 {len(detailed)} 个详细阶段窗中，首读发起至首条 SIMD load 的有符号中位差为 {signed['read']:+.3f} µs，SIMD load 至最后写发起则为 {signed['body']:+.3f} µs。此组相对误差最大的 {worst['shape']}（{worst['protocol']}）中，绝对偏差最大的阶段是 {key}，模型 {worst['predicted_stages_us'][key]:.3f} µs，末核实测 {worst['observed_phase_us'][key]:.3f} µs。")
        if same:
            wrong=[r for r in same if r['predicted_aiv']!=r['measured_aiv']]
            md.append(f"{sku} 在 {len(same)-len(wrong)}/{len(same)} 个完整窗预测正确分核。")
    border=[r for r in rows if r['shape']==[29,4,320] and r['role']=='repeat']
    if len(border)==2:
        md+=['','同一个 [29,4,320] 重复调用可检验核数改变引发的路线差异。', '', '| SKU | 预测路线 | 单核片宽 | 预测 / 实际 AIV | 安装包入口 | 重复预测 / CA µs |','|---|---|---:|---:|---|---:|']
        for r in border:
            md.append(f"| {r['chip']} | {r['route']} | {r['fragment_width']} | {r['predicted_aiv']} / {r['measured_aiv']} | …_{r['kernel'].split('_')[-1]} | {r['candidate_us']:.3f} / {r['ca_us']:.3f} |")
    write(OUT/'stage-errors.json',phase_review)
    md+=['', '[逐里程碑误差](stage-errors.json)', '', '## 状态可辨识性与成本','']
    for sku in src['measured']:
        v=[x for x in cover if x['chip']==sku]
        if v:
            worst=max(v,key=lambda z:z['unknown_minimax_relative_pct'])
            md.append(f"{sku} 的 {len(v)} 对相同输入中，任意无前序单值最坏误差下界最高为 {worst['unknown_minimax_relative_pct']:.2f}%（{worst['shape']}，两次实测 {worst['min_us']:.3f}/{worst['max_us']:.3f} µs）。给同一长期分布而无顺序仍会丢失这一信息。")
    md+=['', '这一压力组只包含同族 Post 的连续、扩核和隔形调用。旧生产链 Pre→Sinkhorn→Post 的 950PR 组在 10 个不同 shape 上无前序 WAPE 为 6.93%，其终点的输入来自其他 kernel。该历史结果只描述生产者前序分布，与本轮首发/同族热态两端的误差分别使用自己的分母。[历史生产链回放](../../v0.22/post-stream/report.md)']
    if speed:
        worst=max(speed,key=lambda r:abs(r['candidate_us']/r['ca_us']-1))
        md+=['',f"按相同调用的 DT 时间 / PR 时间计算 {len(speed)} 个加速比，影子模型的比值 WAPE / 最大为 {fmt(speed_summary['candidate_us'])}，公开 v0.23 为 {fmt(speed_summary['baseline_us'])}。比值误差最大的 {worst['shape']}（{worst['protocol']}），实测 DT/PR = {worst['ca_us']:.3f}，候选 {worst['candidate_us']:.3f}，旧版 {worst['baseline_us']:.3f}。两端分核与具体函数不同，当前重复通道修正只作用在一端，跨芯片相对优势出现反向。两端总 WAPE 的改善与跨芯片相对误差的退化应同时保留。"]

    md+=['',f"预载配置/参数的单次候选查询 {bench['n']} 次，墙钟中位 {bench['median_ms']:.4f} ms、P95 {bench['p95_ms']:.4f} ms、最大 {bench['max_ms']:.4f} ms。冻结时首次候选查询最大 {bench['first_frozen_max_ms']:.4f} ms。上述 {len(costs)} 个采集进程累计 {sum(v['seconds'] for v in costs)/60:.1f} 分钟宿主墙钟，原始总量 {sum(v['raw_bytes'] for v in costs)/2**30:.2f} GiB。", '',
         '[完整冻结同集评分与逐核原始证据入口](evaluation.json) · [实时状态](status.json)']
    if (OUT/'multirow.json').exists():
        w=next(r for r in read(OUT/'multirow.json')['data'] if r['role']=='repeat')
        gap=w['errors_us']
        md+=['','### 五轮 token 热态的负例','',f"950DT [257,4,832] 的末核 {w['critical']} 完整处理 token {w['items'][0]['token']}–{w['items'][-1]['token']}。分核和轮次均正确，公开预测与影子均为 {w['model_us']:.3f} µs，CA 为 {w['ca_us']:.3f} µs。按同一组互斥队列标签记账，MTE2 高估 {gap['MTE2']:.3f} µs、MTE3 高估 {gap['MTE3']:.3f} µs；二者解释总差 {w['model_us']-w['ca_us']:.3f} µs 的主要部分。五轮确实传输了全部输入和输出，跨轮重叠使实际暴露在关键路径上的搬运时间减少。", '',
              '![上方冻结分账与同口径 CA，下方末核五个真实 token 的读写](post-multirow-repeat.png)', '', '[按真实 GM 地址与指令排序核验的五轮里程碑和分账](multirow.md)']
    if (OUT/'topology.md').exists():
        md+=['', '从 1 扩到 4 个 AIV 的 [2,4,448] 在两端实际最晚完成的都是新启用的 core16.veccore1。旧档案也观察到只有一个前序 AIV 时，末核加入较其他扩核慢；新区分只是单次观测到的调度轨迹，物理域被激活的时间与芯片的缓存命中率不能据此等同。隔形回访保留旧策略，与相邻输入重复单独比较。[逐次活动核、兄弟核、最后完成的核](topology.md)']
    if (OUT/'physical-summary.json').exists():
        phys=read(OUT/'physical-summary.json')
        if phys['rows']:
            bysku=phys['summary']
            companions={(r['chip'],r['group'],r['i']):r for r in rows}
            deltas=[abs(companions[(r['chip'],r['group'],r['ordinal'])]['ca_us']-r['ca_us']) for r in phys['rows'] if (r['chip'],r['group'],r['ordinal']) in companions]
            missing=sum(bool(r['missing_synthetic_label']) for r in phys['rows'])
            md+=['','### 独立核查每核的物理指令窗口','',f"用同一口径重读 {phys['windows']} 次调用。从所有活动 AIV 最早 ELF 入口指令到最后真实指令完成，对每个核核对已安装代码中的 END、输出写字节和官方表。{missing} 次缺少至少一个软件生成的 END_LABEL。这个补充时钟统一排除人为标签，并单独评分，与预注册的含标签窗口有 {len(deltas)} 次重叠，两种计时的最大差为 {max(deltas):.5f} µs。" if deltas else f"已完成 {phys['windows']} 次物理窗口复核。"]
            for sku,metrics in bysku.items():
                if metrics['candidate_us']:md.append(f"{sku} 物理窗 {metrics['candidate_us']['n']} 个，候选 WAPE / 最大 {fmt(metrics['candidate_us'])}，v0.23 {fmt(metrics['baseline_us'])}，无前序 {fmt(metrics['unknown_us'])}。")
            md+=['','[独立物理 END 收据索引](physical-summary.json)']
    if (OUT/'post-small-pipeline.png').exists():md+=['','![首发与热态的相同四行写回](post-small-pipeline.png)','', '[384 B 分片与逐行地址证据](small-figure-data.json)']
    if (OUT/'post-ordered-pipeline.png').exists():md+=['','![同一逻辑 tile 的关键路径预测及两次实测](post-ordered-pipeline.png)','', '[行及通道地址映射和图上的数据](figure-data.json)']
    if len(receipts)==8:
        md+=['','## 结论','', '两端均达到 10 个不同新 shape，冷发、核扩展、紧邻热态和多轮执行已分别实测。有前序时，通道路线的立即重复得到改善；大 token 热态的最大误差和跨芯片分块边界仍限制推广。公开 v0.23 同流预测继续作为系统入口，候选仅作为影子验证。44 个完整物理窗已支撑不同状态的比较，使用过的留出集在后续方法调整中应改列为开发证据。']
    (OUT/'report.md').write_text('\n'.join(md)+'\n')
if __name__=='__main__':main()
