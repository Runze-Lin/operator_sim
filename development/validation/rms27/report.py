"""Compare prior protocols, immutable prospective point estimates and full-window CA identity."""
from collections import defaultdict
from pathlib import Path
from statistics import median
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.rms27.landmarks import OLD,OUT
from development.validation.rms27.model import fit,forecast
from development.validation.sku19.campaign import read,write,now
from operator_model import load_chip,estimate
PRICE=OUT/'prices.json'

def metrics(rs,key):
    if not rs:return None
    ys=[v['ca_us'] for v in rs]
    return dict(n=len(rs),wape=100*sum(abs(v[key]-y) for v,y in zip(rs,ys))/sum(ys),
                maximum=100*max(abs(v[key]/y-1) for v,y in zip(rs,ys)))


def summarize(rows):
    out={}
    for sku in read(OUT/'freeze.json')['measured']:
        out[sku]={}
        for role in ('entry','repeat','all'):
            gr=[v for v in rows if v['chip']==sku and (v['role']==role or role=='all')]
            out[sku][role]={k:metrics(gr,k) for k in ('us','old_us','cold_hot_us','known_us')}
    return out


def old():
    data=read(OLD/'independent.json')['rows'];land=read(OUT/'landmarks.json')['rows'];prices=read(PRICE);out=[]
    for r in data:
        # The shape is absent on both SKUs. Also exclude every launch in the assessed host process.
        p=fit([z for z in land if z['shape']!=r['shape'] and (z['process']!=r['process'] or z['chip']!=r['chip'])])
        q=forecast(r['shape'],load_chip(r['chip']),p);a=q['conditional']['first_us'];b=q['conditional']['fully_covered_us']
        out.append(dict(chip=r['chip'],shape=r['shape'],H=r['prior_aiv'],role='entry' if r['role']=='entry' else 'repeat',
                    ca_us=r['ca_us'],known_us=r['known_us'],cold_hot_us=2*a*b/(a+b),trace=r['trace'],**q))
    archived=[]
    for r in read(OLD/'resource-diagnostic.json')['rows']:
        c=load_chip(r['chip']);q=forecast(r['shape'],c,prices);a=q['conditional']['first_us'];b=q['conditional']['fully_covered_us'];h=r['prior_aiv']
        archived.append(dict(chip=r['chip'],shape=r['shape'],H=h,ca_us=r['ca_us'],role='repeat' if h>=r['active'] else 'entry',
                        original_role=r['group'],known_us=estimate('rmsnorm',r['shape'],{'resident_aiv':h},chip=c).latency_us,
                        cold_hot_us=2*a*b/(a+b),trace=r['trace'],source=r['source'],**q))
    write(OUT/'development.json',dict(scope='Old v0.23 pairs with same shape on both chips and all other rows of assessed host process purged before stage-price fitting; other 199 pre-existing windows are diagnostic archive replay. Only the new frozen cohort is prospective.',paired_rows=out,paired_summary=summarize(out),archive_rows=archived,archive_summary=summarize(archived)))


def fmt(x):return '—' if x is None else f"{x['wape']:.2f}% / {x['maximum']:.2f}%"

def render():
    dev=read(OUT/'development.json');reg=read(OUT/'freeze.json');rows=[];miss=[];cost=[];future={(r['group'],r['i']):r for r in reg['rows'] if r['chip']==reg['extrapolated'][0]}
    def adapt(r,obj,origin):
        s=r['stage'];z={k:r[k] for k in ('chip','group','shape','i','role','prior_aiv','active_aiv','known_us','covered_us','upper_us','us','old_us','cold_hot_us','runtime_ms','correctness','host_task')}
        return dict(**z,origin=origin,ca_us=s['ca_us'],kernel=s['kernel'],trace=s['trace'],trace_sha256=s['trace_sha256'],
                    first_tick=s['first_tick'],last_tick=s['last_tick'],elf_text_sha256=obj['elf_text_sha256'],extrapolated_970_us=future[r['group'],r['i']]['us'])
    for file in sorted((OUT/'receipts').glob('*.json')):
        obj=read(file);cost.append(obj['cost']);assert reg['utc']<obj['cost']['started_utc']
        for r in obj['rows']:
            if r['role']=='prefix':continue
            if not r['timing_qualified']:
                miss.append(dict(chip=r['chip'],shape=r['shape'],role=r['role'],reason=r.get('reason')));continue
            rows.append(adapt(r,obj,'original'))
    original=list(rows);recovery_costs=[]
    for file in sorted((OUT/'recovery').glob('*.json')):
        obj=read(file);recovery_costs.append(obj['cost'])
        assert obj['cost']['started_utc']>read(OUT/'recovery-freeze.json')['utc']
        for shape in obj['eligible_shapes']:
            key=(obj['chip'],obj['group'],shape)
            rs=[v for v in obj['rows'] if v['role']!='prefix' and v['shape']==shape]
            good=[v for v in rows if (v['chip'],v['group'],v['shape'])==key]
            assert len(good)<2
            if len(rs)==2 and all(x['timing_qualified'] for x in rs):
                # Use both targets from one replay, including the already valid control.
                rows=[v for v in rows if (v['chip'],v['group'],v['shape'])!=key]
                rows.extend(adapt(r,obj,'same-prefix paired recovery')for r in rs)
    # Both protocols from each SKU/shape must be present for paired metrics and their lower bounds.
    paired=[];bounds={}
    for sku in reg['measured']:
        entries={(tuple(r['shape']),r['group']):r for r in rows if r['role']=='entry' and r['chip']==sku};rs=[]
        for v in [v for v in rows if v['chip']==sku and v['role']=='repeat']:
            ref=entries.get((tuple(v['shape']),v['group']))
            if ref:
                assert v['us']==ref['us'] and ref['prior_aiv']<ref['active_aiv']<=v['prior_aiv']
                rs.extend([ref,v]);paired.extend([ref,v])
        e=rs[::2];h=rs[1::2];dif=[abs(a['ca_us']-b['ca_us']) for a,b in zip(e,h)];tot=[a['ca_us']+b['ca_us'] for a,b in zip(e,h)]
        bounds[sku]=dict(n=len(e),max_lower=100*max((d/t for d,t in zip(dif,tot)),default=0) if e else None,wape_lower=100*sum(dif)/sum(tot) if tot else None)
    summary=summarize(paired);qualified=all(bounds[s]['n']>=10 and all(summary[s][g]['us']['wape']<=25 and summary[s][g]['us']['maximum']<=50 for g in ('entry','repeat'))for s in reg['measured'])
    ratios=[]
    lookup={(r['chip'],r['group'],r['i']):r for r in paired}
    dt,pr=reg['measured']
    for r in paired:
        if r['chip']!=dt:continue
        oth=lookup.get((pr,r['group'],r['i']))
        if oth:
            ratios.append(dict(shape=r['shape'],role=r['role'],**{k:r[k]/oth[k] for k in ('ca_us','us','old_us','cold_hot_us','known_us')}))
    write(OUT/'score.json',dict(utc=now(),measured=rows,rejected=miss,matched_summary=summary,paired_bounds=bounds,promotion_qualified=qualified,host=cost,recovery_costs=recovery_costs,original_valid=original,dt_over_pr=ratios))
    md=['# RMSNorm 无前序点预测','', '| 芯片 | 两个前序都合格的独立 shape | 首次遇到该形，候选 / 旧默认 WAPE、最大 | 同形立即复跑，候选 / 旧默认 WAPE、最大 |','|---|---:|---|---|']
    for s in reg['measured']:
        k=summary[s];md.append(f"| {s} | {bounds[s]['n']}/10 | {fmt(k['entry']['us'])} · {fmt(k['entry']['old_us'])} | {fmt(k['repeat']['us'])} · {fmt(k['repeat']['old_us'])} |")
    md+=['',f"已返回 {len(cost)}/8 个宿主进程、{len(rows)}/40 个核验合格的目标窗。公开 shape-only 继续采用当前估计。候选接入所需的两端独立完整配对、两种前序各 WAPE≤25% 且最大≤50% 的判定为{'通过' if qualified else '尚未通过'}。970DT 均为外推。",'','## 同一形状的两个资源端点','',
     r'对完全覆盖的热态，令 $A$ 为本次活动 AIV 数、$K=\lceil D/128\rceil r$ 为本核处理 $r$ 行的向量组数。首条 gamma 搬运发起中位周期为 $G=g_0+g_A(A-1)$，各核发起的离散幅度按在同一 rstd 扇区的核数分配。x 描述符在 gamma 发起后 96 周期发出。等两条读都退休才能开始归约，数据就绪时刻还要满足本核读取总字节的端口下界。',
     '',r'令 $t_r$ 为归约后发出 rstd 写的时刻。归约到该时刻的成本按固定开销和每向量组的服务相加；从 $t_r$ 到 y 写发起另计规范化。对于同一个 128 B 扇区上的 FP32 rstd，依次按 $d_j=\max(t_{r,j}+L_{solo},d_{j-1}+I_{sector})$ 退休。最终取 y 与 rstd 的最晚完成，加退出周期，并满足 MTE、L2、HBM 下界。训练单元是某条 DMA 的发起、响应或 VF 到写发起的间隔。每次启动在端口/向量拟合中只贡献一次统计。没有使用 shape 的实测总时定价。超过一个 UB tile 的核使用既有多段排程。','',
     r'上端点为单独首次调用 $T(H=0)$ 和已用一个 AIV、开始启用第二分区 $T(H=1)$ 中较慢者。下端点为完全覆盖时的资源估计。两端假设时延为 $l,u$，相对误差平衡点 $2lu/(l+u)$ 满足 $p/l-1=1-p/u$。这只是两个模型场景之间的决策，真实 CA 可以落在两端之外。其他前序与同时工作的其他流也会改变终点。输入仍只有 shape、dtype、chip。各 AIV 的计算量按核数和 tile 上界展开，训练不要求未来芯片上提供结果。',
     '','## 先前配对中的偏差','',
     '旧的冷、热谐均参考在 950PR [4,608] 给 2.219 µs。H=1→4 会打开第二个派发分区；CA 4.578 µs，单独冷启动的预测只有 3.081 µs，显式给这个扩核前序则为 4.166 µs。末核 x 读约 3.37–3.88 µs；同形紧接重复则是约 1.03–1.17 µs，完整窗 1.762 µs。相反，两核 [2,2944] 的热态已知 H 预测 1.852 µs、实测 1.610 µs。其末核 y 写发起、退休约 1.49、1.60 µs，而旧单条热 y 描述符预算约 0.209 µs。分别估首发和热态的确切指令响应可以同时处理两个偏差。','',
     '| 数据角色 | SKU | 首次或扩核候选 | 完全覆盖候选 | 冷/热端点谐均，前 / 后 | 旧默认，前 / 后 |','|---|---|---:|---:|---|---|']
    for key,title in [('paired_summary','v0.23 同形双端与同进程剔除开发'),('archive_summary','早期 199 窗开发回放')]:
        for s,g in dev[key].items():
            a=g['entry'];h=g['repeat'];md.append(f"| {title} | {s} | {fmt(a['us'])} | {fmt(h['us'])} | {fmt(a['cold_hot_us'])} · {fmt(h['cold_hot_us'])} | {fmt(a['old_us'])} · {fmt(h['old_us'])} |")
    p=read(PRICE);md+=['','早期回放的“完全覆盖”包括已覆盖全部所需 AIV 的普通前序；新实验的第二窗为同形同一流的立即复跑。两种覆盖协议的缓存与指令响应可能有差异。单次真实芯片资源响应尚不能覆盖所有这些隐状态。','',
        '| 热态阶段 | 基础周期 | 增量周期 | 单位 |','|---|---:|---:|---|']
    for key,unit in [('gamma_issue','每多一块活动 AIV'),('issue_spread','每多一块活动 AIV'),('x_response','每 KiB'),('gamma_response','每 KiB'),('reduce','每 128 通道向量组'),('normalize','每 128 通道向量组'),('y_response','每 KiB')]:
        a,b=p['curves'][key];md.append(f'| {key} | {a:.2f} | {b:.3f} | {unit} |')
    md+=['',f"来源为 18 个旧的双端热态单 UB 启动。x 与 gamma 发起间隔 {p['x_lead']:.0f} 周期，rstd 最短独占响应的逐窗中位数 {p['rstd_solo']:.1f} 周期。同一扇区后继退休间隔沿用 v0.22 的 {p['shared_retire_ii']:.0f} 周期，退出 {p['drain']:.0f} 周期。响应的基础项衡量端口固定开销，增量项按目标端口字节价或每核每周期 FP32 吞吐缩放。冷/开分区取现有已校准分支。",'',
        '## 独立逐窗','', '| 芯片 | [T,D] | 真实前序 H | 时序 | 当前默认 / 冷热谐均 / 新候选 µs | 已知 H / CA µs | 新候选偏差 | 970DT 外推 µs |','|---|---|---:|---|---:|---:|---:|---:|']
    for r in rows:md.append(f"| {r['chip']} | {r['shape']} | {r['prior_aiv']} | {r['role']} | {r['old_us']:.3f} / {r['cold_hot_us']:.3f} / {r['us']:.3f} | {r['known_us']:.3f} / {r['ca_us']:.3f} | {100*(r['us']/r['ca_us']-1):+.2f}% | {r['extrapolated_970_us']:.3f} |")
    md+=['','## 当真实前序已知时','',
         '| SKU | 协议 | 仅按 shape 的新候选 | 按真实 H 的生产估计 | 冻结热端点（仅复跑） |','|---|---|---:|---:|---:|']
    for sku in reg['measured']:
        for role in ('entry','repeat'):
            matched=[x for x in paired if x['role']==role and x['chip']==sku]
            md.append(f"| {sku} | {role} | {fmt(metrics(matched,'us'))} | {fmt(metrics(matched,'known_us'))} | {fmt(metrics(matched,'covered_us')) if role=='repeat' else '—'} |")
    md+=['','下界假设在知道每个实测对以后，可任意选择一个同时服务这对的点。单对相对最坏下界为两端差的绝对值除以两端之和；两个协议不能区分时，两组 WAPE 较差者的下界为每对绝对差求和除以每对两端总和。该事后下界不参与模型定价。','',
         '| SKU | 完整配对 | 最大误差下界 | 两协议较差 WAPE 下界 |','|---|---:|---:|---:|']
    for sku,v in bounds.items():
        pair_vals=(f"{v['max_lower']:.2f}% | {v['wape_lower']:.2f}%" if v['n'] else '— | —')
        md.append(f"| {sku} | {v['n']} | {pair_vals} |")
    total=sum(c['seconds']for c in cost+recovery_costs)
    md+=['',f"冻结时三种芯片 {len(reg['rows'])} 次独立候选查询墙钟中位数 {median([x['runtime_ms'] for x in reg['rows']]):.3f} ms，最大 {max(x['runtime_ms'] for x in reg['rows']):.3f} ms。已返回进程的 CA 墙钟 {total:.1f} 秒。", '',
        '[冻结的三端预测、硬件与程序版本](freeze.json) · [分阶段参数](prices.json) · [开发逐窗回放](development.json) · [候选与 CA 核对](score.json) · [实际指令窗口](landmarks.json)']
    md+=['','## 同序号双端速度比','',
         '速度比取 DT 耗时除以 PR 耗时。两端均有完整同形前后双窗的形状才进入对照。','',
         '| 前序 | 双端对数 | 新无历史比值 WAPE / 最大 | 旧无历史比值 WAPE / 最大 | 给定真实 H 的比值 WAPE / 最大 |','|---|---:|---:|---:|---:|']
    for role in ('entry','repeat'):
        subset=[v for v in ratios if v['role']==role]
        md.append(f"| {role} | {len(subset)} | {fmt(metrics(subset,'us'))} | {fmt(metrics(subset,'old_us'))} | {fmt(metrics(subset,'known_us'))} |")
    if miss or recovery_costs:
        md+=['','初采的缺口与补测分开保存。补测只有在同一新进程原样重放输入前缀、相邻的两次目标都通过时才能计入整对。', '',
            '| 初采未合格目标 | 原因 | 当前可评分的替代对 |','|---|---|---|']
        for v in miss:
            used=[x for x in rows if x['shape']==v['shape'] and x['chip']==v['chip'] and x['origin']!='original']
            md.append(f"| {v['chip']} {v['shape']} {v['role']} | {v['reason']} | "+('两窗均合格' if len(used)==2 else '待合格')+' |')
        md+=['','[补测登记与单槽预算](recovery-freeze.json)']
    md+=['','## 更多轮次与完整 prefill','',
         '早期归档中的 41,943,040 个元素有两种形状：8192×5120 与 32768×1280。每个样例直接列该形的完整 CA 窗。当前新对照集按相同形状双前序采集，历史的大形只有各自原有协议。','',
         '| 历史档案 shape | 950DT 新 / CA µs | 950PR 新 / CA µs |','|---|---:|---:|']
    for shape in ([8192,5120],[32768,1280]):
        cells=[]
        for sk in reg['measured']:
            v=next((x for x in dev['archive_rows'] if x['chip']==sk and x['shape']==shape),None)
            cells.append(f"{v['us']:.3f} / {v['ca_us']:.3f}" if v else '—')
        md.append(f'| {shape} | '+' | '.join(cells)+' |')
    md+=['','以仓中 [DeepSeek Flash](../../../docs/workloads/deepseek-flash.md) 与 [DeepSeek Pro](../../../docs/workloads/deepseek-pro.md) 工作负载的独立 input_layernorm 为例，按整批一次派发：$T=8S$，$D$ 分别为 5120、7168。下表只给候选外推。单次独立 RMS 的这个大形没有该合同的完整 CA；实际分块 prefill 需按实际每次调度行数重新估计。','',
         '| 模型位置 | 每请求 S | 单卡 [T,D] | 950DT / PR / 970DT 外推 µs |','|---|---:|---|---:|']
    p=read(PRICE)
    for w,label in ((5120,'Flash'),(7168,'Pro')):
        for seq in (32768,131072,262144):
            shape=[8*seq,w];forecasts=[forecast(shape,load_chip(sk),p)['us'] for sk in [*reg['measured'],*reg['extrapolated']]]
            md.append(f"| {label} input_layernorm | {seq} | {shape} | "+' / '.join(f'{v:.2f}' for v in forecasts)+' |')
    md+=['','在这些多 UB 块情景，输入 x 和 y 读写量随 T 成正比，多轮重叠、最后一轮余量及缓存容量仍影响短区间的增量。Kimi 自定义的融合归一化须按其实际 kernel 合同单独验证。']
    from development.validation.rms27.limits import group
    limits={'criteria':{'protocol_wape_pct':25,'per_window_max_pct':50},
            'old_rms23':group(dev['paired_rows'],reg['measured']),'new_rms27':group(paired,reg['measured'])}
    write(OUT/'feasibility.json',limits)
    md+=['','## 两种误差约束能否同时做到','',
         r'即便知道所有实测端点，取同一时延同时服务两次调用时，每形 $p_i$ 的可行区间为 $[l_i,u_i]=[\max(\min(E_i,R_i),0.5\max(E_i,R_i)),\min(\max(E_i,R_i),1.5\min(E_i,R_i))]$。此区间同时保留最大误差 50% 和不受支配的点。若非空，该形首次绝对误差可在两个端点到 $E_i$ 的距离之间连续取值。两个协议的绝对误差之和固定为 $D=\sum |E_i-R_i|$。将所有首次误差的最小值、最大值分别求和，所得区间必须与 $[D-0.25\sum R_i,0.25\sum E_i]$ 有交集。这个检验给出同时满足 WAPE 和最大偏差的充要条件。小 footprint 的物理分组只覆盖两派发分区各自首个物理核，950 的核内有两个 AIV，即本次活动核 $A\le4$。','',
         '| 数据 | 芯片 | 全部配对 WAPE 下界（含 50% 限制） | 小 footprint 配对数 | 小 footprint 下界 | 小组两协议同时达标 |',
         '|---|---|---:|---:|---:|---|']
    for key,label in [('old_rms23','旧配对开发'),('new_rms27','冻结后独立')]:
        for sku,g in limits[key].items():
            allgroup=g['all'];small=g['first physical core per partition']
            def pct(x):
                return f"{x['constrained_wape_lower_pct']:.2f}%" if x.get('constrained_wape_lower_pct') is not None else '—'
            md.append(f"| {label} | {sku} | {pct(allgroup)} | {small['n']} | {pct(small)} | "+('可行' if small.get('simultaneous_feasible') else '不可行' if small['n'] else '待测')+' |')
    md+=['','小 footprint 的结论只适用于这些同输入但不同前序的完整配对。较大负载改变总体 WAPE 的权重，可以使总集跨过门槛。[逐组精确可行区间和窗口下界](feasibility.json)']
    from development.validation.rms27.physical import run as physical
    aux=physical()
    md+=['','## 原始全核物理终点复核','',
         '只用各原进程，从全片第一条指令到最后一条真实 ELF 指令完成为止。每个核都按同一规则去掉工具合成的 END_LABEL。次级口径不混入主验收的分母；各回执仍要求两个全输出正确、正式 ELF、所有核的入口/END、总写回字节及两份指令记录一致。','',
         '| SKU | 完整不同 shape | 首次或扩核 WAPE / 最大 | 同形复跑 WAPE / 最大 |','|---|---:|---:|---:|']
    for sku,g in aux['summary'].items():
        md.append(f"| {sku} | {aux['bounds'][sku]['all']['n']} | {fmt(g['entry']['us'])} | {fmt(g['repeat']['us'])} |")
    md+=['','[全部原始物理终点](physical.json)']
    pair=OUT/'receipts/partition-Ascend950PR_957b.json'
    if pair.exists() and not (OUT/'pr-second-partition.json').exists():
        selected=[x for x in read(pair)['rows'] if x['shape']==[4,688]]
        if len(selected)==2 and all(x['timing_qualified'] for x in selected):
            from development.validation.rms27.figure import build,draw
            evidence=build();draw(evidence);write(OUT/'pr-second-partition.json',evidence)
    if (OUT/'pr-second-partition.png').exists():md+=['','![同形、两种启动顺序，上模型下 CA](pr-second-partition.png)','','[每个阶段、时钟与实际 core](pr-second-partition.json)']
    (OUT/'report.md').write_text('\n'.join(md).rstrip()+'\n')

if __name__=='__main__':
    if 'develop' in sys.argv:old()
    render()
