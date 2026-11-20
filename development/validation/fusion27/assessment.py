"""Score only registered golden/whole-device windows; retain unresolved cases and data roles."""
import argparse
import hashlib
from pathlib import Path
import time
from observe import ROOT, OUT, ORIG, read, save, utc

STANDALONE = ROOT / 'results/v0.26/sink'
CHIPS = ('Ascend950DT_9582', 'Ascend950PR_957b')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def metrics(rows, key, gold='ca_us'):
    usable = [r for r in rows if r.get(key) is not None]
    if not usable:
        return None
    return dict(n=len(usable), wape_pct=100*sum(abs(r[key]-r[gold]) for r in usable)/sum(r[gold] for r in usable),
                max_pct=100*max(abs(r[key]/r[gold]-1) for r in usable))

def summary(rows, keys):
    return {'distinct_shapes':len({tuple(r['shape']) for r in rows}),'windows':len(rows),
            **{k:metrics(rows,k) for k in keys}}

def audit():
    sf=STANDALONE/'register.json'; sr=STANDALONE/'independent.json'; fr=OUT/'register.json'
    fd=read(fr);sd=read(sr)
    if sd['registration_sha256']!=sha(sf) or fd['original_sha256']!=sha(ORIG/'freeze.json'):
        raise ValueError('registration identity')
    sink=[];fused=[];statuses=[];seen=set()
    sfjobs={(r['chip'],r['group'],r['index']):r for r in read(sf)['jobs']}
    for r in sd['rows']:
        j=sfjobs[r['chip'],r['group'],r['index']]
        src=ROOT/r['receipt'];d=read(src);x=d['rows'][r['index']]
        if not(x['qualified'] and d['register_sha256']==sha(sf) and x['stage']['trace_sha256']==r['raw_sha256']
               and r['shape']==x['shape']==j['shape'] and r['h']==x['h']==j['prior_h']
               and r['known_us']==j['known']['latency_us'] and r['unknown_us']==j['unknown_us'] and r['v025_us']==j['v025_us']):
            raise ValueError(r['receipt'])
        sink.append({**r,'source_sha256':sha(src)})
    for j in fd['jobs']:
        old=ORIG/(j['id']+'.json'); new=OUT/'receipts'/(j['id']+'.json')
        if old.exists() and new.exists():
            raise ValueError('duplicate registered receipt '+j['id'])
        path=old if old.exists() else new
        result=dict(id=j['id'],sku=j['sku'],shape=j['shape'],cohort=j['cohort'],status='pending')
        if path.exists():
            r=read(path); expect=fd['original_sha256'] if path==old else sha(fr)
            if (r.get('freeze_sha256',r.get('registration_sha256'))!=expect
                    or (r['shape'],r['chip'])!=(j['shape'],j['sku'])):
                raise ValueError('wrong contract '+str(path))
            result.update(receipt=str(path.relative_to(ROOT)),source_sha256=sha(path),
                          status='qualified' if r['qualified'] else 'failed',failure=r.get('reason'),wall_s=r['cost']['seconds'])
            if r['qualified']:
                s=r['stage'];p=j['predictions'][j['sku']];future=next(k for k in j['predictions'] if '970' in k)
                if s['kernel']!=j['kernel'] or s['binary_sha256']!=j['elf'] or len(r['correctness'])!=3 or p!=r['prediction']:
                    raise ValueError('wrong ELF, goldens or forecasts')
                fused.append({**result,**{k:p[k] for k in ('candidate_us','v025_us')},'ca_us':r['ca_us'],
                              'candidate_970_us':j['predictions'][future]['candidate_us'],'raw_sha256':s['trace_sha256'],
                              'active_aic':s['active_aic'],'active_aiv':s['active_aiv'],
                              'core_retirement_spread_cycles':max(r['measured_retire_offsets_cycles'].values())-min(r['measured_retire_offsets_cycles'].values())})
        statuses.append(result)
    for r in (*sink,*fused):
        if r['raw_sha256'] in seen:raise ValueError('duplicate qualified raw trace '+r['raw_sha256'])
        seen.add(r['raw_sha256'])
    groups={};fkeys=('candidate_us','v025_us');skeys=('known_us','unknown_us','v025_us')
    for chip in CHIPS:
        a=[r for r in sink if r['chip']==chip];b=[r for r in fused if r['sku']==chip]
        groups[chip]={'standalone_common':summary([r for r in a if r['v025_us'] is not None],skeys),
                      'standalone_above_v025_domain':summary([r for r in a if r['v025_us'] is None],skeys[:2]),
                      'fused':summary(b,fkeys),'fused_T_at_most_32':summary([r for r in b if r['shape'][0]<=32],fkeys),
                      'fused_T_above_32':summary([r for r in b if r['shape'][0]>32],fkeys)}
        for state in ('first','expand','resident'):
            groups[chip]['standalone_'+state]=summary([r for r in a if r['state']==state],skeys)
    pairs={}
    for name,rows,keys,sk in [('standalone',sink,skeys,'chip'),('fused',fused,fkeys,'sku')]:
        ix={(tuple(r['shape']),r.get('group'),r.get('index'),r[sk]):r for r in rows};paired=[]
        for (shape,g,i,chip),d in ix.items():
            p=ix.get((shape,g,i,CHIPS[1]))
            if chip!=CHIPS[0] or not p:continue
            paired.append(dict(shape=shape,group=g,index=i,**{k:(d[k]/p[k] if None not in (d[k],p[k]) else None) for k in ('ca_us',*keys)}))
        pairs[name]={'quantities':'DT/PR window ratio; >1 means PR faster', 'ratios':paired,
                     'metrics':{k:metrics(paired,k) for k in keys},
                     'same_direction':{k:sum((r[k]>1)==(r['ca_us']>1) and (r[k]<1)==(r['ca_us']<1) for r in paired if r[k] is not None) for k in keys}}
    paired_shapes={tuple(r['shape']) for r in pairs['fused']['ratios']}
    for chip in CHIPS:
        both=[r for r in fused if r['sku']==chip and tuple(r['shape']) in paired_shapes]
        groups[chip]['fused_paired_shapes']=summary(both,fkeys)
    return dict(utc=utc(),role='Postfreeze independent, shape and prior-family stream history kept separate. 970DT extrapolated.',
                registrations={'standalone':sha(sf),'original_fused':fd['original_sha256'],'continued_fused':sha(fr)},
                standalone=sink,standalone_pending=sd['pending_or_failed'],fused=fused,fused_status=statuses,groups=groups,paired=pairs)

def fmt(record):
    return f"{record['wape_pct']:.2f}% / {record['max_pct']:.2f}%" if record else 'N/A'

def render(d):
    lines=['# Sinkhorn 冻结模型的独立检查','',f"计分截止于 {d['utc']}。表内为绝对加权误差 WAPE / 最大单窗绝对相对误差。各端单独按 shape 去重计数，换前序后的重复启动仅增加窗数。",'',
           '| 算子与输入合同 | SKU | 不同 shape / 完整窗 | 冻结候选 | 相同输入的 v0.25 |','|---|---|---:|---:|---:|']
    for sku in CHIPS:
        g=d['groups'][sku]
        for label,key,c in [('独立 Sinkhorn，调用方给同家族 H','standalone_common','known_us'),('独立 Sinkhorn，省略 H 的候选','standalone_common','unknown_us'),('独立 Sinkhorn，超出 v0.25 原域','standalone_above_v025_domain','known_us'),('融合 PreSinkhorn','fused','candidate_us')]:
            x=g[key];lines.append(f"| {label} | {sku[6:11]} | {x['distinct_shapes']} / {x['windows']} | {fmt(x[c])} | {fmt(x.get('v025_us'))} |")
    lines+=['','## 模型与实测的对应关系','','已接入的显式 H 入口及 T>256 的四形外推资格见[独立 Sinkhorn 的公开接口、公式和算例](public-sink.md)。融合分支的历史分组、匹配独立误差、跨芯片速度比及默认接口见[融合 PreSinkhorn 的公开算例](public-fused.md)。','',
        '独立库核 aclnnMhcSinkhorn 在每核分配 R=⌈T/AIV⌉ 个 4×4 FP32 矩阵，单 UB 块内按 G=⌈R/8⌉ 个寄存器组处理。I 是行列归一化的迭代数。实际每个工作核发起 2I+2 个 VF，包括初始化、softmax、交替行列归一化和输出重排。冻结价格把首对和稳态相邻两轮单列为 248G、254G 个参考 Vector 周期，再用每核每周期 FP32 吞吐率和芯片频率换算。输入/输出各搬 64T 字节，阶段用本地 DMA 与片外带宽的较大下界。H 是调用方在同一调用流中显式给出的此前同家族驻留 AIV 高水位。H=0 取首次启动，低于当前工作核数取扩核，已覆盖当前工作核取热态。H 未知时的三状态均值单独计分。多 UB 块没有校准。','',
        '图的横轴在每次启动第一条设备指令处归零。上窗列出关键资源工作的发起点，下窗读取最后退休核的真实描述符，虚线为所有核的整窗结束。双端同 shape 同 I 的两种前序保留了原始核和聚合 trace 散列。','',
        '[950DT 对照与数值](950dt-same-input.json) · [950DT 图](950dt-same-input.png) · [950PR 对照与数值](950pr-same-input.json) · [950PR 图](950pr-same-input.png)','',
        '融合库核 aclnnMhcPreSinkhorn 使用 BF16 [T,4,4096]、20 次迭代，三个返回张量逐元素检查。候选在 T<128、AIV:AIC=2 的分支上单列 AIV cast、分块 K 投影的 AIC MMAD 和第二阶段的每核归一化行数。其它配置沿用原估计；970DT 的 1:1 比例只作外推。新增冻结覆盖共八个 T，配合原有 T72、T113 达到十种，T91 是事前登记的后备，主组 T127 DT 物理退休检查失败后，在其采集开始前用补充登记启用双端。本轮最初冻结的选择线为每端十种、WAPE≤20%、最大≤40% 且优于原预测，用户的最新目标为双端 WAPE≤25%、最大≤50%。','',
        '## 分组与跨芯片方向','',
        '| 芯片 | 首次 H=0 | 扩核 | 热态 | 融合 T≤32 | 融合 T>32 |','|---|---:|---:|---:|---:|---:|']
    for sku in CHIPS:
        g=d['groups'][sku]; cells=[fmt(g['standalone_'+k]['known_us']) for k in ('first','expand','resident')]+[fmt(g['fused_'+k]['candidate_us']) for k in ('T_at_most_32','T_above_32')]
        lines.append('| '+' | '.join([sku[6:11],*cells])+' |')
    for name,key in [('standalone','known_us'),('fused','candidate_us')]:
        p=d['paired'][name];n=len(p['ratios']);lines+=['',f"{name} 的 DT/PR 耗时比有 {n} 个双端匹配启动，候选的比值 WAPE / 最大为 {fmt(p['metrics'][key])}，数值快慢方向相符 {p['same_direction'][key]}/{n}。"]
    lines+=['','两个 SKU 都取得成功回执的融合核同形集（所有正式合格但尚无双侧匹配的行仍保留在上表）。','',
            '| 芯片 | 成对不同 shape | 冻结候选 WAPE / 最大 | 同集旧版 WAPE / 最大 |','|---|---:|---:|---:|']
    for chip in CHIPS:
        g=d['groups'][chip]['fused_paired_shapes']
        lines.append(f"| {chip[6:11]} | {g['distinct_shapes']} | {fmt(g['candidate_us'])} | {fmt(g['v025_us'])} |")
    failure=OUT/'fused-t127-dt-physical.json'
    if failure.exists():
        detail=read(failure);late=detail['descriptors_beyond_original_epilogue']['core15.veccore1']['latest'][0]
        lines+=['',f"T127 DT 的冻结退休门限是 3 周期，实际比较得到 {detail['offset_spread_frozen_cycles']:.6f} 周期。core15.veccore1 的 {late['name']}（END PC−8）比 END 晚 3 周期。逐一对齐全部实际描述符的残余差为 {detail['offset_spread_all_raw_cycles']:.6f} 周期。保留原独立失败；T91 使用事前预测、原门限和原预算，双端金标及全窗合格后计入匹配集。[T127 原始指令](fused-t127-dt-physical.json) · [T91 前向启用登记](reserve-activation.json)"]
    lines+=['','## 融合核的独立算例','','950PR [72,4,4096] 的分块预测为 22 个 AIC、44 个 AIV，最大 K 分块为 768，最慢 AIC 发起 8 次 MMAD，与逐核观测一致。模型预计投影阶段 5.046–5.995 µs，CA 的投影描述符为 5.860–7.170 µs，最后 AIV 的第二阶段从 9.748 µs 开始，比模型的 10.316 µs 更早。整窗 14.423 / 13.649 µs 有两段偏差部分抵消。单核 FIXP 描述符包含等依赖的时间，其全窗与模型预算在图中分别标记。','',
        '[融合核模型与实测同轴图](fused-t72-pr-timeline.png) · [每个 VF、MMAD 和 FIXP 的时刻及原始散列](fused-t72-pr-timeline.json)','',
        '## 离线采集与主机资源','',
        'T72 PR 在 Linux 原亲和性 16 CPU 下启动 71 个原生线程，12 秒快照中平均同时获 CPU 15.94 个，同时等待可用 CPU 19.02 个。确认原进程、原二进制和工作目录并等待空载后，将该进程的工作线程改为 96 CPU 的 SCHED_IDLE；原 PID、45 分钟上限和单槽不变。随后 15 秒快照测得平均同时获 CPU 29.44 个，等待下降到 6.87 个。线程等待计数证实主机排队，不能据此归因为某个具体软件 barrier。T72 PR 退出码 0，墙钟 1937.38 秒、CPU 32964.94 秒。66 个核的退休条件和三个输出的全元素检查均通过。','',
        '原评分进程在 CA 正常结束后载入同名 campaign 模块，使流式评分器缺少 read、PR、sha、math，未生成回执。已使用原聚合 trace、原终止 PC 条件、原价格与原单槽锁恢复；原捕获状态和原文件保留，后续每个 shape 的成本、错误及回执分开。不重复启动已存在 raw 的 shape。新任务在 Pro、FA 和 Flash 允许、系统连续 120 秒满足余量后依次运行。','',
        '[恢复依据与冻结散列](scoring-resume.json) · [后台队列状态](state.json) · [滚动结果](assessment-live.json) · [运行和复核方式](../../../development/validation/fusion27/README.md)','',
        '[资源调整前](native-before.json) · [资源调整后](native-after.json) · [逐线程亲和性](fused-t72-pr-widened.json) · [独立冻结、预测、预算](register.json) · [逐窗状态与耗时](assessment.json)','']
    return '\n'.join(lines)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    prefix='assessment-live' if args.watch else 'assessment';previous=None
    while True:
        d=audit();key=[(r['raw_sha256'],r.get('source_sha256')) for r in d['standalone']+d['fused']]+[(s['id'],s['status']) for s in d['fused_status']]
        if key!=previous:
            save(OUT/(prefix+'.json'),d);(OUT/(('report-live' if args.watch else 'report')+'.md')).write_text(render(d));previous=key
        if not args.watch or (read(OUT/'state.json')['state']=='finished' and not d['standalone_pending']):return
        time.sleep(60)

if __name__=='__main__':main()
