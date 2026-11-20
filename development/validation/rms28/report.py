"""Rebuild the RMS operator report from frozen prices, verified CA receipts and role splits."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.rms28.campaign import OUT,REG,prior,read
from development.validation.rms28.evaluate import evaluate

def fmt(v):
    return '待测' if v is None else f"{v['wape']:.2f} / {v['max']:.2f}"
def report():
    evaluate()
    frozen=read(REG);ev=read(OUT/'evaluation.json');hist=read(OUT/'development-replay.json');cost=read(OUT/'query-cost.json')
    dt,pr,future=prior.CHIPS
    text=['# RMSNorm 校准与误差','',
        '|芯片｜状态|配对形数|旧公共无 H WAPE / 最大 %|v0.27 候选无 H|v0.28 公共无 H|公共已知 H|专项已知 H、立即复跑可见|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for sku in (dt,pr):
        for state in ('entry','repeat','all'):
            m=ev['summary'][sku][state]
            text.append('|'+('950DT' if sku==dt else '950PR')+' · '+{'entry':'首遇/扩核','repeat':'立即复跑','all':'合并'}[state]+'|'
                +str((m['us'] or {}).get('shapes',0))+'|'+'|'.join(fmt(m[k]) for k in ('old_us','prior27_us','us','known_us','specific_context_us'))+'|')
    pending=len(ev['pending']);failed=len(ev['failed'])
    gate=('两端首遇和复跑的预登记总体门槛均已通过。' if ev['passes'] else
        '未满足两端首遇和复跑同时 WAPE ≤25%、最大 ≤50% 的原预登记门槛。' if ev['terminal'] else
        f'单槽采集尚余 {pending} 个进程组。当前统计只含已通过的同进程完整双次窗口。')
    text+=['',gate+f'共计 {ev["paired"]} 个可配对窗口，出现 {failed} 条失败记录。A≤4 的稀疏组单独列出；合并数据仅用于描述这里的等频工作负载。实际流里的首遇与复跑比例可以完全不同。','',
        '“首遇”是这个形状在该进程第一次出现；它可能发生在进程首发，也可能是 H>0 的扩核。每个实验形状依次立即调用两次。上表前 3 列预测只看 shape 和芯片，两次返回相同值。后 2 列传入前序，专项还知道是否紧接同合同成功调用。','',
        '## 每个 shape 的预测、完整 CA 与跨端误差','',
        '每格依次为 **旧公共无 H / v0.27 候选无 H / v0.28 公共无 H / v0.28 条件 / CA**，单位 µs。括号是 v0.28 无 H 相对误差，正号表示高估。970DT 仅列 v0.28 无 H 与条件外推。最后一列是 DT/PR 同态耗时比，依次为无 H 预测、条件预测、CA。','',
        '|shape|状态|950DT|950PR|970DT 外推|DT/PR|','|---|---|---|---|---|---|']
    observations={(r['chip'],tuple(r['shape']),r['role']):r for r in ev['measured']}
    defs={(r['chip'],tuple(r['shape']),r['role']):r for r in frozen['rows'] if r['role']!='prefix'}
    for row in frozen['rows']:
        if row['chip']!=dt or row['role']=='prefix':continue
        shape=tuple(row['shape']);role=row['role'];cells=[]
        for sku in (dt,pr):
            pred=defs[sku,shape,role]
            ca=observations.get((sku,shape,role))
            cell=' / '.join(f'{pred[k]:.3f}' for k in ('old_us','prior27_us','us','specific_context_us'))+' / '
            cell+=f"{ca['ca_us']:.3f} ({100*(pred['us']/ca['ca_us']-1):+.1f}%)" if ca else '待测'
            cells.append(cell)
        pred=defs[future,shape,role];cells.append(f"{pred['us']:.3f} / {pred['specific_context_us']:.3f}")
        a,b=(observations.get((sku,shape,role)) for sku in (dt,pr))
        cells.append(' / '.join(f'{a[k]/b[k]:.3f}' for k in ('us','specific_context_us','ca_us')) if a and b else '待测')
        text.append(f"|{shape[0]} × {shape[1]}|{'首遇' if role=='entry' else '复跑'}|"+'|'.join(cells)+'|')
    text+=['','## 稀疏与多轮分开看','',
        '稀疏为 A≤4（每个调度分区至多首个物理核及它的 sibling），多轮为至少两个 UB 块。每格依次为独立形状数、无 H WAPE/最大、条件 WAPE/最大。','',
        '|分组|950DT|950PR|','|---|---|---|']
    for role in ('entry','repeat'):
        for group,label in [('sparse','稀疏'),('one UB; A>4','普通单块'),('multi UB','多块')]:
            cells=[]
            for sku in (dt,pr):
                v=ev['summary'][sku][role+' / '+group];cells.append(f"{(v['us'] or {}).get('shapes',0)} · {fmt(v['us'])} · {fmt(v['specific_context_us'])}")
            text.append('|'+('首遇' if role=='entry' else '复跑')+' / '+label+'|'+'|'.join(cells)+'|')
    text+=['','对于相同可见输入 x，在首遇与复跑各有完整窗 c 和 h，任何单一预测点的最大相对误差至少为 |c−h|/(c+h)。若还要求每一组 WAPE≤25% 且逐形最大≤50%，可在每个形状上取两个 ±50% 区间的交集，再在线段上精确分配总绝对误差。结果是允许事后任选每个 shape 的最佳值时仍须承受的下界。','',
        '|SKU|集合|形数|两状态 WAPE 最优下界|单形最大误差下界|两个 WAPE 与最大误差条件在数学上可同时达到|','|---|---|---:|---:|---:|---|']
    for sku in (dt,pr):
        for k,v in ev['bounds'][sku].items():
            if not v['n']:continue
            lower=v['constrained_wape_lower_pct']
            text.append(f"|{sku[6:11]}|{k}|{v['n']}|"+(f'{lower:.2f}%' if lower is not None else '最大误差条件已不相容')+
                        f"|{v['max_error_lower_pct']:.2f}%|"+('可' if v['simultaneous_feasible'] else '不可')+'|')
    text+=['','## 哪些阶段真正影响误差','',
        '下表重算的是整套旧批次，全部用于开发。表里首次/复跑前两行以 v0.27 旧独立十形为基线；它们在 v0.28 开发中已可见。整批交叉重放在另一批次学习阶段参数后评估本批；同一系列软件/任务设计对结果仍有影响。','',
        '|开发目标集|SKU / 协议|旧公共|v0.27 候选|专项|只用另一批定价|去掉扇区仲裁|令 VF 串行退休|去掉分区启动端点|换回旧热态|去掉本轮共享首读项|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for batch,data in reversed(list(hist.items())):
        for sku in (dt,pr):
            for role in ('entry','repeat'):
                s=data['summaries'][sku][role]
                cells=[fmt(s[k]) for k in ('old_us','prior27_us','us')]+[fmt(data['crossfit'][sku][role])]
                cells += [fmt(s[k]) for k in ('sector arbitration','VF issue overlap','partition wake','RMS-specific warm','shared first payload')]
                text.append(f"|{batch}|{sku[6:11]} {'首遇' if role=='entry' else '复跑'}|"+'|'.join(cells)+'|')
    text+=['','v0.27 的原未知点在首遇低估和热态高估之间存在取舍。本轮把 γ/x 的响应、归约到 rstd 的发起间隔、不同 VF 间的发起依赖和每核写回独立定价，首遇改善，复跑误差在若干组上有所增加。每状态的条件预测提供更直接的参考。共享 x 流量项在这批单值消融中几乎没有独占关键路径，保留它的依据是字节守恒与芯片带宽，不能据这批合计 WAPE 声称其系数已经跨代验证。','',
        '### 同一逻辑输出行的流水','']
    for file,title in [('partition-repeat','第二分区启动与同形复跑 · [4,784]'),('multi-ub-repeat-dt','950DT 双 UB 块与短尾 · [539,4000]'),('multi-ub-repeat','950PR 双 UB 块与短尾 · [539,4000]')]:
        if (OUT/(file+'.json')).exists():
            text+=['!['+title+']('+file+'.png)','']
            v=read(OUT/(file+'.json'))['entries'];d=v[0];h=v[1]
            first=d['lane']['ops'];rep=h['lane']['ops']
            text += [f"{title}。相同输出起始行为 {d['lane']['row0']}，首次核 {d['lane']['lane']}，复跑核 {h['lane']['lane']}。首个 γ 发起 {first['gamma'][0]['b']} → {rep['gamma'][0]['b']} 周期，第一块 x 响应 {first['x'][0]['e']-first['x'][0]['b']} → {rep['x'][0]['e']-rep['x'][0]['b']} 周期，归约 VF 响应 {first['reduce'][0]['e']-first['reduce'][0]['b']} → {rep['reduce'][0]['e']-rep['reduce'][0]['b']} 周期。先查发起和读响应，再解释整窗变化。虚线为全设备终点，展示的核在复跑时可能更早结束。[对应时戳、GM 地址和 ELF 事件]("+file+'.json)','']
            if len(d['model_tiles'])>1:
                mt=d['model_tiles'];rt=first['x'];mv=mt[0]['x']
                text += [f"首遇时最晚核处理 {mt[0]['rows']}+{mt[1]['rows']} 行，第一块 x 的模型发起/响应时长为 {mv[0]:.0f}/{mv[1]-mv[0]:.0f} 周期，实测为 {rt[0]['b']}/{rt[0]['e']-rt[0]['b']} 周期。下一个 x 在首读尚余 {max(0,rt[0]['e']-rt[1]['b'])} 周期时已发起，首块归约、第二块归约实测于第 {first['reduce'][0]['b']}、{first['reduce'][1]['b']} 周期发起。模型末块 y 提交的预测完成为 {mt[-1]['y'][1]:.0f} 周期，实测为 {first['y'][-1]['e']}。比较的是同逻辑核；其他核或共享下界也可能决定模型的总窗。",""]
                issue_bias=mv[0]-rt[0]['b']
                service_bias=mv[1]-mv[0]-(rt[0]['e']-rt[0]['b'])
                text += [f"第一笔 x 发起偏差 {issue_bias:+.0f} 周期，读响应偏差 {service_bias:+.0f} 周期，合在一起的返回点偏差 {issue_bias+service_bias:+.0f} 周期。总窗相近时仍应检查各自的偏差与抵消。",""]
    text+=['## 按三个模型配置展开长输入','',
        '这里假设单卡一个归一化调用实际提交整批 B8，且图中有独立的全 hidden RMSNorm。若调度分块，应传每次实际提交的本卡 token 数。Flash 读取 text_config.hidden_size=5120，Kimi K3 和 Pro 的文本 hidden 分别从对应配置读出，均为 7168。表中是既有多轮 RMS 后端的公共无 H 设备预测；本轮没有这 27 项的完整 RMS CA。每栏单位为毫秒，括号内是每核最大 UB 块数。','',
        '|模型|B8 输入|单卡形状|950DT 模型|950PR 模型|970DT 外推|','|---|---:|---|---:|---:|---:|']
    grouped={}
    for r in cost['large']:
        grouped.setdefault((r['model'],tuple(r['shape'])),{})[r['chip']]=r
    for (model,shape),skus in grouped.items():
        text.append(f"|{model}|{shape[0]//8192}k|{shape[0]} × {shape[1]}|"+
            '|'.join(f"{skus[c]['public_us']/1000:.3f} ({skus[c]['iterations']})" for c in (dt,pr,future))+'|')
    text += ['', '较大输入在固定宽度下逐渐由每波相同的稠密工作和带宽约束决定，模型可按完整波次数乘以稳态增量。表中的近线性趋势是多轮模型预测。在这三个整批 RMS 形状上还缺少真实完整窗。','']
    text += ['### 有完整旧 CA 的中大数据量控制','',
        'v0.20 的独立窗口在本轮作为回放，以数值金标数和归档的 kernel 身份核对。每格为公共无 H 点 / 明示首次 / CA，单位 µs；后面的带宽是算法逻辑读写字节除以 CA，包含可能留在 L2 的输出。','',
        '|shape|950DT 预测 / 首次 / CA|DT 逻辑 GB/s|950PR 预测 / 首次 / CA|PR 逻辑 GB/s|','|---|---:|---:|---:|---:|']
    archived={}
    for r in cost['historical_large_archive']:archived.setdefault(tuple(r['shape']),{})[r['chip']]=r
    for shape,skus in archived.items():
        cells=[]
        for c in (dt,pr):
            r=skus[c];cells.extend([f"{r['unknown_us']:.3f} / {r['first_us']:.3f} / {r['ca_us']:.3f}",f"{r['logical_gbs']:.0f}"])
        text.append(f"|{shape[0]} × {shape[1]}|"+'|'.join(cells)+'|')
    text += ['', '固定 D=5120，行数从 1024 到 8192 增加 8 倍，DT 整窗为 4.49 倍，PR 为 5.85 倍。固定元素总数 41,943,040 时，[8192,5120] 和 [32768,1280] 的实测 DT 为 44.868/44.585 µs，PR 为 70.485/69.729 µs。这两组里大数据量的时间更接近按总字节比例增长，首发开销仍影响从较小数据量起算的倍率。PR 的逻辑带宽可超过配置的物理 HBM 带宽，端到端逻辑字节包含片上交付，不能据此反算 HBM 利用率。[旧金标与采集成本](../../v0.20/rms/records.json)','']
    elapsed=sum(z['seconds'] for z in ev['costs']);raw=sum(z['raw_bytes'] for z in ev['costs'])/2**30
    text+=['## 计时口径与成本','',
        f"从数值入口到 CA 导出，完成的 {len(ev['costs'])}/8 个串行进程用时 {elapsed/60:.2f} 分钟，保存 {raw:.2f} GiB 原始文件。每进程上限 60 分钟和 12 GiB。A5 CANN 9.2.1 `aclnnRmsNorm` 安装库 key 5000 的相同 ELF .text、实际目标平台和有序任务均经核验。逐调用验证全部 y 与 rstd、完整输出写字节、真实全核 entry/END、官方每核工作窗与 aggregate/shard 一致。计时取最早物理指令至最晚物理 END；预登记的物理口径不要求另加可视化合成终止标，与 v0.27 的合成标记口径相差数个周期。不同模型的同组比较使用同一金标。","",
        f"预测在主机上每个组合同一进程重复 5 次；无 H {cost['unknown']['n']} 次询问的中位、p95、最大分别为 {cost['unknown']['median_ms']:.3f}、{cost['unknown']['p95_ms']:.3f}、{cost['unknown']['max_ms']:.3f} ms。显式 H 的完整分阶段诊断最大 {cost['context_with_diagnostics']['max_ms']:.3f} ms。冻结实验代码与在线专项模块在 {cost['frozen_rows']} 个位置对无 H/条件定价的最大差 {cost['max_repricing_delta_us']:.2g} µs。[查询记录](query-cost.json)","",
        '[专用公式与参数](method.md) · [所有逐形独立回执](evaluation.json) · [冻结三端预测与消融](freeze.json) · [旧批次逐形回放](development-replay.json)']
    if ev['passes']:
        pkg=read(OUT/'installed-wheel.json')
        text+=['','## 公共入口和发布资格','',
            '无 H、每核一或两块时，`estimate("rmsnorm",shape,chip=...,dtype="bf16")` 使用本轮冻结的情景中心。已知 H 的 `estimate` 和超过两块的既有多轮默认继续按已有独立资格计价。调用 `rmsnorm_pipeline(...,resident_aiv=H,immediate_repeat=True,diagnostics=True)` 可查看确切复跑时的逐核 DAG。未知前序的公共 `stages_us` 仅标明情景中心总量；三种可观察条件的实际模型路径放在 `timing_diagnostics.history_scenarios`，逐核细节通过专项入口取回。']
        if pkg['public_expected']=='us':
            text+=['',f"从源码目录外安装 v{pkg['version']} wheel 后，{pkg['rows']} 条两实测 SKU 与 970 外推的冻结登记逐条复核，公共无 H 最大绝对算差 {pkg['public_max_abs_delta_us']:.2g} µs，旧显式 H 最大算差 {pkg['known_public_max_abs_delta_us']:.2g} µs，专项条件诊断与冻结最大算差 {pkg['specialized_inspect_max_abs_delta_us']:.2g} µs。[离线安装回执](installed-wheel.json)"]
    (OUT/'report.md').write_text('\n'.join(text).rstrip()+'\n')

if __name__=='__main__':report()
