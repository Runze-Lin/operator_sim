"""Replay frozen Sinkhorn phases against the public point and audit register eligibility; never launch CA."""
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/v0.28/fusion'
sys.path.insert(0,str(ROOT))
from operator_model import estimate as public, load_chip
from operator_model.mhc_sink_resource import parameters, sink as legacy_price
from operator_model.sink_registers import price as register_price, registers_per_helper
from schedule28 import estimate as schedule

DT,PR,FUT = 'Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'
KEYS=('old_unknown_us','range_clamp_us','outside_harmonic_us','correct_groups_us','point_us','old_known_us','known_us')
def read(name):return json.loads((OUT/name).read_text())
def metrics(rows,key):
    data=[v for v in rows if v.get(key) is not None and v.get('ca_us') is not None]
    if not data:return None
    dif=[abs(x[key]-x['ca_us']) for x in data]
    return dict(n=len(data),shapes=len({(v['t'],v['iters'])for v in data}),
                wape=100*sum(dif)/sum(x['ca_us']for x in data),max=100*max(e/x['ca_us'] for e,x in zip(dif,data)))
def score(rows):return {k:metrics(rows,k)for k in KEYS}
def qualified_known(data):
    return {label:score([r for r in data if criterion(r)]) for label,criterion in [
        ('all',lambda r:True),('G=1',lambda r:r['g']==1),('G>1',lambda r:r['g']>1),
        ('legacy supported',lambda r:r['old_unknown_us'] is not None),
        ('old T>64/I10 development',lambda r:r['t']>64 and r['iters']==10 and r.get('role')=='historical development')]}
def cost(t,it,sku,h):
    c=load_chip(sku);s=[t,4]
    try:
        legacy,_,_,old=legacy_price(s,c,{'num_iters':it})
    except ValueError:
        legacy,old=None,None
    b=public('mhc_sinkhorn',s,{'num_iters':it,'resident_aiv':h},chip=c,dtype='fp32')
    allstates=schedule(s,it,c);conditional=schedule(s,it,c,h,h)
    l,u=allstates['range_us'];known=next(iter(conditional['states'].values()));g=known['groups'];r=known['rows']
    # Cold traffic illustration: both operands cross HBM. With cache residency,
    # useful bytes need not all be external traffic. Local port bound stays separate.
    external_us=128*t/(1000*c.hbm_bandwidth_gbs)
    reduced=clamped=outside=stages=None
    packed=ceil(r*16/64)
    if legacy is not None:
        plan=old['target_plan'];assert plan['packed_chunks']==packed and plan['rows_per_core']==r
        stages=old['stages_us'];fixed=parameters()['phases']['mhc_sinkhorn']['SIMD'][0]*load_chip().frequency_ghz/c.frequency_ghz
        variable=stages['SIMD']-fixed
        # Replace the slope's charged iteration count only. Preserve its fixed
        # helper dispatch, target-clock conversion and all other public costs.
        reduced=register_price(s,c,{'num_iters':it})[0]
        assert abs(reduced-(legacy-variable*(packed-g)/packed))<1.e-11
        clamped=min(u,max(l,legacy))
        outside=legacy if l<=legacy<=u else allstates['harmonic_us']
    return dict(sku=sku,t=t,iters=it,h=h,g=g,rows=r,active=known['active'],state=known['state'],
        old_unknown_us=legacy,old_known_us=b.latency_us,known_us=known['us'],range_us=[l,u],
        point_us=allstates['point_us'],range_clamp_us=clamped,outside_harmonic_us=outside,correct_groups_us=reduced,
        legacy_packed_chunks=packed,full_cold_hbm_us=external_us,legacy_stages_us=stages,
        conditional_epochs_us=known['epochs_us'])

def all_rows():
    legacy=read('replay.json')['rows'];history=[]
    for v in legacy:
        m=cost(v['t'],v['iters'],v['sku'],v['h'])
        for k in ('old_unknown_us','old_known_us','known_us','point_us'):
            assert (m[k] is None and v[k] is None) or abs(m[k]-v[k])<1e-11,(k,m,v)
        history.append(dict(**m,id=v['id'],role=v['role'],source=v['source'],ca_us=v['ca_us']))
    reg=read('register.json');measured=read('independent.json')['rows'];obs={(v['sku'],v['group'],v['index']):v for v in measured}
    paper=[]
    for j in reg['jobs']:
        s=j['shape'];sku=j['chip']
        m=cost(s[0],j['iters'],sku,j['prior_h'])
        for k in ('old_unknown_us','old_known_us','point_us'):
            assert (m[k] is None and j[k] is None) or abs(m[k]-j[k])<1.e-11
        assert abs(m['known_us']-j['conditional']['us'])<1.e-11
        v=obs.get((sku,j['group'],j['index']))
        if v:
            assert (v['t'],v['iters'],v['h'],v['state'])==(s[0],j['iters'],j['prior_h'],m['state'])
            assert v['check']['active'] and v['check']['vf_per_core'] and v['check']['bytes_written']==64*s[0] and v['check']['golden_elements']==16*s[0]
        paper.append(dict(**m,group=j['group'],index=j['index'],role='frozen stage prices',ca_us=v['ca_us'] if v else None,
            receipt=v['receipt'] if v else None,trace_sha=v['id'] if v else None,
            nature='extrapolated' if sku==FUT else 'qualified' if v else 'pending'))
    return history,paper

def hardware_evidence():
    evidence=[]
    for v in read('landmarks.json'):
        if v['sku']!=PR or v['shape'][0] not in (67,241,251,449,4097):continue
        path=ROOT/v['trace'];raw=path.read_bytes()
        assert sha256(raw).hexdigest()==v['trace_sha256']
        events=[e for e in json.loads(raw)['traceEvents'] if e.get('ph')=='X']
        scat=[e for e in events if e['name']=='RV_VSCATTER']
        n=len(scat);vf=sum(e['name']=='VF' for e in events)
        assert n==8*v['groups'] and vf==2*v['iters']+2 and len(set(e['args']['pc_addr'] for e in scat))==1
        evidence.append(dict(t=v['shape'][0],iters=v['iters'],sku=v['sku'],g=v['groups'],h=v['history'],
            dma_bytes=v['dma_bytes'],scatter_subinstructions=n,scatter_pc=scat[0]['args']['pc_addr'],vf=vf,
            steady_issue_cycles=v['milestones']['steady'],trace=v['trace'],sha=v['trace_sha256']))
    return evidence

def register_replay_audit(history):
    archive={}
    for sku in (DT,PR):
        rows=[v for v in history if v['sku']==sku and v['old_unknown_us'] is not None]
        unchanged=sum(v['correct_groups_us']==v['old_unknown_us'] for v in rows)
        wins=sum(abs(v['correct_groups_us']-v['ca_us'])<abs(v['old_unknown_us']-v['ca_us']) for v in rows)
        losses=sum(abs(v['correct_groups_us']-v['ca_us'])>abs(v['old_unknown_us']-v['ca_us']) for v in rows)
        matched=[v for v in rows if v['t']==67 and v['iters']==10]
        assert len(matched)==2 and {v['state'] for v in matched}=={'first','resident'}
        low,high=sorted(r['ca_us'] for r in matched)
        archive[sku]=dict(n=len(rows),bitwise_unchanged=unchanged,strict_improvements=wins,strict_regressions=losses,
                          paired_t67_i10=dict(low_us=low,high_us=high,same_shape_minimax_relative_pct=100*(high-low)/(high+low)))
    return archive


def targeted():
    if not (OUT/'eligibility.freeze.json').exists():
        return None
    reg=read('eligibility.freeze.json');digest=sha256((OUT/'eligibility.freeze.json').read_bytes()).hexdigest()
    subset=[];costs={}
    for j in reg['jobs']:
        path=OUT/f"eligibility.receipt-{j['chip']}.json"
        row=dict(**j,ca_us=None)
        if path.exists():
            receipt=read(path.name);assert receipt['registration_sha256']==digest
            costs[j['chip']]=dict(receipt['cost'])
            cpu=ROOT/'runs/sink28-register-group'/j['chip']/'cpu.txt'
            if cpu.exists():
                fields=cpu.read_text().strip().splitlines()[-1].split()
                if len(fields)==2:
                    costs[j['chip']].update(user_cpu_s=float(fields[0]),system_cpu_s=float(fields[1]))
            v=next((v for v in receipt['rows'] if v['index']==j['index']),None)
            if v and v['qualified']:
                assert v['shape']==j['shape'] and v['iters']==j['iters'] and v['h']==j['prior_h']
                a=v['stage'];assert a['active_aiv']==j['plan']['active_aiv'] and a['written_bytes']==64*j['shape'][0]
                row.update(ca_us=a['ca_us'],trace_sha=a['trace_sha256'],trace=a['trace'],all_output_elements=v['all_output_elements'])
        subset.append(dict(**row,t=j['shape'][0]))
    return dict(registration_sha256=digest,rows=subset,ca_cost=costs,
        metrics={c:{k:metrics([v for v in subset if v['chip']==c],k) for k in ('old_unknown_us','corrected_unknown_us','conditional_us')} for c in (DT,PR)},
        state=read('eligibility.state.json') if (OUT/'eligibility.state.json').exists() else None)


def boundary_examples():
    return [dict(cores=v,first_legacy_extra_token=4*v+1,first_second_register_pass_token=8*v+1,
                 checks=[dict(t=t,rows=ceil(t/v),logical=ceil(ceil(t/v)/4),registers=registers_per_helper(ceil(t/v)))
                         for t in (4*v,4*v+1,8*v,8*v+1)]) for v in (32,56,64,120)]


def main():
    history,paper=all_rows()
    forward=[r for r in paper if r['ca_us'] is not None]
    eligibility=dict(scope='All archived cases replayed. Clamp / out-of-envelope replacement / correcting the packed SIMD count are diagnostic alternatives reviewed after the earlier measurements; the distinct targeted register-count check is frozen separately and uses only future endpoint receipts.',
        data=history,forward=paper,hardware=hardware_evidence(),
        history={c:qualified_known([v for v in history if v['sku']==c])for c in (DT,PR)},
        frozen={c:qualified_known([v for v in forward if v['sku']==c])for c in (DT,PR)},
        register_replay=register_replay_audit(history),targeted=targeted(),register_boundaries=boundary_examples(),state=read('state.json'),registration_sha256=sha256((OUT/'register.json').read_bytes()).hexdigest())
    out=OUT/'eligibility.json';out.write_text(json.dumps(eligibility,indent=2)+'\n')
    render(eligibility)
def fmt(x):return f"{x['wape']:.2f} / {x['max']:.2f}"if x else 'N/A'
def render(d):
    hist=d['data'];paper=d['forward'];forward=[r for r in paper if r['ca_us'] is not None]
    lines=['# Standalone mHC Sinkhorn 的默认点与已知前序资格','',
        '|开发回放|窗 / 不同形|旧无 H WAPE / 最大 %|压到条件端点|越界时用端点谐均|按真实 G 校正旧 SIMD|已知 H 旧 / 专项|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for sku in (DT,PR):
        for label in ('legacy supported','old T>64/I10 development'):
            q=d['history'][sku][label];n=q['old_unknown_us']
            lines.append(f"|{sku[6:11]} · {label}|{n['n']} / {n['shapes']}|"+'|'.join(fmt(q[k])for k in KEYS[:4])+'|'+fmt(q['old_known_us'])+' → '+fmt(q['known_us'])+'|')
    lines+=['','旧无 H 的 PR 最大偏差来自 **[241,4]、I=16、resident**。最忙核 R=5，旧公共式按每 64 个 FP32 字分包，`C=ceil(16R/64)=2`；安装库中的单 VF 内真实寄存器工作的单位为 `G=ceil(R/8)=1`。已在同一 PC 验证 scatter 的八条子指令与稳态的 254 周期发起，两者和 R=2 的 G=1 核一致。这个真实差别能解释旧模型额外付出一遍迭代费用。','',
        '|950PR 末核开发窗 T/I|R / G|VF 数|scatter 子指令数|稳态发起间隔 cycles|每方向 MTE 载荷 B|','|---|---:|---:|---:|---:|---:|']
    for r in d['hardware']:
        lines.append(f"|{r['t']}/{r['iters']} · H={r['h']}|{r['dma_bytes']//64} / {r['g']}|{r['vf']}|{r['scatter_subinstructions']}|{r['steady_issue_cycles']}|{r['dma_bytes']}|")
    lines+=['','## 三条资源对照','',
        '冻结阶段排程用同一实现的可达 H 算出区间 [L,U]。区间是模型情景，不是 CA 保证区间。把旧估计 p 压到最近端点可以保留区间内全部样本，但越界后保留的偏差方向仍可能同号。第二个对照仅在越界时改取 2LU/(L+U)，其余情况原样返回。两者仅用目标资源，不需要目标芯片有 CA。','',
        '第三个对照用实际寄存器组数改旧估时器的 SIMD 段。令 F 为旧 SIMD 固定发起成本（按 target frequency 换算），S 为旧 SIMD 全段，改用 `p′ = p − (S−F)(C−G)/C`。保留原控制、读写和各段固定成本，不改任何阶段系数。R≤4 时 C=G=1，任何芯片都得到旧值。旧无 H 机制只覆盖 T≤256，在 DT 的原支持域 R≤4，因此 DT 的 45 个老窗均保持原价。本项修订源自旧窗的指令数，独立资格由下文 T233 的两端冻结实验单列。','',
        '|PR 发生变化的旧窗|状态|旧价 / CA µs|[L,U]|端点 / 越界谐均 / 校正 G µs|旧 SIMD µs|读写都走 HBM 的理想时间 µs|','|---|---|---:|---|---|---:|---:|']
    for r in hist:
        if r['old_unknown_us'] is None or abs(r['correct_groups_us']-r['old_unknown_us'])<1e-10:continue
        lo,hi=r['range_us']
        lines.append(f"|{r['t']}/{r['iters']}|{r['state']}|{r['old_unknown_us']:.3f} / {r['ca_us']:.3f}|{lo:.3f}–{hi:.3f}|{r['range_clamp_us']:.3f} / {r['outside_harmonic_us']:.3f} / {r['correct_groups_us']:.3f}|{r['legacy_stages_us']['SIMD']:.3f}|{r['full_cold_hbm_us']:.4f}|")
    lines+=['','例如 T241 时读与写各 64T=15,424 B，即便按全部发生于 1400 GB/s HBM，理想服务仅 0.0220 µs。吞吐只给出服务时间下限；它无法为高估的 7.991 µs 提供物理上限。原大偏差中可定位到错误迭代工作量，仍有未知前序造成的剩余误差。DT 在原 T>64/I10 三个窗里，T68 首发、T65 和 T66 均 resident，三条旧估计全在条件区间内。端点压缩和修正寄存器数都不能消除其 25.64% WAPE。','',
        '## 四个事前冻结几何的双端逐窗表','',
        '每一格的格式为 **旧无 H / 冻结无 H / 旧已知 H / 冻结已知 H / 完整 CA**，均为 µs。“待测”与“N/A”分别表示尚无合格回执与旧无 H 源本就没有定价。I 是迭代数。合格实测必须通过每核终止与所有输出检查。无 H 备选公式的开发回放与下表原本已冻结的两个预测独立区分。','',
        '|真实队列 / 序号|T / I|950DT|950PR|970DT 外推无 H / 已知 H|','|---|---|---|---|---|']
    ids=sorted({(r['group'],r['index'])for r in paper});by={(r['group'],r['index'],r['sku']):r for r in paper}
    for g,i in ids:
        cells=[]
        for sku in (DT,PR,FUT):
            r=by[g,i,sku]
            terms=('point_us','known_us')if sku==FUT else ('old_unknown_us','point_us','old_known_us','known_us','ca_us')
            cells.append(' / '.join((f"{r[k]:.3f}"if r[k] is not None else '待测' if k=='ca_us' else 'N/A') for k in terms))
        lines.append(f"|{g}/{i}|{r['t']} / {r['iters']}|"+'|'.join(cells)+'|')
    lines+=['','|已取得的独立样本|窗 / 不同几何|冻结无 H WAPE / 最大|旧无 H 同可定价窗|旧 H|专项 H|','|---|---:|---:|---:|---:|---:|']
    for c in (DT,PR):
        for group in ('all','G>1'):
            z=d['frozen'][c][group];n=z['known_us']
            lines.append(f"|{c[6:11]} · {group}|{n['n'] if n else 0} / {n['shapes'] if n else 0}|"+'|'.join(fmt(z[k]) for k in ('point_us','old_unknown_us','old_known_us','known_us'))+'|')
    lines+=['','本次已登记的四个几何只有 T69、T137 落在旧无 H 合同内。两端在这两个几何上都满足 C=G=1 且原价在 [L,U] 内，所以即使全跑完，三种替换规则也不会在这些待测窗里与旧价分离。','',
        '## 已知 H 的多寄存器组接口','',
        '现行 `estimate("mhc_sinkhorn",[T,4],{"num_iters":I,"resident_aiv":H},chip=...,dtype="fp32")` 接受系统提供的同族前序，不要传入别的 kernel 使用的 AIV 数。公开资源模型可以在经验证的小形上返回已知 H 估计，也能在工作区可容纳的更大形上给出标为外推的旧估计。v0.28 的多寄存器专项用 helper 固定发起加 G−1 份边际工作；校准只使用 DT 的发起与 scatter 指令数据。','',
        '|多组 G>1 的历史回放|窗数|旧已知 H WAPE / 最大|专项 WAPE / 最大|','|---|---:|---:|---:|']
    for c in (DT,PR):
        z=d['history'][c]['G>1'];lines.append(f"|{c[6:11]}|{z['known_us']['n']}|{fmt(z['old_known_us'])}|{fmt(z['known_us'])}|")
    observed=sum(r['g']>1 for r in forward)
    lines+=['',f"本快照共有 {observed} 个已核验的 G>1 前向窗。原登记四形中仅 T1153/I6、T2257/I4 刺激新公式，不能将其他两种 G=1 几何计作它的多组独立证据。公开默认和显式已知 H 接口维持现有资格，开发侧可调 `schedule28.estimate([T,4],I,chip,H,H)` 查看逐发起时间；其依赖版本化验证资料，当前没有作为已验证的新增 wheel 接口发布。[冻结条件、源码、旧批次和同轴图](report.md)",'',
        '[逐窗数值、源码指纹及 CA 来源](eligibility.json) · [原四形登记](register.json) · [正在执行的同一单槽](state.json)']
    lines+=['','## 旧小形的逐窗影响','',
        '|同集旧窗|窗|原价逐位相同|严格改善|严格退化|同形 T67／I10 首次、覆盖复跑 (µs)|该配对的最小可能最大相对误差|','|---|---:|---:|---:|---:|---:|---:|']
    for sku,v in d['register_replay'].items():
        x=v['paired_t67_i10']
        lines.append(f"|{sku[6:11]}|{v['n']}|{v['bitwise_unchanged']}|{v['strict_improvements']}|{v['strict_regressions']}|{x['high_us']:.3f}、{x['low_us']:.3f}|{x['same_shape_minimax_relative_pct']:.2f}%|")
    lines+=['','仅有 shape 时，同一 T/I 的冷、热窗都对应一个预测 p。两种状态绝对相对误差相等时最小化最大误差，可得 p=2LU/(L+U)，下界为 (U−L)/(U+L)。上表使用 T67 的真实双窗；23.22% 与 21.34% 都低于 50% 门槛，两点不能证明整个算子的 WAPE 达标或不可能。','',
        '|DT 已超目标的 T>64/I10 旧小组|R / C / G|H|CA / 旧与修订未知价 / 已知 H (µs)|旧与修订无 H 偏差|','|---|---:|---:|---:|---:|']
    for r in hist:
        if r['sku']!=DT or r['t']<=64 or r['iters']!=10 or r['role']!='historical development':continue
        ca=r['ca_us'];u=r['old_unknown_us']
        lines.append(f"|T{r['t']} / {r['state']}|{r['rows']} / {r['legacy_packed_chunks']} / {r['g']}|{r['h']}|{ca:.3f} / {u:.3f} / {r['known_us']:.3f}|{100*(u/ca-1):+.2f}%|")
    lines+=['','三形都只需一个寄存器组。首次低估、热态两点高估，能够由同族系统前序解释的部分可直接通过已公开的 `resident_aiv` 估价重现。在只有单次形状的 API 上，修复寄存器计数保留该小组原值。']
    lines+=['','## 目标芯片分核改变时的换组点','',
        '安装包 CalcSoftmax、CalcColNorm、CalcRowNorm 和 ScatterOutFromNorm 都从 `GetVRegSize()/GetUbBlockSize()` 得到逐 VF 的 `blockCnt`，以 `ceil(handleNum/blockCnt)` 驱动内层循环。950 采用 256 B 寄存器、32 B UB 块，每一遍覆盖 8 个矩阵。64 B 是完整 4×4 FP32 输入矩阵的大小，用它除寄存器长度会提前一倍跨组。各代在同一实现假设下由目标 AIV 数 V 确定 R=ceil(T/V)。如果底层寄存器或 UB 事务大小改变，需要按目标实现重定价。','',
        '|可用 AIV|旧规则首次误计第二遍的 T|底层首次需要第二遍的 T|边界前后真实组数|','|---:|---:|---:|---|']
    for v in d['register_boundaries']:
        lines.append(f"|{v['cores']}|{v['first_legacy_extra_token']}|{v['first_second_register_pass_token']}|"+'、'.join(f"T{x['t']}→{x['registers']}" for x in v['checks'])+'|')
    if d['targeted']:
        q=d['targeted'];lines+=['','## 新形 [233,4,4] / 16 次归一化的双端验证','',
            '这个新形在 PR 每核至多处理五个矩阵，在 DT 至多四个。原四形队列完成后才开始同一设备、同一流的首次与立即复跑，每端只增加一种几何。每次通过实际 SKU、完整输出、ELF、所有物理 AIV 的 END 和写回字节后填入 CA。已知 H 可降低前序不确定性；该列只用来解释发起与首次/复跑差异。','',
            '|芯片|H / 物理状态|活跃 AIV / R / G|旧无 H / 新无 H / 已知 H (µs)|CA (µs)|旧 / 新有符号误差|','|---|---|---:|---:|---:|---:|']
        for r in q['rows']:
            p=r['plan'];ca=r['ca_us'];future=r['chip']==FUT
            old=r['old_unknown_us'];new=r['corrected_unknown_us']
            lines.append(f"|{r['chip'][6:11]}{' 外推' if future else ''}|{r['prior_h']} / {r['state']}|{p['active_aiv']} / {p['rows_per_core']} / {p['register_passes']}|{old:.3f} / {new:.3f} / {r['conditional_us']:.3f}|"+(f"{ca:.3f}|{100*(old/ca-1):+.2f}% / {100*(new/ca-1):+.2f}%" if ca is not None else ('N/A|N/A' if future else '待测|待测'))+'|')
        lines+=['','|已测新形首次与复跑|合格窗 / 不同形|旧无 H WAPE / 最大 %|新无 H WAPE / 最大 %|已知 H WAPE / 最大 %|','|---|---:|---:|---:|---:|']
        for c in (DT,PR):
            z=q['metrics'][c];n=z['old_unknown_us']
            lines.append(f"|{c[6:11]}|{n['n'] if n else 0} / {n['shapes'] if n else 0}|"+'|'.join(fmt(z[k])for k in ('old_unknown_us','corrected_unknown_us','conditional_us'))+'|')
        lines+=['','[事前预测与资格条件](eligibility.freeze.json) · [持久队列状态](eligibility.state.json)']
    lines+=['','## PR T241 旧窗的发起与写回','', '![950PR 上方条件机制发起与两种无前序总价，下方为同域 CA 关键核](eligibility.trace.png)','', '已知 H=53 的行列循环约每 0.158 µs 发起一对，CA 为 254 个周期即 0.154 µs。完整 CA 是 4.419 µs，条件模型为 4.234 µs。上图两个空心三角仅标无 H 的旧价 7.991 与改组数后的 5.479 µs；模型不提供这两个总价对应的逐指令时刻。CA 淡色矩形含依赖等待。[图内原始数值与校验指纹](eligibility.trace.json)']
    if (OUT/'eligibility.cost.json').exists():
        bench=read('eligibility.cost.json')
        lines+=['','## 在线计算和离线采集开销','',
            '|芯片|候选计价调用次数|中位 / P95 / 最大 (ms)|','|---|---:|---:|']
        for sku,v in bench['by_chip'].items():
            lines.append(f"|{sku[6:11]}|{v['n']}|{v['median_ms']:.4f} / {v['p95_ms']:.4f} / {v['max_ms']:.4f}|")
        lines+=['','这是已加载配置和价格文件后、每次重新计算的候选计价函数耗时。一次物理模型调用与两个前序状态的 CA 采集各自计时，离线每端一次进程预算 1,200 秒、2,200 MiB。合格回执记录实际墙钟和目录占用，CPU 来自同一进程的 time 记录。[预测开销](eligibility.cost.json)']
        if d.get('targeted'):
            for sku,costs in d['targeted']['ca_cost'].items():
                cpu=sum(costs[k] for k in ('user_cpu_s','system_cpu_s')) if 'user_cpu_s' in costs else None
                lines.append(f"\n{sku} 实收 {costs['seconds']:.1f} s 墙钟、{costs['raw_bytes']/2**20:.1f} MiB"+(f"、{cpu:.1f} s CPU。" if cpu is not None else '。'))
    (OUT/'eligibility.md').write_text('\n'.join(lines).rstrip()+'\n')
def benchmark():
    """Measure model service time after chip/config load, on the frozen target only."""
    import platform
    from statistics import median
    from time import perf_counter_ns
    data={}
    for j in read('eligibility.freeze.json')['jobs'][::2]:
        chip=load_chip(j['chip']);shape=j['shape'];attrs={'num_iters':j['iters']}
        assert register_price(shape,chip,attrs)[0]==j['corrected_unknown_us']
        samples=[]
        for _ in range(500):
            start=perf_counter_ns();register_price(shape,chip,attrs);samples.append((perf_counter_ns()-start)/1e6)
        samples.sort()
        data[chip.name]=dict(n=len(samples),median_ms=median(samples),p95_ms=samples[474],max_ms=max(samples))
    (OUT/'eligibility.cost.json').write_text(json.dumps(dict(python=platform.python_version(),
       scope='Candidate pricing function, loaded chip and warm immutable price file; one fresh estimate per call, no result cache. Does not include public API dispatch or simulation.',by_chip=data),indent=2)+'\n')


def render_example():
    """Replay the last physical AIV while keeping point-only estimates off the issue lanes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    v=next(x for x in read('landmarks.json') if x['sku']==PR and x['shape'][0]==241 and x['iters']==16)
    m=next(x for x in read('eligibility.json')['data'] if x['id']==v['aggregate_sha256'])
    assert sha256((ROOT/v['trace']).read_bytes()).hexdigest()==v['trace_sha256']
    core=Path(v['trace']).parent.name
    e=m['conditional_epochs_us']
    model=[(0,e['tiling_begin'],e['tiling_end']), (1,e['read_begin'],e['read_end']),
           (2,e['setup_vf'],None),(3,e['softmax_vf'],None),(4,e['first_col_vf'],None)]
    for i in range(v['iters']-1):
        row=e['first_row_vf']+(0 if i==0 else e['first_pair_period']+(i-1)*e['steady_pair_period'])
        model.extend([(5,row,None),(6,row+(e['first_row_to_col'] if i==0 else e['steady_row_to_col']),None)])
    model.extend([(7,e['scatter_vf'],None),(8,e['write_begin'],e['write_end'])])
    vf=[o for o in v['events'] if o['name']=='VF']
    dma=[o for o in v['events'] if o['name'].startswith('MOV_SRC_TO_DST')]
    assert [o['size'] for o in dma]==[96,320,320] and len(vf)==34
    ca=[(lane, o['begin']/1650, o['end']/1650) for lane,o in list(zip([0,1,8],dma))+list(zip([2,3,4]+[5,6]*15+[7],vf))]
    labels=['Tiling · MTE2','Input · MTE2','Setup · VF','Softmax · VF','First column · VF','Normalize row · VF','Normalize column · VF','Scatter · VF','Output · MTE3']
    paths=OUT/'eligibility.trace'
    chart=dict(sku=PR,shape=[241,4,4],iters=v['iters'],history=53,core=core,labels=labels,
               origin='first instruction of all physical AIVs; microseconds',model=model,ca=ca,conditional_us=m['known_us'],
               ca_us=m['ca_us'],without_history_us={'legacy':m['old_unknown_us'],'registers':m['correct_groups_us']},
               trace=v['trace'],trace_sha256=v['trace_sha256'],aggregate_sha256=v['aggregate_sha256'],receipt=v['receipt'])
    paths.with_name(paths.name+'.json').write_text(json.dumps(chart,indent=2)+'\n')
    colors=['#177c83','#177c83','#546b80','#5364a2','#7959a7','#476bac','#7959a7','#647a89','#b14d2c']
    fig,axs=plt.subplots(2,1,figsize=(12,7.5),sharex=True)
    fig.subplots_adjust(left=.19,right=.95,top=.78,bottom=.14,hspace=.48)
    fig.text(.19,.935,'950PR  |  MhcSinkhorn · 241 × 4 × 4',size=23,weight='bold',color='#1b3540')
    fig.text(.19,.884,'49 AIVs · up to 5 matrices per core · 1 register group · 16 iterations · 53 resident AIVs',size=12,color='#3d5765')
    fig.text(.19,.837,'Whole-operator estimates without H     △  5.479 µs  corrected register count       △  7.991 µs  old count',size=11,color='#58545d')
    for ax,spans,title,end,is_ca in zip(axs,(model,ca),('Resource model · conditioned on H = 53',f'CA trace · {core} (last to finish)'),(m['known_us'],m['ca_us']),(False,True)):
        ax.set_title(title,loc='left',color='#203640',fontsize=13,weight='bold')
        ax.set_title(f'{end:.3f} µs',loc='right',color='#203640',fontsize=13,weight='bold')
        for lane,x,y in spans:
            if y is not None:ax.barh(lane,y-x,left=x,height=.36,color=colors[lane],alpha=.34 if 2<=lane<=7 else .95,zorder=2)
            if 2<=lane<=7:ax.vlines(x,lane-.24,lane+.24,lw=1.7,color=colors[lane],zorder=3)
        ax.axvline(end,color='#293e49',ls=(0,(3,3)),lw=1.3)
        if not is_ca:
            for z,caption,color in [(m['correct_groups_us'],'Corrected G=1','#267a7d'),(m['old_unknown_us'],'Old C=2','#a36646')]:
                ax.plot(z,-.35,'^',ms=8,mfc='none',mec=color,mew=1.6,clip_on=False)
                ax.text(z,.55,caption,ha='center',va='top',color=color,fontsize=9)
        ax.set_yticks(range(9),labels,fontsize=9);ax.set_ylim(8.6,-.7);ax.tick_params(axis='y',length=0)
        ax.grid(axis='x',lw=.7,color='#d3dfe3');ax.spines[['left','right','top']].set_visible(False)
        ax.spines['bottom'].set_color('#adc1c9')
    axs[-1].set_xlim(0,8.4);axs[-1].xaxis.set_major_locator(mticker.MultipleLocator(1))
    axs[-1].set_xlabel('Device time since first instruction on any AIV (µs)',fontsize=12)
    fig.text(.19,.061,'Ticks mark VF issue. Pale CA bars show descriptor lifetimes. Triangles denote whole-operator point estimates only.',fontsize=10,color='#435d69')
    fig.savefig(paths.with_name(paths.name+'.png'),dpi=165);plt.close(fig)


if __name__=='__main__':
    if '--benchmark' in sys.argv[1:]:
        benchmark()
    main()
    if '--figure' in sys.argv[1:]:
        render_example()
