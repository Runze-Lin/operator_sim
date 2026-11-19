"""Refresh current per-shape tables from frozen predictions and completed CA receipts."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate
from development.validation.sku19.campaign import CHIPS,read,write
from development.validation.sku19.report import metrics
from development.validation.sku17.publish import metric

V=ROOT/'results/v0.20'
OLD=ROOT/'results/v0.19'
A='<!-- current-v020:start -->'
B='<!-- current-v020:end -->'
LABEL={'rmsnorm':'RMSNorm','sigmoid_fp32':'Kimi KDA beta Sigmoid','cast_bf16_fp32':'Cast BF16→FP32'}


def query(op,s):
    return {k:estimate(op,s,chip=c,timing_mode='resource').latency_us for k,c in CHIPS.items()}


def row(op,shape,role,targets,old=None,**extra):
    return dict(op=op,shape=shape,role=role,**{k+'_us':targets[k] for k in CHIPS},
        old_dt_us=old.get('dt') if old else None,old_pr_us=old.get('pr') if old else None,
        dt_ca_us=None,pr_ca_us=None,**extra)


def receipt(r,side,ref):
    if not ref:return
    r[side+'_ca_us']=ref['ca_us']
    r[side+'_observation']={k:ref[k] for k in ('kernel','trace_sha256','binary_sha256','correctness',
        'active_aic','active_aiv','trace','seconds','peak_bytes') if k in ref}


def collect():
    rows=[]
    measured={(tuple(v['shape']),v['chip']):v for v in read(V/'rms/records.json')}
    for j in read(V/'rms/freeze.json')['jobs']:
        o=j['outputs'][0]
        r=row('rmsnorm',j['shape'],'独立', {s:o[s+'_us'] for s in CHIPS},
            {s:o['old_'+s+'_us'] for s in CHIPS},api='aclnnRmsNorm',library='ops-nn')
        for s in ('dt','pr'):
            v=measured.get((tuple(j['shape']),CHIPS[s]))
            if v:receipt(r,s,dict(**v['stages'][0],correctness=v['correctness'],seconds=v['cost']['seconds'],peak_bytes=v['cost']['peak_bytes']))
        rows.append(r)
    source=read(V/'large/registration.json')['jobs']
    contract=read(V/'large/cast-v20-freeze.json')
    designs=list(contract['jobs'])
    sealed={j['id']:contract['utc'] for j in designs}
    if (V/'large/pro-freeze.json').is_file():
        pro=read(V/'large/pro-freeze.json');source.append(pro['job']);designs.append(pro['current'])
        sealed[pro['current']['id']]=pro['utc']
    for j in designs:
        job=next((v for v in source if v['id']==j['id']),None)
        if job is None:continue  # Other instrument reports a wrapper boundary.
        roles=' '.join(j['roles'].values())
        role='开发回放' if 'development' in roles else '并行回放' if 'parallel' in roles else '独立'
        r=row('cast_bf16_fp32',j['shape'],role,{s:j['targets'][c]['us'] for s,c in CHIPS.items()},
            {s:job['targets'][c]['us'] for s,c in CHIPS.items()},api='aclnnCast',library='ops-math')
        for s in ('dt','pr'):
            f=V/'large'/(job['id']+'_'+CHIPS[s]+'.json')
            if not f.exists():continue
            v=read(f)
            if v.get('qualified'):
                if role=='独立':assert v['cost']['started_utc']>=sealed[j['id']]
                receipt(r,s,dict(**v['stage'],correctness=v['correctness'],seconds=v['cost']['seconds'],peak_bytes=v['cost']['peak_bytes']))
        rows.append(r)
    for j in read(V/'newop/evaluation.json')['rows']:
        devices=j['devices']
        r=row('sigmoid_fp32',j['shape'],'独立',{s:devices[c]['model_us'] for s,c in CHIPS.items()},
            api='aclnnSigmoid',library='ops-nn')
        for s in ('dt','pr'):
            v=devices[CHIPS[s]]
            if v.get('status')=='qualified':receipt(r,s,dict(ca_us=v['ca_us'],kernel=v['kernel'],active_aiv=v['ca_aiv'],seconds=v['ca_wall_seconds']))
        rows.append(r)
    for r in rows:
        r['speedup_pred']=r['dt_us']/r['pr_us']
        r['speedup_ca']=r['dt_ca_us']/r['pr_ca_us'] if r['dt_ca_us'] and r['pr_ca_us'] else None
        r['speedup_old']=r['old_dt_us']/r['old_pr_us'] if r['old_pr_us'] else None
        r['pr_reuse_us']=r['dt_us']
        r['id']=r['op']+'_'+'_'.join(map(str,r['shape']))
    return rows


def display(x):
    return '—' if x is None else f'{x:.3f}'


def summary(r):
    def compare(k,ref):
        return metric([v for v in r if v.get(k) and v.get(ref)],k,ref)
    return dict(dt=compare('dt_us','dt_ca_us'),pr=compare('pr_us','pr_ca_us'),
        speed=compare('speedup_pred','speedup_ca'),v019_dt=compare('old_dt_us','dt_ca_us'),
        v019_pr=compare('old_pr_us','pr_ca_us'),v019_speed=compare('speedup_old','speedup_ca'),
        dt_reuse_pr=compare('pr_reuse_us','pr_ca_us'))


def detail_row(r):
    def cell(s):
        v=r[s+'_ca_us'];p=r[s+'_us'];old=r['old_'+s+'_us']
        return f'{display(old)} / {p:.3f} / {display(v)}'+(f' ({p/v-1:+.1%})' if v else '')
    dims='×'.join(map(str,r['shape']))
    return (f"| {LABEL[r['op']]} | {dims} | {r['role']} | {cell('dt')} | {cell('pr')} | "
            f"{display(r['speedup_pred'])} / {display(r['speedup_ca'])} | {r['future_us']:.3f} |")


def sections(rows):
    sections=['## 整批 prefill、实际库与规模误差','',
        '取 8 个请求共用一个本卡 attention 副本的情景。在整批调度下，32k、128k、256k 每次分别提交 262,144、1,048,576、2,097,152 个 token。Flash 的 indexer 权重复制投影固定输出 32 个数；Kimi K3 的 69 个 KDA 层在 TP8 下每个 token 有 12 个 beta；Pro 的 hidden 为 7168。每个 kernel 标明整批或宽通道 chunk，真实部署是否融合需实跑日志确认。[模型逐层形状和并行假设](prefill-shapes.md)','',
        '下表两端的每一格均为 **v0.19 预测 / 当前冻结预测 / 完整 CA（当前有符号误差）**，单位 µs。横线表示尚无合格值或旧版没有此路线。独立形状的两端分别采集；加速比只用两端都合格的相同输入。Cast 的 32k 用于定位根因，128k 开始采集时新版尚未冻结，256k 整批、两个宽通道 chunk 以及 Pro 的整批 [262144,64] 均先冻结预测。原生 wrapper 另列，不计入独立误差。','',
        '| 算子 | 本卡 shape | 数据角色 | 950DT 旧 / 新 / CA | 950PR 旧 / 新 / CA | DT/PR 新 / CA | 970DT 外推 |',
        '|---|---|---|---|---|---|---:|',*[detail_row(r) for r in rows],'',
        '误差表分母只包含已完成的正式全核指令窗口。下表汇总同组独立样本的 WAPE / 最大绝对相对误差。直接复用 DT 列把当前 DT 预测作为 PR 预测。','',
        '| 独立留出 | DT v0.19 → 当前 | PR v0.19 → 当前 | DT/PR v0.19 → 当前 | 直接复用 DT 预测 PR |',
        '|---|---|---|---|---|']
    for op,title in LABEL.items():
        items=[r for r in rows if r['op']==op and r['role']=='独立'];d=summary(items)
        def pair(k):return metrics(d['v019_'+k])+' → '+metrics(d[k])
        sections.append(f"| {title} | {pair('dt')} | {pair('pr')} | {pair('speed')} | {metrics(d['dt_reuse_pr'])} |")
    replay=read(V/'historical-replay.json')['summary']
    sections+=['','此前 10 个小 shape 的同集回放另计。RMSNorm 的 DT / PR WAPE 为 '
        +f"{replay['rmsnorm']['dt']['wape']:.2%} / {replay['rmsnorm']['pr']['wape']:.2%}，Cast 为 "
        +f"{replay['cast_bf16_fp32']['dt']['wape']:.2%} / {replay['cast_bf16_fp32']['pr']['wape']:.2%}。"
        +'[旧版、本版及配对 CA 明细](../results/v0.20/historical-replay.json)','','**RMSNorm。** 安装包 ops-nn 的 `aclnnRmsNorm`，每个样本对全部 BF16 输出和每行 FP32 rstd 做数值核验。开发回放 [2048,5120] 的 PR 从 31.655 改为 16.669 µs，CA 17.820 µs；DT 维持 13.186，对应 CA 14.337 µs。分核、MTE3 描述符与 Vector 价格没有参与重拟合。变化来自全片写出是否必须在末条指令前离开 L2。在 PR 上把全部输出强制写回会高估 77.64%，允许在容量约束内驻留后低估 6.46%。PR 尚无该 kernel 的 HBM 事务分解。[模型与实际临界核](examples/rmsnorm.md) · [冻结预测](../results/v0.20/rms/freeze.json) · [逐核与数值检查](../results/v0.20/rms/records.json) · [各次 CA 墙钟及字节开销](../results/v0.20/rms/costs.json)','',
        '**Cast。** CANN 9.2.1 ops-math 的 `aclnnCast`。Flash indexer 的 32k 整批共 8,388,608 个 BF16 输入、同量 FP32 输出。旧模型在 DT/PR 的逻辑流量带宽下界已经超过实际窗口。该 shape 在原生 DT 的 128 个通道中合计读取 16,822,272 B，写命令数为 0，逻辑输入、输出分别为 16 MiB、32 MiB。通道计数包括可能的额外元数据。当前模型分别为 MTE、L2 和片外可见字节定价，PR 的驻留仍是可替换的假设。更宽更长的输入开始溢出时可检验这一规律。[每卡 kernel、全元素金标、容量拐点及两种计时口径](../results/v0.20/large/report.md) · [控制器计数](../results/v0.20/large/cast_262144_32_native-dram.json)','',
        '**Kimi beta Sigmoid。** `aclnnSigmoid` 操作连续 FP32 张量，与 MoE router 的 sigmoid Top-K 分属不同的调用。核数命中仍可能出现明显时延偏差，例如 32k 整批的共享带宽按读写全额收费，预测的末轮退休过晚。每端均校验全部输出，图上保留 VF 和 MTE 指令。三颗芯片上的数据量和每核 UB 轮数可从冻结预测复算。[两张卡每个 shape 的 kernel 和正确性、代价](../results/v0.20/newop/measurements.md) · [32k 时间轴与参数](examples/sigmoid_fp32.md)','',
        '**读流水图。** Matmul 按两个真实输出块、每块两页 K 对齐；Quant 按五轮 UB 将相同的读/VF/写映射到同一行；FA 按首个 KV 页和稳态 II 绘制。上部仅在模型能确定的粒度定位事件，下部统一从全核首指令对齐，标出最后结束的核。[Matmul](examples/matmul.md) · [Quant](examples/quant.md) · [FA](examples/fa.md)','']
    return '\n'.join(sections)


def render():
    rows=collect()
    byop={op:summary([r for r in rows if r['op']==op and r['role']=='独立']) for op in LABEL}
    old=read(OLD/'evaluation.json')
    history=[{k:r[k] for k in ('id','op','shape','attrs','api','library','dt_us','dt_ca_us','pr_us','pr_ca_us','future_us') if k in r}
             for r in old['rows'] if r['op'] not in LABEL]
    write(V/'evaluation.json',dict(version='0.20.0',previous_version='0.19.0',rows=rows,summary=byop,
        unchanged_routes_v019=history,unchanged_summary={k:v for k,v in old['summary'].items() if k not in LABEL},
        previous_evidence='results/v0.19/evaluation.json',future_role='970DT projection only'))
    replay=[]
    for previous in old['rows']:
        if previous['op'] not in ('cast_bf16_fp32','rmsnorm'):continue
        op=previous['op'];r=row(op,previous['shape'],'历史重放',query(op,previous['shape']),
            {s:previous[s+'_us'] for s in CHIPS}, id=previous['id'])
        for s in ('dt','pr'):
            r[s+'_ca_us']=previous[s+'_ca_us'];r[s+'_observation']=previous[s+'_observation']
        r['speedup_pred']=r['dt_us']/r['pr_us'];r['speedup_ca']=r['dt_ca_us']/r['pr_ca_us']
        r['speedup_old']=r['old_dt_us']/r['old_pr_us'];r['pr_reuse_us']=r['dt_us']
        replay.append(r)
    write(V/'historical-replay.json',dict(role='v0.19 matched small-shape replay, excluded from independent v0.20 totals',rows=replay,
        summary={k:summary([r for r in replay if r['op']==k]) for k in ('rmsnorm','cast_bf16_fp32')}))
    path=ROOT/'docs/error-analysis.md';text=path.read_text()
    if A in text:text=text[:text.index(A)]+text[text.index(B)+len(B):].lstrip('\n')
    lines=text.splitlines()
    lines=[s for s in lines if not s.startswith('| Kimi KDA beta Sigmoid |')]
    for op,name in LABEL.items():
        d=byop[op];rs=[r for r in rows if r['op']==op and r['role']=='独立'];future=[r['future_us']for r in rs]
        label=f"v0.20 冻结留出，已完成端 / 登记 {len(rs)} 形"
        entry=f"| {name} | {label} | {d['dt']['n']} | {metrics(d['dt'])} | {d['pr']['n']} | {metrics(d['pr'])} | {metrics(d['speed'])} | {min(future):.3f}–{max(future):.3f} |"
        idx=next((i for i,l in enumerate(lines) if l.startswith('| '+name+' |')),None)
        if idx is None:idx=next(i for i,l in enumerate(lines) if l.startswith('| Kimi AttnResFwd |'));lines.insert(idx,entry)
        else:lines[idx]=entry
    text='\n'.join(lines)+'\n'
    anchor='## 芯片资源及模拟条件'
    text=text.replace(anchor,A+'\n'+sections(rows)+B+'\n\n'+anchor,1)
    path.write_text(text)
    return rows

if __name__=='__main__':render()
