"""Report each actual physical frontier; separate old failed numerics and frozen holdouts."""
from collections import defaultdict
from pathlib import Path
import re
from series import OUT,ROOT,DT,PR,REG,read,save,probe


def frac(a,b):return 'N/A' if b is None or a is None else f'{a/b:.4f}'
def display(x):return 'N/A' if x is None else f'{x:.4f}'
def cells(row):return '| '+' | '.join(map(str,row))+' |'
def core(s):return f"{s.get('active_aic','?')}+{s.get('active_aiv','?')}"
def mem(s):
    by=s['resource']['descriptor_bytes']
    return sum(v for k,v in by.items() if k.startswith('OUT->')),sum(v for k,v in by.items() if k.endswith('->OUT'))

def update():
    reg=read(REG);prior=OUT/'reused-ascend950dt-t2-h1.json'
    group=[('original',prior)]+[('original',OUT/(c['id']+'.json')) for c in reg['cases']]
    for role in ('development','holdout','supplement'):
        p=OUT/('precision-'+role+'-register.json')
        if p.exists():group +=[(role,OUT/(c['id']+'.json')) for c in read(p)['cases']]
    rows=[(role,read(path),path) for role,path in group if path.exists()]
    lines=['# FLA V2 实际分块与数值对照','',
      'B=1，Q/K/V 与原始 gate 是 BF16，头维 K=V=128，分块 C=64。表内三元组依次为 Prepare、FwdH、Finalize。总耗时采用同一条流各 kernel 的完整全核窗口之和，单位为 µs。输入的正负号混合，初态非零且不对称。这里使用 FLA V2 直接入口，所有开发与留出小形状的 Kimi 自动分派会选 V1。','',
      '| 数据角色 | 芯片 | T / H | 活动 AIC+AIV | MMAD 指令 | 已发起 MMAD 运算量（Mop） | 合格总耗时 µs | 记录 |',
      '|---|---|---|---|---|---|---:|---|']
    for role,r,path in rows:
        case=r['case'];s=[s for s in r.get('stages',[]) if s.get('family')]
        if len(s)!=3:continue
        lines.append(cells([{'original':'原协议开发','development':'舍入对照开发','holdout':'首批冻结独立','supplement':'补充冻结独立'}[role],case['sku'][6:11],f"{case['shape'][1]} / {case['shape'][3]}",
           ' / '.join(core(v) for v in s),' / '.join(str(v['mmads']) for v in s),
           ' / '.join(f"{v['resource']['issued_mmad_ops']/1e6:.3f}" for v in s),display(r.get('serialized_device_us')),'['+('通过' if r.get('qualified') else '数值或时窗未通过')+']('+path.name+')']))
    lines+=['','## 原始参考为何随 T 增大产生差异','',
       '数学参考将每个通道的两侧指数衰减合并后以双精度直接点积。发布库的 Prepare 每 16 行取中间行的 gate 前缀为基准，分别计算左、右衰减，将乘过衰减的 Q、K 两个左矩阵和 K 右矩阵舍入 BF16，再相乘累加。Aqk 将缩放后的因果分数转 BF16。Akk 以未舍入的 KK 累加和构造下三角矩阵，计算逆后转 BF16。两种算法在无限精度下相等，混合符号点积发生抵消时对 BF16 中间舍入敏感。attention 和最终 VK 状态仍逐元素对完整数学递推。所有 BF16 逐元素门槛保持 abs ≤ 2e−6 + |ref|/64，FP32 状态保持 abs ≤ 2⁻¹⁶ + |ref|/1024。分组额外绝对上限仍为 1 与 0.01。','',
       '发布测试程序以自然对数的逐行 gate 前缀除以 ln 2，使用 exp2。每行 r 所在带的中间行为 m，左侧 Q⁺ᵣ = BF16(Qᵣ × 2^(gᵣ−gₘ))、K⁺ᵣ 同理，右侧 K⁻ⱼ = BF16(Kⱼ × 2^(gₘ−gⱼ))。Aqkᵣⱼ = BF16(scale × Q⁺ᵣ·K⁻ⱼ)。令 Lᵣⱼ = betaᵣ × K⁺ᵣ·K⁻ⱼ（仅 j < r），Akk = BF16((I+L)⁻¹)。这里的 g、乘积均按通道计算，最后才求点积。[固定提交的独立 CPU 参考](https://github.com/flashserve/flash-linear-attention-npu/blob/5cc699069c7f91a35aabd583032c8720a0b626fa/tests/atk/chunk_kda_fwd_prepare/executor_chunk_kda_fwd_prepare.py)','',
       '| 原数值失败的样本 | attention 超限/总数 | Aqk 超限/总数 | Akk 超限/总数 | final state 超限/总数 | 同一原进程的诊断 |','|---|---:|---:|---:|---:|---|']
    for role,r,path in rows:
        if role!='original' or r.get('golden_qualified'):continue
        d=OUT/'numeric-detail'/path.name
        groups=read(d)['groups'] if d.exists() else {}
        links='[可复核原始文本](numeric-detail/'+path.name+')' if d.exists() else '原始标准输出未刷盘，分组不可恢复'
        lines.append(cells([r['case']['id']]+[f"{groups[n]['bad']}/{groups[n]['count']}" if groups else 'N/A' for n in ('attention','aqk','akk','state_vk')]+[links]))
    lines+=['','[原进程数学参考与公开 CPU 舍入步骤的逐形状汇总复核](numeric-explanation.json)','','## 同一输入的双参考','',
      '坐标顺序是本地头、序列行、该行在 chunk 内的列。下列数值直接来自新驱动保存的设备 BF16 输出。相对误差的分母只使用非零参考元素，最大绝对误差与最大相对误差可出现在不同位置。','',
      '| 样本 | 矩阵 / 参考 | 超限/总数 | 最大绝对差及其坐标 | 设备实际值 | 参考值 | 最大相对差 |','|---|---|---:|---|---:|---:|---:|']
    for role,r,path in rows:
        for line in r.get('auxiliary_diagnostics',[]):
            x=dict(re.findall(r'([a-z_]+)=([^ ]+)',line))
            lines.append(cells([r['case']['id'],x['name']+' / '+x['reference'],x['bad']+'/'+x['count'],
            x['max_abs']+' @ ('+', '.join(x[v] for v in ('h','t','column'))+')',x['got'],x['expected'],x['max_rel_nonzero']]))
    lines+=['','[保留的实际 BF16 输出与逐元素 CPU 重算](actual-array-checks.json)','','## 冻结留出对照','',
      '| 芯片 | T / H | 预测 µs | CA µs | 相对误差 | Prepare 预测 / CA µs | FwdH 预测 / CA µs | Finalize 预测 / CA µs |','|---|---|---:|---:|---:|---|---|---|']
    pred=read(ROOT/'results/kda-resource23/freeze.json')['predictions']
    if (OUT/'supplement-predictions.json').exists():pred+=read(OUT/'supplement-predictions.json')['predictions']
    scores=[];pairs=defaultdict(dict)
    for role,r,path in rows:
        if role not in ('holdout','supplement'):continue
        c=r['case'];a=r.get('serialized_device_us')
        f=next(q for q in pred if c['sku']==q['chip'] and c['shape']==q['shape']);t,h=c['shape'][1],c['shape'][3];p=f['serialized_device_us']
        phases=[s for s in r['stages'] if s.get('family')]
        lines.append(cells([c['sku'][6:11],f'{t} / {h}',display(p),display(a),'N/A' if a is None else f'{(p/a-1)*100:+.2f}%',
           *[display(s['device_us'])+' / '+display(v.get('ca_us') if a is not None else None) for s,v in zip(f['stages'],phases)]]))
        scores.append(dict(selection=role,shape=c['shape'],chip=c['sku'],qualified=r['qualified'],prediction_us=p,ca_us=a,signed_pct=None if a is None else (p/a-1)*100,
                   receipt=path.name,receipt_sha256=probe.digest(path)))
        if a is not None:pairs[t,h][c['sku']]=(p,a)
    lines+=['','| T / H | 预测 DT/PR | 实测 DT/PR | 加速比相对差 |','|---|---:|---:|---:|']
    for (t,h),v in pairs.items():
        if not all(k in v for k in (DT,PR)):continue
        p,a=(v[DT][i]/v[PR][i] for i in range(2));lines.append(cells([f'{t} / {h}',display(p),display(a),f'{(p/a-1)*100:+.2f}%']))
    summary={}
    for chip in (DT,PR):
        valid=[r for r in scores if r['chip']==chip and r['qualified']]
        if valid:
            summary[chip]=dict(n_shapes=len(valid),wape_pct=100*sum(abs(r['prediction_us']-r['ca_us']) for r in valid)/sum(r['ca_us'] for r in valid),
               maximum_abs_pct=max(abs(r['signed_pct']) for r in valid))
    lines+=['','| 芯片 | 已完成的独立 shape 数 | WAPE | 最大绝对相对误差 |','|---|---:|---:|---:|']
    for chip,q in summary.items():
        lines.append(cells([chip[6:11],q['n_shapes'],f"{q['wape_pct']:.2f}%",f"{q['maximum_abs_pct']:.2f}%"]))
    save(OUT/'holdout-comparison.json',dict(prediction_freeze_sha256=probe.digest(ROOT/'results/kda-resource23/freeze.json'),rows=scores,by_chip=summary))
    lines+=['','## 970DT 外推','','| T / H | 970DT 预测 µs | 芯片实测 |','|---|---:|---|']
    target_shapes=sorted({(r['shape'][1],r['shape'][3]) for r in scores})
    for t,h in target_shapes:
        q=next(v for v in pred if v['chip']=='Ascend970DT' and (v['shape'][1],v['shape'][3])==(t,h))
        lines.append(cells([f'{t} / {h}',display(q['serialized_device_us']),'N/A，架构假设外推']))
    lines+=['','## 指令计数和搬运口径','',
      '这三段分配的核数可以不同。单头 T=65 比 T=64 多一个 chunk，Prepare、Finalize 的新增工作可以分到另一组核。FwdH 会在同一个头上串行推进状态，须分别考虑其两次更新。增加独立头 H 能提供并行工作组。OUT 表示指令地址空间；下列读写字节数按描述符请求累计，包含缓存可能命中的请求，真实外存字节数未知。','',
      '| 样本 | Prepare GM读/写 B | FwdH GM读/写 B | Finalize GM读/写 B |','|---|---:|---:|---:|']
    for role,r,path in rows:
        if not r.get('qualified'):continue
        lines.append(cells([r['case']['id']]+[' / '.join(map(str,mem(s))) for s in r['stages'] if s.get('family')]))
    lines+=['','设备 trace 的每个时钟、MMAD PC/形状、DMA 发起间隔、描述符时延与最晚退休核可逐条核对。[完整资源记录](resources/)']
    target=OUT/'mechanisms.md';temp=target.with_suffix('.tmp');temp.write_text('\n'.join(lines)+'\n');temp.replace(target)

if __name__=='__main__':update()
