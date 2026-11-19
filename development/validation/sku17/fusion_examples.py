"""Numerical walkthrough of seven public DT -> scenario projections. No calibration."""
from math import ceil, isclose
from pathlib import Path
from statistics import median
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate,load_chip
from operator_model import fusion_v8 as fusion,mhc_v9
from operator_model.backend import _call_attrs
from operator_model.predictor import _profile,predict
from operator_model.transfer import _queue_factors,_tasks
from operator_model.workload import account_work

OPS={fusion.IDX,fusion.FUSED,fusion.FIRST,fusion.SECOND,fusion.CHAIN,'mhc_pre','mhc'}
Q=('CUBE','SIMD','MTE1','MTE2','MTE3','FIXP','SCALAR','IDLE')


def _numbers(mapping):
    return ', '.join(str(mapping[q]) for q in Q)


def _check(x,y):
    if not isclose(x,y,rel_tol=1e-10,abs_tol=1e-9):
        raise ValueError(f'Walkthrough {x} differs from implementation {y}')


def _physical(part,shape,attrs):
    """What was multiplied/moved, as opposed to the template's scheduling proxy."""
    w=account_work(part,shape,attrs)
    if part in (fusion.FIRST,fusion.SECOND):
        m,k,f,e=shape
        width,red=(2*f,k) if part==fusion.FIRST else (k,f)
        live=sum(c>0 for c in attrs['expert_counts'])
        detail=(f'按每个本地专家所分配的有效行数求和，点积共 2×{m}×{red}×{width} 次标量运算。'
                f'读字节=({m}+{live}×{width})×({red}+2⌈{red}/64⌉)+8×{e}，'
                '依次包含输入、非空专家权重、每 64 个归约元素的两字节 scale，以及专家分组表。')
        if part==fusion.FIRST:
            detail+=f'SwiGLU 五种逐元素算术共 5×{m}×{f} 次；写出 FP8 和 scale，共 {m}×{f}+2×{m}⌈{f}/64⌉ 字节。'
        else:
            detail+=f'BF16 写回 2×{m}×{k} 字节。'
    elif part==fusion.IDX:
        b,q,l,g,d=shape
        valid=b*(q*l-q*(q-1)//2 if q<=l else l*(l+1)//2)
        detail=(f'右下对齐因果掩码覆盖 {valid} 个有效 Q/K 对；点积 2×{valid}×{g}×{d}。'
                f'{g} 个头分别执行 ReLU 与乘权，求和还需 {g-1} 次加法；每对合计 {3*g-1} 次向量通道运算。')
    elif part in (fusion.FUSED,fusion.PRE,'mhc'):
        t,n,d=shape
        detail=(f'Pre 投影把宽 {n*d} 的残差映到 {n*n+2*n} 个系数，'
                f'HF32 计算量 2×{t}×{n*d}×{n*n+2*n}。')
    else:
        detail='组合的读写按子 kernel 接口累加，物化的中间结果计一次写、一次读。'
    return (detail+f"有用 Cube 运算 {w['cube_ops_by_dtype'] or '{}'}、Vector 运算 {w['vec_ops_by_dtype'] or '{}'} "
            f"（FMA 算 2），逻辑读/写 {w['logical_read_bytes']}/{w['logical_write_bytes']} B。字节数不含缓存重读。")


def _units(op,sh,attrs):
    if op==fusion.IDX:
        b,q,l,g,d=sh;w=ceil(q/32)
        return f'Q={q} 路、每路 {g} 个 Query 头，头维 {d}，KV 长度 {l}。DT 调度波 w=⌈{q}/32⌉={w}，读/矩阵按 ⌈{l}/128⌉={ceil(l/128)} 块，top512 按 ⌈{l}/512⌉={ceil(l/512)} 块；Cube 代理 w·⌈L/128⌉·⌈G/16⌉，SIMD 代理 w·⌈L/512⌉。未建模依赖数据值的 top-k 比较路径。'
    if op in (fusion.FIRST,fusion.SECOND):
        m,k,f,e=sh;cnt=attrs['expert_counts'];nx=sum(ceil(c/16) for c in cnt)
        width,red=(2*f,k) if op==fusion.FIRST else (k,f)
        return f'{e} 个本地专家，token 行数 {list(cnt)}，和为 {m}；每专家各自按 16 行补块，合计 {nx} 个行块。输出列={width}、归约宽={red}，Cube jobs={nx}×⌈{width}/128⌉={nx*ceil(width/128)}；DT 波 w=⌈jobs/32⌉={ceil(nx*ceil(width/128)/32)}，Cube 单位 w×⌈{red}/64⌉，SIMD 模板单位 ⌈Σ⌈行数×{f}/256⌉/64⌉={ceil(sum(ceil(x*f/256) for x in cnt)/64)}。空专家不产生数学工作；GMM2 可能仍有向量样式的队列代理，此数不是 SIMD 指令计数。'
    t,n,d=sh;wc=ceil(t/(32 if op in (fusion.PRE,fusion.FUSED) else 64))
    desc=f'T={t}, N={n}, D={d}。工作按 w={wc} 个 DT 波归一化。'
    if op in (fusion.PRE,fusion.FUSED):
        desc+=f'投影按 ⌈ND/128⌉={ceil(n*d/128)}。'
    if op!='split_sink':
        desc+=f'通道处理按 ⌈D/128⌉={ceil(d/128)}。'
    if op in (fusion.FUSED,'split_sink'):
        desc+='20 次 Sinkhorn 是所选合同的一部分，模板未区分任意迭代数。'
    return desc


def _factors(key,child,shape,attrs,raw_us,ref,target):
    shortshape=shape[:2] if child=='mhc_sinkhorn' else shape
    w=account_work(child,shortshape,attrs);notes=[]
    f,active=_queue_factors(key,shape,attrs,{'predicted_us':raw_us},w,ref,target,notes)
    cj,vj=_tasks(key,shape,attrs,{'predicted_us':raw_us})
    bs=w['logical_read_bytes']+w['logical_write_bytes']
    pressure=min(1.,bs/(1000*ref.hbm_bandwidth_gbs*raw_us))
    text=(f"分配代理 Cube/Vector jobs={cj}/{vj}，Cube 波 {ceil(cj/ref.cube_cores)}→{ceil(cj/target.cube_cores)}，"
          f"Vector 波 {ceil(vj/ref.vector_cores)}→{ceil(vj/target.vector_cores)}；970 活跃核代理 (C,V)={active}。"
          f"逻辑读写 {bs} B，阶段 DT 时间 {raw_us:.5f} µs，"
          f"外存权重 p=min(1,{bs}/(1000×{ref.hbm_bandwidth_gbs:g}×{raw_us:.5f}))={pressure:.5f}。"
          f"目标时价系数 `{' / '.join(Q)}` = `{' / '.join(f'{f[q]:.5f}' for q in Q)}`。")
    for unit in ('cube','vector'):
        peaks=[]
        for dt in w['cube_ops_by_dtype' if unit=='cube' else 'vec_ops_by_dtype']:
            bits=next((b for b in ('32','16','8','4') if b in dt),None)
            a,b=getattr(ref,unit+'_tops').get(bits),getattr(target,unit+'_tops').get(bits)
            if a and b:
                ca,cb=getattr(ref,unit+'_cores'),getattr(target,unit+'_cores')
                peaks.append(f'{dt} ({a:g}/{ca})/({b:g}/{cb})={(a/ca)/(b/cb):.5f}')
        if peaks:
            text+=f' {unit} 参考/目标每核峰值比（整片 TOPS÷同类核数）：'+', '.join(peaks)+'。'
    if notes:
        text+=' 有缺项时按相同每核每周期处理量回退：'+', '.join(sorted({x.split(':')[0] for x in notes}))+'。'
    return f,text


def _masks(t,work,fac):
    # Maximum core in this particular template and for this particular target.
    rows=[]
    for phases in t['cores']:
        terms=[(mask,h,max(work[q]/t['work'][q]*fac.get(q,1.) for q in mask.split('+'))) for mask,h in phases.items()]
        rows.append((sum(h*s for _,h,s in terms),terms))
    i=max(range(len(rows)),key=lambda i:rows[i][0])
    return i,*rows[i]


def _template(part,shape,attrs,ref,target,dt_us):
    model=_profile('results/v0.8/fusion/model.json');templates=model['templates'][part]
    ww=fusion.work(part,shape,attrs)
    child={'split_sink':'mhc_sinkhorn','split_post':'mhc_post'}.get(part,part)
    f,ft=_factors(part,child,shape,attrs,dt_us,ref,target)
    lines=[_units(part,shape,attrs),'',f"本例工作代理按 `{' / '.join(Q)}` 排列为 `{_numbers(ww)}`。SCALAR/IDLE 按波计。这是队列工作单位，而非有用运算数。",'',ft,'',
           '对每个 DT 参考 trace 的每个核，把队列活动集合不变的时间段记作 (S,h)，其中 h 是实测微秒。设 U 是本例的队列工作、U₀ 是表中参考工作，f 是上面目标时价（DT 时全取 1）。一个时间段耗时 h·max(q∈S){Uq/U₀q·fq}；同一核逐段相加、同一模板取最晚核、多个模板取中位数，偶数个取中间两值的平均。以下是实际参与该次计算的全部模板，序号按参数原顺序，核索引从 0 开始。','',
           '| 模板 / DT 参考 shape / DB SHA 前12位 | 参考 U₀（同上队列顺序） | DT 最晚核 / µs | 970 最晚核 / µs |',
           '|---|---|---:|---:|']
    tdt=[];t970=[];allterms=[]
    for i,t in enumerate(templates):
        idx,d,ds=_masks(t,ww,{});ix,v,ts=_masks(t,ww,f)
        tdt.append(d);t970.append(v)
        lines.append(f"| {i} / {t['shape']} / `{t['db_sha256'][:12]}` | {_numbers(t['work'])} | {idx} / {d:.5f} | {ix} / {v:.5f} |")
        # Experts can reproduce the complete critical-core subtotal without implying all cores coincide.
        allterms+=['',f'模板 {i}，970 核 {ix} 的逐项 (h × 最大缩放)，µs：','',
                   '；'.join(f'`{mask}` {h:.5f}×{s:.5f}' for mask,h,s in ts)+f'，和为 **{v:.5f}**。']
    lines+=['','<details><summary>展开各模板在 970 关键核上的实际并发项</summary>','',*allterms,'','</details>','',
           f"DT 候选中位数 {median(tdt):.5f} µs，目标候选中位数 {median(t970):.5f} µs。"
           f"参考价格、每个核的活动集合及完整 DB SHA 在 `results/v0.8/fusion/model.json → templates.{part}`。"]
    _check(median(tdt),dt_us)
    return lines,median(tdt),median(t970)


def _mhc(part,shape,attrs,ref,target,dt_us):
    branch,units=mhc_v9.partition(part,shape);coeff=mhc_v9.profile()['stages'][part][branch]
    t,n,d=shape;k=n*d
    if part=='pre':
        desc=(f'K=N·D={k}；K≥512 的 MK 分支按 Kc=256⌈K/(32×256)⌉={256*ceil(k/(32*256))}，'
              f'单核计算代理 ⌈T/16⌉⌈Kc/32⌉；K<512 的 ND 分支则用 ⌈T/32⌉⌈K/256⌉。'
              f'向量 ⌈T/64⌉⌈D/256⌉；流量 [4K(N²+2N)+2TK]/1024 KiB；控制 ⌈T/16⌉。')
    elif part=='sink':
        desc='固定 20 轮，向量代理 ⌈T/64⌉·20N²，搬运 4TN²/1024 KiB，控制 ⌈T/64⌉。'
    else:
        j=1
        while t<32 and ceil(d/j)>256 and t*j<=32:j*=2
        desc=f'短 token 沿 D 方向按 2 的幂切 {j} 份，w=⌈T×{j}/64⌉={ceil(t*j/64)}；向量 w⌈D/(64×{j})⌉N(N+1)，搬运 4TND/1024 KiB，控制 w。'
    key={'pre':'mhc_pre','sink':'split_sink','post':'split_post'}[part]
    child={'pre':'mhc_pre','sink':'mhc_sinkhorn','post':'mhc_post'}[part]
    f,ft=_factors(key,child,shape,{},dt_us,ref,target)
    qq=dict(compute='CUBE',vector='SIMD',traffic='MTE2',control='SCALAR')
    lines=[f'阶段 `{part}`、分支 `{branch}`。{desc}', '',ft,'',
        '下表每行的截距 a（µs）是可见固定成本；b（µs/单位）乘对应工作 x。DT 项=a+bx；970 项=a·fSCALAR+bx·f队列。各行对应互斥时间，不能再取最大值。','',
        '| 资源项 / 目标队列 | x | a | b | DT µs | 970 µs |','|---|---:|---:|---:|---:|---:|']
    dts=[];ts=[]
    for name,c in coeff.items():
        x=units[name];a=c['startup_us'];b=c['unit_us']
        u=a+b*x;v=a*f['SCALAR']+b*x*f[qq[name]];dts.append(u);ts.append(v)
        lines.append(f"| {name} / {qq[name]} | {x:g} | {a:.6f} | {b:.6f} | {u:.5f} | {v:.5f} |")
    lines+=['',f"参考参数：`results/v0.9/mhc/model.json → stages.{part}.{branch}`，归档窗口及身份见同目录 `freeze.json`。"
            f"本段加和 DT {sum(dts):.5f}、970 {sum(ts):.5f} µs。0 斜率只说明独占阶段未识别出这一项；依赖 Cube 的等待可以被计入控制。"]
    _check(sum(dts),dt_us);return lines,sum(dts),sum(ts)


def explain(op,shape,attrs=None):
    """Markdown text for a supported case; validates every final sum against the public API."""
    if op not in OPS:raise ValueError(f'No fusion example for {op}')
    shape=tuple(shape);attrs,dtype=_call_attrs(op,attrs or {},None)
    ref,target=load_chip(),load_chip('Ascend970DT')
    raw=predict(op,shape,attrs);dt=estimate(op,shape,attrs,chip=ref);future=estimate(op,shape,attrs,chip=target)
    if dt.latency_us is None or future.latency_us is None:
        return f"输入不在已实现的库合同内：{dt.reason or future.reason}；无有效时延，不建立数字推导。"
    lines=[f"该例为 `{op}{list(shape)}`，dtype={dtype}，属性 `{attrs}`。"+_physical(op,shape,attrs),'',
           f"DT 为 {ref.cube_cores}/{ref.vector_cores} 个 C/V 核、{ref.frequency_ghz} GHz，970 为 {target.cube_cores}/{target.vector_cores}、{target.frequency_ghz} GHz。"
           f"对固定参考库路线，计算价用每核峰值比 × 目标/参考波数比；控制价用时钟比 {ref.frequency_ghz:g}/{target.frequency_ghz:g}={ref.frequency_ghz/target.frequency_ghz:.5f}。搬运价=(1−p)·本地价+p·{ref.hbm_bandwidth_gbs:g}/{target.hbm_bandwidth_gbs:g}，缺 MTE 带宽时用时钟与波数作本地价。"
           '只有 Cube/Vector 总峰值且可比时才用其比值；混合 dtype 以各自工作量加权。该路线没有直接定价 970 的 L2 命中或重新编译 tiling；970 时间是 DT 阶段模板外推，不是独立的目标硬件资源仿真。']
    dtparts=[];parts=[]
    for part,dt_us in raw['stages_us'].items():
        lines+=['',f'**{part}：从输入到阶段净时延**','']
        sub,dt_part,v=(_mhc(part,shape,attrs,ref,target,dt_us) if part in ('pre','sink','post') else _template(part,shape,attrs,ref,target,dt_us))
        if op==fusion.CHAIN:lines+=[_physical(part,shape,attrs),'']
        lines+=sub;dtparts.append(dt_part);parts.append(v)
    _check(sum(dtparts),dt.latency_us);_check(sum(parts),future.latency_us)
    fmt=lambda z:' + '.join(f'{n:.5f}' for n in z)
    lines+=['',f'按实际阶段顺序相加：DT {fmt(dtparts)} = **{dt.latency_us:.5f} µs**；970 {fmt(parts)} = **{future.latency_us:.5f} µs**。'
            +('组合只含本 rank 的库 kernel 时间；跨卡通信与 kernel 之间的主机开销不在窗口内。' if len(parts)>1 else '窗口只含本 rank 的单个库 kernel。')]
    return '\n'.join(lines)
