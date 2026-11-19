"""Render arithmetic worked examples from the same public chip configurations."""
from math import ceil, log2
from pathlib import Path
import json

from operator_model import estimate,load_chip
from operator_model.backend import _evaluate
from operator_model.chips import REFERENCE_NAME
from operator_model import post_v12,rms_v12,vector_v8,vector_v9,wide_v10
from operator_model.post_v11 import cohort
from operator_model.rms_v11 import pressure
from operator_model.quant_resource import _costs as quant_costs
from operator_model.transfer import _queue_factors

ROOT=Path(__file__).resolve().parents[3]
PHASES=('arrival','SCALAR','SIMD','MTE2','MTE3','other')
UNITS={'arrival':'log₂(A)','SCALAR':'每核额外行','SIMD':'向量工作组','MTE2':'每核主输入 KiB',
       'MTE3':'全片主输出 KiB','other':'无可变单位'}


def _table(header,items):
    rows=['| '+' | '.join(header)+' |','|'+'|'.join('---' for _ in header)+'|']
    rows.extend('| '+' | '.join(str(v) for v in row)+' |' for row in items)
    return '\n'.join(rows)


def _profile(path):
    return json.loads((ROOT/path).read_text())


def _post(shape,attrs,raw):
    t,n,d=shape
    route,a,w,u=vector_v9.schedule('mhc_post',shape,attrs)
    j=a//ceil(t/w)
    family=cohort(route,t,a)
    p=post_v12.parameters()
    costs=p['phases'][route]
    times=raw['context_predictions_us']
    state=raw['context']
    txt=[f'按每核 W=ceil({t}/64)={w} 行分配。T<32 时从 J=1 起，只要 ceil(D/J)>256 且 T·J≤32 就把 J 翻倍。此例 J={j}，A=J·ceil(T/W)={a} 个 AIV，每片 d=ceil({d}/{j})={ceil(d/j)} 列。每条 64 列向量组要做 N 路残差混合与一路外部贡献。SIMD 单位 W·ceil(d/64)·N(N+1)={u["SIMD"]}，每核主读 2WNd/1024={u["MTE2"]:.4f} KiB。',
         '从 DT 最晚结束的 AIV 取各阶段暴露时间。43 个开发 shape 按分支校准，逐阶段作非负 soft-L1 拟合（转折 0.1 µs）。三截距 b 对应首次、扩核、复用，单位 µs，s 为 µs/工作单位，e 为 µs/多一行。每格使用 b+sX+e(W−1)。参数来自 results/v0.12/post/profile.json。']
    txt.append(_table(('阶段','b 首 / 扩 / 热 (µs)','s','X 与单位','e','本例阶段 µs'),
        [(q,' / '.join(f"{costs[q][k]:.5f}" for k in ('first','expand','reuse')),f"{costs[q]['service']:.6f}",
          f"{u[q]:.4f} {UNITS[q]}",f"{costs[q]['per_extra_row']:.5f}",f"{raw['stages_us'][q]:.5f}") for q in PHASES]))
    totals='、'.join(f'{times[k]:.4f}' for k in ('first','expand','reuse'))
    if state=='relative_minimax':
        lo,hi=min(times.values()),max(times.values())
        choose=f'缺省 resident_aiv 的切通道组取两端相对误差相等的点 2×{lo:.4f}×{hi:.4f}/({lo:.4f}+{hi:.4f})。慢、快阶段分别按 {lo/(lo+hi):.4f}、{hi/(lo+hi):.4f} 加权。'
    elif 'resident_aiv' in attrs:
        choose=f'前序覆盖核数 H={attrs["resident_aiv"]}，相对于 A={a} 选择 {state}。'
    else:
        cnt=p['context_counts'][family]
        choose=f'该分组首/扩/热的观察次数为 {cnt["first"]}/{cnt["expand"]}/{cnt["reuse"]}，按状态总时长排序取频数加权中位数，选中 {state}。'
    txt.append(f'首/扩/热总设备时间分别 {totals} µs。{choose}')
    return txt


def _rms(shape,raw):
    t,d=shape
    _,a,w,u=vector_v9.schedule('rmsnorm',shape,{})
    data=rms_v12.parameters()
    old=wide_v10.parameters()['phases']['rmsnorm']
    weights=raw['context_weights']
    state=tuple(weights[k] for k in ('first','expand','reuse'))
    g=ceil(d/128)
    near=pressure(a,w,data['window_bytes'])
    timers=data['ticks_per_us']
    intercept=sum(x*y for x,y in zip(state,data['simd'][:3]))
    txt=[f'每个 token 独立按 {d} 列归约再归一化。W=ceil({t}/64)={w} 行/最忙核，A=ceil({t}/{w})={a} 核。每行 {g} 个 128 列组，SIMD 共 W×{g}={u["SIMD"]} 组。',
         'arrival、SCALAR、MTE2、other 使用 23 条 DT 开发 trace 的逐阶段暴露时长，按状态拟合固定截距与工作项（非负 soft-L1，尺度 0.12 µs）。未指定覆盖状态时，把开发序列的前序 AIV 高水位归入首次、扩核、复用三组。本例的三项权重为 '+', '.join(f'{x:.4f}' for x in state)+'。参数与先验来自 results/v0.10/wide/profile.json。']
    txt.append(_table(('阶段','三态截距 b (µs)','服务斜率 s (µs/单位)','本例 X','Σ权重·b+sX (µs)'),
        [(q,' / '.join(f"{old[q][k]:.5f}" for k in ('first','expand','reuse')),
          f"{old[q]['service']:.6f}",f"{u[q]:.4f} {UNITS[q]}",f"{raw['stages_us'][q]:.5f}")
        for q in ('arrival','SCALAR','MTE2','other')]))
    params=data['simd']
    txt.append(f'SIMD 和写回另用 43 条 DT 临界 AIV 时间线，按实际 SIMD、rstd 写、y 写和两段计算边界作非负 soft-L1 拟合（转折 80 周期）。SIMD 的三态截距为 {params[0]:.3f}/{params[1]:.3f}/{params[2]:.3f} 周期，组增量为 {params[3]:.4f} 周期/组。加权截距 {intercept:.3f} 加 {u["SIMD"]}×{params[3]:.4f}，除 {timers} 周期/µs 得 {raw["stages_us"]["SIMD"]:.5f} µs。')
    m=16*ceil(d/16)
    fold=1<<((m-1).bit_length()-1)
    aux=32*ceil(ceil(fold/64)/32)
    cap=(253952-1024-2048-2*m)//(8*m+12+aux)
    txt.append(f'每轮先写 FP32 行统计量，再写 BF16 y。16 列对齐宽度 {m}，归约折叠宽度 {fold}，辅助区单位 {aux}，UB 每轮容纳 floor((253952−1024−2048−2×{m})/(8×{m}+12+{aux}))={cap} 行，实际分块 {raw["ub_tile_rows"]}。统计量共享的有效窗口为 128 B；每行 4 B，32 行可落同一窗口。邻近写者代理 E=max(0,min(A,32/W)−1)·min(1,A/32)={near:.4f}。')
    rstd0,rstd1=data['rstd']; y0,y1=data['y']
    n0,n1=data['normalize']; z0,z1=data['reduce']
    spans=[]
    cursor=0.
    for i,n in enumerate(raw['ub_tile_rows']):
        x=n*g
        start=cursor+(z0+z1*x if i else 0.)
        rend=start+rstd0+rstd1*near
        cursor=start+n0+n1*x
        yend=max(rend,cursor+y0+y1*(2*d*n/1024))
        spans.append((i+1,n,x,4*n,2*d*n,f'{start:.1f} → {rend:.1f}',f'{cursor:.1f} → {yend:.1f}'))
    txt.append(f'相对首个 rstd 发起的周期轴上，第 2 轮起先加 reduce={z0:.2f}+{z1:.4f}×组数，rstd 服务={rstd0:.2f}+{rstd1:.3f}×E，normalize={n0:.2f}+{n1:.4f}×组数。y 的服务={y0:.2f}+{y1:.4f}×y字节/1024，完成时刻取 rstd 完成和 y 自身完成的较晚值。y 参数仅用可辨识的 17 笔写拟合。来源 results/v0.12/rms/profile.json。')
    txt.append(_table(('轮次','行','组','rstd B','y B','rstd 周期','y 周期'),spans))
    txt.append(f'合并写区间共 {raw["write_union_ticks"]:.2f} 周期，与 SIMD 重叠 {raw["write_simd_overlap_ticks"]:.2f} 周期（以上限 SIMD 忙时裁剪），暴露 MTE3=两者之差/{timers}={raw["stages_us"]["MTE3"]:.5f} µs。')
    return txt


def _linear(op,shape,attrs,raw):
    if op=='kda_recurrent':
        _,s,_,_,k,v=shape
        _,u,w=vector_v8.features(op,shape,attrs)
        p=_profile('results/v0.8/vector/profile.json')[op]
        fields=[(q,p['phases'][q]['startup_us'],p['phases'][q]['service_us'],
                 vector_v8.phase_work(q,1,u,w,op)) for q in PHASES]
        intro=f'一条状态链占一个 AIV。每 token 更新 {k}×{v} 状态，{s} 个 token 顺序执行。128×128 作为一个工作单位，U={s}×{k}×{v}/16384={u:g}。除 arrival=log₂(1)=0，计算和控制阶段都用 U，搬运用 U·log₂(1+1)=U。'
        source='三个 DT 开发 DB 取临界 AIV 暴露时间，分别拟合非负截距 a 和斜率 b；a 单位 µs，b 单位 µs/状态工作单位，arrival 用 µs/log₂(核数)。此模型的截距包含在整次调用内，只支付一次。来源 results/v0.8/vector/profile.json。'
    else:
        t,n=shape
        _,a,w,u=vector_v9.schedule(op,shape,attrs)
        i=attrs.get('num_iters',20)
        p=vector_v9.parameters()['phases'][op]
        fields=[(q,*p[q],u[q]) for q in PHASES]
        intro=f'{t} 个 token 的 {n}×{n} FP32 矩阵独立，{i} 轮归一化存在顺序依赖。W=ceil({t}/64)={w} 行/最忙核，A=ceil({t}/{w})={a} 核。SIMD 单位为 {i}×ceil(16×{w}/64)={u["SIMD"]} 个矩阵向量组。MTE2 取 {w}×64/1024={u["MTE2"]:.4f} KiB/核，MTE3 取 {t}×64/1024={u["MTE3"]:.4f} KiB/芯片。arrival 用 log₂(A)，SCALAR 用 W−1，other 为常数。'
        source='13 个旧 DT shape 的每阶段暴露时长单独拟合 a+bX，a 单位 µs，b 分别是 µs/向量组、µs/KiB 等对应单位。采用非负 soft-L1（转折 0.15 µs），输入工作长期不变时令 b=0、a 取阶段中位数。来源 results/v0.9/vector/profile.json。'
    return [intro,source,_table(('阶段','a µs','b µs/单位','X','a+bX µs'),
                                [(q,f'{a:.5f}',f'{b:.5f}',f'{x:.5f}',f"{raw['stages_us'][q]:.5f}")
                                 for q,a,b,x in fields])]


def _projection(op,shape,attrs,raw,dt,target):
    chip=load_chip()
    future=load_chip('Ascend970DT')
    notes=[]
    f,_=_queue_factors(op,shape,attrs,raw,target.work,chip,future,notes)
    chipbytes=target.work['logical_read_bytes']+target.work['logical_write_bytes']
    pressure=min(1,chipbytes/(1000*chip.hbm_bandwidth_gbs*raw['predicted_us']))
    mapping=target.projection.get('resident_aiv_mapping')
    extra=f' 已将目标前序覆盖 {mapping["target"]} 核换算为 DT 覆盖 {mapping["reference"]} 核。' if mapping else ''
    lines=[f'970DT 的算子实现与分片延用 DT 模板。先算 DT 各段的暴露时间，共 {sum(raw["stages_us"].values()):.5f} µs。设备配置取 configs/chips/Ascend970DT.json。目标有 120 个 AIV、1.8 GHz，本例预计用 {target.active_aiv} 核。控制与到达按频率比 1.65/1.8={f["SCALAR"]:.5f}；SIMD 按实际 dtype 的每核峰值比及波数，倍率 {f["SIMD"]:.5f}。{extra}',
           f'每次调用逻辑总读写 {chipbytes} B，芯片级带宽占比代理 p=min(1,{chipbytes}/(1000×4000×{raw["predicted_us"]:.5f}))={pressure:.5f}。MTE2/MTE3 倍率是 (1−p)×局部价格比+p×4000/14400，得到 {f["MTE2"]:.5f}/{f["MTE3"]:.5f}；局部部分在未指定两端 MTE 速率时取频率比×波数比。该项使用全片逻辑读写量作代理。']
    projected=target.projection['stages_us']
    if op=='rmsnorm':
        hidden=min(raw['stages_us']['SIMD'],raw['write_simd_overlap_ticks']/1650)
        physical=raw['write_union_ticks']/1650*f['MTE3']
        lines.append(f'RMSNorm 在 970 上将写区间并集单独重定价为 {raw["write_union_ticks"]:.2f}/1650×{f["MTE3"]:.5f}={physical:.5f} µs，可隐藏项为 min({physical:.5f},{hidden:.5f}×{f["SIMD"]:.5f},{projected["SIMD"]:.5f})，其差才是暴露的写回。')
    lines.append(_table(('阶段','DT 暴露 µs','目标价格倍率','970 外推 µs'),
                        [(q,f"{raw['stages_us'][q]:.5f}",f'{f[q]:.5f}',f'{projected[q]:.5f}') for q in PHASES]))
    lines.append(f'六段相加得到 DT {dt.latency_us:.5f} µs、970DT {target.latency_us:.5f} µs。arrival 是最晚结束核的到达偏移，other 是未归入已知流水的区间；DT 原始重叠已按 SIMD→MTE3→MTE2→SCALAR 唯一归属，合计时只算一次。')
    return lines


def _quant(shape):
    p=quant_costs()
    lines=[f'单卡 BF16 输入为 {shape[0]}×{shape[1]}，按连续 32 个数共用一份 E8M0 scale，FP8 数据逐元素输出。输入读 {2*shape[0]*shape[1]} B，输出数据 {shape[0]*shape[1]} B、scale {shape[0]*ceil(shape[1]/32)} B。',
           '分核沿用 950 尾轴实现。先测试 R·ceil(C/512)≤0.75×核数，成立则以 256 列为一个工作片，否则用 512 列。行、列切分在目标 AIV 数内最小化空片；UB 取本芯片容量，预留 2048 B，剩余空间按八组一份的双缓冲配额选列块与行块。以下两台目标都直接按资源重新排程。',
           '来源为 results/v0.17/quant.json 和 quant-dt-evidence.json。13 个 DT shape 的首读、VF、两笔写与尾部指令用 1650 周期/µs 换算。全程按阶段校准。启动对 log₂(活跃核数) 拟合；首读截距由第一笔数据读估计，长首读最快每核斜率作为可达速率；连续多读用相邻发起测间隔。向量每循环按 VF 发起至首写区间拟合。数据写用非负仿射拟合；无共享 scale 观测给出启动/尾部中位数，按共享生产者数与并发压力拟合剩余 scale 服务。']
    for chip in map(load_chip,('Ascend950DT_9582','Ascend970DT')):
        result=estimate('quant',shape,chip=chip,timing_mode='resource')
        d=result.projection['timing_diagnostics']; plan=d['target_plan']; s=d['first_tile_cycles']
        f=1000*chip.frequency_ghz
        a=plan['active_aiv']
        r,c=plan['rows_per_core'],plan['cols_per_core']
        tr,tc=plan['tile']; h,w=min(r,tr),min(c,tc)
        effective=shape[0]*shape[1]/(r*c);share=d['shared_bytes_per_cycle']/effective
        sched=d['stages_us']
        iterations=plan['iterations']
        lines.append(f'**{chip.name}**。配置含 {chip.vector_cores} 个 AIV、{chip.frequency_ghz:g} GHz、{chip.vector_tops["16"]} 个 BF16 TOP/s 与 {chip.hbm_bandwidth_gbs:g} GB/s 显存带宽。选 {plan["slice_cols"]} 列工作片，A={a}，最忙核负责 {r}×{c}，UB tile={tr}×{tc}，循环 {iterations} 轮。等效满载核 R·C/({r}×{c})={effective:.4f}。全片 min(HBM,L2)/GHz={d["shared_bytes_per_cycle"]:.3f} B/周期，平分到每个等效核为 {share:.3f} B/周期。')
        lines.append(f'首块 {h}×{w}，g={h}×ceil({w}/32)={s["groups"]} 组，读 x={s["read_bytes"]} B，FP8 写 y={h*w} B、scale 写 g={s["groups"]} B。n=ceil(g/8)+ceil(g/128)={s["loops"]} 轮。每核向量峰值 P=1000×TOPS/(核数×GHz)={d["vec_ops_per_cycle_per_core"]:.3f} 等效操作/周期，校准每轮 κ={d["effective_vec_ops_per_loop"]:.2f} 等效操作。')
        lines.append(_table(('首块项目','数值代入（周期）'),[
            ('启动',f'{p["launch_cycles"]:.2f}+{p["arrival_log2_cycles"]:.3f}×log₂({a})={sched["dispatch"]*f:.2f}'),
            ('读就绪',f'max({p["read_latency_cycles"]:.2f}+{s["read_bytes"]}/{d["read_bytes_per_cycle_per_core"]:.3f}, {s["read_bytes"]}/{share:.3f})={s["read_latency"]:.2f}'),
            ('相邻读发起',f'{s["read_bytes"]}/min({p["read_issue_bytes_per_cycle"]:.3f},{share:.3f})={s["read_ii"]:.2f}'),
            ('VF 就绪至首写',f'{p["vf_dispatch_cycles"]:.2f}+{p["vf_dependency_cycles"]:.2f}+{s["loops"]}×κ/P={s["vector"]:.2f}'),
            ('数据写 D',f'{p["write_latency_cycles"]:.2f}+{h*w}/{d["write_bytes_per_cycle_per_core"]:.3f}={s["data_write_cycles"]:.2f}'),
            ('scale 写 S',f'max({p["scale_base_cycles"]:.2f}+{s["groups"]}/{d["write_bytes_per_cycle_per_core"]:.3f},D+{p["scale_tail_cycles"]:.2f}) + {p["scale_peer_cycles"]:.2f}×({d["scale_peak_writers"]}−1)×log₂({a})={s["scale_write_cycles"]:.2f}'),
            ('写窗口',f'max(D, {p["scale_issue_gap_cycles"]:.2f}+S, {s["write_bytes"]}/{d["write_bytes_per_cycle_per_core"]:.3f})={s["write"]:.2f}')]))
        if iterations==1:
            schedule=f'单轮流水为读就绪+VF+写窗口={s["read_latency"]+s["vector"]+s["write"]:.2f} 周期。'
        else:
            schedule=(f'完整行块 {r//tr}、余行 {r%tr}，每行块完整列块 {c//tc}、余列 {c%tc}。尾块按实际 h、w 重算上表。'
                      '设第 i 块发读 Uᵢ、结束计算 Vᵢ、结束写 Eᵢ，两槽复用使 Uᵢ=max(Uᵢ₋₁+Iᵢ₋₁,Vᵢ₋₂)，Vᵢ=max(Uᵢ+Lᵢ,Vᵢ₋₁,Eᵢ₋₂)+Sᵢ，Eᵢ=max(Vᵢ,Eᵢ₋₁)+Wᵢ；此处 I/L/S/W 分别为相邻读发起、读就绪、VF 和写窗口。')
        lines.append(schedule+f'逐块排程与全片总 I/O 下限 {d["chip_io_floor_cycles"]:.2f} 周期取大，得到 {sched["ub_pipeline"]*f:.2f} 周期；最后加 {p["end_cycles"]:.2f} 周期。启动、流水、尾部分别为 {sched["dispatch"]:.5f}+{sched["ub_pipeline"]:.5f}+{sched["drain"]:.5f}={result.latency_us:.5f} µs。')
    lines.append('scale 的写者数由连续 32 B 范围内不同 AIV 的输出地址所有权求得；这是短片段竞争代理。动态共享端口无法从 DMA 完成事件唯一辨识。970 为假设同实现的资源外推。')
    return lines


def explain(op,shape,attrs=None):
    """Return checkout-generated Markdown for one of the five Vector operators."""
    attrs=dict(attrs or {})
    if op=='quant':
        return '\n\n'.join(_quant(shape))
    dt,raw=_evaluate(op,shape,attrs,REFERENCE_NAME,None,None)
    target,reference=_evaluate(op,shape,attrs,REFERENCE_NAME,'Ascend970DT',None)
    if dt.latency_us is None or target.latency_us is None:
        raise ValueError((op,shape,dt.reason,target.reason))
    if op=='rmsnorm':
        lines=_rms(shape,raw)
    elif op=='mhc_post':
        lines=_post(shape,attrs,raw)
    elif op in ('mhc_sinkhorn','kda_recurrent'):
        lines=_linear(op,shape,attrs,raw)
    else:
        raise ValueError(op)
    lines+=_projection(op,shape,attrs,reference,dt,target)
    return '\n\n'.join(lines)
