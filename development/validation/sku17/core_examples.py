"""Numerical substitutions for the three remaining timed schedules and ChunkKda."""
from operator_model import estimate, load_chip
from operator_model.backend import _evaluate
from operator_model.addmm_v9 import profile
from operator_model.transfer import _queue_factors
from operator_model.matmul_resource import _costs
from operator_model.fa_resource import _costs as fa_params

DT='Ascend950DT_9582'
PR='Ascend950PR_957b'
FUT='Ascend970DT'


def terms(d):
    return '，'.join(f'{k}={v:.3f}' for k,v in d.items())


def matmul(op,s,a):
    m,n,k=s
    lines=[f'每个输出是一行 A 和一列 B 的点积。有用工作 2MNK=2×{m}×{n}×{k}={2*m*n*k:,} ops。每个 BF16 元素占 2 B。先用目标核数选择 ASW 主块，再按真实边界拆最后一波；每个输出子块独立遍历 K。']
    c=_costs()
    lines.append(f'950DT 指令 trace 定价的首读、搬入 L0、末条 MMAD、排空分别为 {c["read_latency_cycles"]:.2f}、{c["l0_latency_cycles"]:.2f}、{c["mmad_latency_cycles"]:.2f}、{c["drain_cycles"]:.2f} cycles；读和 MMAD 发起下限分别 {c["read_ii_cycles"]:.2f}、{c["mmad_ii_cycles"]:.2f} cycles。各芯片继承这些周期；按目标的频率和每核 BF16 峰值重算服务。')
    for chip in (PR,FUT):
        e=estimate(op,s,a,chip=chip,timing_mode='resource')
        d=e.projection['timing_diagnostics']; p=d['target_plan'];r=d['rates'];share=d['shared_traffic']
        lines.extend([f'#### {chip} 的输出网格与搬运',f'主块 {p["tile"]}，父块 {p["output_tiles"]} 个，分给 {e.active_aic} AIC，共 {p["waves"]} 波。最后一波沿 M/N 切数 {p["tail_split_mn"]}，实际子块 [M,N,重复数] 为 {p["tail_wave_histogram"]}。全程 {d["mmads_total"]} 次 MMAD，每个输出块分 {d["pages_per_output_tile"]} 个读页。',
        f'目标吞吐为 {terms(r)}。前三项是每核 B/cycle，L2、HBM 为全片 B/cycle，Cube 为每核 ops/cycle。每波读取按 max(最重本地描述符字节/MTE2，各核合计 L2 字节/L2，暴露的 HBM 字节/HBM)；Cube 操作数按 16 对齐后除每核算力。写出同样取本地 FIX、L2、暴露的 HBM 中最慢者。全程 L2 读 {d["l2_to_l1_read_bytes"]:,} B、HBM 场景读 {d["hbm_read_bytes_scenario"]:,} B，输出 {d["write_bytes"]:,} B，其中预计本次设备窗内 HBM 写 {share["hbm_write_bytes_exposed"]:,.0f} B。输出驻留比例上限 {share["output_retained_fraction_assumption"]}；实际 cache 命中率未知。',
        'R/L 分别为首读和 L0 准备，P 为随后页的累计间隔，C 为末页计算，W 为写完成；Ri/Ci/Wi 分别为连续读、计算、写的服务间隔，单位周期。'])
        for which,x in d['wave_cost_cycles'].items():
            if x:lines.append(f'{which} 波：{terms({k:v for k,v in x.items() if k!="write_bytes"})}。')
        lines.append(f'后波最早计算=max(本波就绪，上波计算资源可用)，最早写=max(本波计算完成，上波写端口可用)。两预取槽还限制读间隔≥(R+L)/2。下表各项已折成 µs，共享链路仅计超过流水的差额。{terms(d["stages_us"])}；相加为 **{e.latency_us:.5f} µs**。有效 HBM payload {share["hbm_effective_gbs"]:.2f} GB/s，占标称 {share["hbm_payload_fraction_of_nominal"]:.2%}。')
    lines.append('[参数及原始 DT trace 来源](../../results/v0.16/matmul-dt-provenance.json) · [末波调度和参数敏感性](../../results/v0.17/matmul-method.md)')
    return '\n\n'.join(lines)


def fa(op,s,a):
    q,kv,d,b,h=s
    lines=[f'Q、K、V 为 BF16 稠密 SBH。QK 与 PV 各需 2BHQ·KV·D，合计 {4*b*h*q*kv*d:,} 个有用 Cube ops。模型按实际尾块与 16 对齐计算硬件工作。']
    for chip in (DT,FUT):
        e=estimate(op,s,a,chip=chip,timing_mode='resource');z=e.projection['timing_diagnostics'];p=z['target_plan']
        pars=fa_params()['routes']['wide' if p['tile'][1]==256 else 'short']
        lines+=[f'#### {chip} 的首块、循环和排空',f'主 tile [Q,KV]={p["tile"]}；并行任务 B·H·ceil(Q/tq)={p["jobs"]}，{p["waves"]} 波，每波遍历 ceil(KV/tk)={p["iterations"]} 块。AIC/AIV={e.active_aic}/{e.active_aiv}。Cube {z["cube_ops_per_cycle"]:.3f} ops/(cycle·AIC)，Vector 的 softmax 指令混合折合 {z["vector_equiv_ops_per_cycle"]:.3f} 等效操作/(cycle·AIV)，由 DT 指令流校准后按目标 FP32 每核峰值缩放。']
        for w in z['wave_types']:
            f=w['first'];last=w['last'];steps=p['iterations'];count=w['waves']
            lines.extend([f'这一类波重复 {count} 次；每块 {w["q_rows"]} 个 Q 行，每个活跃 AIC 配 {w["aiv_per_aic"]:.2f} AIV，K 读 {f["k_bytes"]:,} B、V 同量。每核可用读带宽 min(本地 MTE2, 共享带宽/占用核)={w["read_bytes_per_cycle"]:.3f} B/cycle。首读基准延迟 {pars["read_intercept"]:.2f} cycles，Q 和 K 就绪后再搬 L0。K/V 首就绪 {w["k_ready_cycles"]:.2f}/{w["v_ready_cycles"]:.2f} cycles；首次 QK/PV 开始 {w["qk_start_cycles"]:.2f}/{w["pv_start_cycles"]:.2f} cycles。首块结束为 PV 开始加 {f["pv"]:.2f} cycles。',
            f'每个完整 KV 块各资源的发起服务（cycles）为 {terms(f["demands_cycles"])}。II=max(这些服务)={f["ii_cycles"]:.2f} cycles，受 {f["steady_bottleneck"]} 限制；尾块 II={last["ii_cycles"]:.2f}。首块之后累计 max(0,{steps}−2)×完整块 II 加一个尾块 II（仅当有第二块）。写侧按输出 {w["output_write_bytes"]} B / {w["write_bytes_per_cycle"]:.3f} B/cycle 加末次 Vector 和控制排空。'])
        lines.append(f'{terms(z["stages_us"])} µs，相加为 **{e.latency_us:.5f} µs**。')
    lines.append('[DT 指令发起、读延迟和队列吞吐参数](../../results/v0.16/fa.json) · [改变绑定方式及 Q、KV 后的瓶颈](../../results/v0.17/fa-scenarios.md)')
    return '\n\n'.join(lines)


def addmm(op,s,a):
    dt,r=_evaluate(op,s,a,DT,None,None)
    p=r['plan'];h=profile();u=p['units']
    lines=[f'输入 A[M,K]、B[K,N] 和列偏置 [N]。输出[M,N]共 {s[0]*s[1]:,} 项，每项含一次融合偏置加。主 tile={p["tile"]}、分支={p["branch"]}、{p["active_aic"]} AIC、{p["waves"]} 波。读处理量含偏置的 2·min(N,tileN) B。每队列服务用 950DT 单核 trace 区间拟合 a+bX（µs）。',
    '| 队列 | X（读写 KiB、Cube 以 131072 ops 为一单位） | a (µs) | b (µs/单位) | 服务 (µs) |','|---|---:|---:|---:|---:|']
    for q,unit in [('CUBE','cube'),('MTE2','read'),('MTE1','read'),('FIXP','fix')]:
        c=h['queue_costs'][q]
        lines.append(f'| {q} | {u[unit]:.5f} | {c["intercept_us"]:.5f} | {c["slope_us_per_unit"]:.6f} | {r["stage_services_us"][q]:.5f} |')
    lines+=['',f'控制底噪 {r["control_us"]:.5f} µs，核到达偏斜 {h["arrival_us_per_log_core"]:.5f}·log₂({p["active_aic"]})={r["arrival_us"]:.5f} µs，额外队列暴露系数 η={h["exposed_fraction"]:.5f}。全时长=控制+偏斜+波数×最长服务+η(服务和−最长服务)={dt.latency_us:.5f} µs。η 来自关键核 trace 上各队列活动并集，描述该实现的重叠程度。']
    for name in (PR,FUT):
        target=load_chip(name);f,_=_queue_factors(op,s,a,r,dt.work,load_chip(DT),target,[])
        e=estimate(op,s,a,chip=name)
        lines+=['',f'{name} 按目标频率、每核峰值、有限任务波数和访存压力缩放服务：{terms({q:f[q] for q in ("CUBE","MTE2","MTE1","FIXP","SCALAR")})}。重算最长队列后 {terms(e.projection["stages_us"])} µs，总计 **{e.latency_us:.5f} µs**。']
    lines+=['','[DT 各阶段参数及训练 trace](../../results/v0.9/addmm/profile.json)']
    return '\n'.join(lines)


def chunk(op,s,a):
    _,r=_evaluate(op,s,a,DT,None,None)
    plan=r['structural_plan'];p=plan['tiling'];b,t,hq,hv,k,v=s
    lines=[f'块长 {a["chunk_size"]}，每序列 {p["chunks_per_sequence"]} 块，B×Hv={b}×{hv}={p["state_chains"]} 条独立状态链。每条链的 FP32 状态为 K×V，总状态 {4*b*hv*k*v:,} B。实块长 L={t} 时，每头保留下三角 QK 点积 L(L+1)/2、严格下三角 KK 点积 L(L−1)/2，各点积 2K 次操作。',
    '', '| 阶段 | 依赖 | 资源 | 本例工作（标量等效 ops） |','|---|---|---|---:|']
    for stage in r['stages']:
        lines.append(f'| {stage["id"]} | {", ".join(stage["dependencies"]) or "输入"} | {stage["resource"]} | {stage["work"]:,} |')
    lines+=['','其中 WU 和 finalize 是稠密等效工作。状态跨块串行；三角求逆等未取得执行级映射。依赖图可以输出，设备时间、实际分核、稳态发起间隔和计时误差均为 N/A。','', '```mermaid','flowchart LR','  QK --> KK --> WU --> StateUpdate --> Decay --> Finalize','```','','[归档结构与验证证据](../../results/v0.8/cube/chunk-structure-validation.json)']
    return '\n'.join(lines)
