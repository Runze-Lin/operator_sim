"""Numerically reconstruct an RMSNorm resource estimate in Chinese for one case."""
from math import ceil, isclose

from operator_model import estimate, load_chip
from operator_model.rms_v11 import pressure
from operator_model.rms_v12 import parameters


def explain(op, shape, attrs=None):
    if op != 'rmsnorm':
        raise ValueError('Expected RMSNorm')
    tokens,width=shape
    text=[f'本卡 BF16 输入 {tokens} 行，每行 {width} 列，输出包含每行 1 个 FP32 rstd。'
          f'冷输入和 gamma 共 2×{width}×({tokens}+1)={2*width*(tokens+1)} B，'
          f'输出 2×{tokens}×{width}+4×{tokens}={2*tokens*width+4*tokens} B。'
          '每行先规约求逆均方根，再归一化并乘 gamma，因此统计量写与主输出写可以部分重叠。',
          '', '| 芯片 | AIV预算 / 活跃 | 最忙核行数 / UB 各轮 | SIMD组 | 邻近统计量写者代理 | 写在途 / 可隐藏周期 | 首次 / 扩核 / 热态（µs） | 返回 µs |',
          '|---|---:|---|---:|---:|---:|---|---:|']
    details=[]
    for soc in ('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'):
        chip=load_chip(soc)
        r=estimate(op,shape,attrs or {},chip=chip,timing_mode='resource')
        if r.latency_us is None:
            text.append(f'| {soc} | — | {r.reason} | — | — | — | — | — |')
            continue
        d=r.projection['timing_diagnostics'];p=d['target_plan'];rows=p['rows_per_core']
        a=r.active_aiv;near=pressure(a,rows,parameters()['window_bytes'])
        state=d['conditional_us'];st=d['stages_us'];context=d['residency']
        times=' / '.join(f"{state[k]:.4f}" for k in ('first','expand','reuse'))
        text.append(f'| {soc} | {chip.vector_cores} / {a} | {rows} / {p["ub_tile_rows"]} | {p["vector_groups"]} | {near:.3f} | {d["store_union_cycles"]:.2f} / {d["store_eligible_overlap_cycles"]:.2f} | {times} | {r.latency_us:.4f} |')
        parts=' + '.join(f'{k} {v:.5f}' for k,v in st.items())
        details.append(f'**{soc}**。W=⌈{tokens}/{chip.vector_cores}⌉={rows}，A=⌈{tokens}/{rows}⌉={a}，'
                       f'组数 W⌈{width}/128⌉={rows*ceil(width/128)}。统计量写描述符服务为 '
                       f'335.886+117.018×{near:.3f} 周期，主输出写按最忙核各轮字节计价。'
                       f'变动的向量服务按同频每核峰值比 {d["vector_price_cycles_per_reference_cycle"]:.5f} 缩放。'
                       f'全片冷 I/O 窗口下界 {d["shared_io_floor_cycles"]:.2f} 周期。'
                       f'本次状态为 {context}，互斥阶段（µs）为 {parts} = {r.latency_us:.5f}。')
        assert isclose(sum(st.values()),r.latency_us,abs_tol=1e-10)
    text.extend(['','未提供 resident_aiv 时以三种条件时间的最小值 l、最大值 u 计算 2lu/(l+u)，使两个端点的相对误差对称；给出状态时直接选择对应条件。',*['\n'+s for s in details],
                 '', '阶段截距和 128 B 统计量写者模型来自 DT 临界核 trace。32/56/120 等活跃核数均由输入分区计算；970 只有情景预测。'])
    return '\n'.join(text)
