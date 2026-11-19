"""Render a worked Quant resource schedule on all three configured chips."""
from math import ceil
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate,load_chip
from operator_model.quant_fragments import fragment_costs
from operator_model.quant_resource import _costs


def explain(op,shape,attrs=None):
    if op!='quant':
        raise ValueError('Expected quant')
    rows,cols=shape;attrs=dict(attrs or {})
    p=_costs();frag=fragment_costs()
    out=[f'BF16 [{rows},{cols}] 读 {2*rows*cols:,} B，E4M3FN 数据写 {rows*cols:,} B，E8M0 scale 写 {rows*2*ceil(cols/64):,} B。每 32 个数一组，偶数个 scale 槽构成一行。']
    for name in ['Ascend950DT_9582','Ascend950PR_957b','Ascend970DT']:
        chip=load_chip(name)
        r=estimate(op,shape,attrs,chip=chip,timing_mode='resource')
        if r.latency_us is None:
            out.append(f'**{name}** {r.status}，{r.reason}。')
            continue
        d=r.projection['timing_diagnostics'];s=d['first_tile_cycles'];w=d['fragment_wave']
        plan=d['target_plan'];rr,cc=plan['rows_per_core'],plan['cols_per_core'];tr,tc=plan['tile']
        price=1000*chip.frequency_ghz;stages=d['stages_us']
        out.extend([f'**{name}** 频率 {chip.frequency_ghz} GHz，AIV {chip.vector_cores} 个，'+
                    ('970DT 为无目标 CA 的情景外推。' if name=='Ascend970DT' else '下列时间是资源模型预测。'),
                    f'分核的基础列宽 {plan["slice_cols"]}，每核最多 {rr}×{cc}，实际 {r.active_aiv} 核。UB tile {tr}×{tc}，最大 {plan["iterations"]} 轮。'])
        out.append(f'关键首块读/数据写/scale 写 {s["read_bytes"]}/{s["write_bytes"]-s["scale_bytes"]}/{s["scale_bytes"]} B。读端每核 {d["read_bytes_per_cycle_per_core"]:.2f} B/cycle、整片有效上限 {d["shared_bytes_per_cycle"]:.2f} B/cycle，首响应=max({s["local_read_cycles"]:.2f}, {s["shared_read_floor_cycles"]:.2f})={s["read_latency"]:.2f} cycles，相邻发读间隔 {s["read_ii"]:.2f} cycles。VF 工作 {s["loops"]} 轮 × {d["effective_vec_ops_per_loop"]:.2f} 等效操作/轮，目标每核 {d["vec_ops_per_cycle_per_core"]:.2f} 操作/cycle。')
        if w:
            c=w['critical'];i=c['rank']
            out.append(f'短 scale 描述符共享 {w["shared_grains"]} 个 32 B 粒度。各核提交顺序用行列坐标的秩近似。关键核起点在行 {c["row"]}、列 {c["col"]}，秩 {i}，它在首个核之后 {c["read_issue_cycles"]:.2f} cycles 发读。前端准备与读并行，较慢者就绪才开始 SIMD。按 DT 指令间隔估计的前端 {c["frontend_cycles"]:.2f} cycles，读完后到数据写发起 {s["vector"]:.2f} cycles。')
            out.append(f'关键 scale 请求所触粒度的生产者数 {c["grain_owners"]}。先在每个粒度上按 {w["grain_ii_cycles"]:.2f} cycles 发起节拍排队，再将各描述符按最早可完成时刻排序，按共享返回节拍 {w["fabric_ii_cycles"]:.2f} cycles 排队。该请求单独可在 {c["scale_ready_cycles"]:.2f} cycles 完成，排队后为 {c["scale_end_cycles"]:.2f}，增加 {c["extra_queue_cycles"]:.2f}。两笔写窗口 {s["write"]:.2f} cycles。时间原点是最早的 payload 读，候选关键核是模型选择。')
        else:
            out.append(f'沿用双槽 UB 流水。VF 服务 {s["vector"]:.2f} cycles，数据写 {s["data_write_cycles"]:.2f}，scale 写 {s["scale_write_cycles"]:.2f}，发起差 {s["scale_issue_gap_cycles"]:.2f}。粒度最大写者 {d["scale_peak_writers"]}，估计竞争 {d["scale_queue_cycles"]:.2f}；两笔写的完成窗 {s["write"]:.2f} cycles。行余块 {rr%tr}、列余块 {cc%tc} 按实际几何重算。第三块复用第一块的槽位须等消费结束。')
        out.append(f'整片冷读写下限 {d["chip_io_floor_cycles"]:.2f} cycles。设备预测=(启动 {stages["dispatch"]*price:.2f} + 流水与全片下限取大 {stages["ub_pipeline"]*price:.2f} + 排空 {stages["drain"]*price:.2f})/{price:.0f} = **{r.latency_us:.5f} µs**。')
    out.append(f'周期参数来源为 950DT 阶段观测。常规路径的首读、写启动、排空分别为 {p["read_latency_cycles"]:.2f}、{p["write_latency_cycles"]:.2f}、{p["end_cycles"]:.2f} cycles，来自 13 个 shape。共享短写路径还从 10 个不同 shape 提取首核启动 {frag["launch_cycles"]:.2f}、每核发起跨度斜率 {frag["spread_cycles"]:.2f}、前端准备 {frag["frontend_cycles"]:.2f}、SIMD 固定部分 {frag["vector_cycles"]:.2f}、每粒度/共享返回节拍 {frag["grain_ii_cycles"]:.2f}/{frag["fabric_ii_cycles"]:.2f} cycles；各 shape 等权取中位数。实际端口调度、核顺序随芯片改变时可覆盖这六个参数。')
    return '\n\n'.join(out)
