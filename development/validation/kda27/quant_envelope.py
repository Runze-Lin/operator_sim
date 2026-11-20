"""Audit a native all-AIV MX quant envelope and earlier instruction-window replays."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model import estimate
from operator_model.chips import load_chip
from operator_model.quant_resource import resource_quant

DT, PR, FUTURE = 'Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT'
BIG = 'results/v0.20/full-quant'
RAW = ROOT / 'runs/kimi-full-quant20'
CONTROLS = 'results/v0.20/large'
HIST = 'results/v0.27/replay.json'
OUT = ROOT / 'results/v0.27'
REGION_BYTES = 1 << 20
HEADER = re.compile(rb'Current time: (\d+), tDFI: 1\.0000 ns, DMC Total bytes: (\d+), DDR Total bytes: (\d+)')
WIDTH = re.compile(rb'Total (\d+)B (Read|Write)\s*:\s*(\d+)')
TIME = re.compile(r'core: (\d+), subcore: (\d+), start: (\d+), tick: (\d+), kernelId: 0, blkID: (\d+), blkDim: 64,.*?done')


def load(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def floors(read, write, cap, bandwidth_gbs):
    bytes_per_us = 1000 * bandwidth_gbs
    return dict(cold_and_drained=(read + write) / bytes_per_us,
                cold_retained=(read + max(0, write - cap)) / bytes_per_us,
                unknown_retained=(max(0, read - cap) + max(0, write - cap)) / bytes_per_us)


def wrapper(native):
    targets = {k: tuple(v) for k, v in native['core_ticks'].items()}
    spans, busy, blocks = {}, {}, set()
    for path in sorted((RAW / 'log_ca').glob('core*_summary_log')):
        for part in path.read_text().split('kernal total ticks :')[1:]:
            core, sub, start, stop, block = map(int, TIME.search(part).groups())
            key = f'{core}.{sub}'
            assert key not in spans and block == 2*core + sub-1 and block not in blocks
            blocks.add(block)
            spans[key] = (start, stop)
            pipes = {}
            for tag, name in [('MTE2', 'mte2_veccore{}_su_busy_cycle'), ('MTE3', 'mte3_veccore{}_su_busy_cycle'),
                              ('SIMD', 'rvec_veccore{}_simd_busy_cycle')]:
                ticks = int(re.search(name.format(sub-1) + r'\s*\|\s*(\d+)', part)[1])
                assert 0 < ticks <= stop-start
                pipes[tag] = ticks
            busy[key] = pipes
    assert spans == targets and len(spans) == 64 and blocks == set(range(64))
    mhz = 1000*load_chip(DT).frequency_ghz
    start, end = min(a for a, b in spans.values()), max(b for a, b in spans.values())
    t = (end-start)/mhz
    assert math.isclose(t, native['core_envelope_us'])
    raw_log = (RAW/'run.log').read_text()
    checks = re.findall(r'^correctness count=(\d+) bad=(\d+)$', raw_log, flags=re.M)
    assert 'SMOKE_PASS' in raw_log and f'runtime_soc={DT}' in raw_log and 'hbm3_8000M.ini' in raw_log
    assert checks == [(str(n), '0') for n in native['output_counts']] and native['qualified_core_envelope']
    assert native['instruction_ca_us'] is None and native['cost']['exit_code'] == 0
    fractions = {}
    for tag in ('MTE2', 'MTE3', 'SIMD'):
        per = [100*p[tag]/(spans[k][1]-spans[k][0]) for k,p in busy.items()]
        fractions[tag] = dict(min=min(per), median=statistics.median(per), max=max(per))
    return dict(start_tick=start, end_tick=end, ticks_per_us=mhz, wrapper_us=t, active_aiv=len(spans),
                last_aiv=[k for k,(_,b) in spans.items() if b==end], output_counts=native['output_counts'],
                invoice_cost=native['cost'], instruction_us=None, per_aiv_busy_ticks=busy,
                per_aiv_pipe_busy_pct_of_own_wrapper=fractions)


def selected_panels(path, selected):
    """Read bounded ends only. Select complete tables and integer cumulative size-class counts."""
    with path.open('rb') as inp:
        head = inp.read(REGION_BYTES)
        inp.seek(-REGION_BYTES, 2)
        tail = inp.read()
    records = {}
    header = None
    for portion in (head, tail):
        for panel in portion.split(b' START ')[1:]:
            if b' END ' not in panel:
                continue
            h = HEADER.search(panel)
            when = int(h[1])
            if when not in selected:
                continue
            rate, width = map(int, re.search(rb'hbm3 (\d+)Mbps, x(\d+), DMC Data Width:',panel).groups())
            assert header is None or header==(rate,width)
            header = (rate,width)
            bins = {(int(n), d.decode().lower()):int(c) for n,d,c in WIDTH.findall(panel)}
            assert sorted(bins)==sorted((n,d) for n in (32,64,128,256) for d in ('read','write'))
            cmd = list(map(int,re.search(rb'Total reads\s*:\s*(\d+)\s*\| Write\s*:\s*\d+\s*\| Total writes\s*:\s*(\d+)',panel).groups()))
            assert cmd == [sum(c for (n,di),c in bins.items() if di==d) for d in ('read','write')]
            records[when] = (bins,hashlib.sha256(panel.split(b' END ')[0]).hexdigest())
    assert records.keys()==selected
    return header,records


def controller_data(saved, win):
    horizon = saved['last_controller_time_ns']
    # Regardless of the devices' clock origins, if the wrapper is contained in the
    # reporting horizon, every possible placement contains [horizon-duration, duration].
    # Round inwards to whole DFI panels. Never assign these bytes to a single instruction.
    duration_ns = win['wrapper_us']*1000
    lo = math.ceil((horizon-duration_ns)/1000)*1000
    hi = math.floor(duration_ns/1000)*1000
    assert 0 < lo < hi < horizon
    selected = {lo, hi, horizon}
    totals = {t: Counter() for t in selected}
    file_evidence, header_dist = {}, Counter()
    for path in sorted((RAW/'log').glob('lpddr_state*.log')):
        (rate, width), panels = selected_panels(path, selected)
        header_dist[(rate,width)] += 1
        file_evidence[path.name] = {str(t):h for t,(b,h) in panels.items()}
        for t,(b,h) in panels.items():
            totals[t].update(b)
    assert len(file_evidence)==saved['channel_files']==128 and len(header_dist)==1
    rate, width = next(iter(header_dist))
    wire_gbs = rate*width*len(file_evidence)/8/1000

    def directional(counts):
        return {d+'_bytes':sum(n*c for (n,di),c in counts.items() if di==d) for d in ('read','write')}
    all_bins = totals[horizon]
    for d in ('read','write'):
        assert sum(c for (n,di),c in all_bins.items() if di==d)==saved[d+'_commands']
    whole = directional(all_bins)
    assert sum(whole.values()) == saved['dmc_bytes'] == saved['ddr_bytes']
    for k in all_bins:
        assert 0 <= totals[lo][k] <= totals[hi][k] <= all_bins[k]
    middle = directional(totals[hi]-totals[lo])
    return dict(whole=whole, inner=middle, inner_dfi_ns=[lo,hi], whole_dfi_ns=[0,horizon],
                config_peak_gbs=load_chip(DT).hbm_bandwidth_gbs, rate_mbps_per_pin=rate, width_bits=width,
                channels=len(file_evidence), wire_peak_gbs=wire_gbs,
                final_counts={str(n):{d:all_bins[(n,d)] for d in ('read','write')} for n in (32,64,128,256)},
                selected_complete_dfi_panel_sha256=file_evidence)


def main():
    reg, native, saved = [load(BIG+'/'+x+'.json') for x in ('freeze','native','controllers')]
    shape=reg['shape']
    win=wrapper(native)
    dram=controller_data(saved,win)
    r,y,s=(reg['logical_bytes'][k] for k in ('input','y','scale'))
    w=y+s
    assert (r,y,s)==(2*math.prod(shape),math.prod(shape),2*shape[0]*math.ceil(shape[1]/64))
    latency = win['wrapper_us']
    observed=sum(dram['whole'].values())
    costs = {}
    for sku in (DT,PR,FUTURE):
        chip=load_chip(sku)
        bw,cap=chip.hbm_bandwidth_gbs,chip.details['l2_bytes']
        resource,_,active,info=resource_quant(shape,chip)
        auto=estimate('quant',shape,chip=sku)
        pred=estimate('quant',shape,chip=sku,timing_mode='resource')
        assert abs(pred.latency_us-resource)<1e-8
        portion=dict(assumed_read=info['assumed_offchip_read_bytes'],assumed_write=info['assumed_offchip_write_bytes'],
                     assumed_dirty_output=info['output_retained_bytes'], stages_us=info['stages_us'])
        costs[sku]=dict(auto_us=auto.latency_us,resource_us=resource,active_aiv=active,l2=cap,hbm_gbs=bw,
                        floors_us=floors(r,w,cap,bw),resource_assumptions=portion,
                        envelope_us=latency if sku==DT else None,instruction_us=None)
    g = costs[DT]
    data_bytes = {'logical':r+w, 'whole_controller':observed, 'inner_controller':sum(dram['inner'].values())}
    curves = {}
    for name,p in [('default',g['auto_us']),('resource',g['resource_us']),('wrapper',latency)]:
        curves[name]=dict(us=p, implied_tb_s={k:v/(p*1e6) for k,v in data_bytes.items()},
                          signed_pct_to_wrapper=100*(p/latency-1),
                          under_floor_us={str(bw):{k:v/(bw*1000)-p for k,v in data_bytes.items()}
                                          for bw in (g['hbm_gbs'],dram['wire_peak_gbs'])})
    registry = load(CONTROLS+'/evaluation.json')['controls']
    small=[]
    for c in registry:
        rows,cols=c['shape']; prefix=f'{CONTROLS}/cast_{rows}_{cols}_native'
        ctrl=load(prefix+'.json')
        counts=load(prefix+'-dram.json')
        assert ctrl['qualified_core_envelope'] and abs(ctrl['core_envelope_us']-c['wrapper_us'])<1e-8
        assert counts['write_commands']==0 and counts['dmc_bytes']==counts['ddr_bytes']
        small.append(dict(op='cast_bf16_fp32',shape=c['shape'],active_aiv=ctrl['active_aiv'],
                          instruction_us=c['instruction_us'],wrapper_us=c['wrapper_us'],delta_us=c['wrapper_us']-c['instruction_us'],
                          logical_bytes=counts['logical_read_bytes']+counts['logical_write_bytes'],
                          controller_read_bytes=counts['dmc_bytes'],controller_write_bytes=0,
                          cold_and_drained_us=(counts['logical_read_bytes']+counts['logical_write_bytes'])/(1000*g['hbm_gbs']),
                          sources=[prefix+'.json',prefix+'-dram.json']))
    history={}
    for sku in (DT,PR):
        cfg=costs[sku]
        rows=[]
        for row in load(HIST)['rows']:
            if row['op']!='quant' or row['chip']!=sku:
                continue
            a=estimate('quant',row['shape'],attrs=row['attrs'],chip=sku)
            rr,ww=(a.work[k] for k in ('logical_read_bytes','logical_write_bytes'))
            floor=floors(rr,ww,cfg['l2'],cfg['hbm_gbs'])
            assert math.isclose(a.latency_us,row['latency_us'],abs_tol=1e-8)
            assert len(row['trace_sha256'])==1
            cap_gate=rr+ww>cfg['l2']
            rows.append(dict(shape=row['shape'],trace_sha256=row['trace_sha256'][0], read=rr,write=ww, floor_us=floor,
                             ca_us=row['ca_us'],default_us=a.latency_us, exceeds_l2_combined=cap_gate,
                             exceeds_l2_both=min(rr,ww)>cfg['l2'],
                             capped_default_us=max(a.latency_us,floor['cold_and_drained']) if cap_gate else a.latency_us))
        unique_shapes = len({tuple(r['shape']) for r in rows})
        assert len({r['trace_sha256'] for r in rows})==len(rows)
        def wape(key):
            return 100*sum(abs(row[key]-row['ca_us']) for row in rows)/sum(row['ca_us'] for row in rows)
        history[sku]=dict(n=len(rows),unique_shapes=unique_shapes,max_bytes=max(z['read']+z['write'] for z in rows),
                          over_l2=sum(z['exceeds_l2_combined'] for z in rows), both=sum(z['exceeds_l2_both'] for z in rows),
                          changed=sum(z['default_us']!=z['capped_default_us'] for z in rows),WAPE_baseline_pct=wape('default_us'),
                          WAPE_conditional_pct=wape('capped_default_us'),
                          all_rows=rows)
    inputs=[BIG+'/freeze.json', BIG+'/native.json',BIG+'/controllers.json',BIG+'/capture-registration.json',
            CONTROLS+'/evaluation.json',HIST,'operator_model/backend.py','operator_model/quant_resource.py','operator_model/workload.py']
    inputs += sum((c['sources'] for c in small),[])
    inputs += [f'configs/chips/{c}.json' for c in costs]
    direct=dict(workload=reg['scenario'],shape=shape,api=reg['api'],condition=reg['dispatch_condition'],dtype='bf16 -> e4m3fn + e8m0',
                logical_read=r,logical_y=y,logical_scale=s,wrapper=win,controller=dram,sources_sha256={p:digest(p) for p in inputs},
                raw_root=str(RAW.relative_to(ROOT)),costs=costs,curves=curves,small_cast_controls=small,
                archived_instruction_replay=history,role='Re-analysis. Historical development and one separate quant wrapper; zero new CA.',
                same_shape_quant_PR_us=None,same_shape_quant_970dt_us=None)
    (OUT/'quant-envelope.json').write_text(json.dumps(direct,indent=2,ensure_ascii=False)+'\n')
    write_report(direct)


def write_report(d):
    win,dc=d['wrapper'],d['controller']
    costs,curves,history=d['costs'],d['curves'],d['archived_instruction_replay']
    r,y,s=d['logical_read'],d['logical_y'],d['logical_scale']
    w=y+s
    total=r+w
    whole,inner=(sum(dc[k].values()) for k in ('whole','inner'))
    duration=win['wrapper_us']
    bw=costs[DT]['hbm_gbs']
    wire=dc['wire_peak_gbs']
    f=costs[DT]['floors_us']
    dmc=dc['whole']
    r0=costs[DT]['resource_assumptions']
    pipes=win['per_aiv_pipe_busy_pct_of_own_wrapper']
    md=f"""# Kimi-K3 整批预填充的 MX 量化流量与时间

B8 × 32 Ki、不切 chunk、hidden=7168 时，混合 QKV 在 MXFP8 动态量化、`scale_alg=0` 的分支可独立调用 `aclnnDynamicMxQuant`。950DT 一次运行的 64 个 AIV 从最早启动到最后完成共 **{duration:.3f} µs**。默认预测 {costs[DT]['auto_us']:.3f} µs（相对包络 {curves['default']['signed_pct_to_wrapper']:+.2f}%），显式资源模型 {costs[DT]['resource_us']:.3f} µs（{curves['resource']['signed_pct_to_wrapper']:+.2f}%）。以配置的 4.000 TB/s 为准，原生通道记录支持两者低估所需的总线服务时间。按 CA 引脚配置计算 4.096 TB/s 后，只有默认值仍低于实测流量下限。该形状的标准逐指令延迟仍为 N/A，这些偏差不计入指令窗 WAPE。

## 实际使用的字节与窗口

- 输入 BF16 `[262144,7168]` 有 {r:,} B。FP8 值有 {y:,} B。每 32 个通道产生一个 E8M0 scale，按每行两字节打包后的大小为 `2 × rows × ceil(cols/64)`，本例为 {s:,} B。两个输出逐元素核对，共 {sum(win['output_counts']):,} 个值、0 错。热态、输入复用、脏行延迟写回均可使调用窗口中的 HBM 字节与这 {total:,} B 有别。
- 原生窗口取 64 个 AIV 的最早 start {win['start_tick']:,} tick 到最晚 done {win['end_tick']:,} tick，除以 {win['ticks_per_us']:g} tick/µs。末核为 `{', '.join(win['last_aiv'])}`（物理 core.子核）。复验全套 64 个 block ID 与两个 golden。大例保留的是原生核汇总，无法从中提取逐 tile 的指令时序。
- 原生模拟从启动到最终控制器报告共 {dc['whole_dfi_ns'][1]/1000:.3f} µs。128 个通道各自登记完整的 32/64/128/256 B 读写事务数。各尺寸乘实际字节后求和，精确对上已归档的 DMC 与 DDR 两个总计。不能直接将读写命令个数乘一个固定常数。

| 跨全部通道 | BF16 输入 / 两个输出逻辑 B | DMC 全进程 B | 相差 B |
|---|---:|---:|---:|
| 读 | {r:,} | {dmc['read_bytes']:,} | {dmc['read_bytes']-r:+,} |
| 写 | {w:,} | {dmc['write_bytes']:,} | {dmc['write_bytes']-w:+,} |
| 合计 | {total:,} | {whole:,} | {whole-total:+,} |

DMC 全进程的实际平均吞吐为 **{whole/(dc['whole_dfi_ns'][1]*1000):.3f} TB/s**，占芯片配置峰值 {100*whole/(dc['whole_dfi_ns'][1]*bw):.2f}%。按逻辑字节除以原生 wrapper 的算法吞吐是 {total/(duration*1e6):.3f} TB/s。两者均含空闲和控制阶段，无法从全程平均值反推 MTE 发起间隔。

即使不假设两种时钟原点相同，只要 wrapper 位于同一次控制器采集期间，长度为 $W$ 的 wrapper 在长度为 $D$ 的采集内的任意位置均覆盖 $[D-W,W]$。向内取完整 1 µs 控制器采样区间，得到 [{dc['inner_dfi_ns'][0]/1000:g}, {dc['inner_dfi_ns'][1]/1000:g}] µs。其间确有读 {dc['inner']['read_bytes']:,} B 与写 {dc['inner']['write_bytes']:,} B。它们覆盖全进程字节的 {100*inner/whole:.3f}%。这是 wrapper 时间范围内的全通道事实，仍含同一进程中的元数据及可能的背景流量。无法把余下时间或任一字节归给具体 tile。逐区间哈希和末期事务分布列于 [JSON](quant-envelope.json)。

## 哪一条下界成立

令 $R$、$W_b$ 为输入和输出字节数，$C$ 为共享 L2 容量，$B$ 为读写合计峰值（GB/s，十进制）。任意指定穿过该端口的 $Q$ 字节最少占 $Q/(1000B)$ µs。选择 $Q$ 需要说明缓存起止状态和测量边界。

| 950DT 的前提 | 纳入的字节 | 以 4.000 TB/s 计的下限 µs |
|---|---:|---:|
| 输入已冷且结果全写回 | $R+W_b$ | {f['cold_and_drained']:.3f} |
| 输入已冷，最终允许最多一份 L2 写缓 | $R+\\max(0,W_b-C)$ | {f['cold_retained']:.3f} |
| 初始允许缓存输入，退出又允许缓存输出 | $\\max(0,R-C)+\\max(0,W_b-C)$ | {f['unknown_retained']:.3f} |
| 同一次 CA 中、wrapper 必含的完整报告区间 | {inner:,} B | {inner/(1000*bw):.3f} |
| 完整控制器运行 | {whole:,} B | {whole/(1000*bw):.3f} |

容量两式假设唯一数据的可跨调用留驻储存由 L2 代表，并限定相同地址空间的单次独立调用。若系统把输入和输出留在其他缓冲区、与上下游融合、改变写策略或 HBM 另有独立读写通道，需重新编写下界。按未知驻留的容量下界，资源模型通过，默认预测仍偏低。全进程额外捕获的时间不能补入标准指令窗。

每份通道原始头是 `hbm3 {dc['rate_mbps_per_pin']}Mbps, x{dc['width_bits']}`，理论引脚和 $128×32×8/8={wire:g}$ GB/s。芯片配置取 {bw:g} GB/s。引脚和值还未扣除总线转向、刷新及协议成本。使用引脚和，完整控制器服务下限下降到 {whole/(1000*wire):.3f} µs，必含区间为 {inner/(1000*wire):.3f} µs。依据量化单次运行无法断定配置口径与 CA 控制器绝对峰值完全等价。

| 模型 / 实测 | 时间 µs | 逻辑流量 / 时间 TB/s | 整体 DMC 流量 / 所列时间 TB/s |
|---|---:|---:|---:|
"""
    for name,label in [('default','默认'),('resource','资源'),('wrapper','原生 wrapper')]:
        x=curves[name]
        md+=f"| {label} | {x['us']:.3f} | {x['implied_tb_s']['logical']:.3f} | {x['implied_tb_s']['whole_controller']:.3f} |\n"
    md+=f"""
表中不同时间作为分母仅用于比较，实测的全进程平均带宽在上一节给出。

## 资源模型中的误差来源

此形状的 950DT 默认路由沿用参考预测器。用目标芯片资源重算时，分核为 64 AIV，最忙核 4,096 行，UB tile 为 [5,7168]，共 820 轮。资源模型已经限制独立 MTE 与共享带宽，读侧取 {int(r0['assumed_read']):,} B。由于预留活跃核的两块输入 tile，估计退出时仍有 {int(r0['assumed_dirty_output']):,} B 输出可留 L2，写侧仅计 {int(r0['assumed_write']):,} B。实际 DMC 读比模型多 {int(dmc['read_bytes']-r0['assumed_read']):,} B、写多 {int(dmc['write_bytes']-r0['assumed_write']):,} B。按配置峰值换算，这两项差额共 {(whole-r0['assumed_read']-r0['assumed_write'])/(1000*bw):.3f} µs。即使把控制器实际总量作为带宽服务成本，其理想下限至 wrapper 仍有 {duration-whole/(1000*bw):.3f} µs，涉及管道执行、请求效率、内存调度和 wrapper 边界，无法只归因为一个固定 launch 或一个输出留驻比例。

真实 64 个 AIV 各自的忙周期占自身 wrapper 的中位数分别为 MTE2 {pipes['MTE2']['median']:.1f}%、MTE3 {pipes['MTE3']['median']:.1f}%、SIMD {pipes['SIMD']['median']:.1f}%。同一时段可以多条流水同时忙，三项不能相加。这些汇总无法重建逐指令关键路径。若需将差距下推至系统仿真，可按是否预热、是否要求落 HBM 进行情景计算，并让目标芯片的共享口径、实际并发和分块决定可用吞吐。本次 {100*whole/(dc['whole_dfi_ns'][1]*bw):.1f}% 仅描述本次离线运行。

## 两个采集控制与历史回放

以下两形均运行 950DT 的 BF16→FP32 Cast。它们核验相同原生采集流程的窗口定义，其中 [262144,32] 留有同次运行逐核边界对照。不同算子和数据驻留使控制组差值无法直接用于量化窗口的校正。

| Cast shape | AIV | 完整指令 µs | wrapper µs | 差 µs | DMC 读 / 写 B | 全冷读且全写回下限 µs |
|---|---:|---:|---:|---:|---:|---:|
"""
    for c in d['small_cast_controls']:
        md+=f"| {c['shape']} | {c['active_aiv']} | {c['instruction_us']:.3f} | {c['wrapper_us']:.3f} | {c['delta_us']:.3f} | {c['controller_read_bytes']:,} / {c['controller_write_bytes']:,} | {c['cold_and_drained_us']:.3f} |\n"
    md+="""
[262144,32] 的输入输出合计不足 L2，wrapper 时间也短于逻辑冷读全写回线。原生控制器记录 0 写。这给出了把每次逻辑写入都算作窗口内 HBM 流量的直接反例。两个控制组的 wrapper 分别比标准窗口多约 2 µs，无法为跨算子的时间差规定严格上界。

v0.27 的已核验量化开发回放采用官方核第一条指令到设备最后一条指令的时间。在同一批历史 shape，离线试算条件下界。若 $R+W_b>C$，延迟取默认预测与 $(R+W_b)/(1000B)$ 的较大值。另统计更严格的读、写各自都大于 L2 的样本数。正式历史回放和上面单个大 shape 的 wrapper 各自统计。

| 950 SKU | 指令窗数 / 不同开发形数 | L2 MiB | 最大逻辑流量 B | 合计超 L2 / 两者各超 L2 | 默认改变数 | WAPE 原值→试算 |
|---|---:|---:|---:|---:|---:|---:|
"""
    for sku in (DT,PR):
        h=history[sku]
        md+=f"| {sku} | {h['n']} / {h['unique_shapes']} | {costs[sku]['l2']/2**20:g} | {h['max_bytes']:,} | {h['over_l2']} / {h['both']} | {h['changed']} | {h['WAPE_baseline_pct']:.2f}% → {h['WAPE_conditional_pct']:.2f}% |\n"
    md+=f"""
DT 的 [4,256] 有两个不同指令窗。WAPE 以归档窗口计算。两端最大同为 [2048,5120]。这组开发回放没有覆盖规则会启动的区域，所以零变化无法证明跨 L2 后的准确性。大形只有 950DT 的一次原生窗口，且属于回顾性诊断。在该大形上强制冷读全回写，默认和资源结果都将变为 {f['cold_and_drained']:.3f} µs，距离原生窗口仍低 {100*(1-f['cold_and_drained']/duration):.2f}%。950PR 同形和 970DT 均未有 CA 窗口。

| [262144,7168] | 峰值 GB/s | 默认 / 资源 µs | 未知驻留 / 冷读全回写线 µs | 原生 wrapper / 标准指令 µs |
|---|---:|---:|---:|---:|
"""
    for sku,g in costs.items():
        a=g['floors_us']
        timing=f"{g['envelope_us']:.3f} / N/A" if g['envelope_us'] is not None else 'N/A / N/A'
        md+=f"| {sku} | {g['hbm_gbs']:g} | {g['auto_us']:.3f} / {g['resource_us']:.3f} | {a['unknown_retained']:.3f} / {a['cold_and_drained']:.3f} | {timing} |\n"
    md+=f"""
950PR 的 1.4 TB/s 是芯片配置按 SKU 档位推得的值。970DT 的 14.4 TB/s 来自用户情景。其 120 AIV 的 MTE 与 UB 细节沿用 950 假设。这两行只描述参数敏感性，未验证跨芯片加速比。当前证据支持优先核对大形默认路由，并在有独立超 L2、不同读写比例的指令窗后再决定条件规则。单次观测无法识别可迁移至后续芯片的单一效率。

**开销与复核。** 原 CA 花费 {win['invoice_cost']['seconds']/3600:.3f} 小时、原始峰值 {win['invoice_cost']['peak_bytes']/2**30:.3f} GiB。本诊断只读取既有数据，新增 CA 为 0 次。复现命令为 `python3 development/validation/kda27/quant_envelope.py`。逐核忙周期、逐形下界、所有输入文件和证据面板的哈希见 [机器数据](quant-envelope.json)。[原生完整回执](../v0.20/full-quant/native.json) · [原始注册及算子合同](../v0.20/full-quant/freeze.json) · [历史标准窗口](replay.json) · [控制组两种采集](../v0.20/large/evaluation.json)
"""
    (OUT/'quant-envelope.md').write_text(md)


if __name__=='__main__':
    main()
