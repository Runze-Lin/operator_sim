"""Replay native 950DT DDR command/service counters; use no latency fit or new CA."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/v0.27'
DT = 'Ascend950DT_9582'
KIMI = 'results/v0.20/full-quant'
LARGE = 'results/v0.20/large'
PRO = 'results/v0.23/pro-prefill/pro-b8-256ki-dt.json'
HEADER = re.compile(rb'Current time: (\d+), tDFI: 1\.0000 ns, DMC Total bytes: (\d+), DDR Total bytes: (\d+)')
NAMES = (b'Read cnt', b'Read with AP', b'Write', b'Write with AP', b'Mask write', b'Mask write with AP',
         b'Row hit', b'Row miss cnt', b'Active cnt', b'RW switch cnt')
FIELDS = re.compile(rb'^(' + b'|'.join(NAMES) + rb') +: (\d+) +\| Total [^:\n]+: (\d+) *$', re.M)
VALID = re.compile(rb'Cmd valid\s*:\s*(\d+)')
PRESSURE = re.compile(rb'DMC access\s*:\s*(\d+)\s*\| Command bp\s*:\s*(\d+)')
DDR_RANK = re.compile(rb'Rank0 Total Idle ratio\s*:\s*([\d.]+)\s*\| Rank0 Total act ratio\s*:\s*([\d.]+)')
ROW_BANK = re.compile(rb'Bank(\d+)\s+read : (\d+)\s+\| write : (\d+)\s+\| total : (\d+)')
DFI = re.compile(rb'Data Read bandwidth\s*:\s*([\d.]+)\s*\| Data Write bandwidth\s*:\s*([\d.]+)')
SIZE_CLASS = re.compile(rb'Total (\d+)B (Read|Write)\s*:\s*(\d+)')
BINS = ('read','write','hits','misses','activates','switches','q0','dmc_valid','dmc_access','dmc_bp',
        'dmc_bytes','ddr_bytes','slots','gap','gap_queued_low','gap_queued_high','qclock_tolerance','row_event_mismatch')
PANEL = b'======================================== START ========================================'


def load(path):
    return json.loads((ROOT/path).read_text())


def cast_id(rows,cols):
    return f'cast_{rows}_{cols}_{DT}'


def inputs():
    yield 'Kimi quant', ROOT/'runs/kimi-full-quant20/log', KIMI+'/controllers.json', KIMI+'/native.json'
    for r,c in [(8192,5120),(2097152,32)]:
        key = cast_id(r,c)
        yield 'Cast '+str([r,c]), ROOT/'runs/prefill20-memory'/key/'log', LARGE+'/'+key+'-official-dram.json', LARGE+'/'+key+'.json'
    yield 'Pro indexer Cast', ROOT/'runs/memory26-controller-snapshots/pro-b8-256ki-dt/log', 'results/v0.26/memory/pro-b8-256ki-dt-controller.json', PRO


def pct(a,b):
    return 100*a/b if b else None


def metric(x):
    return dict(summed_channel_counts=dict(zip(BINS,x)),
                payload_service_pct=pct(x[0]+x[1],x[12]),read_service_pct=pct(x[0],x[12]),write_service_pct=pct(x[1],x[12]),
                row_hit_pct=pct(x[2],x[2]+x[3]), switch_per_channel_us=1000*x[5]/x[12],
                dmc_bp_pct_of_valid=pct(x[9],x[7]), queue_empty_pct=pct(x[6],x[12]),
                fraction_gap_queue_occupied_pct=[pct(x[14],x[13]),pct(x[15],x[13])],
                ddr_gbs=128*x[11]/x[12], payload_mean_chip_us=x[11]/4_096_000,
                gap_mean_chip_us=x[13]/128/1000)


def add(target, v):
    for i,n in enumerate(v):
        target[i] += n


def zero():
    return [0]*len(BINS)


def channel(file, periods, groups):
    raw = file.read_bytes()
    sections = raw.split(PANEL)
    assert not sections[0].strip()
    prev=0
    counter_check = Counter()
    for n,panel in enumerate(sections[1:]):
        assert b' END ' in panel and b'hbm3 8000Mbps, x32, DMC Data Width: 256' in panel[:240]
        h=HEADER.search(panel,0,400)
        end,dmc,ddr=map(int,h.groups())
        dt=end-prev
        assert 0<dt<=1000 and (n+2==len(sections) or dt==1000)
        req_at=panel.index(b' Request Statistics ')
        end_stats=panel.index(b' Latency Statistics ',req_at)
        observed={name:(int(inc),int(total)) for name,inc,total in FIELDS.findall(panel,req_at,end_stats)}
        assert set(observed)==set(NAMES)
        inc=[observed[k][0] for k in NAMES]
        r=sum(inc[:2]); w=sum(inc[2:6]); hit,miss,act,switch=inc[6:]
        for k,(local,_) in observed.items():
            counter_check[k]+=local
        row_mismatch=abs(r+w-hit-miss)
        # DRAM-side data burst size is measured independently against the
        # integer DDR header; rank activity/DFI data bandwidth have different meanings.
        assert 32*(r+w)==ddr, (file.name,end,r,w,ddr)
        rd,wr=map(float,DFI.search(panel,0,1600).groups())
        # The last report may use an inclusive DFI period in its printed rates.
        assert min(abs(32*r/length-rd) for length in (dt,dt+1))<=.0051
        assert min(abs(32*w/length-wr) for length in (dt,dt+1))<=.0051
        pressure_at=panel.index(b' DMC Pressure Statistics ')
        valid=int(VALID.search(panel,pressure_at)[1])
        access,bp=map(int,PRESSURE.search(panel,pressure_at).groups())
        assert access+bp==valid
        # Queue occupancy histogram may count the zero-origin or last clock inclusively.
        qline=panel[panel.index(b'\nCycle  |',pressure_at)+1:].splitlines()[0]
        hist=[int(z) for z in qline.split(b'|')[1:] if z.strip()]
        assert len(hist)==65
        q0=hist[0]; qclock=sum(hist); mismatch=abs(qclock-dt)
        assert mismatch<=1 and r+w<=dt+1
        # Marginal-vs-joint ambiguity: at most all empty samples can coincide with
        # missing data beats. Allow the one boundary sample in either direction.
        gap=max(0,dt-r-w)
        lower=max(0,gap-q0-mismatch)
        upper=min(gap,qclock-q0+mismatch)
        v=[r,w,hit,miss,act,switch,q0,valid,access,bp,dmc,ddr,dt,gap,lower,upper,mismatch,row_mismatch]
        add(periods[end],v)
        if dt==1000:
            flow='idle' if r+w==0 else 'mixed' if r and w else 'read_only' if r else 'write_only'
            queued=('queued_95pct' if (qclock-q0)/qclock>=.95 else 'other_queue')
            add(groups[(flow,queued)],v)
        prev=end
    assert all(total==counter_check[k] for k,(_,total) in observed.items())
    count_sizes = {d:sum(int(n)*int(cnt) for n,dir_,cnt in SIZE_CLASS.findall(panel) if dir_.decode()==d) for d in ('Read','Write')}
    assert count_sizes == {'Read':32*(counter_check[b'Read cnt']+counter_check[b'Read with AP']),
                           'Write':32*sum(counter_check[k] for k in NAMES[2:6])}
    idle,active=map(float,DDR_RANK.search(panel).groups())
    bank={int(b):int(total) for b,r,w,total in ROW_BANK.findall(panel)}
    assert sum(bank.values())==sum(int(cnt) for n,di,cnt in SIZE_CLASS.findall(panel))
    return dict(sha256=hashlib.sha256(raw).hexdigest(), reports=len(sections)-1, last_ns=end, final_rank_idle_pct=idle,
                final_rank_act_pct=active, bank_commands=bank, DDR_read_bytes=count_sizes['Read'],DDR_write_bytes=count_sizes['Write'])


def measure(case,raw,source,receipt_source):
    saved=load(source); receipt=load(receipt_source)
    assert receipt.get('qualified',receipt.get('qualified_core_envelope')) and receipt['chip']==DT
    if 'native_controller_totals' in saved:
        saved=saved['native_controller_totals']
    periods=defaultdict(zero); groups=defaultdict(zero); channels={}
    for file in sorted(raw.glob('lpddr_state*.log')):
        channels[file.name] = channel(file,periods,groups)
    times=sorted(periods)
    assert len(channels)==128 and {c['last_ns'] for c in channels.values()}=={saved['last_controller_time_ns']}
    assert {c['reports'] for c in channels.values()}=={len(times)} and len(times)*128==saved['complete_channel_reports']
    total=zero()
    for t in times: add(total,periods[t])
    assert total[10]==saved['dmc_bytes']==saved['ddr_bytes']==total[11] and total[7]==total[8]+total[9]
    assert total[0]*32==sum(c['DDR_read_bytes'] for c in channels.values())
    assert total[1]*32==sum(c['DDR_write_bytes'] for c in channels.values())
    assert total[12]==times[-1]*128
    # Compare the occupied part of each run without imposing a common clock origin.
    # Select whole reporting periods containing the 5th and 95th volume percentile.
    lo=hi=None
    cum=0
    for i,t in enumerate(times):
        cum+=periods[t][11]
        if lo is None and cum>=.05*total[11]: lo=i
        if hi is None and cum>=.95*total[11]: hi=i
    central=zero()
    for t in times[lo:hi+1]: add(central,periods[t])
    # Bounds from exact period marginals: service and queued/backpressured states may overlap.
    bank_counts=Counter()
    for c in channels.values():bank_counts.update(c['bank_commands'])
    return dict(case=case,shape=receipt['shape'],source=source,receipt=receipt_source,raw=str(raw.relative_to(ROOT)),
                receipt_us=receipt.get('ca_us',receipt.get('core_envelope_us')),
                receipt_scope='wrapper' if 'qualified_core_envelope' in receipt else 'all-core instruction',
                archive_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [source,receipt_source]},
                horizon_ns=times[-1],full=metric(total),center_window_ns=[times[lo-1] if lo else 0,times[hi]],
                center=metric(central),central_fraction_of_bytes=central[11]/total[11],
                channel_1us_bins={f'{k[0]}/{k[1]}':metric(v) for k,v in sorted(groups.items())},
                final_rank_idle_pct_mean=sum(c['final_rank_idle_pct'] for c in channels.values())/128,
                final_rank_act_pct_mean=sum(c['final_rank_act_pct'] for c in channels.values())/128,
                active_bank_ids=sorted(b for b,c in bank_counts.items() if c),bank_command_counts=dict(bank_counts),
                channels={k:{q:v for q,v in c.items() if q!='bank_commands'} for k,c in channels.items()})


def write_report(data):
    rows=data['rows']
    q=rows[0]; full=q['full']; c=full['summed_channel_counts']; q_hi=q['channel_1us_bins']['mixed/queued_95pct']
    q_read=q['channel_1us_bins'].get('read_only/queued_95pct')
    switch_chip=c['switches']/128
    qrate=full['payload_service_pct']; gap=full['gap_mean_chip_us']
    qlo,qhi=full['fraction_gap_queue_occupied_pct']
    dur=q['horizon_ns']/1000
    envelope=load('results/v0.27/quant-envelope.json'); g=envelope['costs'][DT]
    modeled=g['resource_assumptions']
    missing=c['ddr_bytes']-modeled['assumed_read']-modeled['assumed_write']
    report=f"""# Kimi 大形量化的 DDR 实际服务

在 950DT 的 Kimi `[262144,7168]` 上，真实控制器传输 {c['ddr_bytes']:,} B。128 个通道合计的 DDR 读数据拍占 {full['read_service_pct']:.2f}%，写数据拍占 {full['write_service_pct']:.2f}%，合计 **{qrate:.2f}%**。按 CA 引脚峰值，平均到全部通道的有效数据服务为 **{full['payload_mean_chip_us']:.3f} µs**，未传输数据的时隙合计 **{gap:.3f} µs**。后者至少 {qlo:.2f}% 发生在 DMC 请求队列非空时。这些时隙中的大多数已有请求排队。

### 服务量、设备时间与管道

本例资源模型给 HBM 计入 {int(modeled['assumed_read']+modeled['assumed_write']):,} B，少了 {int(missing):,} B。这些额外字节按 4.096 TB/s 线速约占 {missing/4_096_000:.3f} µs。原生 64 AIV wrapper 为 {q['receipt_us']:.3f} µs，全进程控制器覆盖 {dur:.3f} µs。资源预测 {g['resource_us']:.3f} µs 至 wrapper 的差为 {q['receipt_us']-g['resource_us']:.3f} µs。字节低估的理想服务时间远小于这一差值，实际数据总线空拍与该差值处在同一数量级。DDR 处于调度或等请求时，SIMD、MTE 和核内控制仍可能并进，不能将这 {gap:.3f} µs 再全部加到算子预测。量化的完整逐指令时间仍为 N/A。[全核计时口径与逻辑流量](quant-envelope.md)

**计数单位。** 每个通道的每份原始面板包含 `DDR Total bytes`、普通读/写、自动预充电读/写（AP）和掩码写。将普通读与 AP 读相加得到 $N_r$。普通写、AP 写、掩码写和 AP 掩码写之和为 $N_w$。在量化和以下三个 Cast 的**每个通道、每个报告期**，独立核验了 $32(N_r+N_w)=Q_{{DDR}}$ 和两个方向的 `Data Read/Write bandwidth`。这里 32 B 是一个有效 DDR 数据拍对应的载荷量。接口头给出每通道 32 bit × 8 Gbit/s，引脚理论上每 1 ns DFI 时钟可传 32 B。全片瞬时线速按 128 通道为 4.096 TB/s。这一分母与芯片配置 4.000 TB/s 的标称值略有差别。每周期 DMC 接收与 DDR 实际服务可错位，各作业末期两层字节精确闭合。

**可测的等待。** 面板分别给出数据拍数与队列深度直方图，后者可还原空、非空的采样周期数。日志没有逐时钟对齐的联合记录。设同一通道报告期未服务数据的拍为 $I$，空队列的周期数为 $E$，非空为 $P$，采样边界误差为 $e$（至多一个 DFI 周期）。则“无数据且队列非空”的保守范围是 $[\\max(0,I-E-e),\\min(I,P+e)]$。逐通道逐期算界后求和，Kimi 为所有空拍的 **{qlo:.2f}%–{qhi:.2f}%**。排队时也可能等待命令依赖、地址可用或数据返回，队列非空不代表请求已可马上执行。

| 比较范围（950DT） | 数据字节 MB | 全程 DDR 数据服务率 | 中间 5%–95% 字节区间的服务率 | 全程平均 TB/s | 每通道每 µs 换向次数 | 空拍中队列非空范围 |
|---|---:|---:|---:|---:|---:|---:|
"""
    for v in rows:
        a=v['full'];b=v['center']; n=a['summed_channel_counts']
        lo,hi=a['fraction_gap_queue_occupied_pct']
        report+=f"| {v['case']} {v['shape'] if v is q or v['case']=='Pro indexer Cast' else ''} | {n['ddr_bytes']/1e6:.2f} | {a['payload_service_pct']:.2f}% | {b['payload_service_pct']:.2f}% | {a['ddr_gbs']/1000:.3f} | {a['switch_per_channel_us']:.2f} | {lo:.2f}–{hi:.2f}% |\n"
    report+="""
中间区间从累计传输达到总量 5% 的首个报告期开始，到达到 95% 的报告期结束，向外取完整报告期。在 Pro 256 Ki 的 Cast 中，控制器报告还覆盖了较长的无数据尾段。看整份控制器日志的平均值会低估执行阶段的吞吐。三个大 Cast 都有官方全核指令窗口及通过的数值校验。另一个 Flash 全 hidden Cast 的合格原生重跑将控制器日志定向到空设备。较早的原始控制器记录属于未完成的运行，均没有加入本比较。

### 队列有请求时仍能否传满

固定到单个物理通道、完整 1 µs 报告期、至少 95% 周期队列非空，按该期实际出现的 DDR 数据方向分组。下表的服务率按相应组实际经过的 DFI 周期计算。“混合”只表示这一微秒内出现过两种方向。

| 负载 | 混合读写期数 | 混合服务率 | 每通道每 µs 换向次数 | 纯读期数 | 纯读服务率 | 行命中率（全程） | 有效命令中受阻比例（全程） |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for v in rows:
        all_m=v['full']; group=v['channel_1us_bins']; m=group['mixed/queued_95pct']; rd=group.get('read_only/queued_95pct')
        read_periods=rd['summed_channel_counts']['slots']//1000 if rd else 0
        read_rate=f"{rd['payload_service_pct']:.2f}%" if rd else 'N/A'
        report+=f"| {v['case']} | {m['summed_channel_counts']['slots']//1000:,} | {m['payload_service_pct']:.2f}% | {m['switch_per_channel_us']:.2f} | {read_periods:,} | {read_rate} | {all_m['row_hit_pct']:.2f}% | {all_m['dmc_bp_pct_of_valid']:.2f}% |\n"
    report+=f"""
Kimi 的高占用队列混合期共 {q_hi['summed_channel_counts']['slots']//1000:,} 个，连续观察下的数据服务率约 {q_hi['payload_service_pct']:.2f}%。同一次运行的高占用纯读期平均为 {q_read['payload_service_pct']:.2f}%。两组的读写比例、局部地址序列和时间位置均有差别。Cast 同口径的混合期也明显低于线速，有的形状纯读期同样慢，无法据此反推一个“每换向等待固定几个周期”的通用参数。

### 换向和 bank 的可辨识程度

- 量化原生 `Total RW switch cnt` 跨全通道为 **{c['switches']:,} 次**，平均每通道 {switch_chip:.1f} 次。读转写与写转读没有单列，各次换向的阻塞起止也没有输出。以整个未服务时间除换向次数会将排队、刷新、行切换和其他原因一并计入，不能用作换向惩罚。
- 原始 `Total row hit` 与 `Total row miss cnt` 给出量化行命中率 {full['row_hit_pct']:.2f}%。同一报告期间的行事件与数据命令可分别发生，它们的计数不要求逐期相等。分 bank 可观察读写请求的累积个数（本例有流量的 bank ID 为 {q['active_bank_ids'][0]}–{q['active_bank_ids'][-1]}），未提供逐 bank 独占 idle/active 时长。最终的 `Rank0 Total Idle ratio` 与 `Rank0 Total act ratio` 是另一个层级，128 通道平均为 {q['final_rank_idle_pct_mean']:.2f}% 与 {q['final_rank_act_pct_mean']:.2f}%，不可代替 DDR 数据线占用。
- `DMC access + Command bp = Cmd valid` 在每个报告期间成立。量化命令有效的周期中，{full['dmc_bp_pct_of_valid']:.2f}% 被 backpressure 阻挡。这里的命令入口每次接收一个尺寸不等的请求，随后拆成 DDR 数据拍，所以这个比率与出口未传数据的比率不可相加。
- 此次可分别量化传输了多少字节、通道有多少数据空拍、空拍与队列非空重叠的保守范围、发生过多少次换向，以及行命中情况。各类空拍中有多少专由转向、bank 冲突、刷新或上游未及时发起造成，现有计数无法唯一分解。跨芯片时仍需替换通道宽度、数量、时钟和实际请求序列。上述百分比属于各次输入和 CA 配置，未用于拟合系数或改变工具输出。

复现命令为 `python3 development/validation/kda27/quant_service.py`。只读取既有计数日志，新增 CA 为 0。每种形状的各通道 SHA-256、数据窗口、计数器及分组分母见 [机器数据](quant-channel-service.json)。
"""
    (OUT/'quant-channel-service.md').write_text(report)


def main():
    groups=[measure(*job) for job in inputs()]
    data=dict(protocol='Decimal DDR GB/s. 32 bytes/DFI data beat independently checked on each channel in each reporting interval. No new CA.',
              target=DT,rows=groups)
    (OUT/'quant-channel-service.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    write_report(data)


if __name__=='__main__':
    main()
