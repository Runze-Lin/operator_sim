"""Audit the frozen, directly simulated long-prefill Cast on each SKU."""
import argparse
from collections import Counter
import json
import mmap
from pathlib import Path
import re
import time

from large_cast import CHIPS, DT, PR, OUT, RAW, ROOT, read, write


def memory_descriptors(trace):
    # The vendor trace is compact JSON, with args before instruction name. Only
    # decode actual DMA records. A 256ki×8 full trace can exceed 3 GiB.
    marker = b'"name":"MOV_SRC_TO_DST_ALIGNv2"'
    start_marker = b'{"args":{'
    with trace.open('rb') as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
        pos = 0
        while True:
            match = data.find(marker, pos)
            if match < 0:
                return
            start = data.rfind(start_marker, max(0, match-16384), match)
            if start < 0:
                raise ValueError('vendor event encoding differs; cannot extract DMA safely')
            end = data.find(b"}", match) + 1
            event = json.loads(data[start:end])
            if event['ph']=='X' and event['tid'] in ('MTE2','MTE3'):
                detail = json.loads(event['args']['detail'])
                if not detail.get('is_duration_excluded'):
                    yield event['pid'], event['tid'], int(detail['processed_bytes']), event['ts'], event['dur']
            pos = end


def dma_audit(result):
    stage = result['stage']
    finish = max(stage['windows'], key=lambda c:stage['windows'][c][1])
    total, sizes, issuers, last = Counter(), {'MTE2':Counter(), 'MTE3':Counter()}, {'MTE2':set(),'MTE3':set()}, []
    for core, queue, size, start, duration in memory_descriptors(ROOT/stage['trace']):
        total[queue] += size
        sizes[queue][size] += 1
        issuers[queue].add(core)
        if core==finish:
            last.append(dict(queue=queue,bytes=size,ts_us=start,duration_us=duration))
    n=result['shape'][0]*result['shape'][1]
    return dict(shape=result['shape'],chip=result['chip'],terminal=finish,
                logical_bytes={'MTE2':2*n,'MTE3':4*n}, visible_bytes=total,
                visible_transfer_sizes=sizes, issuing_cores={k:len(v) for k,v in issuers.items()},
                terminal_dma=last,trace_sha256=stage['trace_sha256'])


def sample_controls():
    rows=[]
    prior=next(r for r in read(ROOT/'results/v0.19/workloads/cast-qualified.json')
               if r['shape']==[32,4096] and r['sku']==DT and r['qualified'])
    for t,h in ((32,4096),(262144,32)):
        stem=f'cast_{t}_{h}'
        f=OUT/(stem+'_native.json')
        official=OUT/(stem+'_'+DT+'.json')
        if not f.exists() or (t!=32 and not official.exists()):continue
        native=read(f)
        if not native['qualified_core_envelope']:continue
        if t==32: stage=prior['kernel']; cost=prior['cost']
        else:
            other=read(official)
            if not other['qualified']:continue
            stage=other['stage'];cost=other['cost']
        row=dict(shape=[t,h],wrapper_us=native['core_envelope_us'],instruction_us=stage['ca_us'],
                 native_wall_s=native['cost']['seconds'],official_wall_s=cost['seconds'],
                 native_peak_mib=native['cost']['peak_bytes']/2**20,official_peak_mib=cost['peak_bytes']/2**20)
        # Same dispatch: recover the absolute time origin of the official trace
        # from one per-core view and compare ALL wrapper ends to instruction ends.
        wrapper=RAW/(stem+'_'+DT)/'wrapper'
        files=list(wrapper.glob('core*_summary_log'))
        if files:
            windows=stage['windows']
            percore=ROOT/Path(stage['trace']).parent/'core0.veccore0/trace.json'
            first=min(e['ts'] for e in read(percore)['traceEvents'] if e['ph']=='X')
            origin=first-windows['core0.veccore0'][0]
            heads, tails=[], []
            for file in files:
                for c,s,a,b in re.findall(r'core: (\d+), subcore: (\d+), start: (\d+), tick: (\d+)',file.read_text()):
                    l,h=windows[f'core{c}.veccore{int(s)-1}']
                    heads.append(round(1650*(l+origin))-int(a)); tails.append(int(b)-round(1650*(h+origin)))
            assert len(heads)==len(windows)
            row['same_launch_boundaries']=dict(cores=len(heads), wrapper_start_to_first_instr_ticks=[min(heads),max(heads)],
                                               last_instr_to_wrapper_done_ticks=[min(tails),max(tails)],ticks_per_us=1650)
        rows.append(row)
    return rows


def snapshot():
    reg=read(OUT/'registration.json')
    new_freeze=read(OUT/'cast-v20-freeze.json')
    current={r['id']:r for r in new_freeze['jobs']}
    jobs=list(reg['jobs'])
    pro=read(OUT/'pro-freeze.json')
    jobs.append(pro['job']);current[pro['job']['id']]=pro['current']
    cases=[]
    for j in jobs:
        new=current[j['id']]
        row=dict(shape=j['shape'],targets=j['targets'],current=new['targets'],roles=new['roles'],devices={})
        for c in reg['chips']:
            name=j['id']+'_'+c
            file=OUT/(name+'.json')
            entry={'status':'运行中' if (RAW/name/'run.log').exists() else '已排队'}
            if (RAW/name/'paused-by-supervisor').exists():entry['status']='等待前一例释放核仿真槽位'
            if file.exists():
                result=read(file)
                if new['roles'][c].startswith('independent'):
                    assert result['cost']['started_utc']>(pro['utc'] if j['id']==pro['job']['id'] else new_freeze['utc'])
                if result['qualified']:
                    ca=result['ca_us']; stage=result['stage']; t,h=j['shape']
                    audit=OUT/(name+'-dma.json')
                    if not audit.exists():write(audit,dma_audit(result))
                    dma=read(audit)
                    entry=dict(status='全核 CA',ca_us=ca,old_error_pct=100*(j['targets'][c]['us']/ca-1),
                        signed_error_pct=100*(new['targets'][c]['us']/ca-1),
                        kernel=stage['kernel'],active_aiv=stage['active_aiv'],events=stage['total_events'],
                        logical_gbs=6*t*h/ca/1000,wall_s=result['cost']['seconds'],
                        raw_peak_bytes=result['cost']['peak_bytes'],peak_is_lower_bound=result['cost'].get('peak_is_lower_bound',False),source=file.name,
                        visible_read_pct=100*dma['visible_bytes'].get('MTE2',0)/(2*t*h),
                        visible_write_pct=100*dma['visible_bytes'].get('MTE3',0)/(4*t*h),
                        read_counts=dma['visible_transfer_sizes']['MTE2'],write_counts=dma['visible_transfer_sizes']['MTE3'])
                else:entry=dict(status='未通过计时或数值门槛', reason=result.get('reason'),source=file.name)
            row['devices'][c]=entry
        row['speedup']={'predicted':new['targets'][DT]['us']/new['targets'][PR]['us']}
        if all('ca_us' in r for r in row['devices'].values()):
            row['speedup']=dict(predicted=new['targets'][DT]['us']/new['targets'][PR]['us'],
                                measured=row['devices'][DT]['ca_us']/row['devices'][PR]['ca_us'])
        cases.append(row)
    return dict(cases=cases,controls=sample_controls())


def fmt(v):return '—' if v is None else f'{v:.2f}'


def report():
    from dram import archive_native
    for shape in ([32,4096],[262144,32],[262144,5120]):
        archive_native(*shape)
    data=snapshot()
    from compare import compare
    comparison=compare(data)
    write(OUT/'evaluation.json',data)
    from figures import draw
    draw(data)
    from tilefig import draw_tile_windows
    draw_tile_windows(data)
    text=['# 大 prefill 的单次 Cast 验证','','32k、128k 和 256k 均按 1024 的倍数计，单卡有 8 个请求。下表前三行把完整 prefill 全部放入一次 launch。Flash indexer 的 `weights_proj` 是输出 32 个系数的复制线性层，Attention TP 不切这 32 个系数。测量 `.float()` 对应的连续 BF16→FP32 `aclnnCast`，每端逐元素和精确 FP32 位模式比对。此测量只覆盖未融合的类型转换，模型前后各有计算，部署能否融合取决于后端。[Flash indexer 源码](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/models/deepseek_v41/indexer.py#L167-L176)','','两个宽通道样例取 Flash 和 Pro 的完整 hidden 维度，以 8192 / 4096 个单卡已调度 token 测 post-norm 向 FP32 升精度的 eager 路径。二者是宽通道 chunk。Kimi 的模型层结构单独列于场景清单，这里没有把相同 kernel 当成 Kimi 的执行证据。[Flash 回退路径](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/models/deepseek_v41/model.py#L843-L852) · [Pro 回退路径](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/models/deepseek_v4/model.py#L735-L743)','','## 冻结预测与实测','','所有设备耗时单位为 µs，误差为（预测 / CA − 1）×100%。时间窗从最早一核首条指令到最晚一核最后指令结束；每核均核对官方 duration 表。DT/PR 均调用 CANN 9.2.1 ops-math Cast，完整 kernel 名、二进制哈希和 trace 哈希见分 SKU 的 JSON。970DT 仅有资源模型外推。同集比较 v0.19 的全量外存定价和 v0.20 的 L2 写接收模型。独立性按 v0.20 第二次冻结时的进程状态判定。','','|单卡 shape / 角色|逻辑读+写 MiB|DT 旧 → 新 / CA / 新误差|PR 旧 → 新 / CA / 新误差|DT耗时/PR耗时 新 / CA|970DT 外推|','|---|---:|---|---|---|---:|']
    text.insert(text.index('## 冻结预测与实测'),'Pro 在 C4 indexer 的 30 层中用复制线性层保留 64 个系数。乘标量后，A5 的 prepare_dsa_indexer_weights 显式调用 weights.float()，表中单列真实完整 [262144,64]。这一条件调用没有计入 Flash 三种上下文的曲线。[Pro 索引算子](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/models/deepseek_v4/indexer.py#L238-L246) · [950 设备分支](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20b9c45c80b34ae3781440d90904a44023/vllm_ascend/device/device_op.py#L1588-L1590)\n')
    for row in data['cases']:
        t,h=row['shape'];d=row['devices'];pred=row['targets'];speed=row.get('speedup',{})
        def cell(c):
            x=d[c]
            return f"{pred[c]['us']:.2f} → {row['current'][c]['us']:.2f} / "+(f"[{x['ca_us']:.2f}]({x['source']}) / {x['signed_error_pct']:+.1f}%" if 'ca_us' in x else x['status']+' / —')
        role={'development':'开发','parallel replay':'并行回放','independent':'独立'}[row['roles'][DT].split(':')[0]]
        text.append(f"|{t} × {h} · {role}|{6*t*h/2**20:.0f}|{cell(DT)}|{cell(PR)}|{fmt(speed.get('predicted'))} / {fmt(speed.get('measured'))}|{row['current']['Ascend970DT']['us']:.2f}|")
    text+=['','同集汇总保留每个样本的首次官方窗口。WAPE 为绝对误差之和除以 CA 耗时之和，跨芯片比值单独计算。','|角色|DT 数量 / WAPE 旧→新|PR 数量 / WAPE 旧→新|比值数量 / WAPE 旧→新|','|---|---:|---:|---:|']
    for role,label in [('prior_holdout_replay','v0.19 十个小样例回放'),('development','开发'),('parallel_replay','冻结时已运行'),('independent','冻结后启动')]:
        summary=comparison[role]['summary']
        def stats_cell(key):
            a,b=summary[key+'_old'],summary[key+'_current']
            return '0 / N/A' if not a else f"{a['n']} / {a['wape_pct']:.1f}% → {b['wape_pct']:.1f}%"
        text.append('|'+label+'|'+'|'.join(stats_cell(k) for k in ('dt','pr','speedup'))+'|')
    text+=['','[逐 shape 的旧版/新版误差、最大误差和加速比](paired-comparison.json)']
    text+=['','## 瓶颈、分块和采集开销','','逻辑流量包含每元素 2 B 读和 4 B 写。下表的等效带宽用该流量除以完整 CA 耗时，描述算子的吞吐。它可以高于外存带宽，因为入参可在缓存中、输出也可能尚未全部从 L2 回写到外存。队列可见率是 profiler 的 DMA 描述符 `processed_bytes` 之和除以张量语义字节量；即使全部可见，OUT↔UB 仍无法确定 LPDDR/HBM 实际流量。','','|shape|SKU|核数 模型/CA|每核 UB 轮数预测|DT/PR 等效带宽 TB/s|MTE2/MTE3 可见字节 %|宿主墙钟 min|单例采集峰值 GiB（≥ 表示下界）|','|---|---|---:|---:|---:|---:|---:|---:|']
    for row in data['cases']:
        for c, x in row['devices'].items():
            if 'ca_us' not in x:continue
            p=row['targets'][c];t,h=row['shape']
            text.append(f"|{t} × {h}|{'DT' if c==DT else 'PR'}|{p['active_aiv']} / {x['active_aiv']}|{p['plan']['busy_core_waves']}|{x['logical_gbs']/1000:.2f}|{x['visible_read_pct']:.1f} / {x['visible_write_pct']:.1f}|{x['wall_s']/60:.1f}|{'≥' if x.get('peak_is_lower_bound') else ''}{x['raw_peak_bytes']/2**30:.2f}|")
    text+=['','首个整批点的 48 MiB 总逻辑数据可装入 DT 的 128 MiB 或 PR 的 112 MiB L2。按所有字节走 4.0 / 1.4 TB/s 外存计算，光数据项便是 12.58 / 35.95 µs，已经超过两端完整 CA。旧模型强制该项构成设备时间下界，因此高估；仅按核数判断不会发现问题。DT 的 DRAM 交易证据见下表。更大的点逐渐超过 L2，能检验小点上观察到的现象能否延续。','','## 低采集量的 950DT 全宽实验','','额外单次派发 [262144,5120]，对应 8 × 32ki 的完整 hidden。含 1,342,177,280 个元素，输入 2.5 GiB，输出 5 GiB。为它单独登记 4 小时 / 8 GiB 原始日志预算。在同一物理 CA 上直接调用 `aclnnCast`，保留所有核的 wrapper start/done 时间戳和金标。wrapper 包含指令执行前的启动等待，下面以相同 shape 的规范指令窗校验时钟和边界。此实验只提供 wrapper 口径的独立观测，不纳入上表误差。','','|对照 shape|原生 wrapper µs|正式指令窗 µs|原生 / 正式墙钟 s|原生 / 正式峰值 MiB|','|---|---:|---:|---:|---:|']
    for c in data['controls']:
        t,h=c['shape'];text.append(f"|{t} × {h}|{c['wrapper_us']:.3f}|{c['instruction_us']:.3f}|{c['native_wall_s']:.0f} / {c['official_wall_s']:.0f}|{c['native_peak_mib']:.0f} / {c['official_peak_mib']:.0f}|")
    text+=['','原生 DT CA 在 128 个 HBM 控制器的每个采样区间分别给出 DMC 与 DDR 字节和累计读写命令。下表汇总整个进程，因此包含可能的取指、元数据和窗口边界外事务；DRAM 事务的实际大小会变化。','','|DT shape|张量逻辑读 / 写 MiB|控制器 DMC / DDR 总 MiB|外存读 / 写命令|','|---|---:|---:|---:|']
    for t,h in ((32,4096),(262144,32),(262144,5120)):
        fc=OUT/f'cast_{t}_{h}_native-dram.json'
        if fc.exists():
            m=read(fc)
            text.append(f"|{t} × {h}|{m['logical_read_bytes']/2**20:g} / {m['logical_write_bytes']/2**20:g}|[{m['dmc_bytes']/2**20:.3f} / {m['ddr_bytes']/2**20:.3f}]({fc.name})|{m['read_commands']:,} / {m['write_commands']:,}|")
    text+=['','[262144,32] 的模拟 HBM 在全部通道上合计 0 个写命令，16,822,272 B 的实际流量比 16 MiB 输入只多 45,056 B；即使通过金标校验，输出 32 MiB 也没有在这次 CA 的 HBM 事务中出现。PR 还缺少同类物理通道计数。']
    for row in data['cases']:
        t,h=row['shape'];fc=OUT/f'cast_{t}_{h}_{DT}-official-dram.json'
        if fc.exists():
            m=read(fc)
            text+=['',f"DT 同一次正式 CA 对 [{t},{h}] 的 128 个控制器记录了 {m['dmc_bytes']/2**20:.3f} MiB DMC 流量，合计 {m['write_commands']:,} 条写命令。[物理通道记录]({fc.name})"]
    for c in data['controls']:
        if 'same_launch_boundaries' in c:
            b=c['same_launch_boundaries']
            text+=['',f"同一正式 dispatch 的 {b['cores']} 核同时有 wrapper 和指令两组时标。在 1650 tick/µs 时，各核 wrapper start 到首条指令相差 {b['wrapper_start_to_first_instr_ticks']} tick，最后一条指令到 wrapper done 相差 {b['last_instr_to_wrapper_done_ticks']} tick。该差值不能作为跨 shape 的固定校正。"]
    from scaling import terminal_intervals
    intervals=terminal_intervals(data)
    details=['**稳态发起间隔。** 对最后完成的 AIV，只选择完整 UB 块的 DMA，读取相邻同队列请求的发起间隔中位数（µs），另列单条请求的在途时间。两次请求可以同时在途。','','|shape / SKU|模型每满块稳态 µs|CA 读发起间隔 / 单条读在途|CA 写发起间隔 / 单条写在途|','|---|---:|---:|---:|']
    for row in intervals:
        t,h=row['shape'];r=row['MTE2'];w=row['MTE3']
        details.append(f"|{t} × {h} / {'DT' if row['sku']==DT else 'PR'}|{fmt(row['model_full_steady_tile_us'])}|{fmt(r['issue_interval_median_us'])} / {fmt(r['duration_median_us'])}|{fmt(w['issue_interval_median_us'])} / {fmt(w['duration_median_us'])}|")
    if (OUT/'cast_1048576_32_Ascend950DT_9582-official-dram.json').exists():
        details+=['','128ki×8 的 DT 在首次 64 MiB 冷读之外发生了脏写。当前模型预计 69.16 MiB 外存流量，控制器观测 94.88 MiB。即使全按实测字节除以配置 4 TB/s，加入 1753 周期首次发起也只得到约 25.93 µs，仍小于实测 35.53 µs。末核满块发起间隔为 1.23 µs，模型局部稳态为 0.80 µs；这揭示还需要反映大片读写并存时的服务节奏。此比较未修改冻结模型。[各芯片和 shape 的满块发起记录](tile-intervals.json)']
    index=text.index('## 低采集量的 950DT 全宽实验')
    text[index:index]=details+['']
    stream=[row for row in data['cases'] if row['shape'][1]==32]
    increments=[]
    for lower,upper in zip(stream,stream[1:]):
        for sku in (DT,PR):
            a=lower['devices'][sku].get('ca_us');b=upper['devices'][sku].get('ca_us')
            if not a or not b:continue
            small,big=lower['shape'][0],upper['shape'][0]
            increments.append(f"|{small//8192}ki → {big//8192}ki|{'DT' if sku==DT else 'PR'}|{big/small:.1f}×|{b/a:.2f}×|{(b-a)*1000/(big-small):.4f}|")
    if increments:
        index=text.index('## 低采集量的 950DT 全宽实验')
        text[index:index]=['## 数据量扩大后的变化','','相邻两次同实现、同单卡 batch 的完整调用能给出区间斜率。ns/token 针对一个 token 的 32 个 indexer 系数。初次装填流水、L2 可留驻的输出量和 spill 均可能随规模改变，三个点仅能检验这里覆盖的区间。','','|请求上下文|SKU|数据量倍数|CA 时延倍数|差分 ns/token|','|---|---|---:|---:|---:|']+increments+['']
    f=OUT/'cast_262144_5120_native.json'
    if f.exists():
        res=read(f)
        if res['qualified_core_envelope']:
            text+=['',f"完整全宽实测通过全部 {res['correct_elements']:,} 个元素的比较，{res['active_aiv']} 个 AIV wrapper 总窗口为 **{res['core_envelope_us']:.3f} µs**。此 shape 首次冻结的规范窗口预测是 {read(OUT/'native-freeze.json')['jobs'][0]['prediction_us']:.3f} µs，写接收模型复算为 {read(OUT/'cast-v20-freeze.json')['jobs'][-1]['targets'][DT]['us']:.3f} µs。宿主 {res['cost']['seconds']/60:.1f} min，采集峰值 {res['cost']['peak_bytes']/2**30:.2f} GiB。[完整全宽实测]({f.name})"]
        else:text+=['','全宽已结束，尚未通过完整窗口/金标门槛，详见[原始结果]('+f.name+')。']
    else:text+=['','全宽任务已启动；通过全部输出和逐核结束门槛后会单独列出设备窗口。']
    for fn,cap in [('indexer-timeline.png','32ki × 8 的模型和最晚完成 AIV 流水'),('full-hidden.png','整批全宽的模型阶段与全核 CA wrapper'),('indexer-cast.png','整批 Flash Cast 的冻结预测、CA 与等效带宽')]:
        if (OUT/fn).exists():text+=['',f'![{cap}]({fn})']
    for fn,label in [('flash-32ki-tiles.png','Flash 32ki'),('flash-128ki-tiles.png','Flash 128ki'),('flash-256ki-tiles.png','Flash 256ki'),('pro-32ki-tiles.png','Pro 32ki')]:
        if (OUT/fn).exists():
            text+=['',f'**{label} 的双端流水。** 上栏给出模型暴露阶段，以及同一满 UB 块在核内和共享资源中分别对应的服务时间。预测读写配对间隔取二者较大值，它是由吞吐折算的稳态均值。下栏给出末核最早两次、中间和最晚两次可见读写及转换的实测时间。',f'![{label} 末核和预测同轴流水]({fn})']
    costs_file=OUT/'query-wall.json'
    if costs_file.exists():
        q=read(costs_file);worst=max(v['maximum_ms'] for v in q['rows']);p95=max(v['p95_ms'] for v in q['rows'])
        text+=['',f"系统仿真的 Python 首次 API 调用耗时 {q['first_api_ms']:.3f} ms（含资源模块首次加载），其后 21 组查询各重复 31 次，各组 P95 最大 {p95:.3f} ms，最慢单次 {worst:.3f} ms。CA 宿主墙钟见上表。[预测开销](query-wall.json)"]
    text+=['','[v0.20 冻结预测与角色](cast-v20-freeze.json) · [原始预测与预算](registration.json) · [原生对照登记](native-freeze.json) · [机器可读指标](evaluation.json)']
    (OUT/'report.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    previous=None
    while True:
        # A report, two figures and CA audits are regenerated only on a change
        # in outcomes, counter artifacts or the queued/running dispatches.
        state=tuple(sorted(str(p) for mask in ('cast_*_Ascend950*.json','cast_*_native.json','*-dram.json') for p in OUT.glob(mask) if not p.name.endswith('-dma.json')))+tuple(sorted(str(d) for mask in ('*/run.log','*/paused-by-supervisor') for d in RAW.glob(mask)))
        if state!=previous:
            report()
            previous=state
        if not args.watch:break
        complete=all((OUT/(j['id']+'_'+c+'.json')).exists() for j in read(OUT/'registration.json')['jobs'] for c in (DT,PR))
        if complete and all((OUT/('cast_262144_64_'+c+'.json')).exists() for c in (DT,PR)) and (OUT/'cast_262144_5120_native.json').exists():break
        time.sleep(60)
