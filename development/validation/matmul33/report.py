"""Summarize matched archival timings without interpreting offline replay as a shorter CA run."""
from statistics import median
from time import perf_counter_ns
from watch import OUT, digest, read, save

def benchmark(p):
    wall=[]
    for _ in range(100):
        t=perf_counter_ns()
        d=read(p)
        observed = max(v['retire_us'] for v in d['cores'].values()) - min(v['first_us'] for v in d['cores'].values())
        assert observed == d['compact_us'] and sum(v['mmad'] for v in d['cores'].values())==d['all_mmad']
        wall.append((perf_counter_ns()-t)/1e6)
    return median(wall)

def main():
    rows=[]
    native=read(OUT/"native-replay.json")["rows"] if (OUT/"native-replay.json").exists() else []
    native_by_cert={r["archive_certificate"]:r for r in native}
    for p in sorted(OUT.glob('*-certificate.json')):
        d=read(p)
        r={k:v for k,v in d.items() if k not in ('cores','source_log_sha256','reference_sha256','trace_sha256')}
        r.update(certificate=p.name,size_bytes=p.stat().st_size,certificate_sha256=digest(p),
                 certificate_reuse_p50_ms=benchmark(p),active=len(d['cores']))
        fast=native_by_cert.get(p.name)
        if fast:
            assert fast["archive_certificate_sha256"]==r["certificate_sha256"]
        r["native_offline_s"]=fast["native_wall_s"] if fast else None
        rows.append(r)
    rows.sort(key=lambda r:(r['shape'][0],r['sku']))
    assert len(rows)>=8
    save('cost-summary.json',dict(rows=rows,scope='Host costs observed in retained official CA archives. First two intervals partition process wall based on STARS file mtime, not exclusive simulator or export stage time. Re-read and reduce costs are offline.'))
    t=['# 大 Matmul 的设备时间与 CA 采集成本','',
       '同一份 aclnnMatmul 实跑，同时用原全局时间线和逐核归约复核设备窗。下表前两行是 Pro 的 B11、每请求校验 9 个 token；其他行是 Flash 的 indexer.wk。B8 的 M=32,768、262,144、1,048,576 分别对应每请求 4Ki、32Ki、128Ki 个 token 整批一次提交。模型是否选择 chunk 由系统调用方决定。','',
       '| M,N,K | 芯片 | 实际参与核 | 逐核归约 / 原全窗 µs | MMAD | 最长单核表值 µs | STARS 两事件之差（原始 tick） |',
       '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        label=r['sku'].split('_')[0].removeprefix('Ascend')
        t.append(f"| {','.join(str(v) for v in r['shape'])} | {label} | {r['active']} | {r['compact_us']:.5f} / {r['full_us']:.5f} | {r['all_mmad']:,} | {r['longest_table_duration_us']:.2f} | {r['task_ticks'][1]-r['task_ticks'][0]:,} |")
    t+=['','对参与核 $i$ 和其指令 $j$，令 $s_{ij}$ 为首发设备时间、$e_{ij}$ 为含在途描述符的退休时间。全窗 $T=\\max_{i,j} e_{ij}-\\min_{i,j}s_{ij}$。逐核文件必须全部出现，逐核物理 END 各一次、位于本核指令窗内。检查全核 MMAD 计数、逐核名册、官方逐核时长表精度，以及精确 ELF 和所有输出的数值回执。每核只保留首发、退休、物理 END、计数及指纹。归约后的结果可用于复算总时间和工作量；研究哪个 tile 等待哪条读指令，仍需相应核的原始时间线。','',
       '最长单核的起点可能晚于全片首发。两条 STARS 记录属于任务调度边界，记录中没有可独立核实的时钟单位。即使按 1 ns/tick 解释，在 Pro decode 的 PR 端也得到 9.070 µs，全指令窗为 6.888 µs；Flash 128Ki PR 则得 806.786 µs 与 804.867 µs。按绝对微秒补一个常数也与这些不同的端点不符。','',
       '## 墙钟与归档大小','',
       '前段从原采集进程开始到含 TASK_DONE 的 STARS 文件最后一次落盘，涵盖初始化、CA 执行及同时发生的记录和 I/O。后段从该文件最后落盘到原采集进程退出，包括数值回读、归档解析与收尾。文件时间取自保留原目录；这两段相加等于一次运行的总墙钟，不能从它们分离真实执行与同时进行的磁盘写入。离线扫描另在原归档上实测，受当时并发负载和页缓存影响。','',
       '| M | 芯片 | 到 STARS 落盘 / 余下 / 合计 min | 原采集峰值 GiB | 全局 JSON / 逐核 JSON GiB | 逐核离线归约 Python / C++ s | 证书 KiB | 证书复算 P50 ms |',
       '|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        fast=f"{r['native_offline_s']:.2f}" if r['native_offline_s'] is not None else 'N/A'
        t.append(f"| {r['shape'][0]:,} | {r['sku'].split('_')[0][6:]} | {r['wall_to_stars_dump_mtime_s']/60:.2f} / {r['wall_after_stars_dump_mtime_s']/60:.2f} / {r['acquisition_wall_s']/60:.2f} | {r['peak_bytes']/2**30:.2f} | {r['global_file_bytes']/2**30:.3f} / {r['shard_input_bytes']/2**30:.3f} | {r['observed_offline_reduce_s']:.2f} / {fast} | {r['size_bytes']/1024:.1f} | {r['certificate_reuse_p50_ms']:.3f} |")
    t+=['','128Ki 两端原采集均约 95% 的墙钟耗在 TASK_DONE 最后落盘前。证书复算读取本地文件、解析 JSON 并重算全核 min/max 和 MMAD，100 次取中位数。首次归约计时包括逐核读文件、解析和哈希，不含模拟、整片旧回执的生成、写证书或后续绘图。独立的单线程 C++ 读取器用 mmap 每次打开一个核的数据，对八个端的每个核逐项复核了 SHA-256、完整事件数、物理 END、最早首发、最晚退休和 MMAD，逐项完全一致。百万行的原始重扫从 DT/PR 的 42.39/40.99 s 减至 13.82/14.30 s；两次扫描的页缓存和主机负载可能不同，未计一次性编译时间。磁盘用量来自观测峰值，保留文件的大小只是其中一部分。','',
       '## 是否可降低首次采集量','',
       '安装版 msopprof simulator --help 说明 950 的额外指令 dump 默认为关闭，本次原采集已用 --dump=false；帮助中规定 PipeUtilization 为必需项。--core-id 选择参与解析的核；缩减到一个核可能丢掉本例的全片起点或最晚完成者。npusim record 提供全核 SQLite 记录和可选报告，指定 -n 会限制产生日志的核。上一轮小形归档存在 12 个 DB，覆盖 11 个不同几何。重新对每个 DB 做 SQLite 完整性、全核物理 END、所有记录的最小起点与最大退休、内核元数据周期、输出 golden 及记录指纹交叉核对。10 个通过原生口径；其余两个均为 [16,192,384]，表无执行事件，原宿主退出码为 1、252。','',
       '通过的 10 个均位于 950DT，记录内核为 entry_kernel_2，object_path 为空，且其中没有本文长投影。扫描 runs 目录的调用日志，10 个相同几何的 msopprof 配对实际都在 950PR，不能识别同芯片采集路径的差别。进一步从 SQLite 提取每个已执行 PC 的原始机器码；每形 893–901 个不同位置的编码在同形 PR 官方 ELF 的整个 .text 中只匹配一处，地址 0x13b18 对应 high_performance_2 符号。未执行的 ELF 字节及原生宿主加载的对象本体仍未认证。已能交叉验证执行路径，并重建各自的原生设备耗时。首次采集若改用紧凑模式，需要事前固定同 SKU、同 provider/映射到同一安装 ELF、同输入和完整 golden、相同核名册与计时零点并验证退休边界。当前 64Ki 仍沿用原全核采集。复用现有证书无需启动 CA；已出版的算例保留原文件以支持重新检查。首次采集的可节约墙钟和落盘量尚待同形同实现实测。','',
       '[同一次运行的逐核证书、原始输入和复算成本](cost-summary.json) · [逐核无损重扫的实测开销](native-replay.json) · [原生 SQLite 的逐样本复核](compact-db.json) · [原生指令到官方 ELF 的逐条机器码映射](sqlite-kernel-identity.json) · [冻结的 64Ki 监测契约](monitor-freeze.json) · [64Ki 最新状态与终态回执](64ki-state.json)','']
    from watch import MODEL,SHAPE
    prices={row['chip']:row for row in read(MODEL)['rows'] if row['shape']==SHAPE}
    d=read(OUT/'64ki-state.json');measured=d.get('ca_us')
    t+=['## B8 × 64Ki 的运行与校验','',
        '本例输入为 BF16 的 [524288,512]，权重 [512,128]，数值资格要求逐一校验 67,108,864 个输出。v0.32 的 950DT 预测冻结时该 CA 已启动且尚未出结果，列作在途盲评；950PR 预测在其开始采集前冻结；970DT 为芯片参数外推。本次 950DT 原始采集登记的预算为 6 小时和 24 GiB。','',
        '| 目标 | 冻结模型 µs | AIC | MMAD | CA µs | 有符号误差 |','|---|---:|---:|---:|---:|---:|']
    for name,row in sorted(prices.items()):
        actual=measured if name=='Ascend950DT_9582' else None
        label=name.removeprefix('Ascend').split('_')[0]+(' 外推' if name=='Ascend970DT' else '')
        result=f'{actual:.5f} | {100*(row["us"]/actual-1):+.2f}%' if actual else 'N/A | N/A'
        t.append(f'| {label} | {row["us"]:.3f} | {row["aic"]} | {row["plan"]["mmads_total"]:,} | {result} |')
    t+=['','此处的时长只接收同次 DT 完整数值、安装库 ELF 指纹、全核 END 与退休边界校验通过的设备窗；PR 的回执由原串行采集登记维护。设备进度打印没有进入时延栏。[单次 DT 的状态与回执](64ki-state.json) · [两端原采集登记](../../v0.30/matmul-dispatch/device/freeze.json) · [三芯片事前预测和盲评口径](../../v0.32/matmul/freeze.json)','']
    host=read(OUT/'host-cost-observation.json') if (OUT/'host-cost-observation.json').exists() else None
    if host:
        t+=[f"在 {host['utc'][:16]} UTC 只读采了一次真实宿主快照。这一 64Ki 运行已用 {host['process_wall_s']/60:.1f} 分钟，该模拟进程挂有 {host['live_threads']} 个宿主线程，允许 {host['allowed_logical_cpus']} 个逻辑 CPU；至该时刻累计 CPU 时间 / 墙钟相当于平均使用 {host['avg_host_cpus_process_only']:.1f} 个逻辑 CPU。整机两秒采样的 idle+iowait 占 {100*host['two_second_host_idle_or_iowait_fraction']:.1f}%。历史上 M 翻倍的 128Ki DT 全次运行仅 98.19 分钟，进程树 CPU 时间 / 墙钟为 38.9。两次运行的亲和范围、同时运行的作业及主机条件未配对，且一个统计单进程，一个统计整棵进程树。墙钟不宜沿 token 数线性外推。此前 128Ki 的设备窗按全核指令计为 405.617 µs。[宿主快照与八次实跑 CPU 账本](host-cost-observation.json)",""]
    (OUT/'report.md').write_text(chr(10).join(t))
if __name__=='__main__':main()
