"""Count all executed instructions by AIC from official aggregated CSVs, without decoding the timeline."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.matmul_dispatch30.device import OUT,ca,save

COLUMNS=('MMAD','MOV_OUT_TO_L1_MULTI_ND2NZ','LOAD_2Dv2','FIX_L0C_TO_DST')


def census(file):
    measurement=ca.read(file)
    if not measurement['qualified']:raise ValueError('A complete reference window is required')
    identity=measurement
    if 'chip' not in identity:
        initial=file.parent/measurement['acquisition_receipt']
        if ca.digest(initial)!=measurement['acquisition_sha256']:raise ValueError('Initial receipt mismatch')
        identity=ca.read(initial)
    stage,=measurement['stages']
    global_trace=ROOT/stage['trace']
    started=perf_counter(); reading=0; aggregates={}; fingerprints={}
    for core in sorted(stage['windows']):
        csv,=(global_trace.parent/core).glob('*_instr_exe*.csv')
        by_name=Counter()
        # The detail field can contain raw commas/quotes; only the fixed prefix is needed.
        # Hash the exact bytes read, so revisiting a model does not require hashing multi-GB timelines.
        content=csv.read_bytes();reading+=len(content)
        fingerprints[core]=hashlib.sha256(content).hexdigest()
        lines=content.splitlines()
        if not lines or lines[0].split(b',')[:4]!=[b'instr',b'addr',b'pipe',b'call_count']:
            raise ValueError(f'Unexpected columns in {csv}')
        for line in lines[1:]:
            name,address,pipe,count,tail=line.split(b',',4)
            n=int(count)
            if n<1:raise ValueError('Invalid executed count')
            by_name[name.decode()]+=n
        aggregates[core]={k:by_name[k] for k in COLUMNS}
    wall=perf_counter()-started
    total={k:sum(v[k] for v in aggregates.values()) for k in COLUMNS}
    if total['MMAD'] != stage['mmad']:raise ValueError('Official instruction census and qualified global trace disagree')
    return dict(chip=identity['chip'],shape=identity['shape'],receipt=str(file.relative_to(ROOT)),ca_us=measurement['ca_us'],
                active_aic=stage['active_aic'],active_aiv=stage['active_aiv'],kernel=stage['kernel'],csv_bytes=reading,
                aggregate_trace_bytes=global_trace.stat().st_size,elapsed_s=wall,
                instruction_counts=total,core_counts=aggregates,csv_sha256=fingerprints,
                reference_mmad=stage['mmad'],global_trace_sha256=stage['trace_sha256'])


def run():
    files=[ROOT/'results/v0.28/matmul/262144-128-512-dt-drain.json',
           ROOT/'results/v0.28/matmul/262144-128-512-pr-drain.json',
           ROOT/'results/v0.28/matmul/long-prefill/131072-dt.json']
    for row in ca.read(OUT/'score.json')['rows']:
        if row['status']=='qualified' and row['shape'][0]<=144:files.append(OUT/row['source'])
    rows=[census(p) for p in files]
    save(OUT/'census.json',dict(scope='Cost of offline work-count extraction. Kernel identity, numerical validity and global timing are taken from separately qualified receipts. No inference of instruction issue/retirement from per-PC sums.',rows=rows))
    header=['# 在大形归档里快速核对实际工作量','',
        '官方逐核 instr_exe.csv 已将同一指令位置的执行合并，call_count 给出实际次数。在全部参与核逐条相加，能复核 MMAD、读页、L0 装载及 FIXP 写出的发起数。每次都与先前合格的全局原 trace MMAD 总数逐项校验。每行的读取 MiB 是程序本次实际消费的压缩前 CSV 字节；对照 GiB 是单份全片 trace.json 的文件长度。','',
        '| M | 芯片 | AIC | MMAD | GM→L1 / L1→L0 / FIXP | CSV MiB / 全片 JSON GiB | 本次离线核对 s |','|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        c=r['instruction_counts']
        header.append(f"| {r['shape'][0]:,} | {r['chip']} | {r['active_aic']} | {c['MMAD']:,} | {c[COLUMNS[1]]:,} / {c[COLUMNS[2]]:,} / {c[COLUMNS[3]]:,} | {r['csv_bytes']/2**20:.2f} / {r['aggregate_trace_bytes']/2**30:.3f} | {r['elapsed_s']:.3f} |")
    header+=['','这是复用已有 trace 的离线成本，受主机文件缓存影响，不是设备仿真或首次采集的墙钟。CSV 缺少全局同步的逐条发起/退休时刻，不能据此把流水周期相加求设备时长。需要定位空泡时，只读取已确认末核的原始指令与依赖。[逐核计数、输入摘要与原始回执](census.json) · [大形关键核算例](../../matmul/report.md)','']
    (OUT/'census.md').write_text('\n'.join(header))


if __name__=='__main__':run()
