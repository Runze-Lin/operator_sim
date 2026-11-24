"""Extract split-row stage prices from the published QSLI device binary's PCs."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

from calibration.mx_matmul.collect import trace_events

SOURCE = Path(__file__).parent
CLOCK = 1650
READ_RATE = 80.06841783750764  # attention_resources.mte_bytes_per_cycle
DEVICE = 'b3a854470fe46368ac752ccccd7ed0bb7ffb087b658373c2ac97bfce027ae4ff'

# Offsets relative to kernel PC base. The paired GM reads/writes follow the
# partial key-bit/ID loops in LdMergeStage, after the two WAIT_FLAG_DEV sites.
PART_WRITE = 0x1054c
MERGE_PART_READ = (0x10ce4, 0x10d00)
MERGE_CARRY_READ = (0x10c50, 0x10c70)
MERGE_OUT = (0x10bd8, 0x10bf8)


def window(raw, parts, wide):
    base, = raw.rglob('simulator')
    elf = (base.parent/'dump/aicore_binary.o').read_bytes()
    assert sha256(elf).hexdigest() == DEVICE
    per_pc = defaultdict(list)
    local_writes, issues, retires = [], [], []
    origin = float('inf')
    for shard in sorted(base.glob('core*.*core*/trace.json')):
        cube_starts = []
        for e in trace_events(shard):
            if e.get('ph') != 'X':
                continue
            start = round(e['ts']*CLOCK)
            origin = min(origin, start)
            end = round((e['ts'] + e['dur'])*CLOCK)
            if e['name'] == 'MMAD_MX' and e['dur'] > 0:
                cube_starts.append(start)
                retires.append(end)
            if e['name'] != 'MOV_SRC_TO_DST_ALIGNv2' or e['dur'] <= 0:
                continue
            pc = int(e['args']['pc_addr'], 16) % (1 << 20)
            if pc == PART_WRITE and shard.parent.name.endswith('.veccore0'):
                local_writes.append(end)
            if shard.parent.name == 'core0.veccore0':
                per_pc[pc].append((start,end))
        if cube_starts:
            issues.append(min(cube_starts))
    assert len(issues) == parts and len(local_writes) == parts
    for stamps in per_pc.values():
        stamps.sort()
    rd0 = per_pc[MERGE_PART_READ[0]]
    rd1 = per_pc[MERGE_PART_READ[1]]
    wa, wb = (per_pc[k] for k in MERGE_OUT)
    carry0, carry1 = (per_pc[k] for k in MERGE_CARRY_READ)
    assert len(rd0) == len(rd1) == parts and len(wa) == len(wb) == (parts+6)//7
    passes = []
    for i in range(len(wa)):
        j = 7*i
        n = min(7, parts-j)
        inputs = rd0[j:j+n] + rd1[j:j+n]
        if i:
            inputs += [carry0[i-1], carry1[i-1]]
        length = (n + int(i>0)) * 512
        first_rd = min(s for s,_ in inputs)
        last_rd = max(e for _,e in inputs)
        passes.append(dict(lists=n, scored_items=length, start=first_rd,
                           read=last_rd-first_rd, select=wa[i][0]-last_rd,
                           write=wb[i][1]-wa[i][0], end=wb[i][1]))
    earliest = min(issues)-origin
    # The earliest MTE2 setup is not isolated. Retain the effective launch
    # control remainder after the modeled three traffic budgets up to MMAD.
    fill = 2*34816/READ_RATE + (32*64 if wide else 0)
    return dict(parts=parts, wide_table=wide, first_mmad_issue=earliest,
                startup_residual=earliest-fill,
                last_mmad_to_local_write=max(local_writes)-max(retires),
                local_write_to_merge=passes[0]['start']-max(local_writes),
                passes=passes)


def summarize(rows):
    passes=[p for row in rows.values() for p in row['passes']]
    # Each input length first contributes its median; robust slopes are over
    # distinct chunk counts to avoid over-weighting 8-list carry passes.
    chunk_samples=defaultdict(list)
    for p in passes:
        chunk_samples[p['scored_items']//256].append(p['select'])
    centers={n:median(values) for n,values in chunk_samples.items()}
    slope=median((y2-y1)/(n2-n1) for n1,y1 in centers.items()
                 for n2,y2 in centers.items() if n1<n2)
    def middle(field):
        return round(median(r[field] for r in rows.values()),3)
    return dict(startup_control_cycles=middle('startup_residual'),
                one_tile_score_drain_cycles=middle('last_mmad_to_local_write'),
                barrier_setup_cycles=middle('local_write_to_merge'),
                read_latency_cycles=round(median(p['read']-6*p['scored_items']/READ_RATE for p in passes),3),
                select_fixed_cycles=round(median(y-n*slope for n,y in centers.items()),3),
                select_cycles_per_256=round(slope,3),
                write_latency_cycles=round(median(p['write']-6*512/READ_RATE for p in passes),3))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',type=Path)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    rows={}
    for sku, teams in [('Ascend950DT_9582',32),('Ascend950PR_957b',28)]:
        for suffix,folder,n,wide in [('static','candidate-v20-aot',2,False),
                ('wide','candidate-v20-aot',2,True),('ipc','candidate-v20-long-row',teams,True)]:
            rows[sku+'-'+suffix]=window(args.runs/folder/(sku+'-'+suffix),n,wide)
    data=dict(role='development replay only; two SKUs with two-way splits plus saturated split rows; no Q32/Q64 holdout input',
              units='CA cycles at 1.65GHz; read/select/write measured at the FD owner AIV',
              device_sha256=DEVICE, records=rows, parameters=summarize(rows))
    args.out.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data['parameters']))

if __name__=='__main__':
    main()
