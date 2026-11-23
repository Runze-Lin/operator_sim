"""Extract per-core FP4 tile phases from each historical physical CA shard once."""
import argparse
from collections import Counter
import json
from pathlib import Path

from calibration.__main__ import read, require
from calibration.mx_matmul.shards import trace_events


def touched_scale_lines(descriptor):
    """Batched descriptors may write several rows at the full output row pitch."""
    lines = set()
    for index in range(descriptor['burst_number']):
        start = descriptor['dst_addr'] + index * descriptor['dst_stride']
        last = start + descriptor['burst_len'] - 1
        lines.update(range(start // 32, last // 32 + 1))
    return lines


def extract(raw):
    stage, = read(raw / 'receipt.json')['stages']
    origin = min(core['first_tick'] for core in stage['cores'].values())
    profile, = (raw / 'profile').glob('OPPROF_*')
    rows, memberships = [], {}
    writers = Counter()
    for core_name, core in stage['cores'].items():
        reads, writes, vectors = [], [], []
        for event in trace_events(profile / 'simulator' / core_name / 'trace.json'):
            if event['ph'] != 'X':
                continue
            desc = json.loads(event['args']['detail'])
            start = round(1650 * event['ts']) - origin
            retire = round(1650 * (event['ts'] + event['dur'])) - origin
            if event['name'] == 'MOV_SRC_TO_DST_ALIGNv2':
                transfers = reads if desc['instr_type'] == 'MTE2' else writes
                transfers.append((start, retire, desc))
            if desc['instr_type'] == 'RVECEX':
                vectors.append((start, retire, event['name']))
        reads.sort(key=lambda transfer: transfer[0])
        require(len(reads) == 2 and len(writes) == 2 and reads[0][2]['processed_bytes'] == 160,
                'Expected the header and payload for one historical UB tile')
        payload, scale = sorted(writes, key=lambda transfer: transfer[2]['processed_bytes'], reverse=True)
        x = reads[1]
        require(x[2]['processed_bytes'] == 4 * payload[2]['processed_bytes'], 'Packed payload size mismatch')
        last_vector = max(v[1] for v in vectors)
        rows.append(dict(core=core_name, first_read=x[0], read_bytes=x[2]['processed_bytes'],
                         read_latency=x[1]-x[0], read_retire=x[1], first_vector=min(v[0] for v in vectors),
                         last_vector=last_vector, zero_vf_markers=sum(op=='RV_PLT' for _,_,op in vectors),
                         last_vf_to_store=payload[0]-last_vector, write_issue=payload[0],
                         data_bytes=payload[2]['processed_bytes'], data_retire=payload[1],
                         data_cycles=payload[1]-payload[0], scale_addr=scale[2]['dst_addr'],
                         scale_bytes=scale[2]['processed_bytes'], scale_cycles=scale[1]-scale[0],
                         scale_retire=scale[1], scale_issue=scale[0], end=core['retire_tick']-origin))
        memberships[core_name] = touched_scale_lines(scale[2])
        writers.update(memberships[core_name])
    for row in rows:
        row['writers'] = max(writers[line] for line in memberships[row['core']])
    return rows


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('root', type=Path)
    cli.add_argument('output', type=Path)
    args = cli.parse_args()
    data = {p.name: extract(p) for p in sorted(args.root.iterdir()) if (p/'receipt.json').is_file()}
    args.output.write_text(json.dumps(data))
