"""Extract MX Cube/Vector phase evidence from the physical-core CA traces."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median
from calibration.mx_matmul.shards import trace_events


def summarize(raw):
    """All times are in microseconds from the first instruction across cores."""
    # CANN can leave symlink aliases for the canonical simulator directory.
    files = sorted({p.resolve() for p in raw.rglob('core*.*core*/trace.json')})
    if not files or len({p.parent.parent for p in files}) != 1:
        raise ValueError('Expected physical-core shards for exactly one device kernel')
    shapes = Counter()
    cube_cores, vector_cores = [], []
    totals = Counter()
    phases = {}
    banks = {}
    earliest, latest = float('inf'), float('-inf')
    for path in files:
        core = path.parent.name
        work = defaultdict(list)
        for event in trace_events(path):
            if event['ph'] != 'X':
                continue
            start = event['ts']
            end = start + event['dur']
            earliest, latest = min(earliest, start), max(latest, end)
            args = event['args'].get('detail')
            if not args:
                continue
            info = json.loads(args)
            if any(info.get(key) for key in ('is_wait', 'is_sync', 'is_duration_excluded')):
                continue
            pipe = info.get('instr_type', '').upper()
            op = event['name']
            if 'MMAD' in op and pipe == 'CUBE':
                key = 'MMAD'
                fragment = tuple(info.get(x, '?') for x in ('m', 'n', 'k', 'matrix_a_dtype', 'matrix_b_dtype'))
                shapes[fragment] += 1
            elif pipe in {'MTE2', 'MTE1', 'MTE3', 'FIXP'} and 'processed_bytes' in info:
                key = pipe
            elif pipe in {'VECTOR', 'RVECEX'}:
                key = 'VECTOR'
            elif pipe.startswith('RVEC'):
                key = pipe
            else:
                continue
            work[key].append((start, end, info.get('processed_bytes', 0), op))
        for key, seq in work.items():
            seq.sort()
            issues = [e[0] for e in seq]
            # Intervals are intra-core, never between independent AICs.
            intervals = [t2-t1 for t1,t2 in zip(issues,issues[1:])]
            summary = dict(count=len(seq), first=issues[0], last=issues[-1],
                           retire=max(e[1] for e in seq), bytes=sum(e[2] for e in seq),
                           instructions=dict(Counter(e[3] for e in seq)))
            if intervals:
                summary['median_issue_interval_us'] = median(intervals)
            banks.setdefault(core, {})[key] = summary
            totals[key] += len(seq)
        if 'MMAD' in work:
            cube_cores.append(core)
        if 'VECTOR' in work:
            vector_cores.append(core)
    for key, count in sorted(totals.items()):
        records = [row[key] for row in banks.values() if key in row]
        phases[key] = dict(count=count, first_issue_us=round(min(r['first'] for r in records)-earliest, 6),
                           last_issue_us=round(max(r['last'] for r in records)-earliest, 6),
                           last_retire_us=round(max(r['retire'] for r in records)-earliest, 6),
                           max_per_core=max(r['count'] for r in records),
                           trace_bytes=sum(r['bytes'] for r in records))
        ops = Counter()
        for row in records:
            ops.update(row['instructions'])
        phases[key]['instruction_names'] = dict(ops)
    terminal = max(cube_cores, key=lambda c: banks[c]['MMAD']['retire']) if cube_cores else None
    detail = {}
    for stage, record in banks.get(terminal, {}).items():
        detail[stage] = {k:round(v-earliest, 6) if k in ('first', 'last', 'retire') else v
                         for k,v in record.items()}
    return dict(schema=1, definition='Non-wait, non-sync, duration-included CA instruction descriptors; '
                'first/last are issues, retire includes dependency waits; descriptor bytes are not HBM traffic.',
                clock_zero='earliest physical-core event', full_window_us=round(latest-earliest,6),
                physical_cores=len(files), active_mmad_aics=len(cube_cores),
                active_vector_instruction_aivs=len(vector_cores), phases=phases,
                fragments=[dict(m=s[0], n=s[1], k=s[2], a=s[3], b=s[4], count=c) for s,c in shapes.items()],
                last_mmad_core=terminal, last_mmad_core_detail=detail)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(summarize(args.raw), indent=2) + '\n'
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end='')
