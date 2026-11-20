"""Check cold-read service feasibility against measured descriptor issue/retire windows."""
import argparse
import heapq
import json
from math import ceil, floor
from pathlib import Path

import numpy as np

from retention import OUT, ROOT, load_chip, payload, read


def peak_interval(reads, cycles_per_us, bandwidth):
    """Any interval must carry all jobs whose complete windows lie inside it."""
    deadlines = np.unique([end for start, end, size in reads])
    deadline_index = {int(t): i for i, t in enumerate(deadlines)}
    by_start = {}
    for start, end, size in reads:
        by_start.setdefault(start, []).append((deadline_index[end], size))
    volumes = np.zeros(len(deadlines), dtype=np.int64)
    best_rate = best_excess = None
    for start in sorted(by_start, reverse=True):
        for column, size in by_start[start]:
            volumes[column] += size
        lo = int(deadlines.searchsorted(start, side='right'))
        ends = deadlines[lo:]
        sizes = volumes[lo:].cumsum()
        # All active jobs have deadline > start; lower columns contain zero bytes.
        durations = (ends - start) / cycles_per_us
        rates = sizes / (1000 * durations)
        excesses = sizes - bandwidth * 1000 * durations
        for i, metric in [(int(rates.argmax()), 'rate'), (int(excesses.argmax()), 'excess')]:
            value = dict(start_us=start / cycles_per_us, end_us=float(ends[i]) / cycles_per_us,
                         bytes=int(sizes[i]), rate_GBps=float(rates[i]), excess_bytes=float(excesses[i]))
            if metric == 'rate' and (best_rate is None or value['rate_GBps'] > best_rate['rate_GBps']):
                best_rate = value
            if metric == 'excess' and (best_excess is None or value['excess_bytes'] > best_excess['excess_bytes']):
                best_excess = value
    return best_rate, best_excess


def earliest_deadline_missed(reads, bytes_per_cycle):
    """Give the whole nominal bandwidth to reads. Expired residual bytes must be served elsewhere."""
    releases = {}
    for start, end, size in reads:
        releases.setdefault(start, []).append((end, float(size)))
    times = sorted(releases.keys() | {end for _, end, _ in reads})
    queue = []
    missed = 0.
    for i, now in enumerate(times):
        while queue and queue[0][0] <= now:
            missed += heapq.heappop(queue)[1]
        for job in releases.get(now, ()):
            heapq.heappush(queue, job)
        if i + 1 == len(times):
            break
        budget = (times[i+1] - now) * bytes_per_cycle
        while queue and budget > 0:
            end, size = heapq.heappop(queue)
            used = min(size, budget)
            budget -= used
            if size > used:
                heapq.heappush(queue, (end, size-used))
    return missed


def examine(path):
    case = read(path)
    chip = load_chip(case['chip'])
    cycles_per_us = chip.frequency_ghz * 1000
    reads = [(floor(a*cycles_per_us)-1, ceil(b*cycles_per_us)+1, n)
             for wr, addr, n, a, b in payload(case) if not wr]
    rate, excess = peak_interval(reads, cycles_per_us, chip.hbm_bandwidth_gbs)
    missed = earliest_deadline_missed(reads, chip.hbm_bandwidth_gbs*1000/cycles_per_us)
    if missed + 1.e-3 < max(0., excess['excess_bytes']):
        raise ValueError('interval and deadline calculations disagree')
    total = sum(n for _, _, n in reads)
    return dict(source=str(path.relative_to(ROOT)), sha256=case['dma_records_sha256'],
                shape=case['shape'], op=case['op'], chip=chip.name, read_descriptors=len(reads),
                read_bytes=total, nominal_GBps=chip.hbm_bandwidth_gbs, peak_rate_interval=rate,
                peak_deficit_interval=excess, undeliverable_read_bytes=missed, fraction=missed/total)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('paths', type=Path, nargs='+')
    args = p.parse_args()
    data = [examine(src.resolve()) for src in args.paths]
    (OUT/'read-windows.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
