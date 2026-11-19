"""Reduce isolated DT launch and copy service from v0.18 and compact DT v0.12.

No end-to-end latency labels or target-chip measurements enter these medians.
"""
import json
from math import ceil
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[3]
def load(name):
    return json.loads((ROOT/name).read_text())


def calibrate():
    trace = load('results/v0.18/sku/trace-evidence.json')['rows']
    rows = [r for r in trace if r['op'] == 'rmsnorm' and r['chip'] == 'Ascend950DT_9582']
    ticks = 1650
    arrival, solo, peers, rates = [], [], [], []
    reads = []
    first_read = []
    for r in rows:
        t, d = r['shape']
        w = ceil(t / 64)
        a = ceil(t / w)
        if a > 1:
            arrival.append(ticks*r['terminal_core_start_us']/(a-1))
        copies = r['critical_core_dma_streams']
        writes = sorted((q for q in copies if q['queue'] == 'MTE3'), key=lambda q:q['first_us'])
        x = max((q for q in copies if q['queue'] == 'MTE2'), key=lambda q:q['first_us'])
        if x['count'] == 1:
            kib = x['raw_processed_bytes']/1024
            reads.append((kib, ticks*x['duration_median_us']))
            first_read.append((kib, ticks*r['queue_payload_union_us']['MTE2']))
        if w == 1:
            if a == 1:
                solo.append(writes[0]['duration_median_us'] * ticks)
            else:
                peers.append((min(32,a)-1, writes[0]['duration_median_us']*ticks))
    historical = (load('results/v0.12/rms/development-evidence.json') +
                  load('results/v0.12/rms/holdout-evidence.json'))
    for r in historical:
        t,d = r['shape']
        for i,s in enumerate(r['stores']):
            if i%2 == 0:
                if t == 1:
                    solo.append(s['end'] - s['start'])
            elif s['end'] > r['stores'][i-1]['end']+16:
                rates.append((s['bytes']/1024, s['end']-s['start']))
    def slope(points):
        # Widely separated payloads make the byte component identifiable above
        # completion jitter. Use the upper half of non-zero pairwise distances.
        pairs=[(x1-x0,(y1-y0)/(x1-x0)) for x1,y1 in points for x0,y0 in points if x1>x0]
        separation=median(a for a,_ in pairs)
        return max(0,median(s for a,s in pairs if a>=separation))
    solo_ticks = median(solo)
    read_slope = slope(reads)
    starts = load('results/v0.19/rms-startup.json')
    return {
        'first_scalar_cycles':median(x['phases_us']['SCALAR']*ticks for x in starts),
        'first_payload_release_cycles':median(x['first_payload_us']*ticks for x in starts),
        'first_read_cycles':median(y-read_slope*x for x,y in first_read),
        'arrival_cycles_per_extra_aiv':median(arrival),
        'rstd_solo_cycles':solo_ticks,
        'rstd_cycles_per_neighbour':median((s-solo_ticks)/p for p,s in peers),
        'read_cycles_per_kib':read_slope,
        'write_cycles_per_kib':slope(rates),
        'sample_counts':dict(arrival=len(arrival),solo=len(solo),peers=len(peers),read=len(reads),write=len(rates))
    }


if __name__ == '__main__':
    print(json.dumps(calibrate(),indent=2))
