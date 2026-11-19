"""Score the two prospective validation rounds using their original freezes."""
from datetime import datetime
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/v0.19/workloads'
sys.path.insert(0, str(ROOT / 'development/validation/sku17'))
from campaign_collect import events, sha

DT, PR, FUTURE = 'Ascend950DT_9582', 'Ascend950PR_957b', 'Ascend970DT'


def read(name):
    return json.loads((OUT / name).read_text())


def metric(predicted, actual):
    assert len(actual) == len(predicted) > 0
    return dict(n=len(actual),
                wape=sum(abs(x-y) for x, y in zip(predicted, actual))/sum(actual),
                max_relative_error=max(abs(x/y-1) for x, y in zip(predicted, actual)))


def score(rows, pairs):
    def side(chip):
        subset = [r for r in rows if r['chip'] == chip]
        return metric([r['pred_us'] for r in subset], [r['ca_us'] for r in subset])
    return dict(summary={chip: side(chip) for chip in (DT, PR)},
                speedup=metric([p['speed'] for p in pairs], [p['ca_speed'] for p in pairs]),
                reuse_dt_as_pr=metric([p['dt_us'] for p in pairs], [p['pr_ca_us'] for p in pairs]),
                speedup_1x=metric([1. for p in pairs], [p['ca_speed'] for p in pairs]),
                same_direction=sum((p['speed'] > 1) == (p['ca_speed'] > 1) for p in pairs))


def costs(attempts):
    values = [r['cost'] for r in attempts]
    # Sum active wall intervals separately from the sum over launch durations.
    # This excludes pauses between rounds and counts overlapping launches once.
    spans = sorted((datetime.fromisoformat(c['started_utc']), datetime.fromisoformat(c['finished_utc'])) for c in values)
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return dict(launches=len(attempts), eligible=sum(r['qualified'] for r in attempts),
                raw_mib=sum(r['raw_bytes'] for r in values)/2**20,
                max_single_raw_mib=max(r['raw_bytes'] for r in values)/2**20,
                simulator_wall_s_sum=sum(r['seconds'] for r in values),
                active_wall_s_union=sum((end-start).total_seconds() for start, end in merged),
                known_cpu_s_lower_bound=sum(r['cpu_seconds'] or 0 for r in values),
                cpu_missing=sum(r['cpu_seconds'] is None for r in values))


def phases(measurement):
    """Pick the last-finishing payload core and include its relative start skew."""
    k = measurement['kernel']
    shape = measurement['shape']
    last = max(k['windows'], key=lambda name: k['windows'][name][1])
    trace = ROOT / k['trace']
    assert sha(trace) == k['trace_sha256']
    starts = []
    first = None
    writes = 0
    for e in events(trace):
        if e['ph'] != 'X':
            continue
        first = e['ts'] if first is None else min(first, e['ts'])
        if e['pid'] != last or e['name'] != 'MOV_SRC_TO_DST_ALIGNv2':
            continue
        details = json.loads(e['args']['detail'])
        if details['instr_type'] == 'MTE2' and details['processed_bytes'] == 2*shape[1]:
            starts.append(e['ts'])
        if details['instr_type'] == 'MTE3':
            writes += 1
    starts = sorted(t-first for t in starts)
    assert len(starts) == 5+shape[3] and writes == 4, (shape, last, starts, writes)
    bounds = [0, starts[0], starts[2], starts[4], starts[-1], k['ca_us']]
    assert bounds == sorted(bounds)
    return last, [b-a for a, b in zip(bounds, bounds[1:])]



def tail_activity(measurement):
    """Union of visible critical-core event intervals after the output-gamma request."""
    last, stage_ca = phases(measurement)
    trace = ROOT / measurement['kernel']['trace']
    es = [e for e in events(trace) if e['ph'] == 'X' and e['pid'] == last]
    origin = min(e['ts'] for e in es) - measurement['kernel']['windows'][last][0]
    start = origin + sum(stage_ca[:-1])
    groups = {'SIMD': [], 'SCALAR': [], 'MTE2': [], 'MTE3': []}
    for e in es:
        name = 'SIMD' if e['tid'].startswith('RVEC') else e['tid']
        if name in groups and e['ts'] + e['dur'] > start:
            groups[name].append((max(start, e['ts']), e['ts'] + e['dur']))
    merged = {}
    for name, intervals in groups.items():
        end = start
        union = 0.
        for a, b in sorted(intervals):
            union += max(0., b-max(a, end))
            end = max(end, b)
        merged[name] = union
    return dict(shape=measurement['shape'], chip=measurement['chip'], core=last,
                ca_tail_us=stage_ca[-1], visible_queue_union_us=merged)

def main():
    initial = read('attn-res-evaluation.json')
    first_rows = [dict(role='first', **r) for r in initial['rows']]
    first_pairs = [dict(role='first', **p) for p in initial['pairs']]
    reg = read('attn-res-extension-freeze.json')
    extra = read('attn-res-extension-holdout.json')
    forecasts = {((f['tokens'], f['hidden'], f['slots'], f['valid']), f['chip']): f for f in reg['scenarios']}
    rows, pairs = [], []
    for m in extra:
        if not m['qualified']:
            continue
        assert datetime.fromisoformat(m['cost']['started_utc']) > datetime.fromisoformat(reg['utc'])
        f = forecasts[(tuple(m['shape']), m['chip'])]
        k = m['kernel']
        assert k['kernel'].endswith('_'+str(f['tiling_key']))
        last, stage_ca = phases(m)
        rows.append(dict(shape=m['shape'], chip=m['chip'], role='supplemental',
                         pred_us=f['latency_us'], ca_us=k['ca_us'], rel_error=f['latency_us']/k['ca_us']-1,
                         predicted_aiv=f['active_aiv'], active_aiv=k['active_aiv'], kernel=k['kernel'],
                         predicted_phases_us=f['stages_us'], ca_phases_us=stage_ca,
                         critical_core=last, trace_sha256=k['trace_sha256']))
    by_shape_sku = {(tuple(r['shape']), r['chip']): r for r in rows}
    assert len(rows) == len(by_shape_sku)
    for dt in (r for r in rows if r['chip'] == DT):
        shape = tuple(dt['shape'])
        pr = by_shape_sku.get((shape, PR))
        if pr:
            pairs.append(dict(shape=dt['shape'], role='supplemental', dt_us=dt['pred_us'], dt_ca_us=dt['ca_us'],
                              pr_us=pr['pred_us'], pr_ca_us=pr['ca_us'], speed=dt['pred_us']/pr['pred_us'],
                              ca_speed=dt['ca_us']/pr['ca_us'], future_us=forecasts[(shape, FUTURE)]['latency_us']))
    all_pairs, all_rows = first_pairs+pairs, first_rows+rows
    assert len(set(tuple(p['shape']) for p in all_pairs)) == len(all_pairs)
    previous = read('attn-res-ca.json')+read('attn-res-micro-ca.json')+read('attn-res-holdout.json')
    result = dict(qualification='Resident priced branch. Full CA and CPU golden. Stages are exposed wall intervals on the latest-finishing core, including its start skew in dispatch. Supplemental inputs were selected after inspecting the first two, at unchanged prices and schedule.',
                  **score(all_rows, all_pairs), first_holdout=score(first_rows, first_pairs), supplemental=score(rows, pairs),
                  rows=all_rows, pairs=all_pairs, cost=costs(previous+extra), supplemental_cost=costs(extra),
                  kimi_tail=[tail_activity(m) for m in read('attn-res-holdout.json')+extra
                             if m['qualified'] and m['shape'][:2] == [1, 7168]])
    (OUT/'attn-res-extension-evaluation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(result['summary'], result['speedup'], result['cost'])


if __name__ == '__main__':
    main()
