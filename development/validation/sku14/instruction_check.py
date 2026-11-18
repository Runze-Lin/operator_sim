"""Audit PR timeline events on the qualified holdout without rerunning the simulator."""
from collections import Counter, defaultdict
import json
from math import ceil
from campaign import OUT, ROOT, load, sha, write


def count(row):
    trace = ROOT / row['pr']['trace']
    if sha(trace) != row['pr']['trace_sha256']:
        raise ValueError(f"Trace changed for {row['id']}")
    queues = Counter()
    mmads = Counter()
    dimensions = Counter()
    endpoints = defaultdict(float)
    for event in load(trace)['traceEvents']:
        if event['ph'] != 'X':
            continue
        core = event['pid']
        queues[event['tid']] += 1
        if event['name'] == 'MMAD':
            mmads[core] += 1
            detail = json.loads(event['args']['detail'])
            dims = tuple(detail.get(axis) for axis in ('m', 'n', 'k'))
            dimensions['×'.join(map(str,dims)) if None not in dims else 'unreported'] += 1
        endpoints[core] = max(endpoints[core], event['ts'] + event['dur'])
    if sum(queues.values()) != row['pr']['events']:
        raise ValueError(f"Timeline count mismatch for {row['id']}")
    last = max(endpoints, key=endpoints.get)
    result = dict(id=row['id'], op=row['op'], shape=row['shape'],
                  last_finishing_core=last, queue_events=dict(queues),
                  total_mmad=sum(mmads.values()), mmads_by_dimensions=dict(dimensions), max_mmad_on_one_core=max(mmads.values(), default=0),
                  total_rvec_queue_events=sum(v for k, v in queues.items() if k.startswith('RVEC')))
    if row['op'] == 'matmul':
        plan = row['projection']['target_plan']
        result['predicted_max_mmad'] = plan['waves'] * ceil(row['shape'][2]/256)
    return result


if __name__ == '__main__':
    rows = load(OUT/'holdout.json')
    if len(rows) != 30:
        raise ValueError('Expected the complete qualified PR holdout')
    write(OUT/'instructions.json', [count(r) for r in rows])
