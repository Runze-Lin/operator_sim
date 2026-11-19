"""Compare the frozen Matmul dispatch to raw MMAD boxes on the four largest PR holdouts."""
from collections import Counter, defaultdict
import json

from campaign import DT, PR, OUT, ROOT, read, sha, write
from campaign_collect import events
from operator_model.chips import load_chip
from operator_model.matmul_resource import resource_matmul
from operator_model.matmul_v15 import matmul_plan
from operator_model.matmul_waves import dispatch


def histogram(counter):
    return {'x'.join(map(str, box)): n for box, n in sorted(counter.items())}


def inspect(pair, side):
    data = pair[side]
    chip = load_chip({'dt':DT, 'pr':PR}[side])
    shape = pair['shape']
    path = ROOT/data['trace'][0]
    assert sha(path) == data['trace_sha256'][0]
    per_core = defaultdict(Counter)
    for e in events(path):
        if e.get('ph') == 'X' and e['name'] == 'MMAD':
            attrs = json.loads(e['args']['detail'])
            box = tuple(attrs[k] for k in ('m','n','k'))
            per_core[e['pid']][box] += 1
    actual = sum(per_core.values(), Counter())
    latency, _, _, details = resource_matmul(shape, chip)
    assert latency == pair[side+'_us']
    full, tail, _ = dispatch(shape, matmul_plan(shape, chip))
    k = shape[2]
    fragment = details['mmad_k']
    expected = Counter()
    for (m,n), count in (full+tail).items():
        if k//fragment:
            expected[m,n,fragment] += count*(k//fragment)
        if k%fragment:
            expected[m,n,k%fragment] += count
    return dict(shape=shape,chip=chip.name,trace=data['trace'][0],sha256=data['trace_sha256'][0],
                active_aic=data['active_aic'],cores_issuing_mmad=len(per_core),plan=details['target_plan'],
                actual=histogram(actual),model=histogram(expected),histogram_matches=actual==expected,
                mmads_by_core={c:histogram(h)for c,h in sorted(per_core.items())})


def main():
    previous=OUT/'campaign-mmad.json'
    records=read(previous) if previous.exists() else []
    known={r['trace'] for r in records}
    for p in read(OUT/'pairs.json'):
        if p['op']!='matmul' or p['shape'][0]<1024:continue
        for side in ('dt','pr'):
            if p[side]['trace'][0] not in known:records.append(inspect(p,side))
    write(previous,records)
    print([(r['shape'],r['chip'],r['plan']['waves'],r['histogram_matches'])for r in records])

if __name__=='__main__':main()
