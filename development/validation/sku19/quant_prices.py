"""Reproduce v0.19 stage prices from development DMA, in 950 cycles."""
from math import ceil, log2
from pathlib import Path
from statistics import median
import json

ROOT = Path(__file__).resolve().parents[3]


def prices(rows):
    launches = {False: [], True: []}
    loops, front, reads, dependencies, spans = [], [], [], [], []
    for capture in rows:
        # Some trace exports lack a data/scale pair on individual cores.
        cores = [c for c in capture['cores'].values() if len(c['write']) >= 2]
        repeated = any(len(c['read']) > 2 for c in cores)
        issue_times = [c['read'][1][0] for c in cores]
        launches[repeated].append(min(issue_times))
        if len(cores)>1:
            spans.append((max(issue_times)-min(issue_times))/log2(len(cores)))
        points = []
        for c in cores:
            issue, ready, payload = c['read'][1][:3]
            groups = ceil(payload / 64)
            points.append((ceil(groups/8)+ceil(groups/128), c['write'][0][0]-c['simd']))
        loops.append(tuple(map(median, zip(*points))))
        # One aligned grain of scale corresponds to 32 groups per descriptor.
        if max(min(w[2] for w in c['write'][:2]) for c in cores) <= 32:
            front.append(median(c['simd']-c['read'][1][0] for c in cores))
            reads.append(median(c['read'][1][1]-c['read'][1][0]-c['read'][1][2]/48.362955988428816 for c in cores))
        dependencies.append(min(c['simd']-c['read'][1][1] for c in cores))
    slope = median((y-v)/(x-u) for x,y in loops for u,v in loops if x>u)
    return dict(quant_launch_cycles=median(launches[False]), quant_arrival_log2_cycles=median(spans),
                quant_prefetch_launch_cycles=median(launches[True]),
                quant_fragment_launch_cycles=median(launches[False]),
                quant_fragment_frontend_cycles=median(front),
                quant_vf_dependency_cycles=median(dependencies),
                quant_fragment_vector_cycles=median(y-slope*x for x,y in loops),
                quant_read_latency_cycles=median(reads),
                quant_loop_work_ops=slope*511.3636363636364)


if __name__ == '__main__':
    captures=json.loads((ROOT/'results/v0.19/quant-dma-samples.json').read_text())
    sides = {s:prices([r for r in captures if r['side']==s]) for s in ('dt','pr')}
    sides['pr']['quant_loop_work_ops'] = sides['dt']['quant_loop_work_ops']
    print(json.dumps(sides, indent=2))
