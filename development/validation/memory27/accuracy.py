"""Audit what a single shape-only point could attain on the intact paired CA protocols."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import OperatorStream


def read(path):
    return json.loads((ROOT/path).read_bytes())


def floor_metrics(rows):
    by_shape=defaultdict(list)
    for r in rows:
        by_shape[tuple(r['shape'])].append(r['ca_us'])
    volume=sum(map(sum,by_shape.values()))
    absolute=bounded=radius=0.
    possible=True
    for ys in by_shape.values():
        lo,hi=min(ys),max(ys)
        radius=max(radius,(hi-lo)/(hi+lo))
        absolute+=sum(abs(median(ys)-y) for y in ys)
        lower,upper=.5*hi,1.5*lo
        if lower>upper:
            possible=False
        else:
            point=max(lower,min(median(ys),upper))
            bounded+=sum(abs(point-y) for y in ys)
    return dict(shapes=len(by_shape),windows=len(rows),unconstrained_min_WAPE_pct=100*absolute/volume,
                minimax_lower_bound_pct=100*radius,minimum_WAPE_at_max50_pct=100*bounded/volume if possible else None,
                joint_target_feasible=possible and bounded/volume<=.25)


def main():
    cohorts={}
    rms='results/rms-unknown23/independent.json'
    post='results/v0.24/post/evaluation.json'
    spec='results/v0.24/post/freeze.json'
    # The missing DT receipt still describes a successfully dispatched call.
    # Replay the entire registered stream, scoring only qualified timing windows.
    predictions={}
    streams={}
    for r in sorted(read(spec)['rows'],key=lambda r:(r['chip'],r['group'],r['i'])):
        key=r['chip'],r['group']
        if key not in streams:
            streams[key]=OperatorStream(r['chip'])
        predictions[key+(r['i'],)]=streams[key].advance('mhc_post',r['shape'],dtype='bf16').latency_us
    for op,filename,rs in [('RMSNorm',rms,[r for r in read(rms)['rows'] if r['role']!='prefix']),
                           ('mHC Post',post,[r for r in read(post)['rows'] if r['timing_qualified']])]:
        for chip in sorted({r['chip'] for r in rs}):
            rows=[r for r in rs if r['chip']==chip]
            metrics=floor_metrics(rows)
            if op=='mHC Post':
                values=[(predictions[chip,r['group'],r['i']],r['ca_us']) for r in rows]
                metrics['public_ordered_replay']=dict(wape_pct=100*sum(abs(x-y) for x,y in values)/sum(y for _,y in values),
                                                       max_pct=100*max(abs(x/y-1) for x,y in values))
            cohorts[op+'/'+chip]=metrics
    out=dict(goal=dict(wape_pct=25,maximum_ape_pct=50),cohorts=cohorts,domain='Confirmed identical chip, shape, dtype, attributes, released kernel and input protocol within each group. Each lower bound is in-sample, not a latency forecast.',
             source_sha256={s:sha256((ROOT/s).read_bytes()).hexdigest() for s in [rms,post,spec]})
    (ROOT/'results/v0.27/history-bounds.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':
    main()
