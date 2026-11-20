"""Identify which hidden dispatch regimes cause residuals on frozen v0.33 prices."""
from collections import defaultdict
from pathlib import Path
from hashlib import sha256
import json

ROOT=Path(__file__).resolve().parents[3]
SRC=ROOT/'results/v0.33/post/replay.json'
OUT=ROOT/'results/v0.34/post'


def stats(points):
    total=sum(ca for ca,_ in points)
    return dict(windows=len(points),absolute_error_us=sum(abs(p-ca) for ca,p in points),
                wape=100*sum(abs(p-ca) for ca,p in points)/total,
                bias=100*sum(p-ca for ca,p in points)/total,
                maximum=100*max(abs(p/ca-1) for ca,p in points))


def main():
    rows=json.loads(SRC.read_bytes())['rows']
    groups=defaultdict(list)
    for r in rows:
        chip=r['chip']
        prior=r['validation_only_actual_prior'].split('_after_')[0]
        groups[chip,r['route'],prior].append((r['ca_us'],r['candidate_us']))
    data=[dict(chip=c,route=t,prior=k,**stats(values)) for (c,t,k),values in sorted(groups.items())]
    checks=[]
    for chip in sorted({r['chip'] for r in rows}):
        # The midpoint minimizes the worst absolute deviation of the two feasible
        # state endpoints. Restrict it to [-50%,+50%] relative to both endpoints
        # where possible, otherwise use equal relative risk on the model bounds.
        sample=[r for r in rows if r['chip']==chip];projected=[]
        for r in sample:
            lo,hi=r['range_us']
            c=.5*hi;d=1.5*lo
            p=max(c,min(d,(lo+hi)/2)) if c<=d else 2*lo*hi/(lo+hi)
            projected.append((r['ca_us'],p))
        checks.append(dict(chip=chip,candidate=stats([(r['ca_us'],r['candidate_us']) for r in sample]),
                           clipped_midpoint=stats(projected)))
    (OUT/'unknown-state.json').write_text(json.dumps(dict(source_sha256=sha256(SRC.read_bytes()).hexdigest(),
        role='development replay, unknown prior labels used for grouped diagnosis only',groups=data,policies=checks),indent=2)+'\n')
    print(checks)


if __name__=='__main__':main()
