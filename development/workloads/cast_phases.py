"""Extract actual terminal-AIV payload release, Vector union and write retirement."""
from collections import defaultdict
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'development/validation/sku17'))
from campaign_collect import events


def widths(spans):
    right=float('-inf')
    total=0.
    for start,end in sorted(spans):
        total+=max(0.,end-max(start,right))
        right=max(right,end)
    return total


def profile(rec):
    stage=rec['kernel']
    core=max(stage['windows'],key=lambda k:stage['windows'][k][1])
    ds=defaultdict(list)
    spans=defaultdict(list)
    origin=float('inf')
    first_payload=float('inf')
    for x in events(ROOT/stage['trace']):
        if x['ph']!='X':
            continue
        start=x['ts']
        origin=min(start,origin)
        q=x['tid']
        if q.startswith(('SIMD','RVEC','VEC')):
            q='SIMD'
        if x['pid']!=core and q!='MTE2':
            continue
        info=json.loads(x['args']['detail'])
        if info.get('is_duration_excluded') or info.get('is_wait') or info.get('is_sync'):
            continue
        end=start+x['dur']
        if x['name']=='MOV_SRC_TO_DST_ALIGNv2':
            n=info['processed_bytes']
            if q=='MTE2' and n>64:
                first_payload=min(first_payload,start)
            if x['pid']==core:
                ds[q].append((start,end,n))
        if x['pid']==core and end>start:
            spans[q].append((start,end))
    return dict(shape=rec['shape'],active=stage['active_aiv'],trace_sha256=stage['trace_sha256'],
                aiv_start_us=stage['windows'][core][0],first_payload_us=first_payload-origin,
                ca_us=stage['ca_us'],queue_union_us={k:widths(v) for k,v in spans.items()},
                payloads={k:[[a-origin,b-origin,n] for a,b,n in sorted(v)] for k,v in ds.items()})


if __name__=='__main__':
    out=ROOT/'results/v0.19/workloads'
    records=json.loads((out/'cast-calibration.json').read_text())
    if (out/'cast-calibration-retry.json').exists():
        records+=json.loads((out/'cast-calibration-retry.json').read_text())
    (out/'cast-phases.json').write_text(json.dumps([profile(r) for r in records if r['qualified']],indent=2)+'\n')
