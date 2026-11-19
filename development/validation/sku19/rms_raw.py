"""Stream historic isolated DT traces, saving payload release and disjoint phases."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'development/validation/sku17'))
from campaign_collect import events


def reduce(row):
    p=ROOT/row['trace']
    with p.open('rb') as f:
        h=hashlib.sha256()
        while block:=f.read(2**20):
            h.update(block)
    assert h.hexdigest()==row['trace_sha256']
    core=row['terminal_core']
    first=float('inf')
    first_data=first
    markers=defaultdict(Counter)
    for event in events(p):
        if event['ph']!='X':
            continue
        start=event['ts']
        first=min(start,first)
        q=event['tid']
        if q.startswith(('SIMD','RVEC','VEC')):
            q='SIMD'
        local=event['pid']==core
        dma=q=='MTE2' and event['name']=='MOV_SRC_TO_DST_ALIGNv2'
        if not (local or dma):
            continue
        info=json.loads(event['args']['detail'])
        if info.get('is_duration_excluded'):
            continue
        if dma and info['processed_bytes']>64:
            first_data=min(first_data,start)
        if local:
            markers[start][q]+=1
            markers[start+event['dur']][q]-=1
    active=Counter()
    totals=Counter()
    prior=min(markers)
    for t,delta in sorted(markers.items()):
        phase=next((q for q in ('SIMD','MTE3','MTE2','SCALAR') if active[q]),'other')
        totals[phase]+=t-prior
        active.update(delta)
        prior=t
    return dict(shape=row['shape'],trace_sha256=h.hexdigest(),first_payload_us=first_data-first,
                phases_us=dict(totals),terminal_start_us=min(markers)-first)


if __name__=='__main__':
    evidence=json.loads((ROOT/'results/v0.18/sku/trace-evidence.json').read_text())['rows']
    rows=[reduce(r) for r in evidence if r['op']=='rmsnorm' and 'DT' in r['chip']]
    (ROOT/'results/v0.19/rms-startup.json').write_text(json.dumps(rows,indent=2)+'\n')
