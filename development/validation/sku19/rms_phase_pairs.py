"""Use only historical equal-geometry first-call pairs, never v0.19 holdouts."""
import json
from collections import defaultdict
from math import ceil
from pathlib import Path
from statistics import median
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'development/validation/sku17'))
from campaign_collect import events
from development.validation.sku19.rms_raw import reduce

OUT=ROOT/'results/v0.19/rms-followup'

def load(path):
    return json.loads(path.read_text())

def clocks(record):
    stage=record['stages'][0]
    core=max(stage['windows'],key=lambda c:stage['windows'][c][1])
    case=dict(trace=stage['trace'],shape=record['shape'],trace_sha256=stage['trace_sha256'],terminal_core=core)
    obs=reduce(case)
    # Segment cold preamble on every AIV at the common instruction addresses.
    origin=float('inf'); starts={}; dmas=defaultdict(lambda:defaultdict(list))
    for e in events(ROOT/stage['trace']):
        if e['ph']!='X':continue
        c=e['pid'];s=e['ts'];q=e['tid'];end=s+e['dur'];starts[c]=min(starts.get(c,s),s)
        origin=min(origin,s)
        if e['name']=='MOV_SRC_TO_DST_ALIGNv2' and q in ('MTE2','MTE3'):
            info=json.loads(e['args']['detail'])
            if not info.get('is_duration_excluded'):
                dmas[c][q].append((s,end,info['processed_bytes']))
    samples=[]
    for c,v in dmas.items():
        reads=sorted(v['MTE2']);writes=sorted(v['MTE3'])
        if len(reads)<3 or len(writes)<2:continue
        meta,gamma,x=reads[:3];stat=writes[0]
        samples.append(dict(core=c,arrival=starts[c]-origin,
            setup_to_meta=meta[0]-starts[c],meta_service=meta[1]-meta[0],
            setup_after_meta=gamma[0]-meta[1],first_input_issue=gamma[0]-starts[c],
            read_service=max(x[1],gamma[1])-gamma[0],
            statistic_service=stat[1]-stat[0],
            statistic_start=stat[0]-starts[c],rs_bytes=stat[2]))
    fields=['arrival','setup_to_meta','meta_service','setup_after_meta','first_input_issue','read_service','statistic_service','statistic_start']
    return dict(active=stage['active_aiv'],kernel=stage['kernel'],binary_sha256=stage['binary_sha256'],CA=stage['ca_us'],
                terminal=obs,median_us={q:median(x[q] for x in samples) for q in fields},
                terminal_dma_us=next(({q:v for q,v in x.items() if q!='core'} for x in samples if x['core']==core),None),
                complete_dma_core_count=len(samples))

def main():
    rows=load(ROOT/'results/v0.18/sku/pairs.json')
    archive=load(ROOT/'results/v0.17/sku/pairs.json')
    selected=[r for r in rows+archive if r['op']=='rmsnorm' and all(r[k]['trace'][0].endswith('/trace.json') for k in ('dt','pr'))]
    out=[]
    for r in selected:
        pair=dict(shape=r['shape'],dt=clocks(r['dt']),pr=clocks(r['pr']))
        pair['equal_geometry']=(pair['dt']['active']==pair['pr']['active'] and ceil(r['shape'][0]/64)==ceil(r['shape'][0]/56))
        out.append(pair)
    (OUT/'old-paired-phases.json').write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':main()
