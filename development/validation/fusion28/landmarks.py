"""Inspect on-core VF issue/retire times and DMA on the last AIV; retain immutable trace digests."""
import hashlib
import json
import sys
from pathlib import Path
from statistics import median

ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/v0.28/fusion';BASE=ROOT/'results/v0.26/sink'
def read(p):return json.loads(p.read_bytes())
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def summarize(shape,iters,sku,events,full_cycles):
    from operator_model import load_chip
    from math import ceil
    from statistics import median
    t,n=shape;chip=load_chip(sku);rows=ceil(t/chip.vector_cores);groups=ceil(rows/8)
    vf=[e for e in events if e['name']=='VF'];dma=[e for e in events if e['name'].startswith('MOV_SRC_TO_DST')]
    if len(vf)!=2*iters+2 or len(dma)!=3:raise ValueError(('wrong kernel',shape,iters,len(vf),len(dma)))
    setup,soft,first=vf[:3];scatter=vf[-1];row=vf[3:-1:2];col=vf[4:-1:2]
    pairs=[b['begin']-a['begin'] for a,b in zip(row,row[1:])]
    last_pair=scatter['begin']-row[-1]['begin'] if row else None
    if last_pair is not None:pairs.append(last_pair)
    fields=dict(tiling_issue=dma[0]['begin'],tiling_wait=dma[0]['end']-dma[0]['begin'],
        setup_issue=setup['begin'],setup_duration=setup['end']-setup['begin'],read_issue=dma[1]['begin'],read_wait=dma[1]['end']-dma[1]['begin'],
        soft_issue=soft['begin'],soft_duration=soft['end']-soft['begin'],first_col_issue=first['begin'],
        first_col_duration=first['end']-first['begin'],first_row_issue=row[0]['begin'] if row else None,
        first_col_to_first_row=(row[0] if row else scatter)['begin']-first['begin'],
        row_to_col=[c['begin']-r['begin'] for r,c in zip(row,col)],
        row_duration=[e['end']-e['begin'] for e in row],col_duration=[e['end']-e['begin'] for e in col],
        pairs=pairs,steady=median(pairs[1:]) if len(pairs)>1 else None,scatter_issue=scatter['begin'],
        scatter_duration=scatter['end']-scatter['begin'],write_issue=dma[-1]['begin'],write_wait=dma[-1]['end']-dma[-1]['begin'],tail=full_cycles-dma[-1]['end'])
    return dict(shape=shape,iters=iters,sku=sku,active=ceil(t/rows),rows=rows,groups=groups,full_cycles=full_cycles,
                dma_bytes=dma[1].get('size'),milestones=fields)

def main():
    sys.path.insert(0,str(ROOT))
    sys.path.insert(0,str(ROOT/'development/validation/sink26'))
    from evidence import shard
    evidence=read(BASE/'trace-evidence.json');old={r['source_sha256']:r for r in read(BASE/'replay.json')['rows']}
    res=[]
    for r in evidence:
        h=old[r['source_sha256']]
        z=summarize(r['shape'],r['iters'],r['chip'],r['events'],r['full_ticks'])
        res.append(dict(**z,state=h['state'],history=h['h'],trace=r['source'],trace_sha256=r['source_sha256'],role='original phase development'))
    for r in read(BASE/'independent.json')['rows']:
        receipt=ROOT/r['receipt'];z=read(receipt)['rows'][r['index']];s=z['stage']
        path,dig,core,origin,first,last,events=shard(s)
        for e in events:
            e['begin']=round(e['begin']-origin);e['end']=round(e['end']-origin)
        events.sort(key=lambda a:a['begin'])
        assert z['qualified'] and (z['iters'],z['shape'],z['h'])==(r['iters'],r['shape'],r['h']) and s['trace_sha256']==r['raw_sha256']
        obs=summarize(r['shape'],r['iters'],r['chip'],events,s['last_tick']-s['first_tick'])
        res.append(dict(**obs,state=r['state'],history=r['h'],trace=path,trace_sha256=dig,aggregate_sha256=s['trace_sha256'],receipt=r['receipt'],
                        events=events,role='v0.26 forward observations now available for mechanism development'))
    save(OUT/'landmarks.json',res)
if __name__=='__main__':main()
