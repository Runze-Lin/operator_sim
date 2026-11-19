"""Score pre-registered Cast issue and cache frontiers as each existing 256Ki job finishes."""
from argparse import ArgumentParser
from math import ceil
from statistics import mean
import time
from prefill import OUT,PRO,POLICIES,read,write,summarize,draw,digest


def score(tag):
    name=f'pro-b8-256ki-{tag}'
    receipt=PRO/(name+'.json')
    src=PRO/(name+'-blocks.json')
    dest=OUT/(name+'-score.json')
    if dest.exists():return True
    if not receipt.exists():return False
    r=read(receipt)
    if not r['qualified']:
        write(dest,dict(qualified=False,receipt_sha256=digest(receipt),reason=r.get('reason'),ca_us=None))
        return True
    if not src.exists():return False
    d=read(src)
    f=read(OUT/'pro-256ki-prefix-forecast.json')['cases'][d['chip']]
    issue=read(OUT/'initial-issues-freeze.json')['cases']['pro256-'+tag]
    assert issue['shape']==d['shape']==r['shape'] and r['chip']==d['chip']
    assert abs(issue['device_us']-d['model_us'])<1e-7 and abs(d['ca_us']-r['ca_us'])<1e-7
    a=d['ca_tiles']
    markers=[row['Read'][0] for row in a[:2]]
    metrics={}
    for policy in POLICIES:
        us=f['totals'][policy]['us']
        prefix=f['prefixes'][policy]
        matching=len(prefix)==len(a)
        err=[b['completion_us']-v['Write'][1] for b,v in zip(prefix,a)] if matching else []
        metrics[policy]=dict(us=us,full_pct=100*(us/d['ca_us']-1),prefix_matches_measured_tile_count=matching,
                             block_write_retire_mae_us=mean(map(abs,err)) if err else None)
    out=dict(qualified=True,shape=d['shape'],chip=d['chip'],case=name,ca_us=r['ca_us'],kernel=d['kernel'],
             frozen_issues_us=issue['issues_us'],actual_issues_us=markers,absolute_origin_errors_us=[v-u for v,u in zip(issue['issues_us'],markers)],
             qualified_ca_blocks=len(a),estimated_busy_core_blocks=len(d['model_tiles']),metrics=metrics,receipt_sha256=digest(receipt),
             blocks_sha256=digest(src),evidence=str(src.relative_to(PRO.parent.parent.parent)))
    if len(d['model_tiles'])==len(a) and d['model_tile_elements'][:-1]==d['ca_tile_elements'][:-1]:
        g=summarize(d,src,'Pro B8 × 256Ki',issue['issues_us'])
        end=max(b for side in ('model','ca') for v in g[side] for a,b in v.values())
        draw(name+'-head',g,2*ceil(end/2))
    write(dest,out)
    return True


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        ready=[score(s) for s in ('dt','pr')]
        if all(ready) or not args.watch:break
        time.sleep(25)
