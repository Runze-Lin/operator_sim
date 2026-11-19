"""A causal service frontier for consecutive Cast UB blocks, evaluated without their CA times."""
import json
from pathlib import Path
import sys

from shadow import ROOT,price
from operator_model import load_chip
from operator_model.cast_resource import resource_cast

OUT=ROOT/'results/v0.25/cache'


def prefix(shape,sku,policy):
    chip=load_chip(sku);_,_,active,d=resource_cast(shape,chip)
    plan=d['target_plan'];s=d['stages_us'];cap=plan['ub_tile_elements'];local=plan['busy_core_elements']
    first_end=sum(s[k] for k in ('arrival','control_to_read','read','vector','dependency_to_write','write'))
    n=shape[0]*shape[1];scale=chip.frequency_ghz*1000;rows=[]
    for i in range(plan['busy_core_waves']):
        local_done=min(local,(i+1)*cap)
        # The balanced grid shares approximately the same fraction of the global stream
        # at this local block ordinal. Device-side launch skew between cores is not observable offline.
        x=2*n*local_done/local;y=4*n*local_done/local
        m=price(chip,active,x,y,policy)
        local_end=first_end+s['steady_tiles']*(local_done-min(cap,local))/max(1,local-cap)
        external=1753/scale+m['external_floor_cycles']/scale
        l2=1753/scale+(m['l2_floor_cycles'] or 0)/scale
        rows.append(dict(tile=i+1,prefix_fraction=local_done/local,logical_prefix_MiB=(x+y)/2**20,
             returned_external_write_MiB=m['modeled_external_write_bytes']/2**20,local_end_us=local_end,
             mixed_memory_end_us=external,L2_end_us=l2,completion_us=max(local_end,external,l2)))
    return rows


def register():
    dst=OUT/'prefix-forecast.json'
    if dst.exists():raise FileExistsError(dst)
    f=json.loads((OUT/'freeze.json').read_bytes());jobs={}
    for j in f['jobs']:
        if not j['op'].startswith('cast'):continue
        raw=OUT/'raw'/j['id']
        if raw.exists() or (OUT/(j['id']+'-result.json')).exists():raise ValueError('prior acquisition')
        jobs[j['id']]={policy:prefix(j['shape'],j['sku'],policy) for policy in ('uniform','pressure')}
        for policy,v in jobs[j['id']].items():
            if abs(v[-1]['completion_us']-j['predictions'][j['sku']][policy]['us'])>1e-8:raise ValueError('forecast endpoint changed')
    from hashlib import sha256
    from datetime import datetime,timezone
    dst.write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),role='Supplemental independent block-completion forecasts on the already frozen Cast jobs; captured before any new CA started.',
                                  basis='target resources only; no measured issue time from these shapes',source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                                  total_freeze_sha256=sha256((OUT/'freeze.json').read_bytes()).hexdigest(),jobs=jobs),indent=2)+'\n')


def replay():
    src=json.loads((ROOT/'results/v0.23/prefill/flash-128ki-pr.json').read_bytes())
    data={p:prefix(src['shape'],src['chip'],p) for p in ('pressure','uniform')}
    for p,rows in data.items():
        for i,r in enumerate(rows):
            r['observed_write_retire_us']=src['ca_tiles'][i]['Write'][1]
            r['error_us']=r['completion_us']-r['observed_write_retire_us']
    d=dict(source='results/v0.23/prefill/flash-128ki-pr.json',physical_frontiers=data)
    for p,r in data.items():
        d[p+'_MAE_all_blocks_us']=sum(abs(t['error_us']) for t in r)/len(r)
    d['uniform_linear_MAE_all_blocks_us']=sum(abs(m['Write'][1]-c['Write'][1]) for m,c in zip(src['model_tiles'],src['ca_tiles']))/len(src['ca_tiles'])
    (OUT/'prefix-replay.json').write_text(json.dumps(d,indent=2)+'\n')
    print({k:v for k,v in d.items() if 'MAE' in k})


if __name__=='__main__':
    {'register':register,'replay':replay}[sys.argv[1]]()
