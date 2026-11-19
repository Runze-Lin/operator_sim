"""Leave each previous paired shape out before estimating the two phase prices."""
import json
from pathlib import Path
from statistics import median
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import load_chip
from operator_model.rms_target_cost import resource_rms_target, RmsFirstDispatch
from operator_model.rms_resource import FIRST_PAYLOAD_RELEASE, FIRST_SCALAR
OUT=ROOT/'results/v0.19/rms-followup'
DT,PR=load_chip(),load_chip('Ascend950PR_957b')


def prices(train):
    def delta(r,k):
        t=[r[side]['terminal'] for side in ('dt','pr')]
        if k=='scalar':return 1650*(t[1]['phases_us']['SCALAR']-t[0]['phases_us']['SCALAR'])
        return 1650*(t[1]['first_payload_us']-t[0]['first_payload_us'])
    return RmsFirstDispatch(FIRST_SCALAR+median(delta(r,'scalar') for r in train),
                            FIRST_PAYLOAD_RELEASE+median(delta(r,'release') for r in train))


def main():
    old=json.loads((OUT/'old-paired-phases.json').read_text())
    out=[]
    for r in old:
        train=[p for p in old if p['shape']!=r['shape'] and p['equal_geometry']]
        config=prices(train)
        shape=r['shape']
        group=dict(shape=shape,equal_geometry=r['equal_geometry'],scalar_cycles=config.scalar_cycles,
                   first_payload_cycles=config.first_payload_cycles,n_train=len(train))
        for key,c in [('dt',DT),('pr',PR)]:
            group[key]=dict(base=resource_rms_target(shape,c)[0],
                            candidate=resource_rms_target(shape,c,first_dispatch=config if key=='pr' else None)[0],
                            ca=r[key]['CA'])
        group['speed']={k:group['dt'][k]/group['pr'][k] for k in ('base','candidate','ca')}
        out.append(group)
    def metric(rs,key,col):
        y=[r[key]['ca'] for r in rs];p=[r[key][col] for r in rs]
        return dict(wape=sum(abs(a-b) for a,b in zip(p,y))/sum(y),max=max(abs(a/b-1) for a,b in zip(p,y)))
    def score(rs):
        o={key:{col:metric(rs,key,col) for col in ('base','candidate')} for key in ('dt','pr','speed')}
        o['correct_direction']={c:sum((r['speed'][c]>1)==(r['speed']['ca']>1) for r in rs) for c in ('base','candidate')}
        o['aggregate_ratio']={c:sum(r['dt'][c] for r in rs)/sum(r['pr'][c] for r in rs) for c in ('base','candidate','ca')}
        return o
    p=prices([r for r in old if r['equal_geometry']])
    payload=dict(source='Only v0.17 official pair and v0.18 official pairs. Paired LOOCV; each SKU-stage parameter estimated with held-out shape excluded.',frozen_candidate_prices=p.__dict__,all_11=score(out),v018=score(out[:10]),
                 equal_geometry=score([r for r in out if r['equal_geometry']]),rows=out)
    (OUT/'crossval.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if k not in ('rows','source')}))

if __name__=='__main__':main()
