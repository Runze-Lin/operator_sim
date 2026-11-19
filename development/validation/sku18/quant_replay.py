"""Compare frozen v0.18 against exactly the archived v0.17 quant predictions."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate


def metric(rows,key):
    return dict(wape=sum(abs(r[key]-r['ca_us']) for r in rows)/sum(r['ca_us'] for r in rows),
                max_ape=max(abs(r[key]/r['ca_us']-1) for r in rows))


def one(shape,chip,previous,ca):
    p=estimate('quant',shape,chip=chip,timing_mode='resource')
    d=p.projection['timing_diagnostics']
    return dict(shape=shape,chip=chip,ca_us=ca,v17_us=previous,v18_us=p.latency_us,active_aiv=p.active_aiv,
                first_tile=d['first_tile_cycles'],fragment=d['fragment_wave'],stages_us=d['stages_us'])


def main():
    out=ROOT/'results/v0.18'
    base=ROOT/'results/v0.17'
    pairs=[p for p in json.loads((base/'sku/pairs.json').read_text()) if p['op']=='quant']
    groups={}
    for side,chip in [('dt','Ascend950DT_9582'),('pr','Ascend950PR_957b')]:
        groups['v17_'+side]=[one(p['shape'],chip,p[side+'_us'],p[side+'_ca_us']) for p in pairs]
    older=json.loads((base/'quant-pr-replay.json').read_text())['rows']
    groups['v16_pr']=[one(p['shape'],'Ascend950PR_957b',p['new_us'],p['ca_us']) for p in older]
    sums={k:{method:metric(rows,method+'_us') for method in ['v17','v18']} for k,rows in groups.items()}
    result=dict(scope='Historical diagnostic replay, no new independent labels. Some v17 DT phase observations are calibration.',
                summary=sums,rows=groups)
    (out/'quant-replay.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(sums,indent=1))
    print([(p['shape'],p['v17_us'],p['v18_us'],p['ca_us']) for p in groups['v16_pr'] if p['fragment']])


if __name__=='__main__':main()
