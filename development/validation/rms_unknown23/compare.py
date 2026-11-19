"""Cross-fit all landmarks by whole compact host process and exclude every matching shape."""
import json
from pathlib import Path
from collections import defaultdict
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.rms_unknown23.probe import states,harmonic
from development.validation.rms_unknown23.model import fit,scenarios
from development.validation.sku19.campaign import write
from operator_model import load_chip
OUT=ROOT/'results/rms-unknown23'

def evaluate():
    raw=json.loads((OUT/'landmarks.json').read_text())['rows'];observations=json.loads((ROOT/'results/v0.22/rms-unknown/audit.json').read_text())['rows'];folds={}; record=[]; by_trace={r['source']:r for r in raw}
    for r in observations:
        name=r['trace'];group=str(Path(name).parent) if name in by_trace else 'external';shape=tuple(r['shape']);chip=load_chip(r['chip']);prior=r['prior_aiv']
        if group not in folds:
            heldout={tuple(k['shape']) for k in raw if str(Path(k['source']).parent)==group};folds[group]=fit([k for k in raw if tuple(k['shape']) not in heldout and str(Path(k['source']).parent)!=group])
        p=folds[group];t,old,a=scenarios(shape,chip,p);_,_,prev,_=states(shape,chip.name)
        low,high=min(t),max(t);hot,cold=t[-1],t[0];run=sum(t[1:]+[hot]*(chip.vector_cores-a))/chip.vector_cores
        calib={'M':harmonic(low,high),'GM':(low*high)**.5,'cold_hot':harmonic(cold,hot),'run':harmonic(run,cold),'clip':max(.6*high,min(1.4*low,old)),'C':harmonic(min(t[1:]),max(prev[0],max(t[1:]))),'old':old,'known':t[min(prior,a)],'known_old':prev[min(prior,a)]}
        record.append({**{k:r[k] for k in ['chip','shape','prior_aiv','ca_us','source','provider','trace']},'active':a,'group':group,'states':t,'model':calib})
    return record

def summary(rr):
    summary={};by=defaultdict(list)
    for r in rr:
        c=r['chip'][6:11];h=r['prior_aiv'];a=r['active'];state='first' if not h else 'expand' if h<a else 'hot';
        for k in [c,c+' '+r['provider'],c+' '+state,c+' '+r['source'].split('/')[1]]:by[k].append(r)
    for k,rs in by.items():
        summary[k]={}
        for n in rs[0]['model']:
            errors=[abs(r['model'][n]-r['ca_us']) for r in rs];summary[k][n]=[len(rs),sum(errors)/sum(r['ca_us'] for r in rs),max(e/r['ca_us'] for e,r in zip(errors,rs))]
        print(k, ' '.join(f'{n}={100*w:.1f}/{100*m:.1f}' for n,(_,w,m) in summary[k].items()))
    return summary
if __name__=='__main__':
    rows=evaluate();s=summary(rows);write(OUT/'resource-diagnostic.json',dict(summary=s,rows=rows))
