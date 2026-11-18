"""Fit phase-level architecture transfer costs only from archived development traces."""
import hashlib
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/v0.15'
sys.path.insert(0,str(ROOT))
from operator_model.model_v2 import plan
from operator_model.model_v6 import Case

QUEUES = {'CUBE':'CUBE','MTE2':'MTE2','MTE3':'MTE3','MTE1':'MTE1','FIXP':'FIXP','RVECEX':'SIMD'}


def analyze(record, keep_streams=False):
    path = ROOT/record['trace']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['trace_sha256']
    if path.suffix == '.db':
        with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as db:
            start,end=db.execute('select min(ExecInstrTickStart),max(ExecInstrTickEnd) from ExecutedInstructions').fetchone()
            first,last=db.execute("select min(ExecInstrTickStart),max(ExecInstrTickEnd) from ExecutedInstructions where ExecInstrName='MMAD'").fetchone()
            return dict(phases_us=[(first-start)/1650,(last-first)/1650,(end-last)/1650])
    es=[e for e in json.loads(path.read_text())['traceEvents'] if e['ph']=='X']
    end=lambda e:e['ts']+e['dur']
    start,stop=min(e['ts'] for e in es),max(map(end,es))
    mm=[e for e in es if e['name']=='MMAD']
    cuts=[start,min(e['ts'] for e in mm),max(map(end,mm)),stop]
    views=[defaultdict(float) for _ in range(3)]
    points=defaultdict(Counter)
    for t in cuts: points[t]
    for e in es:
        q=QUEUES.get(e['tid'])
        if q and e['dur']>0:
            points[e['ts']][q]+=1
            points[end(e)][q]-=1
    active=Counter()
    prev=start
    phase=0
    for t,changes in sorted(points.items()):
        if t>prev:
            mask='+'.join(sorted(q for q,v in active.items() if v)) or 'SCALAR'
            views[phase][mask]+=t-prev
        if t in cuts[1:3]:phase+=1
        active.update(changes)
        prev=t
    out=dict(phases_us=[b-a for a,b in zip(cuts,cuts[1:])],masks_us=views,
             mmads_per_core=max(Counter(e['pid'] for e in mm).values()))
    if keep_streams:
        streams=defaultdict(list)
        for e in es:
            if e['name']=='MOV_OUT_TO_L1_MULTI_ND2NZ' and json.loads(e['args']['detail']).get('n')==256:
                streams[e['args']['pc_addr']].append(e['ts'])
        streams=sorted([sorted(s) for s in streams.values()],key=lambda s:s[0])
        assert [len(s) for s in streams]==[16,16]
        # Ignore initial pipeline setup and final drain; use the median spacing
        # of complete K and V tile starts in the latter half before the final two.
        out['issue_intervals_us']=[median(b-a for a,b in zip(s[8:13],s[9:14])) for s in streams]
        out['kv_issues']=len(streams[0])
    assert abs(sum(out['phases_us'])-record['ca_us'])<1e-6
    return out


def main():
    pairs=[]
    for filename in ('v0.13/sku/pairs.json','v0.14/sku/holdout.json'):
        for row in json.loads((ROOT/'results'/filename).read_text()):
            if row['op']!='fa':continue
            shape=row['shape'] if len(row['shape'])==5 else row['shape']+[1,1]
            it=plan(Case('fa',shape))['iterations']
            pairs.append(dict(shape=shape,iterations=it,**{sku:analyze(row[sku],shape[1]==4096 and sku=='pr')
                         for sku in ('dt','pr')},sha256={s:row[s]['trace_sha256'] for s in ('dt','pr')}))
    costs={}
    for key,pool in [('single',[p for p in pairs if p['iterations']==1]),
                     ('loop',[p for p in pairs if p['iterations']>1 and p['shape'][1]<4096])]:
        # Equal weight per shape; each atom is a fraction of the entire DT window.
        atoms=[Counter() for _ in range(3)]
        for p in pool:
            for group,v in zip(atoms,p['dt']['masks_us']):
                for mask,us in v.items():group[mask]+=us/(len(pool)*sum(p['dt']['phases_us']))
        ratios=[median(p['pr']['phases_us'][i]/p['dt']['phases_us'][i] for p in pool) for i in range(3)]
        costs[key]=dict(dt_fractions=atoms,pr_ratios=ratios,count=len(pool))
    p=next(p for p in pairs if p['shape'][1]==4096)['pr']
    dt32=json.loads((ROOT/'results/v0.5/trace-analysis.json').read_text())
    assert dt32['calibration_shape']==[9,32768,64,1,1]
    dt_ii=max(dt32['costs'][k+'_initiation_interval_us'] for k in ('k','v'))
    first=dt32['qk_pairs'][0][0]
    drain=dt32['costs']['drain_us']
    span=dt32['reference_us']-first-drain
    pr_ii=max(p['issue_intervals_us'])
    costs['long']={'dt':dict(prefix=first,middle_fill=span-127*dt_ii,issue=dt_ii,drain=drain),
                   'pr':dict(prefix=p['phases_us'][0],middle_fill=p['phases_us'][1]-15*pr_ii,
                             issue=pr_ii,drain=p['phases_us'][2])}
    for name,doc in [('fa-diagnosis',dict(pairs=pairs,dt32k_sha256=dt32['db_sha256'])),('fa-phases',costs)]:
        (OUT/(name+'.json')).write_text(json.dumps(doc,indent=2)+'\n')


if __name__=='__main__':
    main()
