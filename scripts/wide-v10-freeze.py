"""Use completed historical traces only. Select and predict cases before acquiring CA evidence."""
from pathlib import Path
from collections import Counter
import json,sys,datetime
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.ca import sha256
from operator_model.vector_v9 import schedule, predict as old
from operator_model.wide_v10 import occupancy,predict
OUT=ROOT/'results/v0.10/wide'
rows=json.loads((OUT/'development-trace.json').read_text())
if (OUT/'freeze.json').exists():raise RuntimeError('already frozen')
prior={};seen={};contracts={}
for r in rows:
    seq=str(Path(r['db_source']).parent); op=r['op']; prior.setdefault(op,Counter()); contracts[op]=r['attrs']; h=seen.get(seq,0)
    route,active,waves,units=schedule(op,r['shape'],r['attrs']);assert active==r['active_aiv'];r.update(route=route,active=active,waves=waves,units=units,context=occupancy(active,h),prior_aiv=h)
    if r['source'].endswith('.db'):prior[op][h]+=1
    seen[seq]=max(h,active)
profile=dict(contracts=contracts,occupancy_prior={op:[[int(k),v/sum(counts.values())] for k,v in sorted(counts.items())] for op,counts in prior.items()},phases={})
for route in sorted(set(r['route'] for r in rows)):
    subset=[r for r in rows if r['route']==route]; phases=profile['phases'][route]={}
    for q in ('SCALAR','MTE2','SIMD','MTE3','arrival','other'):
        # Only fit slopes when the queue has a corresponding scalable workload.
        # State offsets at most per physical queue; sparse token route has no first / expansion mix.
        by_state=q in ('SCALAR','MTE2','SIMD','arrival'); names=sorted({r['context'] if by_state else 'all' for r in subset}); raw=[];y=[]
        for r in subset:
            vals=[float((r['context'] if by_state else 'all')==n) for n in names]
            if q=='SIMD' or (q=='MTE3' and route=='rmsnorm'):vals.append(r['units'][q]);
            if q=='MTE3' and route=='rmsnorm':vals.append(r['waves']-1)
            raw.append(vals);y.append(r['phase_us'][q])
        x=np.asarray(raw);y=np.asarray(y); sol=least_squares(lambda b:x@b-y,np.full(x.shape[1],.01),bounds=(0,np.inf),loss='soft_l1',f_scale=.12)
        sm=dict(zip(names,sol.x));median=float(np.median(list(sm.values()))); phases[q]={k:float(sm.get(k,sm.get('all',median))) for k in ('first','expand','reuse')};
        phases[q]['service']=float(sol.x[len(names)]) if x.shape[1]>len(names) else 0.
        if q=='MTE3' and route=='rmsnorm':phases[q]['per_extra_row']=float(sol.x[-1])
def save(path,x):path.write_text(json.dumps(x,indent=2)+'\n')
save(OUT/'profile.json',profile)
shapes={'rmsnorm':[(2,4608),(7,640),(1,5120),(34,1536),(4,8192),(64,768),(65,384),(2,7168),(96,1280),(3,3072)],'mhc_post':[(2,4,6144),(7,4,384),(1,4,4096),(30,4,768),(4,4,8192),(32,4,384),(34,4,512),(3,4,3072),(72,4,256),(1,4,5120)]}
cases=[]
for op,ten in shapes.items():
    high=0
    for i,s in enumerate(ten+[ten[0]]):
        assert not any(r['op']==op and tuple(r['shape'])==s for r in rows)
        a=contracts[op];p=predict(op,s,a);ctx=predict(op,s,{**a,'resident_aiv':high})
        cases.append(dict(id=f'v10_{op}_{i}',op=op,shape=s,attrs=a,kind='NORM' if op=='rmsnorm' else 'MHC',groups=2 if op=='rmsnorm' else 1,args=[*s,0] if op=='rmsnorm' else s,role='validation' if i<10 else 'paired_control',prediction=p,previous_us=old(op,s,a)['predicted_us'],fixed_us=float(np.median([r['reference_us'] for r in rows if r['op']==op])),sequence_prediction=ctx,prior_aiv=high,group='few_wide' if s[0]<=4 and s[-1]>=2048 else 'controls'))
        high=max(high,p['predicted_active_aiv'])
paths=['operator_model/wide_v10.py','operator_model/vector_v8.py','operator_model/vector_v9.py','results/v0.9/vector/profile.json','results/v0.10/wide/profile.json','scripts/wide-v10-freeze.py','scripts/wide-v10-collect.py','scripts/wide-v10-trace.py','build/vector-v8/NORM','build/vector-v8/MHC','build/capture/libsim_soc_adapter.so','scripts/vec-v9-collect.py','results/v0.10/wide/development-trace.json','scripts/cann-run.sh']
save(OUT/'freeze.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=cases,sha256={s:sha256(ROOT/s) for s in paths},per_batch_seconds=620,disk_budget_gib=3,group_rule='few_wide iff tokens <= 4 and width >= 2048. Default stateless predictions primary; paired repeat is diagnostic only.'))
