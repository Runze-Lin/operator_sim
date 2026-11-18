"""Freeze nonnegative critical-queue service lines before new cases."""
from pathlib import Path
import json,sys,datetime
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.vector_v9 import schedule,predict,PHASES
from operator_model.ca import sha256
from operator_model.vector_v8 import predict as old
out=ROOT/'results/v0.9/vector'
if (out/'freeze.json').exists():raise ValueError('already frozen')
rows=json.loads((out/'development-phases.json').read_text())
old_profile=json.loads((ROOT/'results/v0.8/vector/profile.json').read_text())
profile={'phases':{},'contracts':{op:p['contract'] for op,p in old_profile.items() if op!='kda_recurrent'}}
for route in ('rmsnorm','post_channel','post_token','mhc_sinkhorn'):
    phases=profile['phases'][route]={}
    for phase in PHASES:
        xx=[];yy=[]
        for r in rows:
            key,active,waves,units=schedule(r['op'],r['shape'],r['attrs'])
            if key==route:
                xx.append(units[phase]);yy.append(r['phase_us'].get(phase,0))
        x=np.array(xx);y=np.array(yy)
        if x.max()==x.min():
            phases[phase]=[float(np.median(y)),0.]
        else:
            # Robust service fit as issue skew and warm states differ between captures.
            sol=least_squares(lambda b:b[0]+b[1]*x-y,[.1,.01],bounds=(0,np.inf),loss='soft_l1',f_scale=.15)
            phases[phase]=sol.x.tolist()
def write(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
write('profile.json',profile)
shapes={'rmsnorm':[(2,128),(3,384),(8,1536),(15,3072),(17,7168),(24,4096),(63,512),(66,256),(96,2048),(1,6144)],
        'mhc_post':[(1,4,96),(3,4,256),(6,4,512),(11,4,1024),(17,4,256),(31,4,512),(33,4,256),(63,4,512),(66,4,128),(2,4,4096)],
        'mhc_sinkhorn':[(3,4),(5,4),(7,4),(11,4),(15,4),(21,4),(33,4),(47,4),(66,4),(96,4)]}
cases=[]
for op,targets in shapes.items():
    for i,shape in enumerate(targets):
        assert not any(r['op']==op and tuple(r['shape'])==shape for r in rows)
        attrs=dict(profile['contracts'][op])
        if op=='mhc_sinkhorn':attrs['num_iters']=[2,6,12,20,24,16,20,4,10,20][i]
        args=list(shape)
        if op=='rmsnorm':args.append(0)
        if op=='mhc_sinkhorn':args.append(attrs['num_iters'])
        cases.append(dict(id=f'v9_{op}_{i}',op=op,shape=shape,attrs=attrs,args=args,groups=2 if op=='rmsnorm' else 1,
                          kind='NORM' if op=='rmsnorm' else 'MHC',prediction=predict(op,shape,attrs),
                          previous_us=old(op,shape,attrs,old_profile)['predicted_us']))
paths=['operator_model/vector_v9.py','results/v0.9/vector/profile.json','scripts/vec-v9-collect.py','scripts/vec-v9-freeze.py','build/capture/libsim_soc_adapter.so','build/vector-v8/NORM','build/vector-v8/MHC']
write('freeze.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=cases,
                         per_batch_seconds=1400,disk_budget_gib=8,sha256={p:sha256(ROOT/p) for p in paths}))
