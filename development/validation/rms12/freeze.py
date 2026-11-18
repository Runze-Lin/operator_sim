"""Robust nonnegative calibration from phase/instruction durations only; never total CA."""
import hashlib,json,sys
from datetime import datetime,timezone
from statistics import median
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from operator_model.rms_v12 import predict,tile_rows,pressure
from operator_model.rms_v11 import predict as previous
OUT=ROOT/'results/v0.12/rms'; assert not (OUT/'freeze.json').exists()
rows=json.loads((OUT/'development-evidence.json').read_text());old=json.loads((ROOT/'results/v0.11/rms/rows.json').read_text()); assert len(rows)==len(old)==43
xy={k:[] for k in ('simd','rstd','y','normalize','reduce')}
for r in rows:
    t,d=r['shape'];a=r['active'];h=r['prior'];state=[h==0,0<h<a,h>=a];R=(t+63)//64;lane=(d+127)//128
    s=r['stores'];tiles=[e['bytes']//4 for e in s[::2]];assert sum(tiles)<=R and tiles==tile_rows(sum(tiles),d)
    xy['simd'].append(([*state,sum(tiles)*lane],r['simd_ticks']))
    for j,n in enumerate(tiles):
        st,y=s[2*j:2*j+2];assert y['bytes']==n*2*d
        work=n*lane;norm=y['start']-st['start'];xy['normalize'].append(([1,work],norm));xy['rstd'].append(([1,pressure(a,R,128)],st['end']-st['start']))
        if y['end']>st['end']+16 and (j+1==len(tiles) or y['end']<s[2*j+2]['start']):xy['y'].append(([1,y['bytes']/1024],y['end']-y['start']))
        if len(tiles)==1:xy['reduce'].append(([1,work],max(0,r['simd_ticks']-norm)))

def fit(points):
    x,y=map(np.array,zip(*points));return least_squares(lambda b:x@b-y,[1.]*x.shape[1],bounds=(0,np.inf),loss='soft_l1',f_scale=80).x.tolist()
p={k:fit(points) for k,points in xy.items()}; p.update(ticks_per_us=1650,window_bytes=128,training_counts={k:len(v) for k,v in xy.items()},loss='nonnegative soft-L1; transition 80 ticks',y_censor='completion > rstd end+16 ticks and no pending later rstd; service beyond qualifying bytes extrapolates',fit_source='43 historical critical AIV event timelines; no total timings fitted');(OUT/'profile.json').write_text(json.dumps(p,indent=2)+'\n')
shapes=[[1,4352],[2,5888],[4,5632],[3,6656],[37,1536],[192,3072],[448,4096],[480,4096],[384,6144],[512,6144]];assert len({tuple(s) for s in shapes})==10 and all(s!=r['shape'] for s in shapes for r in rows)
batches=[[0,5],[5,8],[8,10]];cases=[];fixed=median(r['reference_us'] for r in old)
for i,s in enumerate(shapes):
    if i in [x[0] for x in batches]:h=0
    d=predict('rmsnorm',s);cases.append(dict(id=f'v12_rms_{i}',op='rmsnorm',shape=s,attrs={},kind='NORM',groups=2,args=s+[0],role='validation',prediction=d,previous_us=previous('rmsnorm',s)['predicted_us'],fixed_us=fixed,prior_aiv=h,group='few_wide' if s[0]<=4 and s[1]>=2048 else 'controls',sequence_prediction=predict('rmsnorm',s,{'resident_aiv':h})));h=max(h,d['predicted_active_aiv'])
files=['build/vector-v8/NORM','build/capture/libsim_soc_adapter.so','operator_model/rms_v12.py','operator_model/rms_v11.py','operator_model/wide_v10.py','operator_model/vector_v9.py','operator_model/vector_v8.py','results/v0.10/wide/profile.json','results/v0.11/rms/profile.json','results/v0.12/rms/profile.json','results/v0.12/rms/development-evidence.json','development/validation/rms12/freeze.py','development/validation/rms12/collect.py','development/validation/scaling_v11.py'];digest={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
(OUT/'freeze.json').write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),cases=cases,sha256=digest,per_batch_seconds=600,disk_budget_gib=1.5,batches=batches,group_rule='few_wide T<=4 and D>=2048',budget='10 new launches; 3 isolated processes; recovery only for incomplete outputs/helper; at most 4 repeats + 4 conditioning controls',previous='v0.11 frozen public default'),indent=2)+'\n'); print(p)
