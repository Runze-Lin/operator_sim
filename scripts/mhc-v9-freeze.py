"""Calibrate disjoint queue occupancies of the latest finishing core, freeze before CA."""
import hashlib,json,sys,statistics,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.mhc_v9 import partition,predict
from operator_model.fusion_v8 import predict as prev
from operator_model.npusim_v7 import extract_run
from operator_model.model import fit_nonnegative_line
import importlib.util
spec=importlib.util.spec_from_file_location('v8',ROOT/'scripts/fusion-v8-freeze.py')
v8=importlib.util.module_from_spec(spec);spec.loader.exec_module(v8)
OUT=ROOT/'results/v0.9/mhc'
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def freeze():
    if (OUT/'freeze.json').exists():raise FileExistsError('Already frozen')
    legacy=json.loads((ROOT/'results/v0.8/fusion/model.json').read_text())
    old=json.loads((ROOT/'results/v0.8/fusion/rows.json').read_text())
    samples={}
    for r in old:
        if r['op']!='mhc':continue
        path=ROOT/r['source'];res=json.loads(path.read_text())
        m=res['measurement'] or extract_run(path.parent/'profile',path.parent/'run.log',9)
        for st,k in zip(('pre','sink','post'),m['kernels']):
            branch,work=partition(st,r['shape'])
            masks=max(v8.project(k),key=lambda parts:sum(parts.values()))
            times={q:0. for q in ('compute','vector','traffic','control')}
            for mask,us in masks.items():
                q=mask.split('+')
                cls='compute' if 'CUBE' in q else 'vector' if 'SIMD' in q else 'traffic' if any(i.startswith(('MTE','FIX')) for i in q) else 'control'
                times[cls]+=us
            assert abs(sum(times.values())-k['reference_us'])<1e-5
            samples.setdefault((st,branch),[]).append((work,times,k['db_sha256']))
    stages={}
    for (st,branch),rows in samples.items():
        co={}
        for key in rows[0][1]:
            x=[w[key] for w,t,h in rows];y=[t[key] for w,t,h in rows]
            if len(set(x))<2:
                a,b=statistics.median(y),0.
            else:
                fit=fit_nonnegative_line(x,y)
                a,b=fit['intercept_us'],fit['slope_us_per_unit']
            co[key]=dict(unit=key,startup_us=a,unit_us=b)
        stages.setdefault(st,{})[branch]=co
    save(OUT/'model.json',dict(contracts={k:legacy['contracts'][k] for k in ('mhc','mhc_pre')},stages=stages))
    seen={tuple(r['shape']) for r in old if r['op'] in ('mhc','mhc_pre')}
    cases=[]
    for i,(t,d) in enumerate([(2,64),(4,96),(6,128),(5,256),(12,384),(8,512),(12,768),(6,1024),(17,512),(24,1536),(1,160),(2,192),(3,320),(4,384)]):
        s=[t,4,d];assert tuple(s) not in seen
        cases.append(dict(id='mhc-%02d'%i,shape=s,reserve=i>=10,attrs=legacy['contracts']['mhc'],arguments=['mhc',*s],
            predictions={op:predict(op,s)['predicted_us'] for op in ('mhc','mhc_pre')},
            previous={op:prev(op,s,{},legacy)['predicted_us'] for op in ('mhc','mhc_pre')}))
    files=['operator_model/mhc_v9.py','results/v0.9/mhc/model.json','scripts/mhc-v9-freeze.py','scripts/mhc-v9-collect.py','build/smoke/extension_indexer_mhc','build/capture/libsim_soc_adapter.so']
    save(OUT/'freeze.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),timeout_s=360,case_limit_mib=1000,cases=cases,
        sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},training_db={st+'_'+b:[h for w,t,h in rows] for (st,b),rows in samples.items()}))
if __name__=='__main__':freeze()
