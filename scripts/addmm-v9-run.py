"""Freeze before CA, then collect sequential private-directory jobs per worker."""
import argparse
import concurrent.futures
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from operator_model.addmm_v9 import predict
from operator_model.cube_v8 import predict as baseline
from operator_model.ca import sha256

OUT=ROOT/'results/v0.9/addmm'
FREEZE=OUT/'freeze.json'
RUNS=ROOT/'runs/v9addmm'
shapes=[(1,64,32),(3,48,96),(8,96,192),(17,128,128),(32,80,272),(48,256,384),(96,192,256),
        (128,256,384),(256,384,256),(384,512,384),(2,64,32),(24,96,160),(96,256,128),(256,256,384),
        (7,128,64),(64,160,256),(128,224,192),(256,256,128)]
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['freeze','collect'])
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.mode=='freeze':
        if FREEZE.exists():raise SystemExit('already frozen')
        prior=json.loads((ROOT/'results/v0.8/cube/rows.json').read_text())
        assert not any(tuple(r['shape']) in shapes for r in prior if r['op']=='addmm')
        files=['operator_model/addmm_v9.py','results/v0.9/addmm/profile.json','scripts/addmm-v9-run.py',
               'scripts/addmm-v9-calibrate.py','scripts/cube-v8-isolated.py','operator_model/cube_v8.py','operator_model/model_v2.py',
               'operator_model/model.py','operator_model/npusim_v7.py','build/smoke/extension_norm_addmm','build/capture/libsim_soc_adapter.so']
        cases=[]
        for s in shapes:
            attrs={'self_shape':[s[1]],'alpha':1,'beta':1}
            cases.append(dict(op='addmm',id='addmm_'+'_'.join(map(str,s)),shape=s,attrs=attrs,timeout_s=240,
                              predicted_us=predict('addmm',s,attrs)['predicted_us'],previous_us=baseline('addmm',s,attrs)['predicted_us']))
        FREEZE.write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),cases=cases,target=10,
                                          artifact_sha256={f:sha256(ROOT/f) for f in files}),indent=2)+'\n')
    else:
        spec=importlib.util.spec_from_file_location('v8collector',ROOT/'scripts/cube-v8-isolated.py')
        mod=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        frozen=json.loads(FREEZE.read_text())
        assert all(sha256(ROOT/f)==h for f,h in frozen['artifact_sha256'].items())
        RUNS.mkdir(exist_ok=True)
        mod.runs=RUNS
        mod.frozen=frozen
        jobs=iter(frozen['cases'])
        ok=0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            while ok<10:
                batch=[next(jobs,None) for _ in range(min(args.workers,10-ok))]
                batch=[c for c in batch if c]
                if not batch:break
                ok+=sum(r['status']=='ok' for r in pool.map(mod.run,batch))
