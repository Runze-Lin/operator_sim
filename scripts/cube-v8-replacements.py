"""Predesign small replacements for failed originals, without refitting the model."""
import sys,json,time,importlib.util
from datetime import datetime,timezone
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));out=root/'results/v0.8/cube';file=out/'replacement-freeze.json'
from operator_model.cube_v8 import predict
from operator_model.ca import sha256
if sys.argv[1]=='freeze':
    if file.exists():raise SystemExit('already frozen')
    cases=[]
    for op in ('addmm','matmul'):
        for s in ((2,32,32),(32,96,128),(64,160,256),(24,80,64)):
            attrs={'self_shape':[s[1]],'alpha':1,'beta':1} if op=='addmm' else {}
            t=time.perf_counter_ns();pred=predict(op,s,attrs);duration=(time.perf_counter_ns()-t)/1e9
            cases.append(dict(id=op+'_'+'_'.join(map(str,s)),op=op,shape=s,attrs=attrs,prediction=pred,query_seconds=duration,timeout_s=240))
    file.write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),model_freeze_sha256=sha256(out/'freeze.json'),cases=cases),indent=2)+'\n')
else:
    f=json.loads(file.read_text());assert f['model_freeze_sha256']==sha256(out/'freeze.json');spec=importlib.util.spec_from_file_location('collector',root/'scripts/cube-v8-collect.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    for op in ('matmul','addmm'):
        ok=sum(json.loads(p.read_text())['status']=='ok' for p in (root/'runs/v8cube').glob(op+'*/result.json'))
        for case in f['cases']:
            if case['op']==op and ok<10 and not (root/'runs/v8cube'/case['id']).exists():ok+=int(c.run(case)['status']=='ok')
