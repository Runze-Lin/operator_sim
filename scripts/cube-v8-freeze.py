#!/usr/bin/env python3
"""Derive uncovered scalar/control cost from union of arithmetic and DMA intervals."""
from pathlib import Path
import sys,sqlite3,json,time,statistics,hashlib
from datetime import datetime,timezone
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));out=root/'results/v0.8/cube'
from operator_model.cube_v8 import predict
from operator_model.ca import sha256
if (out/'freeze.json').exists():raise SystemExit('already frozen')
rows=[]
for case in ('a7_cal_k256','a7_hold_q9'):
    folder=root/'runs/model-v7'/case
    db=next(folder.rglob('*.db'));cx=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True)
    start,end=cx.execute('select min(ExecInstrTickStart),max(ExecInstrTickEnd) from ExecutedInstructions').fetchone()
    segments=cx.execute("select ExecInstrTickStart,ExecInstrTickEnd from ExecutedInstructions where ExecInstrIsDurationExcluded=0 and json_extract(ExtendParams,'$.queue_type') in ('CUBE','MTE1','MTE2','FIXP','MTE3') order by ExecInstrTickStart")
    union=0;last=start
    for a,b in segments:
        union+=max(0,b-max(a,last));last=max(last,b)
    rows.append({'case':case,'db_sha256':sha256(db),'uncovered_us':(end-start-union)/1650})
(out/'control.json').write_text(json.dumps({'rows':rows,'median_uncovered_us':statistics.median(r['uncovered_us'] for r in rows)},indent=2)+'\n')
shapes=[(1,32,16),(4,80,48),(9,128,256),(16,192,384),(31,64,257),(64,128,512),(128,192,256),(192,384,256),(256,512,256),(512,512,256)]
configs=[(op,s,{} if op=='matmul' else {'self_shape':[s[1]],'alpha':1,'beta':1}) for s in shapes for op in ('matmul','addmm')]
configs += [('kda_chunk',(1,s,1,1,k,k),dict(chunk_size=64,export_gk=True)) for s,k in [(64,128),(1,16),(8,16),(16,16),(63,16),(64,16),(65,16),(96,128),(127,128),(128,128)]]
old=json.loads((root/'results/v0.7/evaluation.json').read_text());seen={(r['op'],tuple(r['shape'])) for r in old['rows']}
cases=[]
for op,shape,attrs in configs:
    assert (op,shape) not in seen
    start=time.perf_counter_ns();prediction=predict(op,shape,attrs) if op!='kda_chunk' else {'predicted_us':None,'status':'CA timing unqualified'};elapsed=(time.perf_counter_ns()-start)/1e9
    case={'id':op+'_'+'_'.join(map(str,shape)),'op':op,'shape':shape,'attrs':attrs,'prediction':prediction,'query_seconds':elapsed,'timeout_s':240}
    cases.append(case)
files=['operator_model/cube_v8.py','operator_model/model.py','operator_model/model_v2.py','operator_model/model_v3.py','operator_model/model_v4.py','operator_model/model_v5.py','operator_model/model_v6.py','operator_model/fast_query.py','operator_model/npusim_v7.py','results/v0.2/model.json','results/v0.6/model.json','results/v0.8/cube/control.json','scripts/cube-v8-freeze.py','scripts/cube-v8-collect.py','build/smoke/three_ops','build/smoke/extension_kda','build/smoke/extension_norm_addmm','scripts/cann-run.sh','build/capture/libsim_soc_adapter.so']
(out/'freeze.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'cases':cases,'artifact_sha256':{f:sha256(root/f) for f in files}},indent=2)+'\n')
