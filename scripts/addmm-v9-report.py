"""Publish frozen comparisons plus the full old development replay."""
from pathlib import Path
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from operator_model.ca import sha256
from operator_model.addmm_v9 import predict

OUT=ROOT/'results/v0.9/addmm'
def load(path):
    return json.loads(path.read_text())
def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2)+'\n')
def metrics(rows,key):
    errors=[abs(r[key]-r['reference_us']) for r in rows]
    return dict(wape=sum(errors)/sum(r['reference_us'] for r in rows),
                max_ape=max(err/r['reference_us'] for err,r in zip(errors,rows)))

rows=[]
for r in load(ROOT/'results/v0.8/cube/rows.json'):
    if r['op']=='addmm':
        rows.append(dict(r,role='development',previous_us=r['predicted_us'],predicted_us=predict('addmm',r['shape'],r['attrs'])['predicted_us']))
attempts=[]
cases={r['id']:r for r in load(OUT/'freeze.json')['cases']}
for p in sorted((ROOT/'runs/v9addmm').glob('*/result.json')):
    r=load(p)
    c=cases[r['id']]
    if r['status']!='ok' and 'SMOKE_FAIL Smoke work budget exceeded' in (p.parent/'run.log').read_text():
        r['error']='caller CPU reference work budget exceeded before kernel dispatch'
    attempts.append(dict(id=r['id'],status=r['status'],source=str(p.relative_to(ROOT)),wall_seconds=r['wall_s'],
                         raw_bytes=r['raw_bytes'],error=r.get('error'),log_sha256=r['log_sha256']))
    if r['status']!='ok':continue
    assert len(r['kernels'])==1 and c['predicted_us']==predict('addmm',c['shape'],c['attrs'])['predicted_us']
    k=r['kernels'][0]
    rows.append(dict(c,role='validation',reference_us=r['reference_us'],kernel=k['kernel'],active_aic=k['active_aic'],
                     active_aiv=k['active_aiv'],correctness=r['correctness'],source=str(p.relative_to(ROOT)),source_sha256=sha256(p),
                     kernels=r['kernels'],api='aclnnAddmm',status='ok',wall_s=r['wall_s']))
for row in rows:
    elapsed=[]
    for _ in range(5):
        start=time.perf_counter_ns()
        predict('addmm',row['shape'],row['attrs'])
        elapsed.append((time.perf_counter_ns()-start)/1e6)
    row['query_max_ms']=max(elapsed)
write('rows.json',rows)
valid=[r for r in rows if r['role']=='validation']
summary={'n':len(valid),'worker_seconds':sum(r['wall_seconds'] for r in attempts),'attempts':attempts}
if valid:
    summary.update(v08=metrics(valid,'previous_us'),v09=metrics(valid,'predicted_us'),query_max_ms=max(r['query_max_ms'] for r in valid))
write('costs.json',summary)
print({k:v for k,v in summary.items() if k!='attempts'})
