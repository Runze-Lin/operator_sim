"""Release only complete independently synchronized CPU-golden cases."""
import json
import math
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.vector_v9 import predict

out=ROOT/'results/v0.9/vector'
rows=json.loads((out/'development-phases.json').read_text())
for row in rows:
    row['role']='development'
    row['previous_us']=row['predicted_us']
    row['prediction']=predict(row['op'],row['shape'],row['attrs'])
    row['predicted_us']=row['prediction']['predicted_us']
costs={}
for path in sorted(out.glob('*-0-10.json')):
    complete=json.loads(path.read_text())
    rows.extend(complete)
    costs[path.stem]=complete[0]['batch_wall_s']
for row in rows:
    times=[]
    for _ in range(10):
        start=time.perf_counter_ns()
        result=predict(row['op'],row['shape'],row['attrs'])
        times.append((time.perf_counter_ns()-start)/1000)
    assert math.isclose(row['predicted_us'],result['predicted_us'],rel_tol=1e-14)
    row['query_max_us']=max(times)
(out/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
(out/'costs.json').write_text(json.dumps(dict(batches=costs,worker_seconds=sum(costs.values())),indent=2)+'\n')
summary={}
for op in sorted(set(r['op'] for r in rows)):
    val=[r for r in rows if r['op']==op and r['role']=='validation']
    if not val:continue
    assert len(set(tuple(r['shape']) for r in val))==10
    def stats(k):
        return {'wape':sum(abs(r[k]-r['reference_us']) for r in val)/sum(r['reference_us'] for r in val),
                'max_ape':max(abs(r[k]/r['reference_us']-1) for r in val)}
    worst=max(val,key=lambda r:abs(r['predicted_us']/r['reference_us']-1))
    summary[op]=dict(v8=stats('previous_us'),v9=stats('predicted_us'),worst=worst['id'],
                     cores_matched=sum(r['active_aiv']==r['prediction']['predicted_active_aiv'] for r in val))
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(summary)
