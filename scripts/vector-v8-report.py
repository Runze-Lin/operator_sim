"""Summarize independently measured Vector cases using their frozen predictions."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.vector_v8 import predict
out=ROOT/'results/v0.8/vector'
prof=json.load(open(out/'profile.json'))
rows=json.load(open(out/'development.json'))
seen=set()
for path in sorted(out.glob('*-*-*.json')):
    for r in json.load(open(path)):
        if r['id'] in seen:raise ValueError('Duplicated validation')
        seen.add(r['id']);rows.append(r)
for r in rows:
    times=[]
    for _ in range(17):
        start=time.perf_counter_ns()
        q=predict(r['op'],r['shape'],r['attrs'],prof)
        times.append((time.perf_counter_ns()-start)/1000)
    assert q['predicted_us']==r['predicted_us']
    r['query_max_us']=max(times)
(out/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
for op in prof:
    valid=[r for r in rows if r['op']==op and r['role']=='validation']
    if valid:print(op,len(valid),sum(abs(r['predicted_us']-r['reference_us']) for r in valid)/sum(r['reference_us'] for r in valid),max(r['query_max_us'] for r in valid))
# An initialization failure can terminate before the wrapper exports a result.
# Use npusim's own observed USER_APP duration for these attempts, not the timeout.
import re
attempts=[]
for folder in sorted((ROOT/'runs/v8vec').glob('*')):
    summary=out/(folder.name+'.json')
    if summary.exists():
        attempts.append({'batch':folder.name,'status':'complete','wall_seconds':json.loads(summary.read_text())[0]['batch_wall_s']})
    elif (folder/'run.log').exists():
        log=(folder/'run.log').read_text(errors='replace')
        match=re.search(r'USER_APP failed after ([\d.]+)s',log)
        if match:attempts.append({'batch':folder.name,'status':'failed','wall_seconds':float(match[1]),'scope':'reported npusim user-app duration; host launch excluded'})
(out/'costs.json').write_text(json.dumps({'attempts':attempts,'worker_seconds':sum(r['wall_seconds'] for r in attempts)},indent=2)+'\n')
