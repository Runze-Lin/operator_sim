#!/usr/bin/env python3
"""Evaluate the frozen quant candidate; never fit to new results."""
import json
import statistics
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.ca import sha256
from operator_model.model_v6 import Case
from operator_model.fast_query import Predictor
from operator_model.model_v5 import Predictor as V5
from operator_model.evaluation_v6 import evaluate_record


def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def metrics(rows,key):
    if not rows:return None
    return {'n':len(rows),'wape':sum(abs(r[key]-r['reference_us']) for r in rows)/sum(r['reference_us'] for r in rows),
        'max_ape':max(abs(r[key]-r['reference_us'])/r['reference_us'] for r in rows)}

def benchmark(new,old,cases):
    out={}
    for mode in ['uncached_scalar','uncached_full','cached_full']:
        timings={'v06':[],'v05':[]}
        for c in cases:new.predict(c);old.predict(c)
        for i in range(3000):
            c=cases[i%len(cases)]
            for key,p in ([('v06',new),('v05',old)] if i%2 else [('v05',old),('v06',new)]):
                t=time.perf_counter_ns()
                if mode=='uncached_scalar':p.predict_us(c,use_cache=False)
                else:p.predict(c,use_cache=mode=='cached_full')
                timings[key].append((time.perf_counter_ns()-t)/1000)
        out[mode]={k:{'median_us':statistics.median(v),'p95_us':sorted(v)[int(.95*(len(v)-1))]} for k,v in timings.items()}
    return out

def main():
    out=ROOT/'results/v0.6';fp=out/'frozen-predictions.json';f=json.loads(fp.read_text())
    if sha256(out/'model.json')!=f['model_sha256']:raise ValueError('Changed frozen model')
    for path,h in f['artifact_sha256'].items():
        if sha256(ROOT/path)!=h:raise ValueError('Changed frozen artifact '+path)
    new=Predictor(out/'model.json');old=V5(ROOT/'results/v0.5/model.json');rows=[];records=[];missing=[]
    for c in f['predictions']:
        case=Case(c['op'],c['shape'])
        if abs(new.predict_us(case,False)-c['prediction']['predicted_us'])>1e-10:raise ValueError('Replay drift')
        path=ROOT/'runs/model-v6'/c['id']/'result.json'
        if not path.exists():missing.append(c['id']);continue
        qualified=path.with_name('qualified-result.json')
        if qualified.exists():
            r=json.loads(qualified.read_text())
            if r['recovery']['source_result_sha256']!=sha256(path):raise ValueError('Recovery source changed')
        else:r=json.loads(path.read_text())
        r['measurement']['rv_count_scope']='helper ExecutedInstructions records; historical CSV used paired log events'
        r['measurement']['rv_count_comparable_to_legacy']=False
        records.append(r)
        row=evaluate_record(c,r,f,sha256(fp))
        if row:rows.append(row)
    diag=json.loads((out/'calibration-diagnostics.json').read_text())
    data={'model_sha256':f['model_sha256'],'freeze_sha256':sha256(fp),'rows':rows,'missing':missing,
        'failed':[r['case']['id'] for r in records if r['status']!='ok'],
        'metrics':{k:metrics(rows,k) for k in ['v06_us','v05_us','constant_us']},
        'development':{'rows':diag['rows'],'metrics':{k:metrics(diag['rows'],k) for k in ['v06_us','v05_us','residual_loo_us']},'scope':diag['scope']},
        'cost':{'attempts':len(records),'new_calibration_ca':0,'wall_seconds':sum(r['wall_seconds'] for r in records),
            'cpu_seconds':sum(r['cpu_seconds'] for r in records),'raw_bytes':sum(r['raw_bytes'] for r in records),
            'extraction_seconds':sum(r['extraction_seconds'] for r in records)},
        'query_implementation':{'path':'operator_model/fast_query.py','sha256':sha256(ROOT/'operator_model/fast_query.py'),'scope':'post-freeze execution optimization; identical frozen timing equations'},
        'benchmark':benchmark(new,old,[Case(c['op'],c['shape']) for c in f['predictions']])}
    write(out/'dataset.json',{'records':records});write(out/'evaluation.json',data)
    print(json.dumps({k:data[k] for k in ['metrics','cost','missing','benchmark']},indent=2))
if __name__=='__main__':main()
