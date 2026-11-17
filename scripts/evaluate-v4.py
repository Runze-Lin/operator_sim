#!/usr/bin/env python3
"""Frozen holdout comparisons and paired benchmarks; no fitting here."""
from datetime import datetime,timedelta
import json
from pathlib import Path
import runpy
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.ca import sha256
from operator_model.model_v3 import Predictor as V3
from operator_model.model_v4 import Case,Predictor
util=runpy.run_path(str(ROOT/'scripts/evaluate-v1.py'));metrics=util['metrics'];quantile=util['quantile']
replay_equal=runpy.run_path(str(ROOT/'scripts/evaluate-v2.py'))['replay_equal']

def paired_benchmark(cases,n=4000):
 ps={'v03':V3(ROOT/'results/v0.3/model.json'),'v04':Predictor(ROOT/'results/v0.4/model.json')}
 for p in ps.values():
  for c in cases:p.predict(c)
 results={}
 for mode in ('uncached','cached'):
  timings={k:[] for k in ps}
  for i in range(n):
   c=cases[i%len(cases)]
   for key in (('v03','v04') if i%2 else ('v04','v03')):
    start=time.perf_counter_ns();ps[key].predict(c,use_cache=mode=='cached');timings[key].append((time.perf_counter_ns()-start)/1000)
  results[mode]={key:{'p50_us':statistics.median(ts),'p95_us':quantile(ts,.95),'queries':len(ts)} for key,ts in timings.items()}
 return results

def main():
 out=ROOT/'results/v0.4';f=json.loads((out/'frozen-predictions.json').read_text());spec=json.loads((ROOT/'configs/sampling-v4.json').read_text())
 if sha256(out/'model.json')!=f['model_sha256'] or sha256(ROOT/'configs/sampling-v4.json')!=f['manifest_sha256']:raise ValueError('Frozen model/manifest mismatch')
 for path,digest in f['artifact_sha256'].items():
  if path!='build/smoke/three_ops' and sha256(ROOT/path)!=digest:raise ValueError('Frozen artifact changed: '+path)
 data=json.loads((out/'dataset.json').read_text());rs={r['case']['id']:r for r in data['records']}
 if len(rs)!=len(data['records']):raise ValueError('Duplicate records')
 p=Predictor(out/'model.json');v3=V3(ROOT/'results/v0.3/model.json')
 if set(p.model['training_keys'])!={r['cache_key'] for r in data['records'] if r['case']['split']=='calibration'}:raise ValueError('Training set changed')
 envkeys=('soc_version','cann_version','package_manifest_sha256','platform_sha256','sim_config_sha256')
 if len({tuple(r['identity'][k] for k in envkeys) for r in data['records']})!=1:raise ValueError('Environment changed')
 if data['manifest_sha256']!=f['manifest_sha256']:raise ValueError('Dataset manifest mismatch')
 rows=[];missing=[]
 for item in f['predictions']:
  c=Case(item['op'],item['shape']);r=rs.get(item['id']);new=item['prediction'];old=item['v03_prediction']
  for a,b in [(new,p.predict(c)),(old,v3.predict(c))]:
   if not replay_equal(a,b):raise ValueError('Frozen prediction mismatch')
  if r is not None:
   if any(r['case'][k]!=item[k] for k in ('op','shape','split')):raise ValueError('Case mismatch')
   if r['identity']['host_binary_sha256']!=f['artifact_sha256']['build/smoke/three_ops']:raise ValueError('Binary mismatch')
   start=datetime.fromisoformat(r['finished_utc'])-timedelta(seconds=r['wall_seconds'])
   if start<=datetime.fromisoformat(f['frozen_utc']):raise ValueError('Holdout predates freeze')
  if r is None or r['status']!='ok':
   failure={**{k:item[k] for k in ('id','op','shape')},'status':r['status'] if r else 'not_run','error':r.get('error') if r else None,'outer_timed_out':r.get('timed_out') if r else None}
   if r:
    output=next((a.split('=',1)[1] for a in r['command'] if a.startswith('--output=')),None)
    log=Path(output).parent/'run.log' if output else None
    if log and log.exists():
     text=log.read_text(errors='replace');failure['profiler_timed_out']='timeout has reached' in text
     failure['log_sha256']=sha256(log)
     failure['diagnostic_lines']=[line for line in text.splitlines() if any(x in line for x in ('timeout has reached','Child process exited','SMOKE_FAIL','SMOKE_PASS'))]
   missing.append(failure);continue
  m=r['measurement'];actual=max(sum(v for k,v in core['dynamic_opcodes'].items() if k.startswith('RV_')) if c.op=='quant' else core['dynamic_opcodes'].get('MMAD',0) for core in m['cores'])
  predcores=[new['plan']['active_aic'],new['plan']['active_aiv']];obscores=[m['active_aic'],m['active_aiv']]
  rows.append({**{k:item[k] for k in ('id','op','shape','selection_reason')},'reference_us':m['reference_us'],'predicted_us':new['predicted_us'],
   'v03_us':old['predicted_us'],'constant_us':new['constant_baseline_us'],'v03_constant_us':old['constant_baseline_us'],
   'ape':abs(new['predicted_us']/m['reference_us']-1),'v03_domain_extension':item['v03_domain_extension'],
   'predicted_cores':predcores,'observed_cores':obscores,'mapping_matches':predcores==obscores,
   'predicted_opcode_count':new['estimated_max_core_selected_opcode_count'],'observed_opcode_count':actual,
   'ca_wall_seconds':r['wall_seconds'],'ca_cpu_seconds':r['cpu_seconds'],'raw_bytes':r['raw_bytes'],
   'kernel':m['kernel'],'trace_sha256':m['trace_sha256'],'phase_schedule':new.get('phase_schedule'),
   'plan':new['plan'],'correctness':m.get('correctness')})
 groups={'all':rows,'quant':[r for r in rows if r['op']=='quant'],'fa':[r for r in rows if r['op']=='fa'],
         'quant_small':[r for r in rows if r['op']=='quant' and r['plan']['work_per_core']<=4096],
         'quant_ub':[r for r in rows if r['op']=='quant' and r['plan']['work_per_core']>4096]}
 agg={g:{key:metrics(rs,key) for key in ('predicted_us','v03_us','constant_us','v03_constant_us')} for g,rs in groups.items()}
 attempts=data['attempts'];cost={k:sum(a[k] for a in attempts) for k in ('wall_seconds','cpu_seconds','raw_bytes')}
 cost.update(attempts=len(attempts),successful_attempts=sum(a['status']=='ok' for a in attempts),reused_shapes=38,reused_quant_traces=13)
 gates={}
 for op in ('quant','fa'):
  a=agg[op]['predicted_us'];expected=sum(x['op']==op for x in spec['cases'])
  gates[op]={'complete':bool(a and a['count']==expected),'accuracy_on_successes':bool(a and a['wape']<=.1 and a['p90_ape']<=.2)}
 cases=[Case(x['op'],x['shape']) for x in f['predictions']]
 workload=json.loads((out/'workload-coverage.json').read_text())
 wq=[Case(x['op'],x['shape']) for x in workload['unique_predictions'].values() if x['op']=='quant']
 bench={'python':sys.version,'protocol':'Paired alternating version order, same preconstructed Cases, 4000 calls/version/mode, after CA. Full predict API includes plan and immutable diagnostics; no trace reads. v0.3 extended shapes bypass only its old Case guard, all formulas/parameters unchanged.',
        'holdout_all':paired_benchmark(cases),'holdout_quant':paired_benchmark([c for c in cases if c.op=='quant']),
        'workload_quant':paired_benchmark(wq)}
 # Scalar-only API is reported separately; not used to claim full-API improvement.
 times=[]
 for i in range(4000):
  c=wq[i%len(wq)];start=time.perf_counter_ns();p.predict_us(c,False);times.append((time.perf_counter_ns()-start)/1000)
 bench['v04_scalar_uncached_workload_quant']={'p50_us':statistics.median(times),'p95_us':quantile(times,.95)}
 gates['speed']={group:bench[group]['uncached']['v04']['p95_us']<=spec['acceptance']['uncached_p95_us'] and bench[group]['cached']['v04']['p95_us']<=spec['acceptance']['cached_p95_us'] for group in ('holdout_all','holdout_quant','workload_quant')}
 gates['budget']={key:cost[key]<=spec['budget'][key] for key in ('wall_seconds','cpu_seconds','raw_bytes')}
 gates['mapping_on_completed']=all(r['mapping_matches'] for r in rows)
 gates['all_planned_complete']=not missing
 result={'frozen_utc':f['frozen_utc'],'model_sha256':f['model_sha256'],'rows':rows,'metrics':agg,'gates':gates,'missing':missing,
         'collection_cost':cost,'benchmark':bench,'failed_attempts':[r for r in attempts if r['status']!='ok']}
 (out/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ('metrics','gates','missing','collection_cost','benchmark')},indent=2))
if __name__=='__main__':main()
