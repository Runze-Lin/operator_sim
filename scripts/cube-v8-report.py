#!/usr/bin/env python3
import json,statistics,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));out=root/'results/v0.8/cube'
from operator_model.cube_v8 import predict
from operator_model.ca import sha256

def dump(name,value):
    (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

frozen=json.loads((out/'freeze.json').read_text());old=json.loads((root/'results/v0.7/evaluation.json').read_text());seen={};rows=[]
for r in old['rows']:
    if r['op'] not in ('matmul','addmm'):continue
    attrs=r.get('attributes',{});p=predict(r['op'],r['shape'],attrs)
    rows.append({'id':r['id'],'op':r['op'],'shape':r['shape'],'attrs':attrs,'role':'development','predicted_us':p['predicted_us'],'reference_us':r['reference_us'],'kernel':r['kernel'],'active_aic':r['observed_cores'][0] if r['op']=='matmul' else r['observed_cores'][0][0],'active_aiv':r['observed_cores'][1] if r['op']=='matmul' else r['observed_cores'][0][1],'correctness':r['correctness'],'source':r['source'],'api':r['api'],'status':'ok'})
failures=[];attempts=[]
by_id={c['id']:c for c in json.loads((out/'safe-chunk-freeze.json').read_text())['cases']+frozen['cases']+json.loads((out/'replacement-freeze.json').read_text())['cases']}
for p in sorted([*root.glob('runs/v8cube/*/result.json'),*root.glob('runs/v8cube-isolated/*/result.json')]):
    c=by_id[p.parent.name]
    r=json.loads(p.read_text());attempts.append({k:r.get(k) for k in ('id','op','status','wall_s','error','raw_bytes')})
    if r['status']!='ok':
        failures.append(dict(attempts[-1],shape=r['shape'],source=str(p.relative_to(root)),log_sha256=r['log_sha256']));continue
    ks=r['kernels'];one=len(ks)==1
    rows.append({'id':r['id'],'op':r['op'],'shape':r['shape'],'attrs':r['attrs'],'role':'validation','predicted_us':c['prediction']['predicted_us'],'reference_us':r['reference_us'],'kernel':' + '.join(k['kernel'] for k in ks),'active_aic':ks[0]['active_aic'] if one else [k['active_aic'] for k in ks],'active_aiv':ks[0]['active_aiv'] if one else [k['active_aiv'] for k in ks],'correctness':r['correctness'],'source':str(p.relative_to(root)),'source_sha256':sha256(p),'kernels':ks,'api':{'matmul':'aclnnMatmul','addmm':'aclnnAddmm','kda_chunk':'aclnnChunkKdaFwd'}[r['op']],'status':'ok','query_seconds':c['query_seconds'],'wall_s':r['wall_s']})
dump('rows.json',rows);dump('attempts.json',{'attempts':attempts,'failures':failures,'sum_ca_wall_s':sum(r['wall_s'] for r in attempts),'sum_raw_bytes':sum(r['raw_bytes'] for r in attempts)})
print('validation counts',[(op,len([r for r in rows if r['op']==op and r['role']=='validation'])) for op in ['matmul','addmm','kda_chunk']]);print(failures)
# Locate phase overlap on the worst complete Addmm validation point. This is a
# post-freeze explanation and never enters the predictions.
import sqlite3
from operator_model.npusim_v7 import extract_db
addrows=[r for r in rows if r['role']=='validation' and r['op']=='addmm']
if addrows:
    worst=max(addrows,key=lambda r:abs(r['predicted_us']/r['reference_us']-1));folder=(root/worst['source']).parent;db=next(folder.rglob('*.db'));k=extract_db(db)
    core=max(k['cores'],key=lambda c:c['end_ticks']);cx=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True)
    sql="SELECT ExecInstrTickStart,ExecInstrTickEnd FROM ExecutedInstructions WHERE CoreTypeId=? AND ExecInstrCoreId=? AND ExecInstrIsDurationExcluded=0 AND json_extract(ExtendParams,'$.queue_type') IN ('CUBE','MTE1','MTE2','FIXP','MTE3') ORDER BY ExecInstrTickStart"
    union=0;last=0
    for a,b in cx.execute(sql,(core['core_type_id'],core['core_id'])):union+=max(0,b-max(a,last));last=max(last,b)
    union/=1650;candidate=predict('addmm',worst['shape'],worst['attrs'])
    dump('worst-addmm.json',{'case':worst['id'],'reference_us':worst['reference_us'],'predicted_us':worst['predicted_us'],'db_sha256':k['db_sha256'],'critical_core_id':core['core_id'],'active_aic':k['active_aic'],'max_core_mmad':k['max_core_mmad'],'critical_data_queue_union_us':union,'critical_uncovered_us':core['span_us']-union,'per_queue_union_us':core['pipeline_union_us'],'estimated_stage_services_us':candidate['stage_services_us'],'estimated_uncovered_us':candidate['control_us']})
