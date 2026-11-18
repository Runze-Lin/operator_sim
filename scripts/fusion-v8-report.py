"""Summarize qualified fusion captures and retain failed validation attempts."""
import json,sys,time,statistics
from pathlib import Path
import re,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.fusion_v8 import predict,PRE,FUSED,IDX,FIRST,SECOND,CHAIN
OUT=ROOT/'results/v0.8/fusion'
model=json.loads((OUT/'model.json').read_text());freeze=json.loads((OUT/'freeze.json').read_text())
for name,digest in freeze['sha256'].items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Frozen artifact changed: '+name)
rows=[];failed=[];cost=0

def add(op,c,role,estimate,kernels,checks,source):
    attrs=model['contracts'][PRE] if op==PRE else c['attrs']
    elapsed=[]
    for i in range(10):
        start=time.perf_counter_ns();prediction=predict(op,c['shape'],attrs,model)
        elapsed.append((time.perf_counter_ns()-start)/1000)
    assert abs(prediction['predicted_us']-estimate)<1e-8
    rows.append({'op':op,'shape':c['shape'],'attrs':attrs,'role':role,'predicted_us':estimate,'reference_us':sum(k['reference_us'] for k in kernels),
        'api':{IDX:'aclnnLightningIndexer',PRE:'aclnnMhcPre',FUSED:'aclnnMhcPreSinkhorn','mhc':'aclnnMhcPre + aclnnMhcSinkhorn + aclnnMhcPost',FIRST:'aclnnGroupedMatmulSwigluQuantV2',SECOND:'aclnnGroupedMatmulV5',CHAIN:'aclnnGroupedMatmulSwigluQuantV2 + aclnnGroupedMatmulV5'}[op],
        'library':'ops-transformer','kernel_records':[dict(kernel=k['kernel'],launch_id=k['helper_info']['launch_id'],reference_us=k['reference_us'],block_dim=k['block_dim'],active_aic=k['active_aic'],active_aiv=k['active_aiv'],max_core_rv_count=k['max_core_rv_count'],max_core_mmad=k['max_core_mmad'],critical_pipeline_us=max(k['cores'],key=lambda core:core['end_ticks'])['pipeline_union_us']) for k in kernels],
        'kernel':'+'.join(k['kernel'] for k in kernels),'active_aic':[k['active_aic'] for k in kernels],'active_aiv':[k['active_aiv'] for k in kernels],
        'correctness':checks,'source':source,'db_sha256':[k['db_sha256'] for k in kernels],'query_us_max':max(elapsed),'query_us_median':statistics.median(elapsed),'stages_us':prediction['stages_us']})

def emit(c,m,source,role):
    op=c['op'];ks=m['kernels'];pairs=[(op,ks)]
    if op==CHAIN:pairs.extend([(FIRST,ks[:1]),(SECOND,ks[1:])])
    if op=='mhc' and role=='validation':pairs.append((PRE,ks[:1]))
    for route,kernels in pairs:
        estimate=c['predictions'][route] if role=='validation' else predict(route,c['shape'],c['attrs'],model)['predicted_us']
        add(route,c,role,estimate,kernels,m['correctness'],source)

for p in sorted((ROOT/'runs/model-v7').glob('*/result.json')):
    r=json.loads(p.read_text())
    if r['case']['op'] in (IDX,PRE,FUSED,CHAIN,'mhc') and r['status']=='ok':
        emit(r['case'],r['measurement'],str(p.relative_to(ROOT)),'development')
reserve=OUT/'reserve.json'
for c in freeze['cases']+(json.loads(reserve.read_text())['cases'] if reserve.exists() else []):
    p=ROOT/'runs/v8cv'/c['id']/'result.json'
    if not p.exists():continue
    r=json.loads(p.read_text());cost+=r['wall_seconds']
    if r['error']=='unverified Pre stage identity':
        # Different shape can legitimately dispatch another kernel. Confirm
        # sequential caller fences and all stage outputs directly in raw logs.
        from operator_model.npusim_v7 import extract_run
        folder=p.parent
        raw=next(folder.glob('profile/npusim_*/npusim.log')).read_text()
        sections=re.split(r'correctness name=(?:hin|chain_sinkhorn|chain_post) count=',raw)
        task_pattern=r'\[TASK_DONE\].*stream_id:(\d+), task_id:(\d+) TASK_DONE'
        tasks=[set(re.findall(task_pattern,section)) for section in sections]
        if len(tasks)==4 and all(len(group)==1 for group in tasks[:3]) and len(set.union(*tasks[:3]))==3 and not tasks[3]:
            m=extract_run(folder/'profile',folder/'run.log',9)
            if [int(k['helper_info']['launch_id']) for k in m['kernels']]==[0,1,2]:
                r=dict(r,error=None,measurement=m)
    if c['op']==CHAIN and not r['error']:
        if [stage['op'] for stage in r['measurement']['stages']]!=[FIRST,SECOND]:
            raise ValueError('MoE attribution incomplete')
    if r['error']:failed.append({'case':c['id'],'error':r['error'],'source':str(p.relative_to(ROOT))})
    else:emit(c,r['measurement'],str(p.relative_to(ROOT)),'validation')
for p in (ROOT/'runs/v8cv').glob('*-infra0/result.json'):
    r=json.loads(p.read_text());cost+=r['wall_seconds']
    failed.append(dict(case=p.parent.name,error=r['error'],source=str(p.relative_to(ROOT))))
(OUT/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
(OUT/'costs.json').write_text(json.dumps({'cumulative_worker_wall_seconds':cost,'failures':failed,'native_ca_attempts':len(list((ROOT/'runs/v8cv').glob('*/result.json')))},indent=2)+'\n')
print('rows',len(rows),'failed',failed)
by_op={}
for op in (IDX,PRE,FUSED,'mhc',FIRST,SECOND,CHAIN):
    observed=[r for r in rows if r['role']=='validation' and r['op']==op]
    if not observed:continue
    worst=max(observed,key=lambda r:abs(r['reference_us']-r['predicted_us'])/r['reference_us'])
    err=sum(abs(r['reference_us']-r['predicted_us']) for r in observed)
    by_op[op]={'n':len(observed),'wape':err/sum(r['reference_us'] for r in observed),
        'max_ape':abs(worst['reference_us']-worst['predicted_us'])/worst['reference_us'],
        'worst_shape':worst['shape'],'worst_kernel':worst['kernel'],'worst_predicted_us':worst['predicted_us'],'worst_ca_us':worst['reference_us'],
        'critical_pipeline_us_at_worst':[k['critical_pipeline_us'] for k in worst['kernel_records']],
        'max_query_ms':max(r['query_us_max'] for r in observed)/1000}
(OUT/'summary.json').write_text(json.dumps(by_op,indent=2)+'\n')
