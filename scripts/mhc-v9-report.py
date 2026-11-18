"""Compare frozen split mHC models on identical new CA calls, keep raw attribution."""
import json,sys,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.mhc_v9 import predict,profile,partition
OUT=ROOT/'results/v0.9/mhc'
def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2)+'\n')
def main():
    rows=[]
    for r in json.loads((ROOT/'results/v0.8/fusion/rows.json').read_text()):
        if r['op'] in ('mhc','mhc_pre'):
            estimate=predict(r['op'],r['shape'],r['attrs'])
            rows.append(dict(r,role='development',previous_us=r['predicted_us'],predicted_us=estimate['predicted_us'],stages_us=estimate['stages_us']))
    attempts=[]
    for c in json.loads((OUT/'freeze.json').read_text())['cases']:
        path=ROOT/'runs/v9mhc'/c['id']/'result.json'
        if not path.exists():continue
        r=json.loads(path.read_text())
        attempts.append(dict(id=c['id'],shape=c['shape'],wall_seconds=r['wall_seconds'],error=r['error'],source=str(path.relative_to(ROOT))))
        if r['error']:continue
        m=r['measurement']
        for op in ('mhc_pre','mhc'):
            ks=m['kernels'][:1] if op=='mhc_pre' else m['kernels']
            timed=[]
            for _ in range(10):
                start=time.perf_counter_ns();res=predict(op,c['shape'],profile()['contracts'][op]);timed.append((time.perf_counter_ns()-start)/1e3)
            assert abs(res['predicted_us']-c['predictions'][op])<1e-9
            rows.append(dict(op=op,shape=c['shape'],attrs=profile()['contracts'][op],role='validation',predicted_us=c['predictions'][op],
                previous_us=c['previous'][op],reference_us=sum(k['reference_us'] for k in ks),correctness=m['correctness'],stages_us=res['stages_us'],
                api='aclnnMhcPre' if op=='mhc_pre' else 'aclnnMhcPre + aclnnMhcSinkhorn + aclnnMhcPost',library='ops-transformer',
                kernel_records=[dict(kernel=k['kernel'],reference_us=k['reference_us'],launch_id=k['helper_info']['launch_id'],block_dim=k['block_dim'],
                    active_aic=k['active_aic'],active_aiv=k['active_aiv'],max_core_mmad=k['max_core_mmad'],max_core_rv_count=k['max_core_rv_count'],
                    critical_pipeline_us=max(k['cores'],key=lambda a:a['end_ticks'])['pipeline_union_us']) for k in ks],kernel='+'.join(k['kernel'] for k in ks),
                active_aic=[k['active_aic'] for k in ks],active_aiv=[k['active_aiv'] for k in ks],source=str(path.relative_to(ROOT)),
                db_sha256=[k['db_sha256'] for k in ks],query_us_max=max(timed),query_us_median=statistics.median(timed)))
    summary={}
    for op in ('mhc_pre','mhc'):
        val=[r for r in rows if r['role']=='validation' and r['op']==op]
        if not val:continue
        assert len(val)==len({tuple(r['shape']) for r in val})
        summary[op]={'n':len(val),'query_us_max':max(r['query_us_max'] for r in val)}
        for key in ('previous_us','predicted_us'):
            diffs=[abs(r[key]-r['reference_us']) for r in val]
            summary[op][key]={'wape':sum(diffs)/sum(r['reference_us'] for r in val),'max_ape':max(x/r['reference_us'] for x,r in zip(diffs,val))}
    diagnosis=[]
    for row in rows:
        if row['role']!='validation' or row['op']!='mhc':continue
        shape=row['shape'];branch,w=partition('pre',shape)
        stages=[]
        for stage,k in zip(('pre','sink','post'),row['kernel_records']):
            stages.append(dict(stage=stage,kernel=k['kernel'],reference_us=k['reference_us'],
                               predicted_us=row['stages_us'][stage],signed_us=row['stages_us'][stage]-k['reference_us']))
        diagnosis.append(dict(shape=shape,pre_branch=branch,estimated_pre_cube_work=w['compute'],
                              max_core_mmad=row['kernel_records'][0]['max_core_mmad'],stages=stages))
    save('stage-checks.json',diagnosis)
    save('summary.json',summary);save('rows.json',rows);save('costs.json',dict(worker_seconds=sum(a['wall_seconds'] for a in attempts),attempts=attempts))
if __name__=='__main__':main()
