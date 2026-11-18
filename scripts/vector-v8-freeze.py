"""Freeze trace phase models and independent 950DT cases before simulation."""
from pathlib import Path
import json,sqlite3,sys,hashlib
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.vector_v8 import predict,features,phase_work
from operator_model.model_v7 import fit_nonnegative
from operator_model.ca import sha256
from operator_model.npusim_v7 import extract_db
OUT=ROOT/'results/v0.8/vector'
ops=['rmsnorm','mhc_post','mhc_sinkhorn','kda_recurrent']
shapes={
 'rmsnorm':[(1,256),(4,512),(16,1024),(9,7168),(36,5120),(32,4096),(64,2048),(65,128),(128,1024),(1,8192)],
 'mhc_post':[(1,4,64),(2,4,256),(4,4,512),(9,4,1024),(16,4,256),(32,4,512),(36,4,128),(64,4,256),(65,4,128),(1,4,7168)],
 'mhc_sinkhorn':[(2,4),(4,4),(8,4),(9,4),(16,4),(31,4),(32,4),(63,4),(65,4),(128,4)],
 'kda_recurrent':[(1,s,1,1,128,v) for v,steps in [(128,(3,5,6,7,8)),(256,(1,2,3,4,6))] for s in steps],
}
kind={'rmsnorm':'NORM','mhc_post':'MHC','mhc_sinkhorn':'MHC','kda_recurrent':'KDA'}

def phases(k):
    c=max(k['cores'],key=lambda c:c['end_ticks'])
    db=sqlite3.connect(Path(k['db_path']).as_uri()+'?mode=ro',uri=True)
    events=defaultdict(Counter)
    sql="""select json_extract(ExtendParams,'$.queue_type'),ExecInstrTickStart,ExecInstrTickEnd from ExecutedInstructions where ExecInstrCoreId=? and CoreTypeId=? and ExecInstrIsDurationExcluded=0"""
    for queue,a,z in db.execute(sql,(c['core_id'],c['core_type_id'])):
        events[a][queue]+=1;events[z][queue]-=1
    start,end=db.execute('select min(ExecInstrTickStart),max(ExecInstrTickEnd) from ExecutedInstructions where ExecInstrCoreId=? and CoreTypeId=?',(c['core_id'],c['core_type_id'])).fetchone()
    db.close();events[end];events[start];last=start;busy=Counter();out=Counter()
    for tick,event in sorted(events.items()):
        name=next((q for q in ('SIMD','MTE3','MTE2','SCALAR') if busy[q]>0),'other')
        out[name]+=(tick-last)/1650
        busy.update(event);last=tick
    out['arrival']=c['start_ticks']/1650
    assert abs(sum(out.values())-k['reference_us'])<1e-5
    return dict(out)

def main():
    if (OUT/'freeze.json').exists():raise RuntimeError('already frozen')
    profile={};cases=[];dev=[]
    for op in ops:
        rows=[r for r in json.load(open(ROOT/'results/v0.7/evaluation.json'))['rows'] if r['op']==op]
        contract=dict(rows[0]['attributes']);contract.pop('num_iters',None)
        p=profile[op]={'contract':contract,'phases':{},'source':[]}
        for r in rows:
            k=extract_db(r['kernels'][0]['db_path'])
            p['source'].append({'id':r['id'],'db_sha256':k['db_sha256'],'phase_us':phases(k)})
        for name in ('arrival','SIMD','MTE3','MTE2','SCALAR','other'):
            xx=[phase_work(name,*features(op,r['shape'],r['attributes']),op) for r in rows]
            yy=[row['phase_us'].get(name,0) for row in p['source']]
            fit=fit_nonnegative(xx,yy)
            p['phases'][name]={'startup_us':fit['startup_us'],'service_us':fit['service_us_per_work']}
        for row in rows:
            dev.append({**{k:row[k] for k in ('id','op','shape','reference_us','correctness','kernel','source')},'attrs':row['attributes'],'role':'development','predicted_us':predict(op,row['shape'],row['attributes'],profile)['predicted_us'],'active_aiv':row['kernels'][0]['active_aiv'],'active_aic':0})
        for i,s in enumerate(shapes[op]):
            a=dict(contract)
            if op=='mhc_sinkhorn':a['num_iters']=[2,5,10,20,32,20,20,5,10,20][i]
            args=list(s) if op=='mhc_post' else ([s[0],s[1],a['num_iters']] if op=='mhc_sinkhorn' else ([s[1],s[-1],0] if op=='kda_recurrent' else [*s,0]))
            cases.append({'id':f'{op}_{i}','op':op,'shape':s,'attrs':a,'args':args,'kind':kind[op],'groups':3 if op=='kda_recurrent' else (2 if op=='rmsnorm' else 1),'prediction':predict(op,s,a,profile)})
    def write(name,x):(OUT/name).write_text(json.dumps(x,indent=2)+'\n')
    write('profile.json',profile);write('development.json',dev)
    paths=['smoke/vector_v8.cpp','smoke/extension_kda.cpp','smoke/extension_norm_addmm.cpp','smoke/extension_indexer_mhc.cpp','operator_model/vector_v8.py','operator_model/npusim_v7.py','scripts/vector-v8-freeze.py','scripts/vector-v8-run.py','results/v0.8/vector/profile.json','build/capture/libsim_soc_adapter.so']+[f'build/vector-v8/{k}' for k in set(kind.values())]
    write('freeze.json',{'cases':cases,'sha256':{path:sha256(ROOT/path) for path in paths},'per_batch_seconds':1200,'disk_budget_gib':8})

if __name__=='__main__':main()
