"""Admit holdouts only after checking both logical outputs, queue-time closure and helper flush."""
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from collections import Counter,defaultdict
import json,sys,runpy,sqlite3,math,re
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));OUT=ROOT/'results/v0.12/rms'
from operator_model.ca import sha256
from operator_model.rms_v12 import predict,tile_rows
from operator_model.rms_v11 import predict as old
from development.validation.rms12.evidence import measure
from development.validation.scaling_v11 import audit_db
from development.validation.rms12.recover import qualify
inspect=runpy.run_path(str(ROOT/'scripts/wide-v10-trace.py'))['inspect']
metrics=runpy.run_path(str(ROOT/'scripts/wide-v10-report.py'))['stats']
def save(file,x): (OUT/file).write_text(json.dumps(x,indent=2)+'\n')
def load(path):return json.loads(path.read_text())
def per_core_tiles(r):
    t,d=r['shape']; a=r['active_aiv'];R=(t+63)//64
    with sqlite3.connect((ROOT/r['source']).resolve().as_uri()+'?mode=ro',uri=True) as db:
        by=defaultdict(list)
        for core,unit,size in db.execute("""select ExecInstrCoreId,CoreTypeId,json_extract(ExtendParams,'$.processed_bytes') from ExecutedInstructions where ExecInstrName='MOV_SRC_TO_DST_ALIGNv2' and json_extract(ExtendParams,'$.dst_mem')='OUT' order by ExecInstrTickStart"""):
            if size<2*d:by[(core,unit)].append(size//4)
    actual=Counter(tuple(x) for x in by.values()); expected=Counter(tuple(tile_rows(min(R,t-i*R),d)) for i in range(a))
    assert expected==actual,(actual,expected);return [{'tiles':list(x),'cores':n} for x,n in sorted(actual.items())]

def main():
    fr=load(OUT/'freeze.json'); assert all(sha256(ROOT/p)==h for p,h in fr['sha256'].items());ds=load(ROOT/'results/v0.11/rms/rows.json');fixed=median(r['reference_us'] for r in ds)
    for r in ds:
        s=r['shape'];p=predict('rmsnorm',s);sp=predict('rmsnorm',s,{'resident_aiv':r['prior_aiv']});r.update(prediction=p,predicted_us=p['predicted_us'],sequence_prediction=sp,sequence_us=sp['predicted_us'],previous_us=old('rmsnorm',s)['predicted_us'],fixed_us=fixed,role='development')
    vs=[];checks=[];bs=[]; ev=[]; originals=[]; recovery={6,7,8,9}
    for i,k in fr['batches']:
        batch=f'rmsnorm-{i}-{k}';folder=ROOT/'runs/v12rms'/batch
        cli=(folder/'run.log').read_text(); wall=float(re.search(r'Simulation SUCCESS\s+·\s+run time ([\d.]+)s',cli)[1]); src=fr['cases'][i:k]; keep={j for j in range(len(src)) if i+j not in recovery}
        original_rows=qualify(folder,[r['shape'] for r in src],wall,keep)
        cases=[];orig=iter(original_rows);dbs=sorted(folder.rglob('*.db'));helper,=folder.rglob('cannsim_helper.log');body=helper.read_text()
        for j,c in enumerate(src):
            if i+j not in recovery:
                r=next(orig);cases.append({**c,**r,'predicted_us':c['prediction']['predicted_us']});originals.append(dict(id=c['id'],shape=c['shape'],status='qualified',source=r['source']))
            else:
                db=dbs[j];t,d=c['shape'];flush=re.search(fr'start launch: id={j} .*?instruction handler flush: ([^\n]*)',body,re.S)[1]
                with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as con:
                    n,raw=con.execute("select count(*),sum(json_extract(ExtendParams,'$.processed_bytes')) from ExecutedInstructions where ExecInstrName='MOV_SRC_TO_DST_ALIGNv2' and json_extract(ExtendParams,'$.dst_mem')='OUT'").fetchone()
                assert 'status=failed' in flush or raw!=t*(2*d+4)
                originals.append(dict(id=c['id'],shape=c['shape'],status='rejected',source=str(db.relative_to(ROOT)),db_sha256=sha256(db),flush=flush,output_sum_bytes=raw or 0,expected_output_bytes=t*(2*d+4),timing_used=False));cases.append(load(OUT/f'recovered-{i+j}.json')['target'])
        assert len(cases)==k-i
        for r in cases:
            assert sha256(ROOT/r['source'])==r['db_sha256'];r.update(inspect(ROOT/r['source']));check=audit_db('rmsnorm',r['shape'],ROOT/r['source']);checks.append(check);gold_n=[int(re.search(r'count=(\d+)',s)[1]) for s in r['correctness']];t,d=r['shape'];assert gold_n==[t*d,t]
            assert check['passed'] and check['aiv']==r['active_aiv']==r['prediction']['predicted_active_aiv'] and check['per_core_rv_min']>0
            r['verified_core_tiles']=per_core_tiles(r);r['sequence_us']=r['sequence_prediction']['predicted_us']; assert math.isclose(sum(r['phase_us'].values()),r['reference_us'],abs_tol=1e-7) and math.isclose(r['previous_us'],old('rmsnorm',r['shape'])['predicted_us'])
            vs.append(r);ev.append(measure(r))
        bs.append(dict(name=batch,wall_s=wall,db_bytes=sum(p.stat().st_size for p in folder.rglob('*.db')),raw_bytes=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())))
    save('original-attempts.json',originals)
    original_time=max(b['wall_s'] for b in bs); recovery_walls=[]
    for i in sorted(recovery):
        meta=load(OUT/f'recovery-cost-{i}.json');folder=ROOT/'runs/v12rms'/meta['name']; bs.append({**meta,'db_bytes':sum(p.stat().st_size for p in folder.rglob('*.db')),'raw_bytes':sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())});recovery_walls.append(meta['wall_s'])
    assert len({tuple(r['shape']) for r in ds+vs})==len(ds+vs)==53
    for r in ds+vs:
        s=r['shape'];wall=[]
        for _ in range(100):
            start=perf_counter_ns();p=predict('rmsnorm',s);wall.append((perf_counter_ns()-start)/1000)
        assert math.isclose(r['predicted_us'],p['predicted_us']);r.update(query_median_us=median(wall),query_max_us=max(wall))
    summary={}
    for role,v in [('development',ds),('validation',vs)]:
        b=summary[role]={}
        for g in ('all','few_wide','controls','large'):
            sel=[r for r in v if g=='all' or r['group']==g or (g=='large' and r['shape'][0]*r['shape'][1]*2>=2**20)]
            b[g]={f:metrics(sel,f) for f in ('predicted_us','previous_us','fixed_us','sequence_us')}
    save('rows.json',ds+vs);save('holdout-audit.json',checks);save('holdout-evidence.json',ev);save('summary.json',{'rmsnorm':summary});save('costs.json',{'rmsnorm':dict(launches=17,distinct_holdouts=10,original_target_launches=10,recovery_target_launches=4,conditioning_controls=3,rejected_original_measurements=4,prior_traces_reused=43,batches=bs,batch_wall_s=sum(b['wall_s'] for b in bs),two_wave_ca_wall_s=original_time+max(recovery_walls),db_bytes=sum(b['db_bytes'] for b in bs),raw_bytes=sum(b['raw_bytes'] for b in bs),query_median_us=median(r['query_median_us'] for r in vs),query_max_us=max(r['query_max_us'] for r in vs),benchmark='100 direct calls per shape; profile cached, shape results not cached; maximum is empirical, not a hard bound. Original CLI wall times and recovery subprocess wall times include shutdown; 3 concurrent originals followed by 4 concurrent targeted recoveries.')});print(summary['validation'])
if __name__=='__main__':main()
