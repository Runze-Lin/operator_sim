"""Join the sealed predictions to verified CA measurements, without any refitting."""
from collections import defaultdict,Counter
import json,math,runpy,sys,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.wide_v10 import predict, parameters
from operator_model.vector_v9 import predict as old
from operator_model.ca import sha256
OUT=ROOT/'results/v0.10/wide'
inspect=runpy.run_path(str(ROOT/'scripts/wide-v10-trace.py'))['inspect']

def stats(rows,k):
    a=[r[k]/r['reference_us']-1 for r in rows];err=[r[k]-r['reference_us'] for r in rows];ref=sum(r['reference_us'] for r in rows)
    return dict(n=len(a),wape=sum(map(abs,err))/ref,max_ape=max(map(abs,a)),signed_bias=sum(err)/ref,mae_us=sum(map(abs,err))/len(a))

def main():
    rows=[]; dev=json.loads((OUT/'development-trace.json').read_text());cost={};pairs=[];summ={}
    for op in ('rmsnorm','mhc_post'):
        previous=[r for r in dev if r['op']==op];seq={};ref=statistics.median(r['reference_us'] for r in previous)
        for r in previous:
            r['role']='development';r['previous_us']=old(op,r['shape'],r['attrs'])['predicted_us'];p=predict(op,r['shape'],r['attrs']);r.update(prediction=p,predicted_us=p['predicted_us'],group='few_wide' if r['shape'][0]<=4 and r['shape'][-1]>=2048 else 'controls',fixed_us=ref)
            stream=str(Path(r['db_source']).parent);h=seq.get(stream,0);r['prior_aiv']=h;r['sequence_us']=predict(op,r['shape'],{**r['attrs'],'resident_aiv':h})['predicted_us'];seq[stream]=max(h,p['predicted_active_aiv'])
        result=json.loads((OUT/(op+'-0-11.json')).read_text());vals=[];seen=set()
        for r in result:
            r.update(inspect(ROOT/r['source']));r['sequence_us']=r['sequence_prediction']['predicted_us']; assert math.isclose(sum(r['phase_us'].values()),r['reference_us'],abs_tol=1e-7)
            assert sha256(ROOT/r['source'])==r['db_sha256']
            if r['role']=='paired_control':pairs.append(r);continue
            assert tuple(r['shape']) not in seen and all(r['shape']!=d['shape'] for d in previous);seen.add(tuple(r['shape']));vals.append(r)
        assert len(seen)==10
        for r in previous+vals:
            us=[]
            for _ in range(100):
                t=time.perf_counter_ns();p=predict(op,r['shape'],r['attrs']);us.append((time.perf_counter_ns()-t)/1000.)
            assert math.isclose(p['predicted_us'],r['predicted_us'],abs_tol=1e-9)
            r['query_max_us']=max(us);r['query_median_us']=statistics.median(us)
        rows.extend(previous+vals);summ[op]={}
        for role,rr in [('development',previous),('validation',vals)]:
            g=summ[op][role]={}
            for grp in ('all','few_wide','controls'):
                sub=[r for r in rr if grp=='all' or grp==r['group']];g[grp]={k:stats(sub,k) for k in ('previous_us','predicted_us','sequence_us','fixed_us')}
        folder=ROOT/'runs/v10wide'/f'{op}-0-11';cost[op]=dict(batch_wall_s=vals[0]['batch_wall_s'],db_bytes=sum(p.stat().st_size for p in folder.rglob('*.db')),raw_bytes=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()),launches=11,distinct_holdouts=10,benchmark='100 uncached shape computations per row with profile loaded; monotonic wall',query_median_us=statistics.median(r['query_median_us'] for r in vals),query_max_us=max(r['query_max_us'] for r in vals))
    def write(n,v):(OUT/n).write_text(json.dumps(v,indent=2)+'\n')
    write('rows.json',rows);write('paired-controls.json',pairs);write('summary.json',summ);write('costs.json',cost)
    for op in summ:
        for grp,k in summ[op]['validation'].items():print(op,grp,'  '.join(f'{n}:{v["wape"]:.1%}/{v["max_ape"]:.1%}' for n,v in k.items()))
    print(cost)
if __name__=='__main__':main()
