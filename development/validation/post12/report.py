"""Evaluate immutable v0.12 and v0.11 predictions only against attributable full CA captures."""
import json, math, runpy, statistics, time
from freeze import ROOT, OUT, OLD, ATTRS, save
from operator_model.ca import sha256
from operator_model.post_v12 import predict
from operator_model.post_v11 import predict as previous
from operator_model.wide_v10 import occupancy

inspect=runpy.run_path(str(ROOT/'scripts/wide-v10-trace.py'))['inspect']
stats=runpy.run_path(str(ROOT/'scripts/wide-v10-report.py'))['stats']
RAW=ROOT/'development/validation/post12/raw'

def summarise(rows):
    groups={'all':rows,'few_wide':[r for r in rows if r['group']=='few_wide'], 'controls':[r for r in rows if r['group']=='controls'],
            'fixed_width_ladder':[r for r in rows if r['shape'][-1]==1536 and r['shape'][0]>=64]}
    groups.update({s:[r for r in groups['few_wide'] if r['sequence_prediction']['context']==s] for s in ('first','expand','reuse')})
    return {g:{k:stats(rr,k) for k in ('predicted_us','previous_us','sequence_us','fixed_us')} for g,rr in groups.items() if rr}

def main():
    manifest=json.loads((OUT/'freeze.json').read_text()); dev=json.loads(OLD.read_text()); benchmark=[]
    for path,digest in manifest['sha256'].items(): assert sha256(ROOT/path)==digest
    for r in dev:
        p=predict('mhc_post',r['shape'],r['attrs']); ctx=predict('mhc_post',r['shape'],dict(r['attrs'],resident_aiv=r['prior_aiv']))
        r.update(role='development',previous_us=previous('mhc_post',r['shape'],r['attrs'])['predicted_us'],
                 predicted_us=p['predicted_us'],prediction=p,sequence_us=ctx['predicted_us'],sequence_prediction=ctx,fixed_us=manifest['cases'][0]['fixed_us'])
        for k in ('query_median_us','query_max_us'): r.pop(k,None) # prior-version measurements
    rows=[]; batches=[]; fails=[]; warm=0
    for batch in manifest['batches']:
        name='batch-'+batch[-1]['id']; rec=json.loads((OUT/(name+'.json')).read_text()); folder=RAW/name
        batches.append(dict(name=name,wall_s=rec['wall_s'],db_bytes=sum(p.stat().st_size for p in folder.rglob('*.db')),
                            raw_bytes=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()),launches=len(batch)))
        fails += [r['id'] for r in rec['timing_failed']]
        for r in rec['completed']:
            c=next(c for c in batch if c['id']==r['id']); assert all(r[k]==v for k,v in c.items()); assert sha256(ROOT/r['source'])==r['db_sha256']
            if not r['output_audit']['qualified']:
                fails.append(r['id']); continue
            if r['role']!='validation': warm+=1; continue
            r.update(inspect(ROOT/r['source'])); assert math.isclose(sum(r['phase_us'].values()),r['reference_us'],abs_tol=1e-7)
            expected='entry_kernel_6' if r['prediction']['route']=='post_token' else 'entry_kernel_5'
            r.update(sequence_us=r['sequence_prediction']['predicted_us'],expected_kernel=expected,kernel_match=expected==r['kernel'],active_match=r['prediction']['predicted_active_aiv']==r['active_aiv'])
            samples=[]
            for _ in range(100):
                t=time.perf_counter_ns(); p=predict(r['op'],r['shape'],r['attrs']); samples.append((time.perf_counter_ns()-t)/1000)
            assert p==r['prediction']; r.update(query_median_us=statistics.median(samples),query_max_us=max(samples)); rows.append(r); benchmark.extend(samples)
    assert len(set(tuple(r['shape']) for r in rows+dev))==len(rows)+len(dev)
    save('rows.json',dev+rows); save('summary.json',{'mhc_post':dict(validation=summarise(rows),development=summarise(dev))})
    save('costs.json',{'mhc_post':dict(batch_wall_s=sum(b['wall_s'] for b in batches),db_bytes=sum(b['db_bytes'] for b in batches),raw_bytes=sum(b['raw_bytes'] for b in batches),
            launches=sum(b['launches'] for b in batches),distinct_holdouts=len(rows),prior_traces_reused=len(dev),batches=batches,
            failed_timing_attempts=len(fails),exclusions=fails,context_warmups=warm,benchmark='100 uncached calls per retained shape, profile loaded; CA sum is process-s, including warmups',
            query_median_us=statistics.median(benchmark),query_max_us=max(benchmark))})
    print('points',len(rows),'failed',fails,'results',summarise(rows)['all'])

if __name__=='__main__':main()
