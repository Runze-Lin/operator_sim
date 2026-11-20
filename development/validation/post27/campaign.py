"""Pre-register changed-entry issued-interval estimates only on untouched processes; no simulator launch."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.post24.freeze import sha
from development.validation.post24.report import stats,fmt
from development.validation.post26.freeze import OUT as V26,OLD as V25
from development.validation.post27.issue import OUT, predict
from development.validation.sku19.campaign import read,write,now
from operator_model import load_chip
REG=OUT/'registration.json'
def register():
    if REG.exists():raise FileExistsError(REG)
    groups=('multi','reserve');dirs=sorted(p.name for p in (ROOT/'runs/post26').iterdir() if p.is_dir())
    assert not any(p.startswith(groups) for p in dirs)
    prices=read(V25/'prices.json');params=read(OUT/'parameters.json');reg=read(V26/'registration.json');rows=[]
    for r in reg['rows']:
        if r['group'] not in groups or r['role']!='entry':continue
        c=load_chip(r['chip']);q=predict(r['shape'],c,r['prior_aiv'],r['previous_shape'],params,prices)
        assert abs(q['base']-r['old_known_us'])<1e-10
        rows.append(dict(**{k:r[k] for k in ('group','i','chip','shape','previous_shape','prior_aiv','active','plan','state')},**q))
    sources=[Path(__file__).resolve(),Path(__file__).with_name('issue.py'),OUT/'parameters.json',OUT/'archive.json',V25/'prices.json',V26/'registration.json',ROOT/'operator_model/post_token_pipeline.py',ROOT/'operator_model/post_resource.py']
    write(REG,dict(utc=now(),raw_prior_groups=dirs,groups=groups,measured=reg['measured'],extrapolated=reg['extrapolated'],rows=rows,
      hashes={str(f.relative_to(ROOT)):sha(f) for f in sources},
      scope='Use only host groups not started on either SKU at freeze. Score changed entry/cold launches, per SKU and paired DT:PR. Repeats retain separate frozen v0.25 deployment-price validation. Current upper bound is five eligible prospective different shapes per SKU, conditional on prior queue running its coverage reserve. No new CA. Unknown state untouched.',
      learning='Previous actual ELF entry category splits the first row-to-row transition; all cohorts share steady read and write issue constraints max(front-end issue price, vector service + retire tail). Curves use observed descriptor issue intervals only. First control interval is measured from last AIV join to payload issue. Parameters never fitted to total latency.'))
def score():
    src=read(REG);assert all(sha(ROOT/f)==h for f,h in src['hashes'].items());idx={(r['group'],r['chip'],r['i']):r for r in src['rows']};a,b=src['measured'];rows=[];miss=[]
    for p in sorted((V26/'receipts').glob('*.json')):
        receipt=read(p)
        if receipt['group'] not in src['groups']:continue
        assert receipt['cost']['started_utc']>src['utc']
        for v in receipt['rows']:
            k=idx.get((v['group'],v['chip'],v['i']))
            if k is None:continue
            assert k['shape']==v['shape']
            if not v['timing_qualified']:miss.append(dict(chip=k['chip'],shape=k['shape'],reason=v.get('reason')));continue
            rows.append(dict(**{q:k[q] for q in ('group','i','chip','shape','previous_shape','us','base','old_pipeline_us','cohort','modified','state') if q in k},
                 ca_us=v['stage']['ca_us'],actual_cores=v['stage']['active_aiv'],kernel=v['stage']['kernel'],trace=v['stage']['trace'],sha256=v['stage']['trace_sha256']))
    summary={s:{k:stats([r for r in rows if r['chip']==s],k) for k in ('us','base','old_pipeline_us')}for s in src['measured']}
    pairs=[]
    for r in rows:
        if r['chip']==a:
            p=next((x for x in rows if x['chip']==b and x['group']==r['group'] and r['i']==x['i']),None)
            if p:pairs.append(dict(shape=r['shape'],ca_us=r['ca_us']/p['ca_us'],**{k:r[k]/p[k] for k in ('us','base','old_pipeline_us')}))
    precise=read(V26/'score.json');repeats={s:precise['summary'][s]['repeat'] for s in src['measured']}
    write(OUT/'score.json',dict(utc=now(),stage='pending' if not rows else 'measured (possibly partial)',measured=rows,rejected=miss,summary=summary,paired_dt_pr=pairs,exact_repeat_policy=precise['support_threshold_met'],exact_repeat_shapes=precise['all_core_repeat_shapes'],repeat_summary=repeats))
    return src,rows,summary,pairs,precise
if __name__=='__main__':
    if 'register' in sys.argv:register()
    score()
