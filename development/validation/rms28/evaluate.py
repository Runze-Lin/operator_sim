"""Score pre-CA prices on paired full windows and separate development-stage replay."""
from collections import defaultdict
from pathlib import Path
from statistics import median
import sys
from time import perf_counter_ns
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.rms28.campaign import ABLATIONS, OUT, REG, prior, read, verify, write
from development.validation.rms28 import fit
from development.validation.rms28.schedule import forecast, schedule
from development.validation.rms27.limits import group
from operator_model.rms_dispatch import price_rms
from operator_model.rmsnorm_pipeline import rmsnorm_pipeline
from operator_model import estimate, load_chip

MODELS = ('us','prior27_us','old_us','specific_context_us','known_us',*ABLATIONS)
def score(rows, key):
    if not rows: return None
    ref = [v['ca_us'] for v in rows]
    diff = [abs(v[key]-c) for v,c in zip(rows,ref)]
    return dict(n=len(rows),shapes=len({tuple(v['shape']) for v in rows}),
        wape=100*sum(diff)/sum(ref),max=100*max(d/c for d,c in zip(diff,ref)))

def summary(rows):
    output = {}
    for sku in prior.CHIPS[:2]:
        sub = [r for r in rows if r['chip']==sku]
        groups = {'all':sub}
        for role in ('entry','repeat'):
            group = [r for r in sub if r['role']==role]
            groups[role] = group
            for name,condition in [('sparse',lambda r:r['active_aiv']<=4),('one UB; A>4',lambda r:r['active_aiv']>4 and r['iterations']==1),('multi UB',lambda r:r['iterations']>1)]:
                groups[role+' / '+name] = [r for r in group if condition(r)]
        output[sku] = {g:{k:score(v,k) for k in MODELS} for g,v in groups.items()}
    return output

def replays():
    stages = read(OUT/'stage-samples.json')
    records = read(OUT/'milestones.json')['rows']
    allprices = read(OUT/'prices.json')
    prev = read(prior.PARAMS)
    result = {}
    for batch in ('v0.23','v0.27'):
        split = fit.fit([s for s in stages if s['version']!=batch])
        rows = []
        for r in records:
            if r['version']!=batch: continue
            t = load_chip(r['chip'])
            shp = r['shape']
            q = forecast(shp,t,allprices)
            old,_,a,d = price_rms(shp,t,{'resident_aiv':r['H']})
            ce = schedule(shp,t,r['H'],r['hot'],allprices,d)['us']
            metrics = {k:forecast(shp,t,allprices,k)['us'] for k in ABLATIONS}
            rows.append(dict(shape=shp,chip=r['chip'],prior_aiv=r['H'],active_aiv=a,
                iterations=q['plan']['iterations'],role='repeat' if r['hot'] else 'entry',ca_us=r['ca_us'],
                old_us=q['old_us'],us=q['us'],crossfit_us=forecast(shp,t,split)['us'],
                prior27_us=prior.forecast(shp,t,prev)['us'],specific_context_us=ce,known_us=old,**metrics))
        cross = {sku:{role:score([r for r in rows if r['chip']==sku and r['role']==role],'crossfit_us')
            for role in ('entry','repeat')} for sku in prior.CHIPS[:2]}
        result[batch]=dict(rows=rows,summaries=summary(rows),crossfit=cross,bounds=group(rows,prior.CHIPS[:2]))
    write(OUT/'development-replay.json',result)

def timing(reg):
    err = 0.
    durations = []
    diag = []
    for row in reg['rows']:
        chip = load_chip(row['chip'])
        shape = row['shape']
        kw = dict(chip=chip,resident_aiv=row['prior_aiv'],immediate_repeat=row['role']=='repeat')
        b = rmsnorm_pipeline(shape,**kw)['us']
        a = rmsnorm_pipeline(shape,chip=chip)['us']
        err = max(err,abs(b-row['specific_context_us']),abs(a-row['us']))
        for collector,opts in ((durations,dict(chip=chip)),(diag,dict(**kw,diagnostics=True))):
            for i in range(5):
                start=perf_counter_ns();rmsnorm_pipeline(shape,**opts);collector.append((perf_counter_ns()-start)/1e6)
    def stats(vals):
        return dict(n=len(vals),median_ms=median(vals),p95_ms=sorted(vals)[int(.95*(len(vals)-1))],max_ms=max(vals))
    large=[]
    for model in ('DeepSeek-V4.1-Flash','Kimi-K3','DeepSeek-V4-Pro'):
        cfg_path = ROOT/'configs/models'/(model+'.json')
        cfg = read(cfg_path)
        width = cfg.get('text_config',cfg)['hidden_size']
        for tokens in (8*32768,8*131072,8*262144):
            shape = [tokens,width]
            for sku in prior.CHIPS:
                start=perf_counter_ns(); q=rmsnorm_pipeline(shape,chip=sku)
                wall=(perf_counter_ns()-start)/1e6
                public=estimate('rmsnorm',shape,chip=sku)
                large.append(dict(chip=sku,model=model,config=str(cfg_path.relative_to(ROOT)),shape=shape,
                    public_us=public.latency_us,scenario_center_us=q['us'],iterations=q['plan']['iterations'],
                    elapsed_ms=wall,status='prediction only, if unfused, all tokens submitted together; previous steady-UB model'))
    archive = []
    for r in read(ROOT/'results/v0.20/rms/records.json'):
        t,w=r['shape']
        if t<1024:continue
        sku=r['chip']; ref=r['ca_us']
        # Numerator is useful algorithmic I/O, not HBM controller transactions.
        logical=4*t*w+2*w+4*t
        archive.append(dict(shape=r['shape'],chip=sku,ca_us=ref,
            unknown_us=estimate('rmsnorm',r['shape'],chip=sku).latency_us,
            first_us=estimate('rmsnorm',r['shape'],{'resident_aiv':0},chip=sku).latency_us,
            logical_gbs=logical/(ref*1000),correctness=r['correctness'],source=r['trace']))
    write(OUT/'query-cost.json',dict(frozen_rows=len(reg['rows']),max_repricing_delta_us=err,
        unknown=stats(durations),context_with_diagnostics=stats(diag),large=large,historical_large_archive=archive))

def evaluate():
    reg = read(REG)
    pending=set((g,c) for g in reg['groups'] for c in reg['measured'])
    observations=[]
    costs=[]
    failed=[]
    allrows={(r['chip'],r['group'],r['i']):r for r in reg['rows']}
    for path in (OUT/'receipts').glob('*.json'):
        rec=read(path)
        assert rec['freeze_sha256']==verify.digest(REG)
        name=rec['chip'],rec['group']
        pending.discard((name[1],name[0]))
        costs.append(dict(chip=name[0],group=name[1],**rec['cost']))
        if rec.get('reason'):failed.append(dict(chip=name[0],group=name[1],reason=rec['reason']))
        for r in rec['rows']:
            if r['role']=='prefix':continue
            if not r['timing_qualified']:
                failed.append(r);continue
            old=allrows[r['chip'],r['group'],r['i']]
            projection=allrows[prior.CHIPS[2],r['group'],r['i']]
            assert old['shape']==r['shape']==projection['shape'] and old['role']==r['role']
            observations.append(dict(**{k:old[k] for k in ('us','old_us','prior27_us','known_us','specific_context_us')},
                **old['abl'],iterations=old['plan']['iterations'],extrapolated_970_us=projection['us'],**r))
    ids={(r['chip'],tuple(r['shape']),r['role']) for r in observations}
    matched=[r for r in observations if (r['chip'],tuple(r['shape']),'entry' if r['role']=='repeat' else 'repeat') in ids]
    summaries=summary(matched)
    complete=not pending
    passes=all(s and s['shapes']==10 and s['wape']<=25 and s['max']<=50
        for sku,cohorts in summaries.items() for s in (cohorts['entry']['us'],cohorts['repeat']['us']))
    ratios=[]
    by={(r['chip'],tuple(r['shape']),r['role']):r for r in matched}
    for key,dt in by.items():
        if key[0]!=prior.CHIPS[0]:continue
        pr=by.get((prior.CHIPS[1],key[1],key[2]))
        if pr:
            ratios.append(dict(shape=key[1],role=key[2],**{k:dt[k]/pr[k] for k in ('us','prior27_us','old_us','known_us','specific_context_us','ca_us')}))
    write(OUT/'evaluation.json',dict(measured=observations,paired=len(matched),pending=sorted(pending),costs=costs,
        failed=failed,terminal=complete,passes=bool(passes) and complete,summary=summaries,bounds=group(matched,prior.CHIPS[:2]),dt_over_pr=ratios))
    return complete

if __name__=='__main__':
    if 'replay' in sys.argv: replays()
    if 'cost' in sys.argv: timing(read(REG))
    evaluate()
