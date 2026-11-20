"""Validate unchanged pre-registered prices; label historical versus new roles per route."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate
from operator_model.post_history import estimate_post_history
from development.validation.post28.parameters import bundle

OUT=ROOT/'results/v0.28/post'
def read(name):return json.loads((ROOT/name).read_text())
def signature(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def stats(rs):
    if not rs:return None
    return dict(windows=len(rs),shapes=len({tuple(r['shape']) for r in rs}),
                **{key:dict(wape=100*sum(abs(r[key]-r['ca_us']) for r in rs)/sum(r['ca_us'] for r in rs),
                    maximum=max(100*abs(r[key]/r['ca_us']-1) for r in rs)) for key in ('selected_us','old_us','stateless_us')},
                measured_inside_scenario_range=sum(r['range_us'][0]<=r['ca_us']<=r['range_us'][1] for r in rs))


def replay():
    v24='results/v0.24/post/evaluation.json';v24reg='results/v0.24/post/freeze.json'
    v27='results/v0.27/post/score.json';v27reg='results/v0.27/post/registration.json'
    v26='results/v0.26/post/score.json';q='results/v0.27/history-bounds.json'
    sources=[v24,v24reg,v26,v27,v27reg,q]
    indexed=lambda name:{(r['chip'],r['group'],r['i']):r for r in read(name)['rows']}
    oldreg,index=indexed(v24reg),indexed(v27reg)
    rows=[]
    def append(source,sku,shape,resident,prev,ca_us,baseline,scenario,trace,sha,kernel,frozen_us=None):
        h=estimate_post_history(shape,chip=sku);f=h.for_predecessor(resident,prev)
        if frozen_us is not None and not math.isclose(f.latency_us,frozen_us,abs_tol=1e-8):
            raise ValueError(('prior frozen prediction changed',source,sku,shape,f.latency_us,frozen_us))
        assert (kernel.endswith('_5') and h.route=='post_channel') or (kernel.endswith('_6') and h.route=='post_token')
        default=estimate('mhc_post',shape,chip=sku)
        cond=next(k for k,v in h.scenarios.items() if v is f)
        rows.append(dict(cohort=source,chip=sku,shape=shape,original_window=trace,sha256=sha,kernel=kernel,scenario=cond,
                         previous_shape=prev,known_resident=resident,ca_us=ca_us,old_us=baseline,stateless_us=default.latency_us,
                         selected_us=f.latency_us,model=f.model,original_role=scenario,range_us=h.range_us))
    for r in read(v24)['rows']:
        s=oldreg[r['chip'],r['group'],r['i']]
        channel_repeat=r['role']=='repeat' and r['route']=='post_channel'
        token_repeat=r['role']=='repeat' and r['route']=='post_token'
        baseline=r['baseline_us'] if not token_repeat else estimate_post_history(r['shape'],chip=r['chip']).scenarios['repeat'].latency_us
        append('v0.24 mixed history replay',r['chip'],r['shape'],s['prior_aiv'],s['previous_shape'],
               r['ca_us'],baseline,r['protocol'],r['trace'],r['trace_sha'],r['kernel'],s['us'] if channel_repeat else None)
    for r in read(v27)['measured']:
        s=index[r['chip'],r['group'],r['i']]
        append('v0.27 original forward ordinary token',r['chip'],r['shape'],s['prior_aiv'],s['previous_shape'],
               r['ca_us'],r['base'],r['state'],r['trace'],r['sha256'],r['kernel'],s['us'])
    for r in read(v26)['independent']:
        if r['role']=='repeat' and r['eligible_repeat']:
            append('published forward exact token repeat',r['chip'],r['shape'],r['resident'],r['previous_shape'],
                   r['ca_us'],r['policy_us'],r['state'],r['trace'],r['trace_sha256'],r['kernel'],r['policy_us'])
    metrics={}
    for s in ('Ascend950DT_9582','Ascend950PR_957b'):
        values=defaultdict(list)
        for r in rows:
            if r['chip']!=s: continue
            values[r['cohort']].append(r);values[r['cohort']+' / '+r['scenario']].append(r)
        metrics[s]={n:stats(rs) for n,rs in values.items()}
    ti=[]
    for s in ('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'):
        for shape in ([1,4,192],[3,4,7168],[157,4,1088],[262144,4,7168]):
            batch=[]
            for _ in range(30):
                t=time.perf_counter_ns();h=estimate_post_history(shape,chip=s);batch.append((time.perf_counter_ns()-t)/1e6)
            ti.append(dict(sku=s,shape=shape,median_ms=statistics.median(batch),max_ms=max(batch),
                           predicted_range_us=h.range_us,route=h.route,target_extrapolated=h.target_extrapolated))
    fresh=[]
    for sku,shape in [('Ascend950DT_9582',[3,4,7168]),('Ascend950PR_957b',[157,4,1088]),('Ascend970DT',[262144,4,7168])]:
        code=('import sys,time,json;from operator_model.post_history import estimate_post_history;'
              't=time.perf_counter_ns(); h=estimate_post_history(json.loads(sys.argv[2]),chip=sys.argv[1]);'
              'print((time.perf_counter_ns()-t)/1e6)')
        us=float(subprocess.check_output([sys.executable,'-c',code,sku,json.dumps(shape)],cwd=ROOT,text=True))
        fresh.append(dict(sku=sku,shape=shape,first_query_after_module_import_ms=us))
    actual={'fresh_python_queries':fresh,'immutable_profile_parity':bundle()==read('results/v0.28/post/parameters.json'),'scope':'Already frozen historical component prices; no new per-shape latency fit. v0.24 counted as replay for combined estimator.',
            'by_sku':metrics,'records':rows,'warm_python_30_queries':ti,'evidence_sha256':{p:signature(p) for p in sources},
            'stateless_floor':{k:v for k,v in read(q)['cohorts'].items() if k.startswith('mHC Post/')}}
    assert actual['immutable_profile_parity']
    (OUT/'history-replay.json').write_text(json.dumps(actual,ensure_ascii=False,indent=2)+'\n')
    return actual


if __name__=='__main__':replay()
