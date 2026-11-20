"""Audit installed/public equivalence against the locked per-stage predictions and qualified windows."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.27/fusion'

def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def scores(rows,key):
    return dict(n=len(rows),wape_pct=100*sum(abs(r[key]-r['ca_us']) for r in rows)/sum(r['ca_us'] for r in rows),
                max_pct=100*max(abs(r[key]/r['ca_us']-1) for r in rows))

def audit(installed=False):
    if not installed:sys.path.insert(0,str(ROOT))
    import operator_model
    from operator_model import estimate,load_chip
    base=ROOT/'results/v0.26/sink';reg=read(base/'register.json');locks=[]
    imported=Path(operator_model.__file__).resolve()
    if installed and ROOT in imported.parents:raise ValueError('workspace source imported in installation test')
    epochs_error=0
    rows=read(base/'independent.json')['rows']
    lookup={(r['chip'],r['group'],r['index']):r for r in rows}
    wall=[];max_us=0; unknown=0
    for j in reg['jobs']:
        chip=load_chip(j['chip']);kwargs=dict(op='mhc_sinkhorn',shape=j['shape'],chip=chip,attrs={'num_iters':j['iters'],'resident_aiv':j['prior_h']})
        start=time.perf_counter_ns();p=estimate(**kwargs);wall.append((time.perf_counter_ns()-start)/1e6)
        for _ in range(20):
            start=time.perf_counter_ns();estimate(**kwargs);wall.append((time.perf_counter_ns()-start)/1e6)
        if p.latency_us is None:raise ValueError(p.reason)
        diag=p.projection['timing_diagnostics']
        err=abs(j['known']['latency_us']-p.latency_us)
        max_us=max(max_us,err)
        epochs_error=max(epochs_error,*[abs(diag['issue_epochs_us'][k]-v) for k,v in j['known']['vf_setup_epochs'].items()])
        old=estimate('mhc_sinkhorn',j['shape'],{'num_iters':j['iters']},chip=chip)
        if j['v025_us'] is not None:
            unknown=max(unknown,abs(old.latency_us-j['v025_us']))
        if j['v025_us'] is None and old.latency_us is not None:raise ValueError('shape-only domain changed')
        obs=lookup.get((j['chip'],j['group'],j['index']))
        locks.append(dict(shape=j['shape'],chip=j['chip'],group=j['group'],index=j['index'],iters=j['iters'],resident_aiv=j['prior_h'],
                          status=p.status,public_us=p.latency_us,frozen_us=j['known']['latency_us'],ca_us=obs and obs['ca_us'],
                          raw_sha256=obs and obs['raw_sha256'],shape_only_us=old.latency_us))
    if max(max_us,epochs_error,unknown)>1e-11:raise ValueError('frozen parity mismatch')
    groups={}
    for chip in sorted({r['chip'] for r in rows}):
        groups[chip]={}
        for label,cond in [('supported',lambda t:t<=256),('wide_one_ub',lambda t:t>256)]:
            a=[r for r in locks if r['chip']==chip and cond(r['shape'][0])]
            groups[chip][label]=dict(distinct_shapes=len({tuple(r['shape']) for r in a}),public=scores(a,'public_us'),
                v025=scores(a,'shape_only_us') if label=='supported' else None)
    pairs={}
    for label,cond in [('supported',lambda t:t<=256),('wide_one_ub',lambda t:t>256)]:
        pr={(r['group'],r['index']):r for r in locks if '950PR' in r['chip'] and cond(r['shape'][0])}
        pair=[]
        for d in locks:
            if '950DT' not in d['chip'] or not cond(d['shape'][0]):continue
            p=pr[d['group'],d['index']];pair.append({k:d[k]/p[k] for k in ('ca_us','public_us')})
        pairs[label]=dict(**scores(pair,'public_us'),correct_direction=sum((r['ca_us']>1)==(r['public_us']>1) and (r['ca_us']<1)==(r['public_us']<1) for r in pair))
    return dict(instrument='Python perf_counter_ns; includes first access and 20 repeated public queries per registered row; offline CA excluded',
                imported_from='installed wheel outside source checkout' if installed else 'workspace source',
                registration_sha256=sha(base/'register.json'),phase_file_sha256=sha(base/'prices.json'),
                raw_window_rows=len(rows),predictions=len(locks),max_time_delta_us=max_us,max_epoch_delta_us=epochs_error,
                max_old_shape_only_delta_us=unknown,query_wall_ms=dict(n=len(wall),median=median(wall),p99=sorted(wall)[int(len(wall)*.99)],max=max(wall)),
                cohorts=groups,paired_dt_over_pr=pairs,rows=locks)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--installed',action='store_true');args=p.parse_args()
    d=audit(args.installed)
    (OUT/('public-sink-installed.json' if args.installed else 'public-sink.json')).write_text(json.dumps(d,indent=2)+'\n')
    print(d['imported_from'],d['predictions'],d['max_time_delta_us'],d['max_epoch_delta_us'],d['query_wall_ms'])
