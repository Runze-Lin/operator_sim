"""Compare the installed fused API to every frozen geometry, all historical windows and matched holdouts."""
import argparse
import json
from pathlib import Path
from statistics import median
import sys
import time
from assessment import CHIPS,OUT,ORIG,ROOT,audit,metrics,read,save,sha

def check(installed):
    if not installed:sys.path.insert(0,str(ROOT))
    import operator_model
    from operator_model import estimate,load_chip
    from operator_model.mhc_sink_resource import fused as legacy
    if installed and ROOT in Path(operator_model.__file__).resolve().parents:raise ValueError('uninstalled API')
    record=audit()
    for chip in CHIPS:
        group=record['groups'][chip]['fused_paired_shapes']
        m=group['candidate_us'];old=group['v025_us']
        if group['distinct_shapes']<10 or m['wape_pct']>20 or m['max_pct']>40 or m['wape_pct']>=old['wape_pct']:
            raise ValueError('preregistered criteria not reached on paired geometry: '+chip)
    history=read(ORIG/'replay.json'); reg=read(OUT/'register.json')
    by_shape={tuple(j['shape']):j['predictions'] for j in reg['jobs']}
    times=[];rows=[];max_delta=0.;max_stage=0.
    for shape,predictions in by_shape.items():
        for sku,p in predictions.items():
            chip=load_chip(sku)
            start=time.perf_counter_ns();r=estimate('mhc_pre_sinkhorn',shape,chip=chip);times.append((time.perf_counter_ns()-start)/1e6)
            for _ in range(20):
                start=time.perf_counter_ns();estimate('mhc_pre_sinkhorn',shape,chip=chip);times.append((time.perf_counter_ns()-start)/1e6)
            max_delta=max(max_delta,abs(r.latency_us-p['candidate_us']))
            max_stage=max(max_stage,*(abs(r.projection['stages_us'][key]-v) for key,v in p['stages_us'].items()))
            if set(r.projection['stages_us'])!=set(p['stages_us']) or max_stage>1e-11 or r.projection['target_plan']!=p['target_plan']:
                raise ValueError('frozen stage/tiling mismatch')
            rows.append(dict(shape=shape,chip=sku,status=r.status,us=r.latency_us,previous_us=p['v025_us'],schedule=p['schedule']))
    for r in history['rows']:
        v=estimate('mhc_pre_sinkhorn',r['shape'],chip=r['sku'])
        max_delta=max(max_delta,abs(v.latency_us-r['candidate_us']))
        if abs(r['v025_us']-legacy(r['shape'],load_chip(r['sku']),{})[0])>1e-11:raise ValueError('historical baseline changed')
    if max_delta>1e-11:raise ValueError('freeze delta')
    return dict(checked_version='installed wheel' if installed else 'local source',frozen_file_sha256=sha(OUT/'register.json'),
                activation_sha256=sha(OUT/'reserve-activation.json'),largest_frozen_delta_us=max_delta,max_stage_delta_us=max_stage,
                new_unique_geometries=len(by_shape),price_parity=rows,history=history['groups'],
                independent_by_chip={c:record['groups'][c]['fused'] for c in CHIPS},
                matched_by_chip={c:record['groups'][c]['fused_paired_shapes'] for c in CHIPS},
                paired_dt_over_pr=record['paired']['fused'],remaining_failure=[x for x in record['fused_status'] if x['status']=='failed'],
                public_query_ms=dict(n=len(times),median=median(times),p99=sorted(times)[int(len(times)*.99)],max=max(times)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--installed',action='store_true');a=p.parse_args()
    d=check(a.installed);save(OUT/('fused-public-installed.json' if a.installed else 'fused-public.json'),d)
    print(d['checked_version'],d['largest_frozen_delta_us'],d['matched_by_chip'],d['public_query_ms'])
