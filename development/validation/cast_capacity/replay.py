"""Offline Cast alternatives. CA inputs are restricted to v0.19 and v0.20."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate, load_chip
OUT=ROOT/'results/cast-capacity'
V20=ROOT/'results/v0.20/large'
DT,PR='Ascend950DT_9582','Ascend950PR_957b'
MIB=2**20
OP='cast_bf16_fp32'


def read(p):
    return json.loads(p.read_text())


def calibration():
    small,big=[262144,32],[2097152,32]
    trace=next(r for r in read(V20/'tile-intervals.json') if r['shape']==small and r['sku']==DT)
    diag=estimate(OP,small,chip=DT).projection['timing_diagnostics']
    ctrl=next(r for r in read(OUT/'controller-periods.json') if r['shape']==big)
    # Take whole 1-us intervals at the center of the bytes actually written.
    # This anchors the mixed-traffic service in counters rather than fitting
    # elapsed kernel time. 10-90% suppresses startup/drain, not shape selection.
    cum=0;rates=[];frequency=load_chip(DT).frequency_ghz
    for point in ctrl['intervals']:
        cum+=point['write_bytes']
        if point['end_ns']%1000==0 and .1 <= cum/ctrl['write_bytes'] <= .9:
            rates.append((point['read_bytes']+point['write_bytes'])/(1000*64*frequency))
    stages=diag['stages_us']
    return dict(mixed_bytes_per_aiv_cycle=median(rates),mixed_samples=len(rates),
      read_latency_factor=trace['MTE2']['duration_median_us']/stages['read'],
      write_latency_factor=trace['MTE3']['duration_median_us']/stages['write'],
      latency_shape=small,controller_shape=big,dt_anchors=2)


def load_rows():
    result=[]
    for p in [ROOT/'results/v0.19/workloads/cast-calibration.json', ROOT/'results/v0.19/workloads/cast-qualified.json']:
        for r in read(p):
            if r['qualified']:
                obs=r['kernel']
                result.append((r['shape'],r['sku'],obs['ca_us'],p,obs['trace_sha256'],'small'))
    for row in read(V20/'tile-intervals.json'):
        s,c=row['shape'],row['sku'];p=V20/f'cast_{s[0]}_{s[1]}_{c}.json';d=read(p)
        assert d['qualified'] and d['stage']['ca_us']==d['ca_us']
        result.append((s,c,d['ca_us'],p,d['stage']['trace_sha256'],'narrow' if s[1]<=64 else 'wide'))
    assert len({r[4] for r in result})==len(result)
    return result


def trace_queue_checks(s,c):
    p=V20/f'cast_{s[0]}_{s[1]}_{c}-dma.json'
    if not p.is_file():return None
    d=read(p)
    reads=sorted((a for a in d['terminal_dma'] if a['queue']=='MTE2'),key=lambda a:a['ts_us'])
    writes=sorted((a for a in d['terminal_dma'] if a['queue']=='MTE3'),key=lambda a:a['ts_us'])
    # One data DMA per tile; omit partial/missing terminal payloads.
    cap=21120
    pairs=[(reads[i+2],writes[i]) for i in range(min(len(reads)-2,len(writes))) if reads[i]['bytes']==2*cap and writes[i]['bytes']==4*cap]
    return dict(terminal=d['terminal'],pairs=len(pairs),
       prefetch_before_write_retirement=sum(a['ts_us']<b['ts_us']+b['duration_us'] for a,b in pairs),
       median_next2_read_minus_current_write_issue_us=median(a['ts_us']-b['ts_us'] for a,b in pairs) if pairs else None)


def evaluate_row(s,c,ca,p,digest,group,params):
    chip=load_chip(c);r=estimate(OP,s,chip=c);diag=r.projection['timing_diagnostics']
    plan,stage=diag['target_plan'],diag['stages_us'];L=chip.details['l2_bytes']
    n=s[0]*s[1];X,Y=2*n,4*n
    cap=plan['ub_tile_elements']
    active=plan['active_aiv'];frequency=chip.frequency_ghz
    local=sum(v for k,v in stage.items() if k!='shared_io')
    newlocal=local
    if plan['busy_core_waves']>1:
        srv=max((stage['read']*params['read_latency_factor']+stage['vector'])/2,
                (stage['write']*params['write_latency_factor']+stage['vector'])/2)
        extra=(plan['busy_core_elements']-cap)/cap
        newlocal+=max(0,srv*extra-stage['steady_tiles'])
    issue=active*frequency*params['mixed_bytes_per_aiv_cycle']
    bound=min(issue,chip.hbm_bandwidth_gbs)
    retention_fraction=Y/(X+Y)
    base_retention=diag['assumed_retained_write_bytes']
    # Candidate hypothesis only: clean lines are displaced first when the
    # external channel falls behind requests from the active cores.
    clean_reserve=L*(1-retention_fraction)*min(1,chip.hbm_bandwidth_gbs/issue)
    pressure_retention=min(Y,L-clean_reserve) if X+Y>L else Y
    launch=1753/(1000*frequency)
    l2=(diag['l2_floor_cycles'] or 0)/(1000*frequency)
    def phase(retain):return launch+max((X+Y-retain)/(1000*bound),l2)
    shared0=launch+max(diag['external_floor_cycles'],diag['l2_floor_cycles'] or 0)/(1000*frequency)
    assert abs(max(local,shared0)-r.latency_us)<1e-8
    spill_lower=max(0,Y-L)
    out=dict(shape=s,chip=c,ca_us=ca,group=group,kernel_sha256_source=str(p.relative_to(ROOT)),
       trace_sha256=digest,active_aiv=active,read_bytes=X,output_bytes=Y,capacity=L,
       uniform_retention_bytes=base_retention,pressure_retention_bytes=pressure_retention,
       min_retention_if_configured_bw_and_cold_read=max(0,X+Y-ca*1000*chip.hbm_bandwidth_gbs),
       candidate_model_us={'baseline':r.latency_us,'two_slot':max(newlocal,shared0),
        'two_slot_mixed_rate':max(newlocal,phase(base_retention)),'pressure':max(newlocal,phase(pressure_retention))},
       unconstrained_retention_range_us=[max(newlocal,phase(min(Y,L))),max(newlocal,phase(base_retention))],
       local_stage_us=local,two_slot_local_us=newlocal,current_shared_us=shared0,
       transfer_pressure_ratio=min(1,chip.hbm_bandwidth_gbs/issue),
       queue=trace_queue_checks(s,c),anchor=c==DT and s in (params['latency_shape'],params['controller_shape']))
    if group!='small':
        hist=next((q for q in read(OUT/'controller-periods.json') if q['shape']==s),None) if c==DT else None
        if hist:
            rd=wr=0;first=None
            for b in hist['intervals']:
                rd+=b['read_bytes'];wr+=b['write_bytes']
                if not first and wr>=MIB/4:first=dict(time_ns=b['end_ns'],read_bytes=rd,write_bytes=wr)
            out['controller']=dict(read_bytes=rd,write_bytes=wr,
              total_dirty_retained_bytes=Y-wr,first_256kib_write=first,
              cold_input_excess_bytes=rd-X)
    return out


def scores(rows):
    if not rows:return None
    dic={}
    for method in rows[0]['candidate_model_us']:
        errors=[(abs(r['candidate_model_us'][method]-r['ca_us']),r['ca_us']) for r in rows]
        dic[method]=dict(wape=sum(e for e,_ in errors)/sum(ref for _,ref in errors),
            max_ape=max(e/ref for e,ref in errors))
    return dict(n=len(rows),scores=dic)


if __name__=='__main__':
    pars=calibration()
    rows=[evaluate_row(*r,pars) for r in load_rows()]
    groups={}
    for c in (DT,PR):
        same=[r for r in rows if r['chip']==c]
        groups[c]={group:scores([r for r in same if pred(r)]) for group,pred in
            [('small',lambda r:r['group']=='small'),('narrow',lambda r:r['group']=='narrow'),
             ('wide',lambda r:r['group']=='wide'),('all_large',lambda r:r['group']!='small'),
             ('large_nonanchor',lambda r:r['group']!='small' and not r['anchor'])]}
    (OUT/'replay.json').write_text(json.dumps(dict(parameters=pars,groups=groups,rows=rows,
        status='Experimental historical replay. No v0.21 measurements used. No production change.'),indent=2)+'\n')
    print(pars)
    for c,group in groups.items():
        for k,v in group.items():print(c,k,v['n'],[(m,round(100*x['wape'],1),round(100*x['max_ape'],1)) for m,x in v['scores'].items()])
