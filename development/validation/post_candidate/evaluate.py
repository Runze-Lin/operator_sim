"""Process-and-shape-purged stage calibration, with a separate zero-target-price test."""
from collections import defaultdict
import math
from pathlib import Path
import statistics
import sys

import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import DT,PR,read,write
from operator_model import load_chip

OUT=ROOT/'results/post-candidate'
PHASES=('arrival','control','read','body','retire')
STATES=('first','expand','reuse','after_pre_sinkhorn')


def basis(q,r,c):
    # Target prices use device microseconds, with 1.65 GHz/DT FP32 TOP/s as
    # reference for cycle/issue scaling. L2 residency itself stays unknown.
    f=1.65/c.frequency_ghz
    width=r['fragment_width'];rows=r['rows_per_core'];active=r['active_aiv'];n=4
    rel_simd=(27/64)/(c.vector_tops['32']/c.vector_cores)
    byte_per_tile=width*2*(n+1) + (80 if r['route']=='post_token' else 0)
    return {
        'arrival':[f*math.log2(active)/6],
        'control':[],
        'read':[f*byte_per_tile/1024],
        'body':[rel_simd*rows*math.ceil(width/64)/32, f*(rows-1)],
        'retire':[]}[q]


def train(rows):
    # Robust regression of individual milestones only. The response, residual
    # and design never use the entire kernel duration. States are observable
    # calibration labels; no labels determine the no-history query.
    curves={}
    for route in ('post_channel','post_token'):
        subset=[r for r in rows if r['route']==route]
        for q in PHASES:
            labels=[s for s in STATES if any(r['state']==s for r in subset)]
            x=[];y=[]
            for r in subset:
                c=load_chip(r['chip'])
                x.append([1.65/c.frequency_ghz if s==r['state'] else 0 for s in labels]+basis(q,r,c))
                y.append(r['phase_us'][q])
            X=np.array(x);y=np.array(y)
            fit=least_squares(lambda p:X@p-y,.01*np.ones(X.shape[1]),bounds=(0,np.inf),
                              loss='soft_l1',f_scale=.1).x
            known=dict(zip(labels,fit[:len(labels)]))
            curves[route,q]=known,fit[len(labels):]
    return curves


def phases(curves,r,state):
    c=load_chip(r['chip']);f=1.65/c.frequency_ghz
    stages={}
    for q in PHASES:
        K,slopes=curves[r['route'],q]
        fallback={'expand':'first','after_pre_sinkhorn':'reuse','first':'expand','reuse':'after_pre_sinkhorn'}[state]
        intercept=K.get(state,K.get(fallback,statistics.median(K.values())))
        stages[q]=float(f*intercept+slopes@basis(q,r,c))
    return stages


def estimate(curves,r):
    possible=[s for s in STATES if not(s=='expand' and r['active_aiv']==1)]
    cond={s:phases(curves,r,s) for s in possible}
    totals={s:sum(q.values()) for s,q in cond.items()}
    a=min(totals,key=totals.get);b=max(totals,key=totals.get)
    w=totals[a]/(totals[a]+totals[b])
    mixed={q:(1-w)*cond[a][q]+w*cond[b][q] for q in PHASES}
    # Offer two independent boundary conditions derived from the caller, never
    # from a held-out timing. An ended Post process need not synchronize caches;
    # the Pre/Sink producer path therefore has its own measured stage classes.
    previous=r.get('prior_aiv')
    if r['state']=='after_pre_sinkhorn':
        stream_state='after_pre_sinkhorn'
    elif previous==0:
        stream_state='first'
    elif previous is not None and previous<r['active_aiv']:
        stream_state='expand'
    elif previous is not None:
        stream_state='reuse'
    else:
        raise ValueError('Missing documented launch history')
    assert stream_state==r['state']
    return dict(no_history_us=sum(mixed.values()),stages_us=mixed,conditional_us=totals,
                given_history_us=totals[r['state']],given_history_stages_us=cond[r['state']])


def metric(rows,key):
    return dict(n=len(rows),n_shapes=len({tuple(r['shape']) for r in rows}),
                wape=100*sum(abs(r['us']-r[key]) for r in rows)/sum(r['us'] for r in rows),
                max_err=100*max(abs(r[key]/r['us']-1) for r in rows))


def run():
    rows=read(OUT/'phase-evidence.json');assert all('failed' not in r and r['active_match'] for r in rows)
    result=[];floor=[];coefs=train(rows)
    folds=defaultdict(list)
    for r in rows:folds[r['process']].append(r)
    for p,hold in folds.items():
        shapes={tuple(r['shape']) for r in hold}
        train_rows=[r for r in rows if r['process']!=p and tuple(r['shape']) not in shapes]
        c=train(train_rows); zero=train([r for r in train_rows if r['chip']==DT]) if any(r['chip']==PR for r in hold) else None
        for r in hold:
            # After the very first issued kernel, the simulator can often tell
            # whether this is an ordinary later Post even if prior AIVs were
            # never counted. Keep expand vs reuse unknown in that case.
            pred=estimate(c,r)
            t=pred['conditional_us']
            if r['state'] in ('first','after_pre_sinkhorn'):
                partial=t[r['state']]
            elif 'expand' not in t:partial=t['reuse']
            else:partial=2*t['reuse']*t['expand']/(t['reuse']+t['expand'])
            row=dict(first_or_producer_known_us=partial,shape=r['shape'],chip=r['chip'],state=r['state'],process=p,role='retrospective grouped cross-validation',
                     trace=r['trace'],us=r['us'],group=r['group'],old_unknown_us=r['old_unknown_us'],actual_phases_us=r['phase_us'],
                     **pred,train_n=len(train_rows),train_processes=len({r['process'] for r in train_rows}))
            if zero is not None:
                row['zero_pr_labels_us']=estimate(zero,r)['no_history_us']
            result.append(row)
    summary=[]
    for chip in (DT,PR):
        pool=[r for r in result if r['chip']==chip]
        for group in ('all','first','expand','reuse','after_pre_sinkhorn','few token/wide','one row/core','multiple rows/core'):
            subset=[r for r in pool if group=='all' or group in (r['state'],r['group'])]
            if subset:
                keys=['old_unknown_us','no_history_us','first_or_producer_known_us','given_history_us']
                if chip==PR:keys.append('zero_pr_labels_us')
                summary.append(dict(chip=chip,group=group,**{k:metric(subset,k) for k in keys}))
    write(OUT/'cross-validation.json',dict(summary=summary,rows=result,
          training_split='Leave one entire CA process out; remove that process AND all matching tensor shapes from every SKU/backend/state from training. Additional PR zero-shot fits to DT labels alone after the same purge.',
          baseline='v0.21 known archive replay; some prior DT rows used by its historical v0.12 calibration. Candidate is out-of-process AND out-of-shape. No claim of independent validation for either.',
          all_archive_parameters=[dict(route=route,phase=q,state_intercept_us={s:float(x) for s,x in ints.items()},resource_slopes=list(coefs)) for (route,q),(ints,coefs) in coefs.items()]))
    print([(r['chip'],r['group'],[(k,round(r[k]['wape'],1),round(r[k]['max_err'],1)) for k in ('old_unknown_us','no_history_us','given_history_us')]) for r in summary if r['group'] in ('all','first','reuse','few token/wide','multiple rows/core')],flush=True)
if __name__=='__main__':run()
