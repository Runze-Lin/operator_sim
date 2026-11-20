"""Offline candidate for installed RegBase tiling key 10005, FP32 bias and K <=64."""
from functools import lru_cache
from math import ceil
from statistics import median
from evidence import OUT,DT,read,save,merge_ops


def line(x,y,a,b):
    slope=(b-y)/(a-x);return [y-x*slope,slope]

def prices():
    import numpy as np
    rs=[x for x in read(OUT/'physical.json')['rows'] if x['role']=='primitive-price development'];single=sorted((x for x in rs if x['shape'][0]==1),key=lambda x:x['shape'][1]);a,b=[r['cores'][0] for r in single]
    calc=lambda key:line(128,a[key][0],896,b[key][0]);merges=a['merge_runs']+b['merge_runs'];xy=[[1,e,r-1] for e,r,y in merges];w=np.linalg.lstsq(xy,[y for e,r,y in merges],rcond=None)[0].tolist()
    cold=[a,b];m=next(x['cores'] for x in rs if x['shape'][0]==65);multi=[x for x in m if x['rows']==2]
    arr=[x['launch_abs']+x['init_to_bias_issue'] for x in m];origin=min(x['launch_abs'] for x in m);d0=median(x['init_to_bias_issue'] for x in cold)
    price=dict(launch_cycles=d0,launch_port_cycles_per_aiv=(max(arr)-origin-d0)/(len(m)-1),first_read_response_cycles=median(x['read_ready'] for x in cold),
        row_begin_cycles=median(x['first_norm_after_ready'] for x in cold),next_batch_response_cycles=median(x['x_response'] for x in multi),
        first_sigmoid_bias_to_sort_cycles=calc('norm_to_sort'),sort32_cycles=calc('sort32_durations'),ub_copy_cycles=calc('ub_copy_cycles'),
        merge_cycles_fixed_per_element_extra_repeat=w,select_and_normalize_cycles=median(x['select_cycles'][0] for x in cold),
        next_row_issue_after_select_cycles=median(x['select_to_next_norm'][0] for x in multi),repeated_sigmoid_extra_cycles=median(x['norm_to_sort'][1]-a['norm_to_sort'][0] for x in multi),
        write_latency_cycles=median(x['write_latency'] for x in cold),final_write_issue_cycles=median(x['write_issue_from_lastselect'] for x in cold),
        max_ub_rows=128,batch_fixed_expert_words=17,reference_fp32_ops_per_cycle=27e12/(64*1.65e9),
        source_sha256=[r['trace_sha256'] for r in rs],selector_role='128 and effective 17 words/expert reproduce UB pointer deltas. These are inferred host selector parameters, not physical bytes occupied by the installed kernel.',
        limitation='Input/output response on later batches extrapolated from isolated historical input descriptors; multi-batch observations pending. Sort engine cycles kept fixed/issue at a new clock unless separate prices available.')
    save(OUT/'prices.json',price)


@lru_cache(maxsize=1)
def profile():return read(OUT/'prices.json')

def predict(shape,chip):
    t,e,k=shape
    if not (0<t and 2<e<=2048 and 0<k<=min(64,e)):raise ValueError('small-K one-group FP32 with nonempty rows required')
    p=profile();a=32*ceil(e/32);kstride=8*ceil(k/8);cap=min(p['max_ub_rows'],int(((chip.details.get('ub_bytes') or 253952)-4*p['batch_fixed_expert_words']*a)//(8*a+16*kstride)))
    if cap<1:raise ValueError('target UB cannot use this template')
    r=ceil(t/chip.vector_cores);active=ceil(t/r);n_batches=ceil(r/cap);freq=chip.frequency_ghz;unit=1000*freq
    mte2=min((chip.details.get('mte2_bandwidth_gbs') or 32*freq)/freq,chip.hbm_bandwidth_gbs/(active*freq));mte3=min((chip.details.get('mte3_bandwidth_gbs') or 32*freq)/freq,chip.hbm_bandwidth_gbs/(active*freq))
    vec_scale=p['reference_fp32_ops_per_cycle']/(chip.vector_tops['32']*1000/(chip.vector_cores*freq))
    aff=lambda key:p[key][0]+a*p[key][1];merge,cp=merge_ops(e);mf,me,mr=p['merge_cycles_fixed_per_element_extra_repeat']
    rank_hw=aff('sort32_cycles')+cp*aff('ub_copy_cycles')+sum(mf+me*elements+mr*(repeat-1) for elements,repeat in merge)
    norm=aff('first_sigmoid_bias_to_sort_cycles')*vec_scale;row_first=p['row_begin_cycles']+norm+rank_hw+vec_scale*p['select_and_normalize_cycles']
    ii=p['next_row_issue_after_select_cycles']+vec_scale*p['repeated_sigmoid_extra_cycles']+norm+rank_hw
    launch=p['launch_cycles']+p['launch_port_cycles_per_aiv']*(active-1)
    batch0=min(r,cap);tail=r-(n_batches-1)*cap
    first=max(p['first_read_response_cycles'],4*e*(1+batch0)/mte2);body=n_batches*row_first+(r-n_batches)*ii
    drain=lambda size:max(p['write_latency_cycles'],8*k*size/mte3)+p['final_write_issue_cycles']
    inter=lambda size:max(drain(cap),p['next_batch_response_cycles'],4*e*size/mte2)
    edges=(max(0,n_batches-2)*inter(cap)+inter(tail)) if n_batches>1 else 0
    stages=dict(launch=launch,first_batch=first,ranking=body,interbatch_edges=edges,tail_write=drain(tail))
    return dict(us=sum(stages.values())/unit,active_aiv=active,rows_per_core=r,batch_capacity=cap,critical_batches=n_batches,tail_rows=tail,
        stages_us={k:v/unit for k,v in stages.items()},merge_elements_and_repeats=merge,ub_copies_per_row=cp,
        primary_sort32_per_row=1,steady_row_initiation_cycles=ii,rank_body_engine_cycles=rank_hw)

if __name__=='__main__':prices()
