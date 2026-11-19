"""Equivalent bounded evaluation for one-fragment rows with a two-slot period.

The frozen oracle remains unchanged. A translation of all six live timestamps
is a sufficient condition for the complete deterministic recurrence to repeat.
"""
from model import ROOT, PARAMS, load, predict, _core


def _periodic(n,d,rank,chip,p):
    probe=16
    if n<=probe:return _core(n,[d],rank,chip,p,False)[0],0
    _,bars,_=_core(probe,[d],rank,chip,p,True)
    by_row=[{} for _ in range(probe)]
    for row,_,_,stage,begin,end in bars:by_row[row][stage]=(begin,end)
    # Bars for drawing are rounded to 0.001 clocks. Reconstruct the six full-
    # precision states from the frozen one-fragment recurrence to prove equality.
    scale=p['vector_reference_ops32_cycle']/(chip['vector_tops']['32']*1000/chip['vector_cores']/chip['frequency_ghz'])
    body=(p['arith_body']['setup']+p['arith_body']['ticks_per_unit']*((d+63)//64))*scale
    def srv(key,nb):
        price=p['dma'][key]
        return price['setup']+price['ticks_per_unit']*nb
    states=[]
    last_x=coeff=cast_end=write=old_write=write_end=0.
    for row in range(probe):
        coeff=(p['first_coeff']+rank*p['issue_fanout_per_aiv'] if not row else
               max(coeff+p['next_row_coeff_issue'],last_x+p['next_row_coeff_issue']-p['payload_after_coeff']))
        coeff_end=coeff+p['coeff_issue_gap']+srv('x',64)
        nominal=coeff+p['payload_after_coeff']
        reused=row>=2 and old_write+p['triple_slot_release']>=nominal
        x=old_write+p['triple_slot_release'] if reused else nominal
        gap=p['triple_slot_release'] if reused else p['initial_read_issue_gap']
        ready=max(coeff_end,x+srv('x' if row else 'first_x',2*d),
                  x+gap+srv('res' if row else 'first_res',8*d))+p['input_ready']
        start=max(ready,cast_end-p['next_head_overlap']) if row else ready
        queue=p['warm_dispatch'] if row else p['first_vf_admission']+rank*p['admission_fanout_per_aiv']
        cast_end=start+queue+body-p['cast_head_overlap']+p['cast_body']*scale
        old_write,write=write,max(write_end+p['triple_slot_release'],cast_end+p['cast_to_write'])
        write_end=write+srv('write',8*d);last_x=x
        state=(x,coeff,cast_end,write,old_write,write_end)
        witness=by_row[row]
        if max(abs(state[j]-z) for j,z in enumerate((witness['x'][0],witness['post'][0],witness['cast'][1],witness['write'][0],old_write,witness['write'][1])))>.000501:
            raise ValueError('State reconstruction differs from frozen scheduler')
        states.append(state)
    for period in range(1,5):
        # Check every timestamp, on two consecutive periods after queue priming.
        for k in range(2+period,probe-period):
            diff=[a-b for a,b in zip(states[k],states[k-period])]
            diff2=[a-b for a,b in zip(states[k+period],states[k])]
            if max(diff+diff2)-min(diff+diff2)>1e-7:continue
            steps,tail=divmod(n-1-k,period)
            if steps<1:continue
            return states[k+tail][-1]+steps*diff[-1]+p['return_after_write'],steps*period
    return _core(n,[d],rank,chip,p,False)[0],0


def accelerated(t,d,sku):
    chip=sku if isinstance(sku,dict) else load(ROOT/'configs/chips'/f'{sku}.json')
    a=min(t,chip['vector_cores'])
    if d>=4096 or t<=16*a or t<1 or d<16 or d%16:return predict(t,d,chip)
    q,r=divmod(t,a);p=load(PARAMS);cohorts=[(q,a-1)]+([(q+1,r-1)] if r else [])
    ticks,skipped,n,rank=max(((*_periodic(n,d,rank,chip,p),n,rank) for n,rank in cohorts),key=lambda v:v[0])
    # All totals are affine in global tokens for a one-fragment row. Coefficient
    # work, physical casting and AIV metadata have distinct multiplicities.
    result=predict(a,d,chip)
    reads=t*(10*d+80);writes=t*8*d;freq=chip['frequency_ghz'];mem=chip['details'].get('l2_bytes',134217728)
    hbm=(reads+max(0,writes-mem))*freq/chip['hbm_bandwidth_gbs']
    l2=chip['details'].get('l2_bandwidth_gbs');l2_ticks=(reads+writes)*freq/l2 if l2 else 0.
    result.update(device_us=max(ticks,hbm,l2_ticks)/(freq*1000),core_ticks=ticks,
                  bandwidth_ticks={'offchip_cold_lower':hbm,'l2_peak_lower':l2_ticks},
                  max_rows=q+bool(r),critical_rows=n,critical_cohort_rank=rank,skipped_steady_rows=skipped,
                  vf_commands=2*t,fp32_useful_ops=36*t*d,bf16_cast_outputs=16384*t,
                  logical_gm_read_bytes=reads,logical_gm_write_bytes=writes,
                  descriptor_counts=dict(meta=a,post=t,comb=t,x=t,res=t,write=t))
    return result
