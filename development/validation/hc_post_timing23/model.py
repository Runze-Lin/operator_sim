"""Experimental vLLM HcPost local resource + two-slot scheduler. Inputs never include CA."""
from functools import lru_cache
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
PARAMS=ROOT/'results/v0.23/hc-timing/parameters.json'


@lru_cache(maxsize=None)
def load(path):
    return json.loads(Path(path).read_bytes())


def _core(rows, widths, cohort, chip, p, bars):
    # All events share the earliest device instruction as tick zero.
    scale=p['vector_reference_ops32_cycle']/(chip['vector_tops']['32']*1000/chip['vector_cores']/chip['frequency_ghz'])
    ceil=math.ceil
    pos=0
    last_x=last_coeff=cast_end=write_begin=old_write=write_end=0.0
    end=0.; row=0; prev_state=None; timeline=[]; cycles_skipped=0
    def record(r,i,w,k,a,b):
        if bars:
            timeline.append([r,i,w,k,round(a,3),round(b,3)])
    def service(k,n):
        base=p['dma'][k]
        return base['setup']+base['ticks_per_unit']*n
    while row<rows:
        coeff=(p['first_coeff']+cohort*p['issue_fanout_per_aiv'] if row==0 else
               max(last_coeff+p['next_row_coeff_issue'],last_x+p['next_row_coeff_issue']-p['payload_after_coeff']))
        comb=coeff+p['coeff_issue_gap']
        # Coefficient response is overlapped with larger payload fetch. Smallest observed
        # GM->UB descriptor's intercept is the unobserved short-transfer approximation.
        coeff_done=comb+service('x',64)
        record(row,0,0,'post',coeff,coeff+service('x',16))
        record(row,0,0,'comb',comb,coeff_done)
        for i,w in enumerate(widths):
            first_x=coeff+p['payload_after_coeff']+(p['split_descriptor_extra'] if len(widths)>1 else 0)
            if i:
                first_x=last_x+p['second_fragment_issue']+(p['tail_descriptor_extra'] if w<4096 else 0)
            blocked=pos>=2 and old_write+p['triple_slot_release']>=first_x
            x=old_write+p['triple_slot_release'] if blocked else first_x
            gap=p['triple_slot_release'] if blocked else p['initial_read_issue_gap']+(p['full_read_issue_extra'] if w==4096 else 0)
            res=x+gap
            # First pair has no previous descriptor group in flight.
            xend=x+service('x' if pos else 'first_x',2*w)
            rend=res+service('res' if pos else 'first_res',8*w)
            vf=max(xend,rend,coeff_done)+p['input_ready']
            if pos: vf=max(vf,cast_end-p['next_head_overlap'])
            body=p['arith_body']['setup']+p['arith_body']['ticks_per_unit']*ceil(w/64)
            queue=(p['warm_dispatch'] if pos else p['first_vf_admission']+cohort*p['admission_fanout_per_aiv'])
            vend=vf+queue+body*scale
            cast=vend-p['cast_head_overlap']
            cast_end=cast+p['cast_body']*scale
            old_write,write_begin=write_begin,max(write_end+p['triple_slot_release'],cast_end+p['cast_to_write'])
            write_end=write_begin+service('write',8*w)
            for k,a,b in [('x',x,xend),('res',res,rend),('arith',vf,vend),('cast',cast,cast_end),('write',write_begin,write_end)]:
                record(row,i,w,k,a,b)
            last_x=x;pos+=1
        last_coeff=coeff;row+=1;end=write_end+p['return_after_write']
        # Exact translation invariance of the two-slot state. Skip identical steady rows.
        state=(last_x,last_coeff,cast_end,write_begin,old_write,write_end)
        if prev_state and row>=3 and not bars and rows>row:
            diff=[now-before for now,before in zip(state,prev_state)]
            if max(diff)-min(diff)<1.e-6:
                skip=rows-row;shift=skip*diff[-1]
                last_x,last_coeff,cast_end,write_begin,old_write,write_end=(v+shift for v in state)
                pos+=skip*len(widths);row+=skip;cycles_skipped+=skip;end+=shift
        prev_state=state
    return end,timeline,cycles_skipped


def predict(t,d,sku,*,timeline=False):
    if type(t) is not int or t<1 or type(d) is not int or d<16 or d%16:
        raise ValueError('T must be positive; dense HcPost evidence requires D multiple of 16')
    chip=sku if isinstance(sku,dict) else load(ROOT/'configs/chips'/f'{sku}.json')
    p=load(PARAMS);active=min(t,chip['vector_cores']);rows,extra=divmod(t,active)
    full,tail=divmod(d,4096)
    widths=[4096]*full+([tail] if tail else [])
    # Tiling assigns extra rows to a prefix; fanout is measured as an arrival envelope,
    # not an assumed mapping to physical CA core IDs.
    candidates=[(rows,active-1)]
    if extra:candidates.append((rows+1,extra-1))
    core_ticks,barlist,skipped,crit_rows,rank=max(
        ((*_core(n,widths,rank,chip,p,timeline),n,rank) for n,rank in candidates),
        key=lambda k:k[0])
    freq=chip['frequency_ghz']; reads=t*(10*d+80);writes=t*8*d
    mem=chip['details'].get('l2_bytes',134217728)
    # At least compulsory input traffic plus dirty data that cannot stay resident.
    # No hardware traffic counters exist for this extension; report the assumed cold bound.
    cold_bytes=reads+max(0,writes-mem)
    hbm_ticks=cold_bytes*freq/chip['hbm_bandwidth_gbs']
    l2=chip['details'].get('l2_bandwidth_gbs')
    l2_ticks=(reads+writes)*freq/l2 if l2 else 0.
    total=max(core_ticks,hbm_ticks,l2_ticks)
    result=dict(device_us=total/(freq*1000),core_ticks=core_ticks,bandwidth_ticks={'offchip_cold_lower':hbm_ticks,'l2_peak_lower':l2_ticks},
                active_aiv=active, max_rows=rows+bool(extra), critical_rows=crit_rows, critical_cohort_rank=rank,
                skipped_steady_rows=skipped, fragments_per_row=len(widths),vf_commands=2*len(widths)*t,
                fp32_useful_ops=36*t*d,bf16_cast_outputs=16384*t*len(widths),
                logical_gm_read_bytes=reads,logical_gm_write_bytes=writes,
                descriptor_counts=dict(meta=active,post=t,comb=t,x=t*len(widths),res=t*len(widths),write=t*len(widths)))
    if timeline:result['bars']=barlist
    return result
