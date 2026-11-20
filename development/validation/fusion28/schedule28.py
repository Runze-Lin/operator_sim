"""Conditional N4 Sinkhorn issue schedule and distribution-free pricing under unknown residency."""
from functools import lru_cache
import json
from pathlib import Path
import sys
from statistics import mean
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/v0.28/fusion'
sys.path.insert(0,str(ROOT))
from operator_model import load_chip
from operator_model.sink_history import price

@lru_cache(maxsize=1)
def params():return json.loads((OUT/'prices.json').read_bytes())

def phases(shape,it,chip,h):
    us,_,a,old=price(shape,chip,{'num_iters':it,'resident_aiv':h})
    if us is None:raise ValueError(old['qualification'])
    plan=old['target_plan'];g=plan['register_passes'];p=params()
    e=dict(old['issue_epochs_us'])
    if g>1:
        ref=load_chip();rate=(chip.vector_tops['32']/chip.vector_cores/chip.frequency_ghz)/(ref.vector_tops['32']/ref.vector_cores/ref.frequency_ghz)
        clock=1000*chip.frequency_ghz;unit=rate*clock
        # Account peak throughput only on pass work. The original input/DMA and
        # launch eligibility come unchanged from the target resource schedule.
        state=p['phase_source'][plan['history_state']]
        setup_to_col=state['first_col_from_setup'][0]+a*state['first_col_from_setup'][1]
        extra_soft=p['soft_extra_cycles']*(g-1)
        soft_gap=max(p['soft_one_floor'],p['soft_start_to_first_col_linear_intercept']+extra_soft)
        e['first_col_vf']=max(e['softmax_vf']+soft_gap/unit,e['setup_vf']+setup_to_col/clock)
        e['first_row_vf']=e['first_col_vf']+(p['col_row_second_pass']+p['col_row_increment']*(g-2))/unit
        e['first_pair_period']=(p['first_pair']+p['extra_pair_cycles']*(g-1))/unit
        e['steady_pair_period']=(p['steady_pair']+p['extra_pair_cycles']*(g-1))/unit
        # The descriptor stays open across the overlap. With several passes the
        # next column can issue before the preceding row's descriptor has retired.
        trailing=(p['column_tail_two']+p['column_tail_inc']*(g-2))/unit
        e['first_row_to_col']=max(p['row_col_one']/unit,e['first_pair_period']-trailing)
        e['steady_row_to_col']=e['steady_pair_period']-trailing
        e['scatter_vf']=e['first_row_vf']+(e['first_pair_period']+max(0,it-2)*e['steady_pair_period'] if it>=2 else 0)
        old_end=old['issue_epochs_us']
        # Additional scatter groups use the issue spacing at the *same* RV_VSCATTER PC;
        # fixed setup is paid once by the helper rather than once each pass.
        scatter_gap=(state['write_gap'][0]+a*state['write_gap'][1]+p['scatter_additional']*(g-1))/unit
        e['write_begin']=e['scatter_vf']+scatter_gap
        e['write_end']=e['write_begin']+old_end['write_end']-old_end['write_begin']
        e['end']=e['write_end']+old_end['end']-old_end['write_end']
    return dict(state=plan['history_state'],active=a,groups=g,rows=plan['rows_per_core'],required_ub_bytes=plan['required_ub_bytes'],epochs_us=e,us=e['end'],legacy_known_us=us)

def estimate(shape,it,chip,lower_h=0,upper_h=None):
    upper_h=chip.vector_cores if upper_h is None else upper_h
    if type(lower_h)is not int or type(upper_h)is not int or not 0<=lower_h<=upper_h<=chip.vector_cores:raise ValueError('invalid integer H interval on target')
    base=phases(shape,it,chip,lower_h);a=base['active'];states={}
    for lower,upper in ((0,0),(1,a-1),(a,chip.vector_cores)):
        h=max(lower_h,lower)
        if h>min(upper_h,upper):continue
        v=base if h==lower_h else phases(shape,it,chip,h)
        states[v['state']]=v
    values=[v['us']for v in states.values()]
    low=min(values);high=max(values);risk=(high-low)/(high+low)
    # If every modeled history can meet the user's 25% relative-error target,
    # choose the feasible point closest to an equal-state mean. If infeasible
    # the interval collapses to the exact relative minimax. No data mix is fitted.
    tol=max(.25,risk);left=(1-tol)*high;right=(1+tol)*low
    return dict(range_us=[low,high],lower_h=lower_h,upper_h=upper_h,states=states,model_uncertainty_pct=100*risk,
                harmonic_us=2*low*high/(low+high),point_us=min(right,max(left,mean(values))),mean_us=mean(values))
