"""Shape-only diagnostic; no process or prior-operation state is an input."""
from math import ceil,sqrt
from operator_model import load_chip
from operator_model.post_resource import resource_post
from operator_model.post_history import _scenarios


def prices_for_shape(shape, chip, model):
    target=load_chip(chip)
    _,_,n,base=resource_post(shape,target,{})
    st={k:v.latency_us for k,v in _scenarios(tuple(shape),target,base).items()}
    plan=base['target_plan']
    if not target.vector_tops['32']:
        return st,dict(scope='uncalibrated vector rate')
    floor=base['shared_floor_us']
    if plan['route']=='post_token':
        return st,dict(scope='token row fill then steady issue',rows=plan['rows_per_core'])
    c=target.frequency_ghz*1000
    v=target.vector_tops['32']*1000/(target.vector_cores*target.frequency_ghz)
    hot=model['hot']; exp=model['expansion'];d=plan['fragment_width'];g=ceil(d/64)
    def unit(port):
        bw=target.details.get(('mte2' if port=='read' else 'mte3')+'_bandwidth_gbs')
        return 1024*target.frequency_ghz/bw if bw else model['source'][port+'_cycles_per_kib']
    read=unit('read')/1024;write=unit('write')/1024;vwork=9*d/v
    ii=max(vwork,2*d*write,hot['scalar_issue_cycles']+g*hot['group_cycles']*hot['reference_ops_per_cycle']/v+
           (hot['aligned_tail_cycles'] if d%64 else 0.))
    hot_ready=max(max(hot['read0_cycles'],2*d*read),hot['read_gap_cycles']+max(hot['read1_cycles'],8*d*read))+hot['vector_handoff_cycles']
    pressure=(n-1)/63  # DT arbitration slope per submitting lane; replace for a different port topology.
    arrival=hot['multi_aiv_arrival_cycles'] if n>1 else 0.
    setup=hot['setup_base_cycles']+hot['setup_per_pressure_cycles']*pressure
    warm=dict(arrival=arrival,first_operand_and_first_write=setup+vwork,reads=hot_ready,
              remaining_write_issues=3*ii,write_response=max(hot['hot_write_response_cycles'],2*d*write),exit=hot['exit_cycles'])
    st['repeat']=max(floor,sum(warm.values())/c)
    if n>1:
        cold=dict(arrival=exp['arrival'],control=exp['control'],read_ready=max(exp['read0']+2*d*read,
            exp['gap']+exp['read1']+8*d*read)+exp['after_read'],first_write=model['expansion_first_write_upper_cycles']+vwork,
            remaining_write_issues=3*ii+exp['fourth_write_extra'],write_response=2*d*write+exp['ack'],exit=exp['exit'])
        st['expand_from_one_resident']=max(floor,sum(cold.values())/c)
    else:cold=None
    return st,dict(scope='four-write channel',fragment_width=d,hot_cycles=warm,expansion_cycles=cold,
                   setup_mad_us=hot['setup_mad_cycles']/c)


def predict(shape,chip,model):
    states,details=prices_for_shape(shape,chip,model)
    lo,hi=min(states.values()),max(states.values())
    relative_minimax=2*lo*hi/(lo+hi)
    if details['scope']=='four-write channel':
        # A simulation that understates cold expansion by more than half can admit
        # twice the intended arrival rate. Publish the explicit provisioning guard.
        point=max(hi/2,relative_minimax)
        policy='relative minimax with cold latency underestimation limited to 50%'
    else:
        # Row-throughput uncertainty is multiplicative. This equalizes p/lo and hi/p.
        point=sqrt(lo*hi)
        policy='minimax multiplicative ratio across row issue-rate scenarios'
    return dict(us=point,relative_minimax_us=relative_minimax,policy=policy,
                conditional_us=states,range_us=[lo,hi],details=details)
