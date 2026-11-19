"""Frozen Post candidate. Five measured milestones and parallel ingress/egress floors."""
import math
from operator_model import load_chip
from operator_model.post_resource import resource_post
from operator_model.rms_resource import READ_PER_KIB


def milestone_base(plan, chip, phase):
    clock = 1.65 / chip.frequency_ghz
    rows, width = plan['rows_per_core'], plan['fragment_width']
    vec = (27 / 64) / (chip.vector_tops['32'] / chip.vector_cores) if chip.vector_tops['32'] else clock
    return {'arrival': [clock * math.log2(plan['active_aiv']) / 6], 'control': [], 'read': [],
            'body': [vec * rows * math.ceil(width / 64) / 32, clock * (rows - 1)], 'retire': []}[phase]


def predict(shape, chip, history, previous_shape, params):
    attrs = {} if history is None else {'resident_aiv': history}
    base, _, active, diag = resource_post(shape, chip, attrs)
    p = diag['target_plan']
    state, route = diag['context'], p['route']
    repeat = (state == 'reuse' and tuple(previous_shape or ()) == tuple(shape))
    # The ordered historical comparison supported the token kernel's cold first-read
    # schedule. Exact repeats of the channel kernel have two observed SKU contexts.
    detailed = (state == 'first' and route == 'post_token') or (repeat and route == 'post_channel')
    if not detailed:
        return dict(us=base, active=active, state=state, mode='public resource stages',
                    stages=diag['stages_us'], plan=p, port=None)
    kind = 'repeat' if repeat else 'first'
    table = params[route][kind]
    clock = 1.65 / chip.frequency_ghz
    st = {}
    for phase in ('arrival','control','body','retire'):
        terms = table['phase'][phase]
        st[phase] = terms['base_us'] * clock + sum(a*b for a,b in zip(terms['slopes'],milestone_base(p,chip,phase)))
    width = p['fragment_width']
    lane = 32 * math.ceil(width * 2 / 32)
    payload = [lane,4*lane] if route=='post_channel' else [64,16,2*width,8*width]
    port = table['port']
    freq = 1000 * chip.frequency_ghz
    bw = chip.details.get('mte2_bandwidth_gbs')
    slope = 1024 * chip.frequency_ghz / bw if bw else READ_PER_KIB
    starts = [0.]
    for wait in port['gap_cycles']:
        starts.append(starts[-1] + wait / freq)
    responses = [(fixed+n*slope/1024)/freq for fixed,n in zip(port['response_fixed_cycles'],payload)]
    st['read'] = max(t+l for t,l in zip(starts,responses)) + port['handoff_cycles']/freq
    st = {k:st[k] for k in ('arrival','control','read','body','retire')}
    st['shared_link_floor'] = max(0.,diag['shared_floor_us']-sum(st.values()))
    return dict(us=sum(st.values()),active=active,state=state,mode=kind+' operand schedule',
                stages=st,plan=p,port=dict(bytes=payload,issue_us=starts,response_us=responses,
                                           handoff_us=port['handoff_cycles']/freq))
