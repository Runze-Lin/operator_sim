"""Conditional timing of the two-fragment built-in Post kernel on an ordered token stream."""
from operator_model.resources import load_parameters
from functools import lru_cache
from math import ceil, log2

from operator_model.chips import load_chip
from operator_model.operators.mhc.repeat import channel_tiles, _prices as row_prices
from operator_model.operators.mhc.resources import resource_post


@lru_cache(maxsize=1)
def _prices():
    return load_parameters('post_slots')['cycles']


def _advance_rows(pair, first, tail, repeat):
    """Each new write needs the previous write and release of its input slot two writes back.

    For each complete row raise the same two-by-two max-plus transition to the
    number of intervening rows. pair=(tail write, first write) of the last row.
    """
    # First fragment W0'=max(W1+II0,W0+slot0).
    # Tail W1'=max(W0'+II1,W1+slot1).
    a = max(first['ii']+tail['ii'], tail['slot_path'])
    b = first['slot_path']+tail['ii']
    c, d = first['ii'], first['slot_path']
    x, y = pair
    while repeat:
        if repeat & 1:
            x, y = max(a+x,b+y), max(c+x,d+y)
        a, b, c, d = max(a+a,b+c), max(a+b,b+d), max(c+a,d+c), max(c+b,d+d)
        repeat >>= 1
    return x, y


def estimate_post_fragments(shape, *, chip, resident_aiv, previous_shape, inspect_rows=(), vf_price=None):
    """Conditional BF16 N=4 resource hypothesis; no target CA or hardware runtime.

    Inputs describe a known same-family high-water mark and the immediately
    preceding successful built-in Post. Two UB fragments, changed token shapes
    and covered AIVs are required. Other cases continue to use their normal
    public entry. Requested row indices address a longest lane; at most eight
    may be inspected. 950 coefficients come from one ordinary DT launch.
    """
    target = load_chip(chip)
    if not target.vector_tops['32']:
        raise ValueError('A target FP32 Vector rate is required')
    _, _, active, diag = resource_post(shape, target, {'resident_aiv':resident_aiv})
    _, _, previous_active, prev = resource_post(previous_shape, target, {})
    plan = diag['target_plan']
    tiles = channel_tiles(shape[2], target)
    if (diag['context'] != 'reuse' or resident_aiv < previous_active or tuple(previous_shape)==tuple(shape)
            or plan['route']!='post_token' or prev['target_plan']['route']!='post_token'
            or not tiles or tiles['count']!=2):
        raise ValueError('Requires a covered changed token shape with two UB fragments')
    return covered_fragment_schedule(target, diag, tiles, inspect_rows, vf_price=vf_price)


def covered_fragment_schedule(target, diag, tiles, inspect_rows=(), vf_price=None):
    """Two-slot costs conditional on a covered, changed token-to-token predecessor.

    The individual previous tensor dimensions do not enter the critical path
    once the caller has established that dispatch condition.
    """
    active = diag['target_plan']['active_aiv']
    rows = diag['target_plan']['rows_per_core']
    requested = tuple(inspect_rows)
    if len(requested)>8 or any(type(i) is not int or not 0<=i<rows for i in requested):
        raise ValueError('Inspect at most eight valid row indices of the critical AIV')

    p = _prices()
    shared = row_prices()
    clock = 1000*target.frequency_ghz
    ref_ops_per_cycle = 27000/(64*1.65)
    target_ops_per_cycle = target.vector_tops['32']*1000/(target.vector_cores*target.frequency_ghz)
    def port(queue):
        rate = target.details.get('mte'+queue+'_bandwidth_gbs')
        return 1024*target.frequency_ghz/rate if rate else shared['read_cycles_per_kib' if queue=='2' else 'write_cycles_per_kib']
    read, write = port('2'), port('3')
    lead = p['slot_release_lead_to_store']
    def fragment(width, previous_width, includes_coeff):
        groups = ceil(width/64)
        vf = max(36*width/target_ops_per_cycle,shared['ready_to_write_per_group']*groups*ref_ops_per_cycle/target_ops_per_cycle)
        if vf_price:
            vf=vf_price('post_mix',1,width,4)
        # Previous payload can hold the single write port. Each first fragment of
        # a new row needs 80 B of coefficients in addition to data.
        ii = max(vf+p['write_gap_over_vec'], (10*width+80*includes_coeff)*read/1024,
                 8*previous_width*write/1024)
        h = 2*width*read/1024
        x = 8*width*read/1024
        spans = {s: max(p[s+'_h_ack']+h,p[s+'_hx_issue']+p[s+'_x_ack']+x)
                 for s in ('prefetch','reuse')}
        handoff = vf+p['first_ready_to_write_over_vec']
        return dict(width=width,ii=ii,slot_path=spans['reuse']+handoff-lead,
                    spans=spans,h=h,x=x,compute=handoff)
    first=fragment(tiles['inner'],tiles['tail'],True)
    tail=fragment(tiles['tail'],tiles['inner'],False)
    arrival=shared['by_state']['reuse']['arrival_per_log2_aiv']*log2(active)
    coeff=arrival+p['first_coeff_after_lane_entry']
    coeff_ready=coeff+max(p['coeff0_fixed']+64*read/1024,p['coeff1_to_coeff2']+p['coeff1_fixed']+16*read/1024)
    h0=coeff+p['coeff_to_first_h'];h1=h0+p['second_h_lookahead']
    w0=max(coeff_ready,h0+first['spans']['prefetch'])+first['compute']
    # A single MTE3 cannot issue a new fragment until the previous issue is served.
    w1=max(w0+tail['ii'],max(coeff_ready,h1+tail['spans']['prefetch'])+tail['compute'])
    last,_=_advance_rows((w1,w0),first,tail,rows-1)
    drain=p['final_write_fixed']+8*tiles['tail']*write/1024
    cycles=last+drain+p['exit_after_last_write']
    stages=dict(arrival=arrival/clock,first_operand=(h0-arrival)/clock,
                first_read_ready=(w0-first['compute']-h0)/clock,first_operand_to_store=first['compute']/clock,
                next_write_issues=(last-w0)/clock,final_write_response=drain/clock,
                retire=p['exit_after_last_write']/clock)
    stages['exposed_shared_floor']=max(0.,diag['shared_floor_us']-cycles/clock)
    timeline=[]
    for i in requested:
        pair=_advance_rows((w1,w0),first,tail,i)
        prev_pair=_advance_rows((w1,w0),first,tail,i-1) if i else None
        for f,frag in enumerate((first,tail)):
            tag='reuse' if i else 'prefetch'
            h=(prev_pair[1-f]-lead) if i else (h0,h1)[f]
            w=pair[1-f]
            timeline.append(dict(row=i,fragment=f,columns=frag['width'],
                h_read_us=[h/clock,(h+p[tag+'_h_ack']+frag['h'])/clock],
                x_read_us=[(h+p[tag+'_hx_issue'])/clock,(h+p[tag+'_hx_issue']+p[tag+'_x_ack']+frag['x'])/clock],
                write_issue_us=w/clock,final_write_ack_us=(w+drain)/clock if i==rows-1 and f==1 else None))
    return dict(model='post_two_fragment_slot_reuse',qualification='conditional engineering estimate',
                us=sum(stages.values()),stages_us=stages,active_aiv=active,critical_rows=rows,
                channel_tiles=tiles,fragment_costs_cycles=[first,tail],timeline=timeline,
                device_work=diag['target_plan'],chip=target.name)
