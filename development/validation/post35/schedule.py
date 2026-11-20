"""First operand response and steady UB issue schedule; no observed predecessor as input."""
from copy import deepcopy
from functools import lru_cache
from math import ceil, sqrt
from pathlib import Path
import importlib.util
import json

from operator_model import load_chip
from operator_model.post_resource import resource_post
from operator_model.post_repeat import channel_tiles, _prices
from operator_model.post_token_pipeline import token_pipeline
from operator_model.post_fragment_pipeline import covered_fragment_schedule

ROOT=Path(__file__).resolve().parents[3]
def legacy_candidate():
    spec=importlib.util.spec_from_file_location("channel33",ROOT/"development/validation/post33/candidate.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m
C=legacy_candidate()
@lru_cache(maxsize=1)
def channel_prices():
    return json.loads((ROOT/"results/v0.33/post/prices.json").read_bytes())

def schedule(shape,chip):
    c=load_chip(chip)
    _,_,_,base=resource_post(shape,c,{})
    p=base["target_plan"]
    if p["route"]!="post_token":
        q=C.predict(shape,chip,channel_prices())
        return dict(us=q['us'],states=q['conditional_us'],model="archived four-write channel schedule",route=p['route'])
    tiles=channel_tiles(shape[2],c)
    if not tiles or tiles['count'] not in (1,2) or not c.vector_tops['32']:
        q=C.predict(shape,chip,channel_prices())
        return dict(us=q['us'],states=q['conditional_us'],model="unpriced operand schedule; previous model",route=p['route'])
    modes=('first','expand','reuse','repeat') if p['active_aiv']>1 else ('first','reuse','repeat')
    floor=base['shared_floor_us'];rates=_prices();states={};stages={}
    if tiles['count']==1:
        for mode in modes:
            pipe=token_pipeline(shape,c,p,mode,rates,floor)
            states[mode]=pipe['us'];stages[mode]=pipe['stages_us']
        ident="one-slot row dependency endpoints"
    else:
        # The observed two-slot loop schedules writes in max-plus time. The first
        # output issue includes dispatch, coefficient/input response and first SIMD.
        # Measure its offset between conditions with one fragment (same CANN family),
        # then process every later fragment only at the shared 2-slot issue interval.
        common=covered_fragment_schedule(c,base,tiles)
        width=tiles['inner']; one={**p,'rows_per_core':1}
        starts={}; stores={}; exits={}
        for mode in modes:
            z=token_pipeline([shape[0],4,width],c,one,mode,rates,0)['stages_us']
            starts[mode]=sum(z[k] for k in ('arrival','control_to_first_read','first_read_to_ready','ready_to_first_write'))
            stores[mode]=z['last_write_response'];exits[mode]=z['exit_after_write']
        old=common['stages_us']
        for mode in modes:
            fill=sum(old[k] for k in ('arrival','first_operand','first_read_ready','first_operand_to_store'))+starts[mode]-starts['reuse']
            tail=max(0.,old['final_write_response']+stores[mode]-stores['reuse'])
            components=dict(first_write=max(0.,fill),later_writes=old['next_write_issues'],drain=tail,exit=exits[mode])
            states[mode]=max(sum(components.values()),floor);stages[mode]=components
        ident="two-slot max-plus loop with one conditional first-write lift and drain"
    # The point policy is exactly the v0.33 token policy, isolating the resource change.
    return dict(us=sqrt(min(states.values())*max(states.values())), states=states,route=p['route'],
                prices=stages,model=ident,rows=p['rows_per_core'],fragments=tiles,actual_predecessor_used=False)
