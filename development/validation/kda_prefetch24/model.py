"""Cross-chunk candidate. Reuses frozen v0.23 primitive prices, never CA totals."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model.chips import load_chip
from operator_model.kda_fla_v2 import FAMILIES, Tile, schedule


def block(rows,chip,active):
    z=Tile(chip,FAMILIES[1],active);v=256*rows;p=16*((rows+15)//16)
    return dict(w=z.dma('OUT->L1',v),h=z.dma('OUT->L1',32768),u=z.dma('OUT->UB',v),
        
        l0=max(z.dma('L1->L0A',256*p),z.dma('L1->L0B',32768)),
        mm0=z.calc([(p,128,128,'bf16')]),fix0=z.dma('L0C->UB',2*v),
        vf1=z.vec(1,rows),right_write=z.dma('UB->OUT',v,v),
        k=z.dma('OUT->L1',v),right=z.dma('OUT->L1',v),l2=max(z.dma('L1->L0A',256*p),z.dma('L1->L0B',256*p)),
        mm2=z.calc([(128,128,p,'bf16')]),fix2=z.dma('L0C->UB',65536),
        state_and_gate=z.dma('OUT->UB',65536,512),gate=z.dma('OUT->UB',512),
        vf3=z.vec(2,rows),write_h=z.dma('UB->OUT',32768),write_vk=z.dma('UB->OUT',65536),
        initial=z.dma('OUT->UB',65536)+z.vec(0,rows)+z.dma('UB->OUT',32768))


def chain(state,part,first,last):
    state_ready,w_slot_free,u_slot_free=state
    d=part
    if first:
        # Median W issue minus initial-state payload issue on the critical
        # workgroup of archived windows. Source clocks stay separate by engine.
        w_slot_free=state_ready+334.5
        state_ready+=d['initial']
        u_slot_free=state_ready+3
    w_end=w_slot_free+d['w']; h_end=state_ready+d['h']
    mm0=max(w_end,h_end)+d['l0']; product=mm0+d['mm0']+d['fix0']
    u_end=u_slot_free+d['u']
    vf1=max(product,u_end)
    wr_start=vf1+d['vf1'];right=wr_start+d['right_write']
    # Kg enters its own fixed L1 slot around MMAD0 issue. Right consumes
    # the shared H/right slot only once the corrected-value store retires.
    k_end=mm0+d['k'];right_end=right+d['right']
    mm2=max(k_end,right_end)+d['l2']
    state_input=right+(d['state_and_gate'] if first else d['gate'])
    delta=max(mm2+d['mm2']+d['fix2'],state_input)
    state_write=delta+d['vf3']
    state_end=state_write+(d['write_vk'] if last else d['write_h'])
    bars=dict(W=[w_slot_free,w_end],H=[state_ready,h_end],K=[mm0,k_end],right=[right,right_end],
              U=[u_slot_free,u_end],mm0=[mm0,mm0+d['mm0']],mm2=[mm2,mm2+d['mm2']],
              corrected=[vf1,wr_start],write_right=[wr_start,right],state=[delta,state_write],
              write_state=[state_write,state_end])
    # Conservatively retire the current MMAD2 before granting W a slot.
    # U can reuse its UB after right retires (six-cycle observed control gap).
    return (state_end,mm2+d['mm2'],right+6),bars


def predict(shape,chip):
    chip=load_chip(chip);ref=schedule(shape,chip); _,s,_,h,_,_=shape
    z=ref['stages'][1];active=z['active_aic'];num=(s+63)//64;heads=(shape[0]*h+active-1)//active
    times=[p['device_us'] for p in ref['stages']];old=times.copy();timeline=[]
    # Host rounds with several heads on one active AIC have different slot/flag
    # semantics. Their cross-head overlap is not validated by these single-head shards.
    if heads==1:
        full=block(64,chip,active); tail=block(s-64*(num-1),chip,active)
        state=(z['first_payload_cycles'],0.,0.)
        if num>1:
            state,record=chain(state,full,True,False);timeline.append(record)
        middle=num-2
        if middle>0:
            state,record=chain(state,full,False,False);timeline.append(record)
            if middle>1:
                after,record2=chain(state,full,False,False);timeline.append(record2)
                # Identical full blocks: a single translated state after one warm
                # transition superposes every subsequent deadline. Translation is
                # asserted before skipping; variable tail is always scheduled.
                remaining=middle-2
                if remaining:
                    nxt,_=chain(after,full,False,False)
                    shifts=[a-b for a,b in zip(nxt,after)]
                    if max(shifts)-min(shifts)<1.e-6:
                        after=tuple(x+remaining*shifts[0] for x in after)
                    else:
                        for _ in range(remaining):after,_=chain(after,full,False,False)
                state=after
        state,end=chain(state,tail,num==1,True);timeline.append(end)
        times[1]=max(state[0]/(1000*chip.frequency_ghz),z['hbm_spill_scenario_us'])
    return dict(sku=chip.name,shape=list(shape),v023_us=sum(old),stages_old=old,
                us=sum(times),stages_us=times,fwd_h_trace_model_ticks=timeline,
                proof_domain='one sequence × head per workgroup; measured release GK FP32-state branch',
                active_aic=[st['active_aic'] for st in ref['stages']])


if __name__=='__main__':
    for t in [2,64,65,129,262144]:
        for sku in ['Ascend950DT_9582','Ascend950PR_957b']:
            p=predict((1,t,3,3,128,128),sku);print(t,sku,p['stages_old'][1],p['stages_us'][1],p['us'])
