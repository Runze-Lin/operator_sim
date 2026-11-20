"""Conditional B-resident L1 schedule for the Ascend 950 BF16 MatMulV3 ASWT path."""
from math import ceil, log2

from operator_model.chips import REFERENCE_NAME, load_chip
from operator_model.operators.matmul.resources import _costs
from operator_model.operators.matmul.tiling import matmul_plan


def estimate_matmul_full_load(shape, *, chip=REFERENCE_NAME, dtype='bf16', layout='ND_ND'):
    """Price a contiguous, non-transposed BF16 multiply with one N rectangle.

    The 950 route is a conditional source-based candidate. Only full output,
    A-page and MMAD tiles are priced. Shapes requiring other dispatches return
    an unavailable result. All latency fields represent device microseconds.
    """
    if not isinstance(shape,(list,tuple)) or len(shape)!=3 or any(type(x)!=int or x<=0 for x in shape):
        raise ValueError('Expected positive integer [M,N,K]')
    target=load_chip(chip)
    m,n,k=shape
    unavailable={'status':'unavailable','latency_us':None}
    if dtype!='bf16' or layout!='ND_ND':
        return {**unavailable,'reason':'Requires contiguous BF16 A[M,K] and B[K,N], no transpose or bias.'}
    if not target.cube_tops.get('16'):
        return {**unavailable,'reason':'BF16 peak is required.'}
    det=target.details; cores=target.cube_cores; clock=target.frequency_ghz
    l1=det.get('l1_bytes') or 524288
    l0a=det.get('l0a_bytes') or 65536
    l0b=det.get('l0b_bytes') or 65536
    l0c=det.get('l0c_bytes') or 262144
    l2=det.get('l2_bytes') or 134217728
    coarse=matmul_plan(shape,target)
    coarse_m,base_n,base_k=coarse['tile']
    if base_n!=n or n%16 or k%16 or ceil(m/coarse_m)<=cores:
        return {**unavailable,'reason':'Requires one aligned N rectangle and repeated M rectangles beyond one wave.'}
    weight=2*n*k
    if weight>3*l1//4 or 2*m*k<=3*l1//4:
        return {**unavailable,'reason':'B must fit the 3/4 L1 guard, and A residency must already be excluded.'}
    l2_rate=(det.get('l2_bandwidth_gbs') or 100*cores*clock)/clock
    hbm=target.hbm_bandwidth_gbs/clock
    edge=l2_rate*clock/(target.cube_tops['16']*1000)
    ratio=l2_rate/hbm
    edge+=max(1,2*(m+n)*k/l2)*(1-ratio)*(1/m+1/n)-(1+ratio)/k
    if 1/coarse_m+1/base_n<=.85*edge:
        return {**unavailable,'reason':'Initial ASWT geometry is already on the Cube-bound side of the host gate.'}
    # DoBL1FullLoad restricts the M tile for two FP32 L0C output buffers.
    # The A and B operands also need two L0 slots, even for a resident B.
    remaining=l1-weight
    max_m=min(remaining//(4*base_k), l0c//(8*base_n))//16*16
    bm=min(coarse_m,max_m,16*ceil(m/(16*cores)))
    if not bm or m%bm or k%base_k:
        return {**unavailable,'reason':'Ragged outer M or K tile not priced by this candidate.'}
    limit=remaining//(4*base_k*bm)
    steps=min(ceil(k/base_k),limit,4)
    if steps==limit and (steps*base_k)%256 and 8*base_k*base_n<=l0b:
        # The optional doubled base-K variant needs its own verified schedule.
        return {**unavailable,'reason':'A different doubled base-K branch is needed.'}
    if steps<1 or 4*bm*base_k>l0a or 4*n*base_k>l0b:
        return {**unavailable,'reason':'Insufficient local double-buffer capacity.'}
    step=1
    while step<steps and (2*bm*base_k*step<48*1024 or base_k*step%128):
        step+=1
    page=base_k*step
    if k%page:
        return {**unavailable,'reason':'A partial last GM read page needs separate pricing.'}

    values=_costs()
    def price(key):
        return det.get('matmul_'+key) or values[key]

    def port(key, default_cycles):
        rate=det.get(key+'_bandwidth_gbs')
        return rate/clock if rate else default_cycles

    # The largest 950DT trace delivered >=63.75 A bytes/cycle on its critical
    # AIC. 64 is a rounded *effective scenario*, not a measured port maximum.
    mte2=port('mte2',64)
    mte1=port('mte1',price('mte1_bytes_per_cycle'))
    fixp=port('mte3',price('write_bytes_per_cycle'))
    a_tile=2*bm*k
    a_page=2*bm*page
    output_tile=2*bm*n
    tiles=m//bm;active=min(cores,tiles);waves=ceil(tiles/active)
    mmads=k//base_k
    per_cycle=target.cube_tops['16']*1000/(clock*cores)
    issue=ceil(2*bm*n*base_k/per_cycle)
    compute_tile=mmads*issue
    l0_tile=2*(bm+n)*k
    per_aic_step=max(compute_tile,a_tile/mte2,l0_tile/mte1,output_tile/fixp)
    output=2*m*n
    # Minimum cold off-chip traffic for non-aliasing A/B/C and an initially cold
    # input. At most the whole L2 can hold dirty C at kernel return. Write
    # sectors may evacuate earlier; report the all-write-through sensitivity too.
    spill=max(0,output-l2)
    start=price('launch_cycles')+price('arrival_log2_cycles')*log2(active)
    l0_first=price('l0_latency_cycles')+2*(bm+n)*base_k/mte1
    first_fill=price('read_latency_cycles')+max((weight+a_page)/mte2,active*(weight+a_page)/l2_rate,(weight+active*a_page)/hbm)
    first_cube=start+first_fill+l0_first
    # Descriptor response extends 24 cycles beyond the initiation interval on
    # all three development tails. This is a terminal pipeline drain, once.
    cube_tail=24
    final_write=price('drain_cycles')+output_tile/fixp
    core_finish=first_cube+(waves-1)*per_aic_step+compute_tile+cube_tail
    l2_bytes=tiles*a_tile+active*weight+output
    l2_finish=start+price('read_latency_cycles')+l2_bytes/l2_rate
    first_to_last_page=step*issue+cube_tail
    def memory_finish(writeback):
        return start+price('read_latency_cycles')+(2*m*k+weight+writeback)/hbm+l0_first+first_to_last_page
    hbm_finish=memory_finish(spill)
    fronts={'core':core_finish, 'l2':l2_finish,'hbm':hbm_finish}
    end=max(fronts.values())+final_write
    scale=1000*clock
    return dict(status='conditional',route='b_full_l1',latency_us=end/scale,
                all_output_writeback_us=(max(core_finish,l2_finish,memory_finish(output))+final_write)/scale,
                bottleneck=max(fronts,key=fronts.get),target_extrapolated=target not in (load_chip(REFERENCE_NAME),load_chip('Ascend950PR_957b')),
                shape=tuple(shape),tile=(bm,n,base_k),a_page_k=page,active_aic=active,waves=waves,
                critical_core_tiles=waves,cube_ops=2*m*n*k,mmad_total=tiles*mmads,mmads_per_tile=mmads,
                work=dict(chip_l2_interface_bytes=l2_bytes,logical_b_bytes=weight,b_lower_link_bytes=active*weight,
                          a_read_bytes=2*m*k,final_bf16_store_bytes=output,fixp_fp32_source_bytes=2*output,
                          minimum_cold_hbm_bytes=2*m*k+weight+spill,all_store_cold_hbm_bytes=2*m*k+weight+output,
                          per_tile_a_bytes=a_tile,per_tile_l0_bytes=l0_tile,per_tile_store_bytes=output_tile),
                cycles=dict(launch=start,first_fill=first_fill,l0_first=l0_first,first_cube=first_cube,
                            cube_issue=issue,compute_tile=compute_tile,per_aic_step=per_aic_step,
                            response_after_last_issue_interval=cube_tail,final_write=final_write,
                            chip_memory_finish=hbm_finish,core_finish=core_finish,l2_finish=l2_finish,total=end),
                rates=dict(bf16_ops_per_aic_cycle=per_cycle,mte2_bytes_per_aic_cycle=mte2,mte1_bytes_per_aic_cycle=mte1,
                           fixp_destination_bytes_per_aic_cycle=fixp,hbm_bytes_per_chip_cycle=hbm,l2_bytes_per_chip_cycle=l2_rate),
                assumption='One aligned output rectangle along N, qualified geometry family on 950 only. B is loaded once per participating AIC, cold A and B, minimum compulsory dirty C eviction. Scope assumes the same ASWT implementation on future chips.')
