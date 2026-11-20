"""Experimental BF16 split-K after reselecting output rectangles for a single wave."""
from math import ceil, log2

from operator_model.chips import load_chip
from operator_model.operators.matmul.resources import _costs
from operator_model.operators.matmul.tiling import matmul_plan
from operator_model.operators.matmul.split_k import estimate_matmul_splitk


def estimate_matmul_retile(shape, *, chip='Ascend950DT_9582'):
    """Return a resource candidate for contiguous BF16 ND, using only target config."""
    target=load_chip(chip)
    current=estimate_matmul_splitk(shape,chip=target)
    ordinary=current['ordinary_us']
    if current['route']!='ordinary' or not 1 < current['output_tiles'] <= target.cube_cores:
        return current
    m,n,k=shape
    if ceil(k/256)*256 < current['threshold_aligned_k'] or not target.vector_tops['32']:
        return current
    details=target.details; cost=_costs(); clock=target.frequency_ghz
    l0a=details.get('l0a_bytes') or 65536
    l0b=details.get('l0b_bytes') or 65536
    l0c=details.get('l0c_bytes') or 262144
    l1=details.get('l1_bytes') or 524288
    # A/B double buffers contain at least one aligned, 64-deep BF16 MMA.
    max_m=(l0a//(4*64))//16*16
    max_n=(l0b//(4*64))//16*16
    if min(max_m,max_n)<16:return current
    mb,nb=ceil(m/max_m),ceil(n/max_n)
    am=16*ceil(m/(16*mb));an=16*ceil(n/(16*nb))
    if 4*am*an>l0c:return current
    groups=mb*nb;cores=target.cube_cores
    splits=min(cores//groups,ceil(k/64))
    if splits<2:return current
    segment=ceil(k/splits);tail=k-segment*(splits-1)
    if tail<=0:return current
    bk=64*(min(l0a//(4*am),l0b//(4*an))//64)
    if not bk:return current
    # One buffered GM page can cover several successive L0/MMAD descriptors.
    # Repeated pages overlap in pairs, following the ordinary MatMulV3 flow shop.
    page=(min(l1//(4*(am+an)),l1//(8*max(am,an)))//bk)*bk
    if not page:return current
    mm=min(am,m);nn=min(an,n)
    def ticks(key):return details.get('matmul_'+key) or cost[key]
    def per_core(name,key):return details[name]/clock if details.get(name) else cost[key]
    mte2=per_core('mte2_bandwidth_gbs','mte2_bytes_per_cycle')
    mte1=per_core('mte1_bandwidth_gbs','mte1_bytes_per_cycle')
    fixp=per_core('mte3_bandwidth_gbs','write_bytes_per_cycle')
    peak=(target.cube_tops['16'] or cost['default_bf16_ops_per_cycle']*clock*cores/1000)*1000/(cores*clock)
    def block(depth):
        frag=min(depth,bk);last=(depth-1)%bk+1;pieces=ceil(depth/bk)
        cube=lambda d:2*am*an*(16*ceil(d/16))/peak
        read=2*depth*max(mm,nn)/mte2
        lead=2*frag*max(mm,nn)/mte1
        mac=(pieces-1)*max(ticks('mmad_ii_cycles')+cube(frag),lead)+ticks('mmad_latency_cycles')+cube(last)
        return read,lead,mac
    rd,ld,compute=block(min(segment,page));end=block((segment-1)%page+1)[2]
    ii=max(2*ticks('read_ii_cycles'),rd,(ticks('read_latency_cycles')+rd)/2,compute)
    body=(ticks('read_latency_cycles')+rd+ticks('l0_latency_cycles')+ld+
          (ceil(segment/page)-1)*ii+end+ticks('drain_cycles')+4*mm*nn/fixp)/(clock*1000)
    launch=(ticks('launch_cycles')+cost['arrival_log2_cycles']*log2(cores))/(clock*1000)
    hbm_bytes=2*k*(m+n)
    l2_read=2*k*(m*nb+n*mb)
    l2_bw=details.get('l2_bandwidth_gbs') or clock*100*cores
    hbm_floor=hbm_bytes/(1000*target.hbm_bandwidth_gbs)
    l2_floor=l2_read/(1000*l2_bw)
    vec_rate=target.vector_tops['32']*1000/(target.vector_cores*clock)
    merge_issue=(1000+60*splits*256/vec_rate)/(clock*1000)
    merge_bytes=2*m*n*(4*splits-1)
    merge_us=merge_issue+merge_bytes/(1000*l2_bw)
    total=launch+max(body,hbm_floor,l2_floor)+merge_us
    # The host selects a template using effective bandwidth versus participating
    # AICs before its detailed issue/retirement time can be modeled.
    baseline=matmul_plan(shape,target);om,on,_=baseline['tile']
    def admission(bm,bn,num_splits,used):
        shared=used/cores
        rep=2*k*(m*ceil(n/bn)+n*ceil(m/bm))-hbm_bytes
        io=hbm_bytes/(1000*target.hbm_bandwidth_gbs*shared)+rep/(1000*l2_bw*shared)
        compute=2*bm*bn*16*ceil(k/(16*num_splits))/(1000*clock*peak)
        return max(io,compute)
    old_price=admission(om,on,1,baseline['active_aic'])
    used=groups*splits
    candidate_price=admission(am,an,splits,used)+(merge_issue+merge_bytes/(1000*target.hbm_bandwidth_gbs))*cores/used
    if candidate_price>=old_price:
        return {**current,'retile_candidate_us':total,'host_prices_us':[old_price,candidate_price]}
    return dict(route='retiled_output_split_k',latency_us=total,ordinary_us=ordinary,
        original_output_tiles=current['output_tiles'],output_tiles=groups,base_m=am,base_n=an,
        host_prices_us=[old_price,candidate_price],
        m_tiles=mb,n_tiles=nb,k_splits=splits,max_segment_k=segment,tail_segment_k=tail,
        base_k=bk,page_k=page,productive_aic=groups*splits,active_aic=cores,active_aiv=target.vector_cores,
        mmads_total=groups*((splits-1)*ceil(segment/bk)+ceil(tail/bk)),cube_ops=2*m*n*k,
        logical_reduction_adds=m*n*(splits-1),input_bytes=hbm_bytes,l2_to_aic_input_bytes=l2_read,
        partial_fixup_bytes=merge_bytes,final_output_bytes=2*m*n,
        launch_us=launch,partial_pipeline_us=body,cold_floor_us=hbm_floor,l2_floor_us=l2_floor,
        merge_us=merge_us,merge_issue_us=merge_issue,threshold_aligned_k=current['threshold_aligned_k'],
        evidence='Retiled one-wave resource candidate; 970 assumes compatible mixed implementation; no target CA')
