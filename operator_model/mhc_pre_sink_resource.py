"""Source-partitioned BF16 Pre + Sinkhorn CV pipeline for the 950 inference ELF."""
from math import ceil

from .chips import load_chip
from .fusion_v8 import FUSED, work
from .mhc_sink_resource import compute_scale, fused as template_fused, partition


# Costs of diagnostic intervals in 950DT clock cycles. The 1-row absolute
# launch anchor comes from the existing three queue-overlap templates.
# results/v0.26/presink/prices.json retains the development PCs and intervals.
CAST_CYCLES_PER_ELEMENT = 0.10070278418564925
MMAD_ISSUE_CYCLES = 76.52407095634511
MMAD_CYCLES_PER_KMAC = 0.49683638109334943
POST_CYCLES_PER_ROW = 3299.000005563721


def cube_fragments(m, k, outputs):
    """Instruction count on one full AIC K-split, including each L1 tail."""
    aligned_m = 16 * ceil(m / 16)
    aligned_n = 16 * ceil(outputs / 16)
    l1_k = (min(32768 // aligned_m, 1024) // 128) * 128
    l0_k = (256 // max(aligned_m, aligned_n)) * 32
    pages, tail = divmod(k, l1_k)
    count = pages * ceil(l1_k / l0_k) + ceil(tail / l0_k)
    return count, l1_k, l0_k


def _cost(shape, target, plan, ref):
    t,n,d=shape
    _,k=plan['tile']
    outputs=n*n+2*n
    count,l1,l0=cube_fragments(t,k,outputs)
    ratio=target.vector_cores/target.cube_cores
    elements=ceil(t/ratio)*k
    simd=compute_scale(ref,target,'vector','32')/(1000*ref.frequency_ghz)
    # Stage 1 converts BF16 to FP32 and reduces along the local K partition.
    # When the target has a known slower DMA port, the streaming lower bound
    # remains active. Inherited 950 rates are included in the measured price.
    read_bw=target.details.get('mte2_bandwidth_gbs')
    cast=max(elements*CAST_CYCLES_PER_ELEMENT*simd, 2*elements/(1000*read_bw) if read_bw else 0)
    aligned_mac=16*ceil(t/16) * (16*ceil(outputs/16)) * k
    issue=count*MMAD_ISSUE_CYCLES/(1000*target.frequency_ghz)
    compute=aligned_mac/1024*MMAD_CYCLES_PER_KMAC*compute_scale(ref,target,'cube','16')/(1000*ref.frequency_ghz)
    rows=plan['rows_per_aiv']
    return dict(cast=cast,projection=issue+compute,post=rows*POST_CYCLES_PER_ROW*simd),dict(
        largest_k=k,critical_aiv_stage1_elements=elements,critical_aic_mmad=count,
        l1_k=l1,l0_k=l0,critical_aic_padded_mac=aligned_mac,stage2_rows=rows)


def fused(shape,target,attrs):
    work(FUSED,shape,attrs)
    plan=partition(shape,target)
    # The current public arch35 tiler reserves a distinct Sinkhorn worker group
    # from 128 tokens on, with different synchronization. No marked windows in
    # this development set cover that branch, or the 1:1 MIX binary.
    if shape[0]>=128 or target.vector_cores!=2*target.cube_cores:
        us,aic,aiv,diag=template_fused(shape,target,attrs)
        diag['schedule']='prior_overlap_template'
        diag['qualification']='Separate Sinkhorn team or alternative C:V template; per-stage timing unverified.'
        return us,aic,aiv,diag
    ref=load_chip()
    anchor_shape=(1,4,4096)
    anchor_us,_,_,_=template_fused(anchor_shape,target,attrs)
    base,_=_cost(anchor_shape,target,partition(anchor_shape,target),ref)
    cost,details=_cost(shape,target,plan,ref)
    fixed=max(0.,anchor_us-sum(base.values()))
    stages=dict(dispatch_and_handoff=fixed,stage1_cast=cost['cast'],stage1_cube=cost['projection'],stage2_rows=cost['post'])
    t,n,d=shape
    io=(2*t*n*d+4*n*d*(n*n+2*n)+2*t*d)/(1000*target.hbm_bandwidth_gbs)
    stages['exposed_hbm_floor']=max(0.,io-sum(stages.values()))
    return sum(stages.values()),plan['active_aic'],plan['active_aiv'],dict(
        stages_us=stages,target_plan={**plan,**details},schedule='k_split_cv2_joint_sinkhorn',
        launch_anchor_us=anchor_us,
        qualification='Combined control and dependency cost anchored on a one-row launch. HF32 per-AIC rate uses the 16-bit peak ratio as a proxy.')
