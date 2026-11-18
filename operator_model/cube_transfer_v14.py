"""Finite BF16 MatMulV3 pipeline, distinguishing descriptor waits from exposed work."""
from math import ceil, log2
from .chips import load_chip
from .model import line
from .predictor import _profile, predict
from .target_tiling import matmul_plan


def _same_hardware(a,b):
    return all(getattr(a,k,None)==getattr(b,k,None) for k in (
        'name','frequency_ghz','cube_cores','vector_cores','hbm_capacity_gb','hbm_bandwidth_gbs',
        'cube_tops','vector_tops','details'))


def _window(shape,p,chip,ref,f):
    costs=_profile('results/v0.14/cube-phases.json')
    m,n,k=shape; tm,tn,_=p['tile']; m1,n1=min(m,tm),min(n,tn)
    pages=ceil(k/512); first=min(512,k); tail=(k-1)%512+1
    # ND->NZ issues A and B back-to-back. The slower descriptor wins the fill race.
    desc=lambda depth:2*max(m1,n1)*depth/1024
    r,t=ref.details.get('mte2_bandwidth_gbs'),chip.details.get('mte2_bandwidth_gbs')
    clock=f['SCALAR']; local_ratio=r/t if r and t else clock
    def price(name,units,ratio):
        cost=costs[name];return cost['intercept_us']*clock + cost['slope_us_per_unit']*units*ratio
    prefix=line(costs['prefix'],log2(1+p['active_aic']))*clock
    fill=price('pair',desc(first),local_ratio)
    l0=price('l0',2*(m1+n1)*min(256,k)/1024,f['MTE1'])
    compute=lambda depth:price('compute',2*m1*n1*depth/131072,f['CUBE'])
    period=max(compute(first),costs['repeat_us_per_max_descriptor_kib']*desc(first)*local_ratio)
    drain=costs['drain']['intercept_us']*f['FIXP']
    stage=dict(prefix_us=prefix,first_pair_us=fill,l0_us=l0,page_intervals_us=(pages-1)*period,
               last_compute_us=compute(tail),drain_us=drain)
    # Additional output waves overlap admission with the preceding page / output tail.
    stage['extra_waves_us']=(p['waves']-1)*(fill+(pages-1)*period+max(compute(tail),l0,drain))
    total=sum(stage.values()); bw=chip.hbm_bandwidth_gbs
    # Scenario floor assumes one compulsory pass over logical BF16 tensors. No miss counter.
    hbm=2*(m*k+k*n+m*n)/(bw*1000) if bw else 0.
    return max(hbm,total), dict(stage,logical_hbm_floor_us=hbm,read_pages=pages)


def project_matmul(shape,factors,reference,target,target_plan):
    """Price exposed stage landmarks at target rates. Retain DT's level for uncalibrated targets."""
    projected,diagnostics=_window(shape,target_plan,target,reference,factors)
    if _same_hardware(reference,load_chip('Ascend950DT_9582')) and _same_hardware(target,load_chip('Ascend950PR_957b')):
        return projected,dict(diagnostics,mode='paired_family_landmarks')
    unit={q:1. for q in ('SCALAR','MTE1','CUBE','FIXP')}
    ref_window,_=_window(shape,matmul_plan(shape,reference.cube_cores),reference,reference,unit)
    base=predict('matmul',tuple(shape))['predicted_us']
    # Reference discrepancy is dimensionless. Applying a ratio preserves positivity,
    # identity, and homogeneous clock scaling without pricing old FIXP waits twice.
    result=max(projected/ref_window*base,diagnostics['logical_hbm_floor_us']) if projected!=ref_window else base
    return result,dict(diagnostics,mode='reference_anchored_landmarks',reference_window_us=ref_window,
                       reference_level_us=base)
