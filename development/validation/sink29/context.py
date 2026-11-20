"""Evaluate source-derived issue schedules under a bounded, observable AIV prefix."""
from math import ceil
from operator_model import estimate


def envelope(shape, iterations, chip, minimum_resident=0):
    """A causal previous high-water lower bound, unknown earlier stream or freshly attached."""
    t,n=shape
    if type(minimum_resident) is not int or not 0<=minimum_resident<=chip.vector_cores:
        raise ValueError('invalid target-AIV prefix')
    attrs={'num_iters':iterations}
    default=estimate('mhc_sinkhorn',shape,attrs,chip=chip,dtype='fp32')
    R=ceil(t/chip.vector_cores);A=ceil(t/R)
    # Use the existing calibrated single-register public state schedules. Respect
    # their shape-only domain. Missing timing always propagates as unknown.
    if default.latency_us is None or n!=4 or R>8:
        return dict(point_us=None,old_us=default.latency_us,states={},minimum_resident=minimum_resident)
    times={}
    for lo,hi in ((0,0),(1,A-1),(A,chip.vector_cores)):
        h=max(lo,minimum_resident)
        if h>hi: continue
        x=estimate('mhc_sinkhorn',shape,attrs|{'resident_aiv':h},chip=chip,dtype='fp32')
        if x.latency_us is None: raise ValueError(x.reason)
        times[x.projection['target_plan']['history_state']]=x.latency_us
    l,u=min(times.values()),max(times.values())
    # Use the v0.28 interval/risk machinery but preserve the old public point
    # whenever it already meets every modeled state's symmetric error budget.
    # This is a pricing objective, not a guarantee about the CA residual or WAPE.
    frac=max(.25,(u-l)/(u+l))
    left=(1-frac)*u;right=(1+frac)*l
    point=min(max(default.latency_us,left),right)
    if len(times)==1:point=l
    return dict(point_us=point,old_us=default.latency_us,states=times,resource_range_us=[l,u],
                tolerance=frac,robust_feasible_us=[left,right],minimum_resident=minimum_resident)


def scenario_mix(states, probabilities):
    """Caller-sourced deployment mixture; expectation of a batch of such dispatches."""
    if set(states)!=set(probabilities) or any(not 0<=p<=1 for p in probabilities.values()) or abs(sum(probabilities.values())-1)>1e-9:
        raise ValueError('provide one normalized external probability per reachable state')
    return sum(t*probabilities[k] for k,t in states.items())
