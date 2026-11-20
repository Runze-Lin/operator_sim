"""950DT split mHC: local M/K tiles, output sharding and trace-priced resources."""
from operator_model.resources import load_parameters
from functools import lru_cache
from math import ceil


@lru_cache(maxsize=1)
def profile():
    return load_parameters('mhc_partition')

def partition(stage, shape):
    t,n,d = shape
    if n!=4 or t<1 or t>256 or d<16 or d>16384 or d%16:
        raise ValueError('Expected split mHC N4, T<=256, 16-aligned D<=16384')
    if stage=='pre':
        k=n*d
        mk=k>=512
        # Within T<=256 the reference 950 M-K grid puts one M tile on a group of 32 AICs.
        kc=256*ceil(k/(32*256)) if mk else min(k,256)
        mtiles=ceil(t/16)
        mm=mtiles*ceil(kc/32) if mk else ceil(t/32)*ceil(k/256)
        # Whole-chip demand proxies include weights K*(N*N+2N) and token inputs.
        return ('mk' if mk else 'nd'), dict(compute=mm, vector=ceil(t/64)*ceil(d/256),
            traffic=(4*k*(n*n+2*n)+2*t*k)/1024, control=mtiles)
    if stage=='sink':
        return 'sink',dict(compute=0,vector=ceil(t/64)*20*n*n,traffic=4*t*n*n/1024,control=ceil(t/64))
    # Split hidden dimension in powers of two only for short-token post path.
    j=1
    while t<32 and ceil(d/j)>256 and t*j<=32:
        j*=2
    waves=ceil(t*j/64)
    return ('row' if t>=32 else 'split'),dict(compute=0,vector=waves*ceil(d/(64*j))*n*(n+1),
        traffic=(4*t*n*d)/1024,control=waves)

def predict(op,shape,attrs=None):
    data=profile()
    if op not in ('mhc_pre','mhc'):
        raise ValueError('Expected split mHC or Pre')
    if any(data['contracts'][op].get(k)!=v or k not in data['contracts'][op] for k,v in (attrs or {}).items()):
        raise ValueError('Unvalidated mHC attributes')
    values={}
    for stage in (('pre',) if op=='mhc_pre' else ('pre','sink','post')):
        branch,work=partition(stage,shape)
        cost=data['stages'][stage][branch]
        values[stage]=sum(c['startup_us']+c['unit_us']*work[c['unit']] for c in cost.values())
    return dict(predicted_us=sum(values.values()), stages_us=values)
