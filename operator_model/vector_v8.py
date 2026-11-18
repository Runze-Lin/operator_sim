"""Critical AIV trace template. Stage costs use disjoint intervals of its timeline."""
from math import ceil, log2


def features(op,shape,attrs):
    """Return core count, local loop work and chip-wide traffic pressure."""
    if not shape or any(type(v) != int or v <= 0 for v in shape):
        raise ValueError('Expected positive integer dimensions')
    if op == 'rmsnorm':
        r,c=shape
        if c>8192 or c%16 or r>512:raise ValueError('Outside full-row BF16 template')
        active=min(64,r)
        waves=ceil(r/active)
        units=waves*ceil(c/128)
    elif op=='mhc_post':
        t,n,d=shape
        if n!=4 or d%16 or d>8192 or t>256:raise ValueError('Outside N4 BF16 template')
        pieces=1
        while ceil(d/pieces)>256 and (t*pieces<=32 or ceil(d/pieces)>=253952/24):
            pieces*=2
        active=min(t*pieces,64);waves=ceil(t*pieces/active)
        units=waves*ceil(d/pieces/64)*n*(n+1)
    elif op=='mhc_sinkhorn':
        t,n=shape
        if n!=4 or t>256 or not 1<=attrs.get('num_iters',20)<=100:raise ValueError('Outside N4 inference FP32 template')
        active=min(t,64);waves=ceil(t/active)
        units=waves*attrs.get('num_iters',20)*n*n
    elif op=='kda_recurrent':
        b,s,hq,hv,k,v=shape
        if (b,hq,hv,k)!=(1,1,1,128) or v not in (128,256) or s>8:raise ValueError('Outside recurrent template')
        active=1;waves=1;units=s*k*v/16384
    else:raise ValueError('Unknown vector op')
    return active,units,waves


def predict(op,shape,attrs,profile):
    try:
        expected=profile[op]['contract']
        given=dict(attrs or {})
        if op=='mhc_sinkhorn':given.pop('num_iters',None)
        if any(k not in expected or expected[k]!=v for k,v in given.items()):raise ValueError('Different caller attributes')
        active,units,waves=features(op,shape,attrs)
    except (ValueError,KeyError,TypeError) as exc:
        return {'predicted_us':None,'status':'unsupported','reason':str(exc)}
    stages={}
    for name,p in profile[op]['phases'].items():
        x=phase_work(name,active,units,waves,op)
        stages[name]=p['startup_us']+p['service_us']*x
    return {'status':'A5_vector_stage_template','predicted_us':sum(stages.values()),'stages_us':stages,
            'predicted_active_aiv':active,'waves':waves,'scope':'cross-core instruction makespan'}


def phase_work(name,active,units,waves,op):
    if name=='arrival':return log2(active)
    if name in ('MTE2','MTE3') and op=='mhc_sinkhorn':return waves*log2(1+active)
    if name in ('MTE2','MTE3'):return units*log2(1+active)
    return units
