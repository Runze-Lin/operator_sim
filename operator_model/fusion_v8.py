"""Compressed resource overlap templates for installed 950DT kernel routes."""
from math import ceil
from statistics import median

PRE='mhc_pre'
FUSED='mhc_pre_sinkhorn'
IDX='lightning_indexer'
FIRST='grouped_matmul_swiglu_quant_v2'
SECOND='gmm2'
CHAIN='moe_ffn'


def work(op, shape, attrs):
    """Critical-wave resource work in logical tile units (not ISA counts)."""
    if any(type(d) is not int or d<=0 for d in shape):
        raise ValueError('Positive dimensions required')
    if op==IDX:
        b,q,l,g,d=shape
        if b!=1 or g!=8 or d!=128 or l-q+1<512:
            raise ValueError('Only verified dense BSND G8 D128 top512 route')
        waves=ceil(q/32)
        cube=waves*ceil(l/128)*ceil(g/16)
        vector=waves*ceil(l/512)
        read=waves*ceil(l/128)
    elif op in (FIRST,SECOND):
        m,k,f,e=shape
        counts=attrs.get('expert_counts',())
        if len(counts)!=e or any(type(x)is not int or x<0 for x in counts) or sum(counts)!=m or k%64 or f%64:
            raise ValueError('Expected all valid rows, grouped ND MX and aligned K/F')
        output,reduction=(2*f,k) if op==FIRST else (k,f)
        waves=ceil(sum(ceil(x/16) for x in counts)*ceil(output/128)/32)
        cube=waves*ceil(reduction/64)
        vector=ceil(sum(ceil(x*f/256) for x in counts)/64)
        read=cube
    else:
        t,n,d=shape
        if n!=4 or d%16 or (op==FUSED and d!=4096):
            raise ValueError('Unsupported mHC smoke contract')
        waves=ceil(t/(32 if op in (PRE,FUSED) else 64))
        cube=waves*ceil(n*d/128) if op in (PRE,FUSED) else 1
        vector=waves*(1 if op=='split_sink' else ceil(d/128))
        read=vector
    return {'CUBE':cube,'MTE1':cube,'FIXP':waves,'SIMD':vector,
            'MTE2':read,'MTE3':vector,'SCALAR':waves,'IDLE':waves}


def stage_prediction(op,shape,attrs,model):
    demand=work(op,shape,attrs)
    predictions=[]
    for ref in model['templates'][op]:
        refs=ref['work']
        domains=[]
        for phases in ref['cores']:
            # Each atom is an interval with a constant set of active queues.
            # Coactive queues can overlap for the duration of their slowest job.
            duration=sum(us*max(demand.get(q,1)/refs.get(q,1) for q in mask.split('+'))
                         for mask,us in phases.items())
            domains.append(duration)
        predictions.append(max(domains))
    return median(predictions)


def predict(op,shape,attrs,model):
    if op not in (IDX,PRE,FUSED,'mhc',FIRST,SECOND,CHAIN):raise ValueError('Unknown fusion route')
    required=model['contracts'][op]
    for key,value in attrs.items():
        if key=='expert_counts' and op in (FIRST,SECOND,CHAIN):continue
        if key not in required or value!=required[key]:raise ValueError('Unsupported attribute '+key)
    stages=(PRE,'split_sink','split_post') if op=='mhc' else ((FIRST,SECOND) if op==CHAIN else (op,))
    times={s:stage_prediction(s,shape,attrs,model) for s in stages}
    return {'predicted_us':sum(times.values()),'stages_us':times,'method':'pipeline_overlap_template',
            'scope':'950DT CA kernel instruction windows; macro-tile partition is a source-informed hypothesis'}
