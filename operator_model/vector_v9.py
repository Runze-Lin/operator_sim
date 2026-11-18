"""950DT critical-AIV timing using separate dispatch, lane and shared-byte work."""
from functools import lru_cache
import json
from math import ceil, log2
from pathlib import Path
from .vector_v8 import features as validate_v8

PHASES = ('arrival','SCALAR','SIMD','MTE2','MTE3','other')


@lru_cache(maxsize=1)
def parameters():
    return json.loads((Path(__file__).resolve().parents[1]/'results/v0.9/vector/profile.json').read_text())


def schedule(op, shape, attrs):
    validate_v8(op,shape,attrs)
    tokens = shape[0]
    rows = ceil(tokens / 64)
    slices = 1
    route = op
    if op == 'rmsnorm':
        width = shape[1]
        work = rows * ceil(width/128)
        local_kib = rows * width * 2 / 1024
        shared_kib = tokens * width * 2 / 1024
    elif op == 'mhc_sinkhorn':
        # The observed loop vectorizes independent four-by-four matrices together.
        iterations = attrs.get('num_iters',20)
        work = ceil(rows * 16 / 64) * iterations
        local_kib = rows * 16 * 4 / 1024
        shared_kib = tokens * 16 * 4 / 1024
    else:
        _, n, width = shape
        route = 'post_token' if tokens >= 32 else 'post_channel'
        if route == 'post_channel':
            while ceil(width/slices)>256 and tokens*slices<=32:
                slices *= 2
        local_width = ceil(width/slices)
        work = rows * ceil(local_width/64) * n * (n+1)
        local_kib = rows * local_width*n*2 / 1024
        shared_kib = tokens * width*n*2 / 1024
    active = ceil(tokens/rows)*slices
    units = {'SIMD':work, 'MTE2':local_kib, 'MTE3':shared_kib,
             'arrival':log2(active), 'SCALAR':rows-1, 'other':0.0}
    return route, active, rows, units


def predict(op, shape, attrs=None):
    attrs = dict(attrs or {})
    models = parameters()
    try:
        if op not in models['contracts']:
            raise ValueError('Unsupported vector route')
        for key,value in attrs.items():
            if op=='mhc_sinkhorn' and key=='num_iters':
                if type(value) is not int: raise ValueError('Expected integer iterations')
                continue
            if models['contracts'][op].get(key,object())!=value:
                raise ValueError('Unsupported attribute '+key)
        route,active,rows,units = schedule(op,shape,attrs)
        stages = {}
        for phase,(intercept,slope) in models['phases'][route].items():
            stages[phase] = intercept + slope * units[phase]
        return dict(predicted_us=sum(stages.values()),stages_us=stages,predicted_active_aiv=active,
                    rows_per_core=rows,route=route,status='950DT critical-AIV estimate')
    except (KeyError,TypeError,ValueError) as exc:
        return dict(predicted_us=None,status='unsupported',reason=str(exc))
