"""Compose repeated double-buffer updates in max-plus algebra for long UB loops."""
import math
from .model import ceildiv, line
from .model_v2 import plan

NEG = -math.inf
IDENTITY = tuple(tuple(0. if i==j else NEG for j in range(5)) for i in range(5))

def compose(after, before):
    return tuple(tuple(max(x+y for x,y in zip(row,column)) for column in zip(*before)) for row in after)

def repeat(matrix, n):
    result=IDENTITY
    while n:
        if n&1:
            result=compose(matrix,result)
        n//=2
        if n:
            matrix=compose(matrix,matrix)
    return result

def price(predictor, case):
    p=plan(case)
    if p['iterations']<256:
        return predictor.predict_us(case,use_cache=False)
    costs=predictor.quant_model['quant']['costs']
    prefix=line(costs['prefix'],math.log2(1+p['active_aiv']))
    drain=costs['write_drain_us']
    rows,cols=p['rows_per_core'],p['cols_per_core']
    tr,tc=p['tile']

    def tile(height,width):
        kib=2*height*width/1024
        groups=height*ceildiv(width,32)
        latency=line(costs['first_read'],kib)
        interval=line(costs['read_interval'],kib)
        service=line(costs['vector'],ceildiv(groups,8)+ceildiv(groups,128))
        # State holds read, vector, write completion, then vector/write one tile earlier.
        return ((interval,NEG,NEG,latency,NEG),
                (interval+service,service,NEG,latency+service,service),
                (interval+service+drain,service+drain,drain,latency+service+drain,service+drain),
                (NEG,0.,NEG,NEG,NEG),(NEG,NEG,0.,NEG,NEG))
    def row(height):
        n,tail=divmod(cols,tc)
        matrix=repeat(tile(height,tc),n)
        return compose(tile(height,tail),matrix) if tail else matrix
    # Seed a virtual previous read so the first read completes at prefix+latency.
    kib=2*min(rows,tr)*min(cols,tc)/1024
    initial=prefix+line(costs['first_read'],kib)-line(costs['read_interval'],kib)
    n,tail=divmod(rows,tr)
    matrix=repeat(row(tr),n)
    if tail:
        matrix=compose(row(tail),matrix)
    body=initial+matrix[2][0]+costs['suffix_us']
    floor=line(costs['control_floor'],min(predictor.quant_model['control_log2_cap'],math.log2(1+p['active_aiv'])))
    return max(body,floor)
