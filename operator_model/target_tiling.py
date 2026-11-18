"""Capacity-dependent dispatch using the 950 implementation policy."""
from .model import align, ceildiv, quant_partition


def quant_plan(shape, capacity, ub_bytes):
    rows, cols = shape
    core = quant_partition(rows, cols, capacity)
    # Reserve 2 KiB of UB then fit eight block32 groups per buffer quantum.
    max_groups = ((ub_bytes-2048)//(256*6+8*2*5))*8
    if max_groups < 2:
        raise ValueError('UB capacity is insufficient for a quant tile')
    groups = ceildiv(core['cols_per_core'],32)
    col_loops = ceildiv(groups,max_groups)
    tile_groups = align(ceildiv(groups,col_loops),2)
    row_loops = ceildiv(core['rows_per_core'],max_groups//tile_groups)
    tile_rows = ceildiv(core['rows_per_core'],row_loops)
    elements = tile_rows*tile_groups*32
    return {**core, 'active_aic':0, 'tile':[tile_rows,tile_groups*32],
            'iterations':row_loops*col_loops, 'waves':1,
            'units':{'read':2*elements/1024,
                     'vector':ceildiv(tile_rows*tile_groups,8)+ceildiv(tile_rows*tile_groups,128)}}


def matmul_plan(shape, capacity):
    m,n,k = shape
    tm,tn = 16,64
    while ceildiv(m,tm)*ceildiv(n,tn)>capacity and (tm<256 or tn<256):
        candidates = []
        for a,b in ((2*tm,tn),(tm,2*tn)):
            if a>256 or b>256:
                continue
            count=ceildiv(m,a)*ceildiv(n,b)
            candidates.append((abs(count-capacity),count*a*b,a+b,a,b))
        _,_,_,tm,tn = min(candidates)
    jobs = ceildiv(m,tm)*ceildiv(n,tn)
    em,en,ek = min(m,tm),min(n,tn),align(k,16)
    branch = 'gemv' if m==1 else 'k_tail' if k%16 else 'coarse_tile' if (tm,tn)!=(16,64) else 'dense'
    return {'active_aic':min(jobs,capacity),'active_aiv':0,'tile':[tm,tn,256],
            'waves':ceildiv(jobs,capacity),'iterations':1,'branch':branch,
            'units':{'read':2*(em*ek+en*ek)/1024,'cube':2*em*en*ek/131072,
                     'fix':2*em*en/1024,'vector':0.,'write':0.}}
