"""Map quantization, MMAD and attention dimensions to reference-device tiles."""
import math

from operator_model.profiles.stage_model import align, ceildiv, quant_partition, schedule


def plan(case):
    if case.op=='quant':
        r,c=case.shape;s=quant_partition(r,c)
        # BF16->FP8 block32 DB capacity from documented 950 host formula.
        max_groups=((253952-2048)//(256*6+8*2*5))*8
        col_groups=ceildiv(s['cols_per_core'],32)
        col_loops=ceildiv(col_groups,max_groups)
        groups_per_col=align(ceildiv(col_groups,col_loops),2)
        row_loops=ceildiv(s['rows_per_core'],max_groups//groups_per_col)
        rows_per_ub=ceildiv(s['rows_per_core'],row_loops)
        groups=rows_per_ub*groups_per_col
        main_loops=ceildiv(groups,8);scale_loops=ceildiv(groups,128)
        iterations=row_loops*col_loops
        elems=groups*32
        branch='ub_loop' if iterations>1 else ('slice512' if s['slice_cols']==512 else 'slice256')
        return {**s,'active_aic':0,'waves':1,'iterations':iterations,'branch':branch,
                'tile':[rows_per_ub,groups_per_col*32],
                'units':{'read':2*elems/1024,'write':(elems+groups)/1024,
                         'vector':main_loops+scale_loops,'cube':0.,'fix':0.},
                'opcode_features':[iterations,iterations*main_loops,iterations*scale_loops],
                'useful_work':r*c,'logical_bytes':r*c*(3+1/32),'work_per_core':s['rows_per_core']*s['cols_per_core']}
    if case.op=='matmul':
        m,n,k=case.shape;tm,tn=16,64
        # Balance elementary output tiles when the grid exceeds 32 AICs.
        # Source-informed hypothesis, checked with actual MMAD shapes/core counts.
        while ceildiv(m,tm)*ceildiv(n,tn)>32 and (tm<256 or tn<256):
            options=[]
            for a,b in ((tm*2,tn),(tm,tn*2)):
                if a>256 or b>256:continue
                count=ceildiv(m,a)*ceildiv(n,b)
                options.append((abs(count-32),ceildiv(m,a)*a*ceildiv(n,b)*b,a+b,a,b))
            _,_,_,tm,tn=min(options)
        tiles=ceildiv(m,tm)*ceildiv(n,tn);active=min(32,tiles);waves=ceildiv(tiles,active)
        em,en=min(m,tm),min(n,tn);ek=align(k,16)
        branch='gemv' if m==1 else ('k_tail' if k%16 else ('coarse_tile' if (tm,tn)!=(16,64) else 'dense'))
        return {'active_aic':active,'active_aiv':0,'waves':waves,'iterations':1,'branch':branch,
                'tile':[tm,tn,256],'units':{'read':2*(em*ek+en*ek)/1024,'cube':2*em*en*ek/131072,
                                         'fix':2*em*en/1024,'vector':0.,'write':0.},
                'opcode_features':[waves*ceildiv(k,256)],
                'useful_work':2*m*n*k,'logical_bytes':2*(m*k+k*n+m*n),'work_per_core':2*em*en*ek*waves}
    q,kv,d,b,h=case.shape
    qbase=64 if b*h*ceildiv(q,128)<32 else 128
    kvbase=256 if (d==64 and q%128==0 and kv%128==0) or kv>1024 else 128
    tq=16 if q==1 else min(q,qbase)
    tk=min(align(kv,16),kvbase)
    total=b*h*ceildiv(q,qbase);active=min(32,total);waves=ceildiv(total,32);iterations=ceildiv(kv,kvbase)
    return {'active_aic':active,'active_aiv':2*active,'waves':waves,'iterations':iterations,
            'branch':f'd{d}_kv{kvbase}','tile':[tq,tk,d],
            'q_tile_capacity':qbase,'kv_tile_capacity':kvbase,
            'units':{'read':2*(tq*d+2*tk*d)/1024,'cube':4*tq*tk*d/131072,
                     'vector':tq*tk/2/256,'fix':4*tq*tk/1024,'write':2*tq*d/1024},
            'opcode_features':[2*iterations*waves],
            'useful_work':4*q*kv*d*b*h,'logical_bytes':2*(2*q*d+2*kv*d)*b*h,
            'work_per_core':4*tq*tk*d*iterations*waves}


def body_time(op,p,stages):
    if op=='quant':
        return sum(stages.values())+(p['iterations']-1)*max(stages.values())
    return schedule(op,p,stages)


def features(p,body):
    return [1.,body,math.log2(1+p['active_aic']+p['active_aiv'])]
