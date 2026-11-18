"""Contiguous BF16 MatMulV3 rebalance policy; all resources supplied by chip config."""
from math import isqrt
from .model import align, ceildiv


def matmul_plan(shape, chip):
    m,n,k=shape; cores=chip.cube_cores; details=chip.details
    # Portable fallback values are 950DT's buffer capacities, not a runtime CANN query.
    l0a=details.get('l0a_bytes') or 65536
    l0b=details.get('l0b_bytes') or 65536
    l0c=details.get('l0c_bytes') or 262144
    l1=details.get('l1_bytes') or 524288
    l2=details.get('l2_bytes') or 134217728
    # Compare rates in consistent decimal GB/s, TFLOP/s. 100 bytes/core/cycle
    # is the 950 library's L2 rate hint, not an independently measured bandwidth.
    bw2=details.get('l2_bandwidth_gbs') or chip.frequency_ghz*cores*100
    bw=chip.hbm_bandwidth_gbs
    power=chip.cube_tops.get('16') or chip.frequency_ghz*cores*8
    edge=bw2/(power*1000)+max(1,2*(m+n)*k/l2)*(1-bw2/bw)*(1/m+1/n)-(1+bw2/bw)/k
    best_m=min(align(m,16),256)
    best_n=max(16,min(align(n,16),l0c//(4*best_m)//16*16))
    memory=1/best_m+1/best_n>edge
    an=128 if k<m*n/(m+n)*bw/bw2 else (64 if memory else 32)
    am=16; ak=align(k,16)
    limit=min(l0a//(4*min(ak,16 if memory else 64)),l0c//64,
              l1//(8*min(128 if memory else 256,ak)))
    def upper(x,a):
        u=min(align(x,a),limit//a*a)
        return min(u,align(x,16)) if x<a else u
    mm,nn=upper(m,am),upper(n,an)
    def balance(a,b):
        jobs=ceildiv(m,a)*ceildiv(n,b); rounds=(jobs-1)//cores
        splits=cores//(jobs-rounds*cores)
        area=(m if m<=16 else a)*(n if n<=16 else b)
        if rounds and area//splits>=4096:
            s=isqrt(splits); last=1/(s*(s+(splits-s*s)//s))
        else:last=1
        return m*n/(cores*area*(rounds+last))
    if mm<16 or nn<16:
        raise ValueError('Insufficient buffers for a BF16 16x16 tile')
    bm=max(16,min(mm,256));bn=max(16,min(nn,(l0c//(4*bm))//an*an))
    score=1/bm+1/bn;bal=balance(bm,bn);edge*=.85
    for a in range(mm,0,-am):
        for b in range(min(nn,(l0c//(4*a))//an*an),0,-an):
            value=1/a+1/b
            if bal>=.9 and value>score and value>edge>0:continue
            ratio=balance(a,b)
            cube=value<=edge and ratio>bal
            if cube or value/ratio<score/bal or (abs(value/ratio-score/bal)<1e-9 and ratio>bal):
                if cube:edge=value
                score,bal,bm,bn=value,ratio,a,b
    bm,bn=min(align(m,16),bm),min(align(n,16),bn)
    max_k=min(l0a//(4*bm),l0b//(4*bn))
    if max_k<16:raise ValueError('Insufficient double-buffered L0 for BF16 operands')
    bk=ak if ak<=max_k else (max_k//128*128 if max_k>=128 else max(x for x in (16,32,64) if x<=max_k))
    page=0
    for step in range(1,min(8,ceildiv(k,bk))+1):
        d=bk*step
        if 4*(bm+bn)*d>l1 or 8*max(bm,bn)*d>l1:break
        short=2*max(bm,bn)*page<48*1024
        if not page or (page%256 and (d%128==0 or short)) or (page%256==0 and d%256==0 and short):
            page=d
    if not page:raise ValueError('Insufficient L1 for BF16 double-buffered input tile')
    jobs=ceildiv(m,bm)*ceildiv(n,bn)
    em,en=min(m,bm),min(n,bn)
    return dict(active_aic=min(jobs,cores),active_aiv=0,tile=[bm,bn,bk],read_page_k=page,
                waves=ceildiv(jobs,cores),iterations=1,branch='gemv' if m==1 else
                'k_tail' if k%16 else 'coarse_tile' if (bm,bn)!=(16,64) else 'dense',
                mmads_total=jobs*ceildiv(k,bk),output_tiles=jobs,
                units=dict(read=2*(em*ak+en*ak)/1024,cube=2*em*en*ak/131072,
                           fix=2*em*en/1024,vector=0.,write=0.))
