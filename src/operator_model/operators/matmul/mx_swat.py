"""Dense ND A8W4 geometry for the Blaze SWAT kernel in CANN 9.2.1.

Source: ops-nn v9.2.1, 28d19e078fd2d4144a681eb20ed3261047256d0c,
QuantBatchMatmulV4WeightQuantMxSwatTiling and WeightQuantMxSwatTilingSolver.
The grid is represented by rectangle counts, so work is bounded by the number
of physical cores in the last wave, independently of sequence length.
"""
from collections import Counter
from dataclasses import dataclass


def div(n, d):
    return (n+d-1)//d


def align(n, d):
    return div(n, d)*d


def _initial_tile(m, n, k, cores, limits):
    bm, bn, bk = min(256,align(m,16)), min(256,align(n,16)), min(k,128)
    mc, nc = div(m,bm), div(n,bn)
    if mc*nc >= cores:
        return bm,bn,bk

    if div(m,16) > div(n,16):
        nc = div(n,align(div(n,nc),16))
        mc = cores//nc
    else:
        mc = div(m,align(div(m,mc),16))
        nc = cores//mc
    def rect(mc,nc):
        return align(div(m,mc),16), align(div(n,nc),16)
    h,w = rect(mc,nc)
    mc,nc = div(m,h),div(n,w)
    while w > 2*h and nc < cores//2 and w > 16:
        proposed = rect(cores//(2*nc),2*nc)
        if proposed == (h,w):
            break
        h,w = proposed
        mc,nc = div(m,h),div(n,w)
    while h >= 2*w and mc < cores//2 and h > 16:
        proposed = rect(2*mc,cores//(2*mc))
        if proposed == (h,w):
            break
        h,w = proposed
        mc,nc = div(m,h),div(n,w)
    depth = min(align(k,128),limits['l0a_bytes']//(2*max(h,w))//128*128)
    if depth >= 128 and 2*h*depth <= limits['l0a_bytes'] and 2*w*depth <= limits['l0b_bytes'] and 4*h*w <= limits['l0c_bytes']:
        return h,w,depth if depth <= 4095 else align(depth//2,256)
    return bm,bn,bk


def _buffers(shape, tile, slots, limits):
    _,n,k = shape
    bm,bn,bk = tile
    l0 = dict(l0a_bytes=2*bm*bk, l0b_bytes=2*bn*bk, l0c_bytes=4*bm*bn)
    if any(size > limits[key] for key,size in l0.items()):
        return
    # The kernel addresses its second L1 bank at the fixed 256-KiB offset.
    half_capacity = min(256*1024,limits['l1_bytes']-256*1024)
    n_weight = min(bn,n)
    n_ub = align(div(n_weight,2),16) if slots==4 and n_weight>16 else n_weight
    for step in range(min(4,div(k,bk)),0,-1):
        page = step*bk
        k_ub = min(page,k)
        ub = slots*(n_ub*align(k_ub,64)//2 + (align(n_ub,16)+1)*align(k_ub,64))
        if ub > limits['ub_bytes']:
            continue
        for factor in range(min(4,div(k,page)),0,-1):
            scale_page = factor*page
            half_used = (bm+bn)*(slots//2*align(page,64)+align(scale_page,64)//32)
            if half_used <= half_capacity:
                return (page,scale_page,n_ub,k_ub,
                        dict(ub_bytes=ub,l1_bytes=2*half_used,**l0))


@dataclass(frozen=True)
class _Axis:
    size: int
    tile: int
    count: int
    merged: int
    main: int

    def extent(self, idx):
        regular = self.count-self.merged
        if idx < regular:
            return self.tile
        if idx < self.count-1:
            return self.main
        return self.size-regular*self.tile-(self.merged-1)*self.main

    def histogram(self):
        result = Counter()
        for extent,count in [(self.tile,self.count-self.merged),
                             (self.main,self.merged-1),(self.extent(self.count-1),1)]:
            if count:
                result[extent] += count
        return result


def _axis(size, tile, optimize):
    count, tail = div(size,tile),size%tile
    merged, main = 1,0
    if optimize and tail and count>1:
        window = min(4,count)
        before = count//window-1
        remaining = count-window*before
        best = (before+1)*tile
        cap = min((tile-tail)//16,count)
        for j,merged_before_last in enumerate(range(remaining-1,cap,window),1):
            candidate = align(div(merged_before_last*tile+tail,merged_before_last+1),16)
            cost = (before+1-j)*tile+j*candidate
            if cost <= best:
                best, merged, main = cost, merged_before_last+1, candidate
    return _Axis(size,tile,count,merged,main)


def _tail_rectangles(ma, na, total, remainder):
    result = Counter()
    window = min(4,ma.count)
    before = ma.count//window-1
    for idx in range(total-remainder,total):
        row = min(idx//(window*na.count),before)
        rows = window if row<before else ma.count-before*window
        pos = idx-row*window*na.count
        mi, ni = row*window+pos%rows, pos//rows
        if row%2:
            ni = na.count-1-ni
        result[ma.extent(mi),na.extent(ni)] += 1
    return result


def _split_rectangles(tail, sm, sn):
    result = Counter()
    for (h,w),count in tail.items():
        hm,wn = align(div(h,sm),16),align(div(w,sn),16)
        mh,nh = _axis(h,hm,False).histogram(),_axis(w,wn,False).histogram()
        for a,ac in mh.items():
            for b,bc in nh.items():
                result[a,b] += ac*bc*count
    return result


def _schedule(shape, tile, cores, conservative):
    m,n,k = shape
    bm,bn,_ = tile
    ma,na = _axis(m,bm,k%128==0 and not conservative),_axis(n,bn,k%128==0 and not conservative)
    rects = Counter({(h,w):mc*nc for h,mc in ma.histogram().items() for w,nc in na.histogram().items()})
    total = ma.count*na.count
    tail = _tail_rectangles(ma,na,total,total%cores)
    sm,sn = 1,1
    pieces = sum(tail.values())
    order = (0,1) if (m-1)%bm+1 >= (n-1)%bn+1 else (1,0)
    while pieces and pieces<cores:
        previous = pieces
        for axis in order:
            cm,cn = sm+(axis==0),sn+(axis==1)
            if cm>div(bm,16) or cn>div(bn,16):
                continue
            used = sum(cnt*div(h,align(div(h,cm),16))*div(w,align(div(w,cn),16))
                       for (h,w),cnt in tail.items())
            if pieces < used <= cores:
                sm,sn,pieces = cm,cn,used
        if pieces==previous:
            break
    rects.subtract(tail)
    rects += _split_rectangles(tail,sm,sn)
    return {'base_grid':[ma.count,na.count], 'tail_split':[sm,sn],
            'tail_merge_m':[ma.merged,ma.main], 'tail_merge_n':[na.merged,na.main],
            'last_wave_tiles':pieces or min(cores,total),
            'rectangles':[(h,w,count) for (h,w),count in sorted(rects.items())]}


def plan(shape, chip, cores):
    """Select the two-slot ND policy first, four-slot then minimal fallback."""
    limits = {key:chip.details.get(key) or 0 for key in
              ('l0a_bytes','l0b_bytes','l0c_bytes','l1_bytes','ub_bytes')}
    m,n,k = shape
    initial = _initial_tile(m,n,k,cores,limits)
    for tile,slots,conservative in ((initial,2,False),(initial,4,False),((16,16,min(k,64)),4,True)):
        result = _buffers(shape,tile,slots,limits)
        if result:
            page,scale_page,ub_n,ub_k,footprint = result
            return {**_schedule(shape,tile,cores,conservative), 'base_tile':list(tile),
                    'l1_k':page, 'scale_l1_k':scale_page,'ub_n':ub_n,'ub_k':ub_k,
                    'l1_slots':slots, 'conservative_fallback':conservative,
                    'required_buffer_bytes':footprint}
