"""Native FP4 ASW tiling. K fragments, L1 buffers and output work have distinct lifetimes."""
from collections import Counter


# Payload bytes per AIC cycle, pooled across the seven old 950DT/PR native
# two-page endpoints. The inter-retirement measurement is described in the
# worked example and does not include the cost of the initial read response.
STREAM_BYTES_PER_CYCLE = 87.22795794625252


def ceildiv(x, y):
    return (x+y-1)//y


def fragment_k(shape, tile, budget, details):
    """V3 default or underfilled-grid BaseBlockCalculator, capacity bounded."""
    m,n,k=shape
    bm,bn=tile
    if ceildiv(m,256)*ceildiv(n,256)>=budget:
        return min(256,ceildiv(k,64)*64)
    capacity=min((details.get('l0a_bytes') or 0)//bm,(details.get('l0b_bytes') or 0)//bn)
    # Two FP4 elements per byte, two L0 slots. AdjustBaseBlockDefault
    # aligns each packed ND inner dimension to 128 bytes (256 FP4 values).
    candidate=min(ceildiv(k,256)*256,capacity//256*256)
    if candidate>4095:
        candidate=ceildiv(candidate//2,256)*256
    return candidate or min(256,ceildiv(k,64)*64)


def _axis(length,size):
    full,tail=divmod(length,size)
    return [(dim,count) for dim,count in ((size,full),(tail,1)) if count and dim]


def _parts(length,splits):
    # ASW last subblock takes the remainder of the ceil-sized partitions.
    width=ceildiv(length,splits)
    return _axis(length,width)


def geometry(shape,tile,budget,details):
    m,n,k=shape
    bm,bn,bk=tile
    mt,nt=ceildiv(m,bm),ceildiv(n,bn)
    grid=mt*nt
    active=min(grid,budget)
    remainder=grid%active
    whole=Counter({(h,w):mh*nw for h,mh in _axis(m,bm) for w,nw in _axis(n,bn)})
    l1=details.get('l1_bytes') or 0
    # Restrict A retention to the evidenced single-row grid. Each AIC
    # consumes the same A slice again on its next N rectangle.
    retained=mt==1 and grid>active and bm*ceildiv(k,64)*32<=l1//2
    if retained:
        left=l1-bm*ceildiv(k,64)*34
        stride=max(1,ceildiv(256,bk))
        b_bytes=bn*(bk+2*ceildiv(bk,64))*stride//2
        step=stride*ceildiv(min(32768,max(1,left//2)),b_bytes)
        if step==1 and k>bk and left>2*b_bytes:
            step=2
    else:
        # GetDepthA1B1 chooses powers of two, CalStepKs caps MX at 4.
        step=1
        for candidate in (2,4):
            data_bytes=(bm+bn)*bk*candidate
            scale_bytes=(bm+bn)*4*ceildiv(max(min(k,4096),candidate*bk),64)
            if data_bytes+scale_bytes<=l1:
                step=candidate
    step=min(max(1,step),ceildiv(k,bk))
    def l1_reservation(s):
        # Data ping-pong and separate E8M0 double-buffered scale window.
        data_depth=s*bk
        scale_bytes=4*ceildiv(min(k,max(data_depth,4096)),64)
        if retained:
            return bm*ceildiv(k,64)*34+bn*(data_depth+scale_bytes)
        return (bm+bn)*(data_depth+scale_bytes)
    while step>1 and l1_reservation(step)>l1:
        step=max(1,step//2)
    l1_k=step*bk
    # Split the last underfilled window across available AICs. Multi-axis
    # windows use the row-major grid as a candidate traversal for edge tiles.
    tail=Counter()
    split_m=split_n=1
    if remainder and grid>active:
        limit=active//remainder
        mx=min(limit,ceildiv(min(m,bm),16))
        nx=min(limit,ceildiv(min(n,bn),16))
        prefer_m=(m-1)%bm>=(n-1)%bn
        if retained:
            split_n=max(1,min(limit,bn//16))
        else:
            while True:
                prev=(split_m,split_n)
                for along_m in (prefer_m,not prefer_m):
                    if along_m and split_m<mx and (split_m+1)*split_n<=limit:
                        split_m+=1
                    if not along_m and split_n<nx and split_m*(split_n+1)<=limit:
                        split_n+=1
                if prev==(split_m,split_n):
                    break
        # Count the suffix without visiting each tile (also for very large chips).
        for hd,hc in _axis(m,bm)[::-1]:
            if remainder==0:
                break
            take=min(hc*nt,remainder)
            fullrows,last=divmod(take,nt)
            for wd,wc in _axis(n,bn)[::-1]:
                count=fullrows*wc+min(last,wc)
                last=max(0,last-wc)
                if count:
                    whole[hd,wd]-=count
                    for h,mult_m in _parts(hd,split_m):
                        for w,mult_n in _parts(wd,split_n):
                            tail[h,w]+=count*mult_m*mult_n
            remainder-=take
    def entries(group):
        return sorted((h,w,count) for (h,w),count in group.items() if count)
    return dict(rectangles=entries(whole+tail),full_rectangles=entries(whole),tail_rectangles=entries(tail),
                tail_split=[split_m,split_n],grid_tiles=grid,l1_k=l1_k,mmad_k=bk,
                resident_a=retained,l1_payload_buffer_bytes=l1_reservation(step),
                tail_assignment='per-core continuation, row-major edge candidate')


def wave_cycles(native,k,active,prices,make_page,stages):
    """Overlap the streams on the critical worker, bounded separately by chip traffic."""
    bk=native['mmad_k']
    page_k=native['l1_k']
    def worker(h,w,repeat=False):
        depth=min(bk,k)
        last=(k-1)%bk+1
        first=make_page(h,w,depth,False)
        remaining=make_page(h,w,last,False)
        nfrags=ceildiv(k,bk)
        first_page=make_page(h,w,min(page_k,k),repeat)
        full=make_page(h,w,k,repeat)
        issue=prices['mmad_issue']
        # Each page transfers into L1 while previous L0 slices are consumed.
        # The dependent first L0 slice fills once; subsequent descriptors issue
        # independently subject to the most heavily loaded engine.
        issue_stream=max(first['cube'],issue)*(nfrags-1)
        prefix=max(0,full['load']-remaining['load'],issue_stream)
        body=max(full['read'], first_page['read']+prefix)
        out=prices['write'](h,w)
        return full,first_page,remaining,body,out,nfrags

    def peak(rects):
        return max((p[:2] for p in rects),key=lambda hw:hw[0]*hw[1])
    h,w=peak(native['full_rectangles'])
    full,first_page,last,body,out,frags=worker(h,w)
    if frags==1 and not native['tail_rectangles'] and native['grid_tiles']<=active:
        return
    stages['read_fill']=prices['read_latency']+first_page['read']
    stages['page_intervals']=max(0, body-first_page['read'])
    stages['l1_to_l0']=prices['load_latency']+last['load']
    stages['cube_tail']=prices['cube_latency']+last['cube']
    stages['write_tail']=prices['drain']+out
    # No cross-core barrier at a wave edge. A core continues when its own
    # previous work releases the reusable operand/accumulator slots. Count
    # whole-tile serial resource demand on its path, not per-wave launch/skew.
    full_waves=max(1,native['grid_tiles']//active)
    repeat,_,_,_,_,_=worker(h,w,native['resident_a'])
    full_ii=max(repeat['read'],repeat['load'],repeat['cube'],frags*prices['mmad_issue'],out)
    extra=(full_waves-1)*full_ii
    if native['tail_rectangles']:
        th,tw=peak(native['tail_rectangles'])
        value,_,_,_,wr,fn=worker(th,tw,native['resident_a'])
        extra+=max(value['read'],value['load'],value['cube'],fn*prices['mmad_issue'],wr)
    stages['additional_waves']=extra
    return dict(first_page_read_cycles=first_page['read'],whole_tile_read_cycles=full['read'],
                after_first_page_cycles=stages['page_intervals'],final_l0_load_cycles=last['load'],
                final_cube_service_cycles=last['cube'],full_worker_interval_cycles=full_ii,
                tail_continuation_cycles=extra-(full_waves-1)*full_ii)
