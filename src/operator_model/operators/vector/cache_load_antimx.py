"""Contiguous AntiMxQuant E2M1 -> BF16. Counts tile classes instead of simulating rows."""
from math import isqrt, log2

from operator_model.chips import load_chip
from operator_model.operators.rmsnorm.resources import READ_PER_KIB, WRITE_PER_KIB
from operator_model.scheduling.memory import stream_memory


def up(a, b):
    return (a+b-1)//b


def core_grid(rows, width, cores):
    """AutoTiling, including floor-quotient split candidates and the smaller-N tie break."""
    blocks = up(width, 512)
    budget = min(cores, rows*blocks)
    cuts = set()
    root = isqrt(budget)
    for i in range(1, root+1+int(root*root != budget)):
        cuts.update((i, budget//i))
    choices = []
    for m in sorted(cuts):
        if not 1 <= m <= rows:
            continue
        n = budget//m
        if not 1 <= n <= blocks:
            continue
        r, c = up(rows,m), up(blocks,n)
        delta = r*c
        if m*n == budget:
            tail_r = r if rows % m == 0 else rows % r
            tail_c = c if blocks % n == 0 else blocks % c
            delta -= tail_r*tail_c
        choices.append((delta, n, m))
    if not choices:
        raise ValueError('No legal row/512-column core partition for this AntiMxQuant shape')
    _, n, m = min(choices)
    r, c = up(rows,m), up(blocks,n)
    return up(rows,r), up(blocks,c), r, c*512


def _lengths(extent, count, normal):
    """Up to two extents with their repetition counts; exact even splits have one."""
    last = extent-(count-1)*normal
    if last==normal:
        return [(last,count)]
    return ([(normal,count-1)] if count>1 else [])+[(last,1)]


def _resource_service(core_rows, core_width, capacity, rd, wr, vector_scale, rates):
    blocks = up(core_width,32)
    col_chunks = up(blocks,capacity)
    col_groups = 2*up(up(blocks,col_chunks),2)
    row_capacity = capacity//col_groups
    row_chunks = up(core_rows,row_capacity)
    normal_rows = up(core_rows,row_chunks)
    counts = dict(read_bytes=0,write_bytes=0,vf_iterations=0,scale_vf_iterations=0,
                  ub_tiles=row_chunks*col_chunks)
    total_read=total_write=total_vector=first_drain=0.
    for w,nw in _lengths(core_width,col_chunks,col_groups*32):
        scale_slots=2*up(w,64)
        for r,nr in _lengths(core_rows,row_chunks,normal_rows):
            n=nw*nr
            reads=r*(w//2+scale_slots)
            writes=2*r*w
            data_iterations=up(r*scale_slots,16)
            scale_iterations=up(r*scale_slots,256)
            counts['read_bytes']+=reads*n
            counts['write_bytes']+=writes*n
            counts['vf_iterations']+=data_iterations*n
            counts['scale_vf_iterations']+=scale_iterations*n
            # Two input descriptors, one write. Measured latency includes both overlapping
            # input responses; the payload term separately prices their issue bandwidth.
            read=rates['read_response_cycles']+reads/rd
            write=rates['write_response_cycles']+writes/wr
            vector=rates['handoff_cycles']+vector_scale*(rates['vf_base_cycles']*scale_iterations+
                                                        rates['vf_data_cycles']*data_iterations)
            total_read+=n*read
            total_write+=n*write
            total_vector+=n*vector
            first_drain=max(first_drain,read+vector+write-max(read,vector,write))
    stages=dict(read_service_cycles=total_read,vector_service_cycles=total_vector,
                write_service_cycles=total_write,fill_drain_cycles=first_drain)
    return max(total_read,total_vector,total_write)+first_drain, counts, stages


# Each cycle covers first input issue, its outstanding response, conversion/handoff, then
# last output retirement. Anchor data and cross-chip assumptions are in the worked example.
RATES=dict(front_cycles=3413.5,arrival_log2_cycles=749/log2(56),
           read_response_cycles=1166-68*READ_PER_KIB/1024,
           write_response_cycles=302-256*WRITE_PER_KIB/1024,
           handoff_cycles=47.,vf_base_cycles=8.,vf_data_cycles=60.,
           tail_cycles=23.,outlier_wait_log2_cycles=1966/log2(56))


def schedule(rows, width, work, chip):
    rg,cg,r,c=core_grid(rows,width,chip.vector_cores)
    active=rg*cg
    capacity=((chip.details.get('ub_bytes') or 253952)//2624)*16
    if capacity<16:
        raise ValueError('AntiMxQuant double buffers need at least one 2624-byte tile')
    f=chip.frequency_ghz
    rd=(chip.details.get('mte2_bandwidth_gbs') or (1024*f/READ_PER_KIB))/f
    wr=(chip.details.get('mte3_bandwidth_gbs') or (1024*f/WRITE_PER_KIB))/f
    peak=chip.vector_tops['16']
    if not peak:
        raise ValueError('Explicit AntiMxQuant requires BF16 Vector peak')
    # Normalize dispatch/convert cost by per-core BF16 throughput per cycle.
    vector_scale=(54/(64*1.65))/(peak/(chip.vector_cores*f))
    candidates=[]
    for rr,_ in _lengths(rows,rg,r):
        for cc,_ in _lengths(width,cg,c):
            time,counts,stages=_resource_service(rr,cc,capacity,rd,wr,vector_scale,RATES)
            candidates.append((time,rr,cc,counts,stages))
    local,rr,cc,counts,stages=max(candidates,key=lambda x:x[0])
    memory=stream_memory(chip,active,work['logical_read_bytes'],work['logical_write_bytes'])
    shared=max(memory['external_floor_cycles'],memory['l2_floor_cycles'] or 0.)
    front=RATES['front_cycles']+RATES['arrival_log2_cycles']*log2(active)
    # The last worker can wait after payload arrival (DT VF dispatch) or after
    # write retirement (PR control). A single exposed wait avoids charging both.
    exposed_wait=RATES['outlier_wait_log2_cycles']*log2(active)
    us=(front+max(local,shared)+exposed_wait+RATES['tail_cycles'])/(1000*f)
    plan=dict(active_aiv=active,active_aic=0,row_partitions=rg,column_partitions=cg,
              critical_core_rows=rr,critical_core_cols=cc,ub_capacity_block32=capacity,
              **counts,**stages,front_cycles=front,local_pipeline_cycles=local,
              shared_floor_cycles=shared,exposed_tail_wait_cycles=exposed_wait,
              tail_cycles=RATES['tail_cycles'],memory=memory,
              mte2_bytes_per_cycle=rd,mte3_bytes_per_cycle=wr,vector_price_scale=vector_scale)
    same_950=any(chip==load_chip(sku) for sku in ('Ascend950DT_9582','Ascend950PR_957b'))
    in_domain=same_950 and rows<=128 and width in (64,128,512,576)
    return us,plan,'estimated' if in_domain else 'extrapolated'
