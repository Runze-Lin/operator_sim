"""Schedule a wave of small MX-scale descriptors on shared aligned grains."""
from operator_model.resources import load_parameters
from collections import Counter
from functools import lru_cache
from math import ceil, log2


@lru_cache(maxsize=1)
def fragment_costs():
    return load_parameters('quant_fragments')['parameters']


def fragment_wave(shape, plan, service, chip, loop_work, peak, scale_base, scale_tail, write_bpc):
    """Return critical single-tile request and wave clocks, or None for bulk writes."""
    rows,cols=shape
    nr,nc=plan['rows_per_core'],plan['cols_per_core']
    active=plan['active_aiv']
    if active<=1 or plan['iterations']!=1 or nr*2*ceil(nc/64)>32:
        return None
    # E8M0 scale storage packs two entries per 64 input elements per row.
    stride=2*ceil(cols/64)
    grains_by_owner=[]
    owners=Counter()
    for row in range(0,rows,nr):
        h=min(nr,rows-row)
        for col in range(0,cols,nc):
            w=min(nc,cols-col)
            written=2*ceil(w/64)
            grains=set()
            for i in range(row,row+h):
                start=i*stride+col//32
                grains.update(range(start//32,(start+written-1)//32+1))
            owners.update(grains)
            grains_by_owner.append((row,col,h,w,grains))
    if max(owners.values())==1:
        return None

    costs={k:chip.details.get('quant_fragment_'+k) or v for k,v in fragment_costs().items()}
    # Issue skew is measured between the earliest / last payload descriptors.
    # The deterministic rank is a proxy for admission order, not a physical core ID.
    spread=costs['spread_cycles']*log2(active)
    requests=[]
    for i,(row,col,h,w,grains) in enumerate(grains_by_owner):
        s=service(h,w)
        issue=spread*i/(active-1)
        compute=costs['vector_cycles']+(s['instruction_body_cycles'] if s.get('instruction_body_cycles') is not None else s['loops']*loop_work/peak)
        ready=issue+max(s['read_latency']+s['dependency_cycles'], costs['frontend_cycles'])+compute
        s['vector']=ready-issue-s['read_latency']
        requests.append((ready,i,row,col,s,grains))
    # Reserve each touched line independently. One descriptor can span two
    # grains; waiting on the first must not stall later arrivals on the second.
    line_ends={};retire=[]
    for ready,i,row,col,s,grains in sorted(requests):
        shared=sorted(g for g in grains if owners[g]>1)
        gap=s['scale_issue_gap_cycles']
        base=max(scale_base+s['scale_bytes']/write_bpc,s['data_write_cycles']+scale_tail)
        local=ready+gap+base
        demand=local
        for g in shared:
            end=max(local,line_ends.get(g,-float('inf'))+costs['grain_ii_cycles'])
            line_ends[g]=end
            demand=max(demand,end)
        retire.append((demand,i,local,ready,row,col,s,shared))
    fabric_end=-float('inf')
    finish=-float('inf')
    for demand,i,local,ready,row,col,s,shared in sorted(retire):
        end=demand
        if shared:
            end=max(demand,fabric_end+costs['fabric_ii_cycles'])
            fabric_end=end
        gap=s['scale_issue_gap_cycles']
        # The two output descriptors can overlap; the port still serves the combined bytes.
        done=max(end,ready+s['data_write_cycles'],ready+s['write_bytes']/write_bpc)
        if done>finish:
            finish=done
            critical=dict(row=row,col=col,rank=i,read_issue_cycles=spread*i/(active-1),
                          scale_ready_cycles=local,scale_end_cycles=end,
                          shared_grains=shared,grain_owners=[owners[g] for g in shared],
                          extra_queue_cycles=end-local,frontend_cycles=costs['frontend_cycles'])
            s['scale_write_cycles']=end-ready-gap;s['write']=done-ready
            first=s
    return costs['launch_cycles'],finish,first,dict(mode='single tile; at most 32 scale bytes per producer; shared aligned grain',
            critical=critical,peak_writers=max(owners.values()),shared_grains=sum(v>1 for v in owners.values()),
            arrival_spread_cycles=spread,grain_ii_cycles=costs['grain_ii_cycles'],fabric_ii_cycles=costs['fabric_ii_cycles'])
