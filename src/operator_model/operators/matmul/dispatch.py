"""BF16 MatMulV3, B resident in L1: issue work and competing memory service."""
from collections import Counter
from math import ceil, log2

from operator_model.operators.matmul.aspect_dispatch import resource_matmul_dispatch as aspect_dispatch
from operator_model.operators.matmul.resources import _costs
from operator_model.operators.matmul.supply import _prices, _mixed_efficiency
from operator_model.operators.matmul.tiling import matmul_plan
from operator_model.profiles.stage_model import align, ceildiv


def _geometry(shape, chip):
    m, n, k = shape
    d, cores = chip.details, chip.cube_cores
    l1, l0a, l0b, l0c = (d.get(key+'_bytes') or default for key, default in
                            (('l1',524288), ('l0a',65536), ('l0b',65536), ('l0c',262144)))
    weight = 2*n*k
    if (n % 16 or k % 16 or weight > 3*l1//4 or 2*m*k <= 3*l1//4
            or m <= 16*cores or not chip.cube_tops.get('16')):
        return None
    coarse = matmul_plan(shape,chip)
    bm0, bn, bk = coarse['tile']
    # One N block; targeted implementation only. Other templates fall back.
    if bn != n or ceildiv(m,bm0) <= cores:
        return None
    clk = chip.frequency_ghz
    l2 = d.get('l2_bytes') or 134217728
    l2_gbs = d.get('l2_bandwidth_gbs') or 100*cores*clk
    ratio = l2_gbs/chip.hbm_bandwidth_gbs
    edge = l2_gbs/(1000*chip.cube_tops['16']) + max(1,2*(m+n)*k/l2)*(1-ratio)*(1/m+1/n) - (1+ratio)/k
    if 1/bm0+1/n <= .85*edge:
        return None
    free = l1-weight
    bm = min(bm0, free//(4*bk)//16*16, l0c//(8*n)//16*16, 16*ceildiv(m,16*cores))
    if bm < 16:
        return None
    limit = free//(4*bk*bm)
    steps = min(ceildiv(k,bk),limit,4)
    if steps == limit and steps*bk%256 and 8*bk*n <= l0b:
        # Other base-K branch is deliberately left with the original provider.
        return None
    if not steps or 4*bk*bm>l0a or 4*bk*n>l0b:
        return None
    step = 1
    while step < steps and (2*bm*bk*step < 48*1024 or bk*step%128):
        step+=1
    page = bk*step
    if k%page:
        return None
    return bm, bk, page


def _tail(full_rows, blocks, splits, last_rows):
    """At most four different fragment sizes regardless of hardware core count."""
    hist=Counter()
    for size, count in ((full_rows,blocks-1),(last_rows,1)):
        if not count:
            continue
        chunk=ceildiv(size,splits)
        whole, rem=divmod(size,chunk)
        hist[chunk]+=count*whole
        if rem:
            hist[rem]+=count
    return hist


def resident_schedule(shape, chip):
    geometry = _geometry(shape,chip)
    if not geometry:
        return None
    bm, bk, page = geometry
    m,n,k=shape
    cores,clk,det=chip.cube_cores,chip.frequency_ghz,chip.details
    costs=_costs()
    price=lambda name:det.get('matmul_'+name) or costs[name]
    clock=1000*clk
    peak=chip.cube_tops['16']*1000/(clk*cores)
    l2=det.get('l2_bytes') or 134217728
    shared=(det.get('l2_bandwidth_gbs') or 100*cores*clk)/clk
    shared*=det.get('matmul_l2_efficiency') or 1.
    hbm=chip.hbm_bandwidth_gbs/clk
    physical=hbm*(det.get('matmul_hbm_efficiency') or 1.)
    mte2=(det.get('mte2_bandwidth_gbs') or 64*clk)/clk
    mte1=(det.get('mte1_bandwidth_gbs') or price('mte1_bytes_per_cycle')*clk)/clk
    fixp=(det.get('mte3_bandwidth_gbs') or price('write_bytes_per_cycle')*clk)/clk
    weight=2*n*k; mm_per=k//bk; nt=ceildiv(m,bm)
    waves=ceildiv(nt,cores); total_last=nt-(waves-1)*cores
    splits=cores//total_last if waves>1 else 1
    tail=_tail(bm,total_last,splits,m-(nt-1)*bm)
    max_tail=max(tail); tail_work=sum(t*r for r,t in tail.items())
    partial_tiles=sum(tail.values()); fragments=(waves-1)*cores+partial_tiles
    # DT counters: saturation occupies only a fraction of a DFI period for useful
    # beats. Scale the measured service ceiling by target shared capacity, not by
    # the different SKU's *rated external* link a second time.
    price_bin,write_decile=_mixed_efficiency(_prices()['saturated_write_fraction_deciles'],n/(k+n))
    ref_dfi_to_shared=4000/(32*1.65*100)
    mixed_shared=min(shared,shared*ref_dfi_to_shared*price_bin['median_eta'])
    retain=det.get('matmul_output_retained_fraction')
    retain=1. if retain is None else retain
    resident_rows=min(m,int(max(0,l2-weight-2*cores*(2*bm*page))*retain)//(2*n))
    prefix_read=2*k*resident_rows
    ingress=det.get('matmul_initial_read_gbs') or chip.hbm_bandwidth_gbs
    credit=min(l2,prefix_read*max(0,1-chip.hbm_bandwidth_gbs/ingress))
    warm=credit/prefix_read if prefix_read else 0.

    def core(rows):
        issue=ceil(2*align(rows,16)*n*bk/peak)
        return max(mm_per*issue,2*rows*k/mte2,2*(rows+n)*k/mte1,2*rows*n/fixp)

    init=price('launch_cycles')+costs['arrival_log2_cycles']*log2(cores)
    fill=price('read_latency_cycles')+max((weight+2*bm*page)/mte2,cores*(weight+2*bm*page)/shared,(weight+2*cores*bm*page)/physical)
    start=init+fill+price('l0_latency_cycles')+2*(bm+n)*bk/mte1
    drain=24+price('drain_cycles')+2*max_tail*n/fixp
    cuts=sorted({max(0,min(waves,v)) for v in (0,1,resident_rows//(bm*cores),resident_rows//(bm*cores)+1,waves-1,waves)})
    phases=[];service=0.
    for begin,end in zip(cuts,cuts[1:]):
        last=begin==waves-1
        rows=tail_work if last else bm*cores
        longest=max_tail if last else bm
        cached=min(rows,max(0,resident_rows-begin*bm*cores))
        spill=rows-cached
        depth=k-page if begin==0 else k
        a0,a1=2*cached*depth,2*spill*depth
        ew=2*spill*n
        # Use the actual active fragments for the final wave on each interface;
        # cache residency and the memory clock are shared across the full chip.
        active=partial_tiles if last else cores
        occupancy=active/cores
        shared_read=a0*(1-warm)/physical
        mixed_external=(a1+ew)/physical
        mixed_interconnect=(a1+ew)/(mixed_shared*occupancy)
        ports=max(shared_read+max(mixed_external,mixed_interconnect),(a0+a1+2*rows*n)/(shared*occupancy))
        local=core(longest)
        cost=max(local,ports)
        service+=(end-begin)*cost
        phases.append(dict(first=begin,end=end,active_aic=active,rows=rows,cached_rows=cached,core_cycles=local,
                           read_prefix_cycles=shared_read,mixed_hbm_cycles=mixed_external,
                           mixed_fabric_cycles=mixed_interconnect,l2_or_memory_cycles=ports,cycles_per_wave=cost))
    stages=dict(first_mmad=start/clock,shared_or_core=service/clock,terminal_drain=drain/clock)
    work=dict(logical_tensor_allocation_bytes=2*(m*k+n*k+m*n),resident_b_gm_to_l1_bytes=cores*weight,a_gm_to_l1_bytes=2*m*k,logical_output_bytes=2*m*n,
              possible_dirty_output_bytes=2*n*resident_rows,scenario_initial_a_credit_bytes=credit,
              physical_mmad_count=fragments*mm_per)
    plan=dict(branch='b_resident_l1',tile=[bm,n,bk],read_page_k=page,output_tiles=nt,effective_output_tiles=fragments,
              waves=waves,active_aic=cores,active_aiv=0,iterations=mm_per,mmads_total=fragments*mm_per,
              tail_wave_pieces=partial_tiles,tail_split_mn=[splits,1],
              tail_wave_histogram=[[r,n,c] for r,c in sorted(tail.items())])
    diag=dict(model='target_resource_b_resident_l1',target_plan=plan,stages_us=stages,internal_work=work,
              service_segments=phases,pricing=dict(l2_bytes=l2,hbm_bytes_per_cycle=hbm,
                  inplace_excluded_tensors_fit_hbm=work['logical_tensor_allocation_bytes']<=chip.hbm_capacity_gb*1e9,
                  mixed_fabric_bytes_per_cycle=mixed_shared,shared_bytes_per_cycle=shared,mixed_write_decile=write_decile,
                  mixed_dfi_observed_efficiency=price_bin['median_eta'],initial_logical_read_scenario_gbs=ingress),
              qualification='B resident in L1, contiguous BF16 ND only. Mixed-fabric ceiling scales an independent 950DT DFI observation by target shared service. Initial read credit is a bounded scenario, not a measured hit rate. Future timings extrapolated.')
    return sum(stages.values()),cores,0,diag


def resource_matmul_dispatch(shape, target):
    if any(v > 2**63-1 for v in shape):
        raise ValueError('MatMulV3 extent exceeds a signed 64-bit tensor dimension')
    result=resident_schedule(shape,target)
    return result if result else aspect_dispatch(shape,target)
