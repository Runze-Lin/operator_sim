"""Target-resource BF16 MatMulV3. No target observations or source-device latency."""
import json
from functools import lru_cache
from math import ceil, log2
from pathlib import Path
from .matmul_v15 import matmul_plan
from .matmul_waves import dispatch

@lru_cache(maxsize=1)
def _costs():
    p=Path(__file__).resolve().parents[1]/'results/v0.16/matmul.json'
    return json.loads(p.read_text())['parameters']


def resource_matmul(shape,chip):
    m,n,k=shape; plan=matmul_plan(shape,chip); am,an,ak=plan['tile']
    active=plan['active_aic']; jobs=plan['output_tiles']; waves=plan['waves']
    bm,bn=min(am,m),min(an,n);page=plan['read_page_k'];pages=ceil(k/page)
    c=_costs(); d=chip.details; ghz=chip.frequency_ghz
    def cost(key):
        return d.get('matmul_'+key) or c[key]
    def bandwidth(key,default):
        # bytes/cycle per participating core; explicit config is decimal GB/s.
        return d.get(key+'_bandwidth_gbs')/ghz if d.get(key+'_bandwidth_gbs') else c[default]
    mte2=bandwidth('mte2','mte2_bytes_per_cycle')
    mte1=bandwidth('mte1','mte1_bytes_per_cycle')
    fix=bandwidth('mte3','write_bytes_per_cycle')
    l2=(d.get('l2_bandwidth_gbs')/ghz if d.get('l2_bandwidth_gbs') else 100*chip.cube_cores)
    hbm=chip.hbm_bandwidth_gbs/ghz
    peak=(chip.cube_tops.get('16') or 0)*1000/(ghz*chip.cube_cores) or c['default_bf16_ops_per_cycle']
    align=lambda v:16*ceil(v/16)
    mm_ops=lambda depth:2*align(bm)*align(bn)*align(depth)
    # Replication at the L2 -> core boundary is known from the output grid.
    # At HBM the ideal cold first pass assumes L2 reuse if the page fits;
    # outside L2 capacity, charge replicated input traffic as an upper scenario.
    cold_input=2*(m+n)
    l2_size=d.get('l2_bytes') or 134217728
    full_hist,last_hist,split=dispatch(shape,plan)
    full_waves=waves-1
    split_jobs=sum(full_hist.values())+sum(last_hist.values())
    total_l2_read=2*k*sum((a+b)*v for hist in (full_hist,last_hist) for (a,b),v in hist.items())
    total_l0_read=total_l2_read
    hbm_input=cold_input*k if cold_input*min(k,page)<=l2_size else total_l2_read
    out_bytes=2*m*n
    hbm_eff=d.get('matmul_hbm_efficiency') or 1.
    l2_eff=d.get('matmul_l2_efficiency') or 1.
    retained_fraction=d.get('matmul_output_retained_fraction')
    if retained_fraction is None:retained_fraction=1.
    # Cache write-back scenario: capacity bounds the possible dirty retention;
    # it is not a hit-rate observation. Caller may force write-through (fraction=0).
    retained=min(out_bytes,max(0,l2_size-min(k,page)*cold_input))*retained_fraction
    visible_write=out_bytes-retained; lh=l2*l2_eff;hh=hbm*hbm_eff
    plan={**plan,'mmads_total':split_jobs*ceil(k/ak),'tail_split_mn':list(split),
          'effective_output_tiles':split_jobs,'tail_wave_pieces':sum(last_hist.values()),
          'tail_wave_histogram':[[a,b,count] for (a,b),count in sorted(last_hist.items())]}
    def wave(hist,repeat):
        # Earlier complete waves use their mean exact bytes, preserving total traffic
        # without looping over the problem size; the final wave is priced separately.
        local_read=max(max(rect) for rect in hist)
        max_area=max(a*b for a,b in hist); max_pad_area=max(align(a)*align(b) for a,b in hist)
        inputs=2*sum((a+b)*v for (a,b),v in hist.items())/repeat
        writes=2*sum(a*b*v for (a,b),v in hist.items())/repeat
        def stages(depth):
            cp=lambda z:2*max_pad_area*align(z)/peak
            frag=min(ak,depth);last=(depth-1)%ak+1;parts=ceil(depth/ak)
            read_work=max(2*depth*local_read/mte2,inputs*depth/lh,
                          inputs*depth*(hbm_input/total_l2_read)/hh)
            l0_work=lambda z:2*z*local_read/mte1
            issue=max(cost('mmad_ii_cycles')+cp(frag),l0_work(frag))
            compute=(parts-1)*issue+cost('mmad_latency_cycles')+cp(last)
            return read_work,l0_work(frag),compute
        first=min(k,page);rd,l0,cmp=stages(first);end=stages((k-1)%page+1)
        pii=max(2*cost('read_ii_cycles'),rd,(cost('read_latency_cycles')+rd)/2,cmp)
        # Local FIXP bandwidth, shared L2 ingress and exposed HBM spill compete.
        wt=max(2*max_area/fix,writes/lh,writes*(visible_write/out_bytes)/hh)
        return dict(R=cost('read_latency_cycles')+rd,L=cost('l0_latency_cycles')+l0,
                    P=(pages-1)*pii,C=end[2],W=cost('drain_cycles')+wt,
                    Ri=max(2*cost('read_ii_cycles'),rd)+(pages-1)*pii,
                    Ci=(pages-1)*pii+max(0,end[2]-cost('mmad_latency_cycles')+cost('mmad_ii_cycles')),
                    Wi=wt,write_bytes=writes)
    tailwave=wave(last_hist,1)
    fwave=wave(full_hist,full_waves) if full_waves else tailwave
    # Flow shop. Request i+1 can enter MTE before MMAD/FIX of i finish. Split the
    # terminal stage into payload service (serial port) and latency (may overlap).
    # Completing a buffer pair every two waves is necessary with two prefetch slots.
    startup=cost('launch_cycles')+c['arrival_log2_cycles']*log2(active)
    base=dict(read_fill=fwave['R'],l0_lead=fwave['L'],page_intervals=fwave['P'],
              last_mmad=fwave['C'],write_tail=fwave['W'])
    elapsed=sum(base.values())
    if full_waves:
        f=fwave;t=tailwave
        rstep=max(f['Ri'],(f['R']+f['L'])/2)
        cstep=max(rstep,f['Ci']);wstep=max(cstep,f['Wi'])
        # Critical frontier just before the final wave. Monotone resource clocks
        # yield a closed form for any number of identical full waves.
        pf=full_waves-1
        prev_c=f['R']+f['L']+pf*cstep
        prev_w=f['R']+f['L']+f['P']+f['C']+pf*wstep
        tail_c=max(full_waves*rstep+t['R']+t['L'],prev_c+f['Ci'])
        tail_w=max(tail_c+t['P']+t['C'],prev_w+f['Wi'])
        elapsed=max(prev_w+f['W'],tail_w+t['W'])
    # A shared link must also carry read+write bytes; this is an integrated
    # lower bound, not an independently recharged duration.
    busfloor=max((total_l2_read+out_bytes)/lh,(hbm_input+visible_write)/hh)
    base['additional_output_waves']=max(0,elapsed-sum(base.values()))
    base['shared_link_floor']=max(0,busfloor-elapsed)
    ticks=dict(base,launch=startup)
    stages={name:value/(ghz*1000) for name,value in ticks.items()}
    total=sum(stages.values())
    info=dict(model='target_resource_matmul',target_plan=plan,stages_us=stages,logical_cube_ops=2*m*n*k,
              physical_ops_per_full_mmad=mm_ops(min(ak,k)),mmads_total=plan['mmads_total'],
              hbm_read_bytes_scenario=hbm_input,l2_to_l1_read_bytes=total_l2_read,
              l1_to_l0_operand_bytes=total_l0_read,write_bytes=2*m*n,logical_read_bytes=cold_input*k,
              read_page_k=page,mmad_k=ak,pages_per_output_tile=pages,
              rate_units='bytes/cycle; MTE per core, HBM/L2 per chip; BF16 arithmetic ops/cycle/core',
              rates=dict(mte2=mte2,mte1=mte1,write=fix,l2=l2,hbm=hbm,cube=peak),
              shared_traffic=dict(hbm_read_bytes=hbm_input,hbm_write_bytes_exposed=visible_write,
                  eventual_write_bytes=out_bytes,l2_write_bytes=out_bytes,
                  output_retained_fraction_assumption=retained_fraction,hbm_efficiency_cap=hbm_eff,l2_efficiency_cap=l2_eff,
                  hbm_effective_gbs=(hbm_input+visible_write)/(1000*total),
                  l2_effective_gbs=(total_l2_read+out_bytes)/(1000*total),
                  hbm_payload_fraction_of_nominal=(hbm_input+visible_write)/(1000*total*chip.hbm_bandwidth_gbs),
                  l2_payload_fraction_of_nominal=(total_l2_read+out_bytes)/(1000*total*l2*ghz)),
              wave_cost_cycles=dict(full=fwave if full_waves else None,tail=tailwave),
              calibrated_on='DT-only phase costs; 950 library scheduling assumptions, checked on historical DT/PR')
    return total,active,0,info
