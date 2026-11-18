"""Target-resource BF16 MatMulV3. No target observations or source-device latency."""
import json
from functools import lru_cache
from math import ceil, log2
from pathlib import Path
from .matmul_v15 import matmul_plan

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
    gm,gn=ceil(m/am),ceil(n/an)
    repeat_input=2*(m*gn+n*gm) # aggregate bytes for one unit of K
    cold_input=2*(m+n)
    l2_size=d.get('l2_bytes') or 134217728
    input_hbm_per_k=cold_input if cold_input*min(k,page)<=l2_size else repeat_input
    def wave(num_tiles):
        share=num_tiles/jobs
        def stages(depth):
            cp=lambda z:mm_ops(z)/peak
            frag=min(ak,depth);last=(depth-1)%ak+1;parts=ceil(depth/ak)
            read_work=max(2*depth*max(bm,bn)/mte2,repeat_input*depth*share/l2,
                          input_hbm_per_k*depth*share/hbm)
            l0_work=lambda z: 2*z*max(bm,bn)/mte1
            # CUBE and MTE1 for the following fragment can overlap.
            issue=max(cost('mmad_ii_cycles')+cp(frag),l0_work(frag))
            compute=(parts-1)*issue+cost('mmad_latency_cycles')+cp(last)
            return read_work,l0_work(frag),compute
        first=min(k,page);read,l0,compute=stages(first);tail=stages((k-1)%page+1)
        # Two buffer pairs: consume one while prefilling the other. Two A/B
        # descriptor issues per page and hold at most two pages in flight.
        ii=max(2*cost('read_ii_cycles'),read,(cost('read_latency_cycles')+read)/2,compute)
        return {'read_fill':cost('read_latency_cycles')+read,'l0_lead':cost('l0_latency_cycles')+l0,
                'page_intervals':(pages-1)*ii,'last_mmad':tail[2],
                'write_tail':cost('drain_cycles')+max(2*bm*bn/fix,2*m*n*share/l2,2*m*n*share/hbm)}
    last_jobs=(jobs-1)%active+1
    full=wave(active);last=full if last_jobs==active else wave(last_jobs)
    ticks={name:max(0,waves-1)*value+last[name] for name,value in full.items()}
    ticks['launch']=cost('launch_cycles')+c['arrival_log2_cycles']*log2(active)
    stages={name:value/(ghz*1000) for name,value in ticks.items()}
    total=sum(stages.values())
    info=dict(model='target_resource_matmul',target_plan=plan,stages_us=stages,logical_cube_ops=2*m*n*k,
              physical_ops_per_full_mmad=mm_ops(min(ak,k)),mmads_total=plan['mmads_total'],
              hbm_read_bytes_scenario=input_hbm_per_k*k,l2_to_l1_read_bytes=repeat_input*k,
              l1_to_l0_operand_bytes=repeat_input*k,write_bytes=2*m*n,logical_read_bytes=cold_input*k,
              read_page_k=page,mmad_k=ak,pages_per_output_tile=pages,
              rate_units='bytes/cycle; MTE per core, HBM/L2 per chip; BF16 arithmetic ops/cycle/core',
              rates=dict(mte2=mte2,mte1=mte1,write=fix,l2=l2,hbm=hbm,cube=peak),
              calibrated_on='950DT only; target resource projection')
    return total,active,0,info
