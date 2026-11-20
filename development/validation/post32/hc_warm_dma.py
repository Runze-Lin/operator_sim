"""Freeze a cyclic steady-DMA candidate from DT descriptor timings before the PR result."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate,load_chip
from operator_model.hc_post_timing import _schedule, _parameters, _dma_prices

OUT=ROOT/'results/v0.32/post'
DT,PR,FUT='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'
def prices(lanes):
    result={}
    for key,n in [('x',2),('r',8)]:
        points=[]
        for i,size in enumerate([4096*n,1024*n]):
            spans=[r[key+str(i)][1]-r[key+str(i)][0] for lane in lanes for r in lane['rows'][2:]]
            points.append((size,median(spans)))
        (x,y),(xx,yy)=points
        slope=(y-yy)/(x-xx)
        result['res' if key=='r' else 'x']=(yy-slope*xx,slope)
    return result

def prediction(shape,name,steady):
    c=load_chip(name)
    v=estimate('mhc_post',shape,{'implementation':'vllm_hc_post'},chip=name)
    plan=v.projection['target_plan']
    if plan['d_tiles_per_row']!=2 or plan['tail_elements']!=1024:
        raise ValueError('Measured two-payload schedule required for this validation')
    p=_parameters()['prices'];d=_dma_prices(c,p)
    ds=dict(d)
    for q,(h,k) in steady.items():
        bw=c.details.get('mte2_bandwidth_gbs')
        ds[q]=(h,c.frequency_ghz/bw if bw else k)
    widths=[4096,1024]
    scale=p['vector_reference_ops32_cycle']/(c.vector_tops['32']*1000/c.vector_cores/c.frequency_ghz)
    cohorts=[(plan['min_rows_per_core'],plan['active_aiv']-1)]
    if plan['larger_cores']:
        cohorts.append((plan['max_rows_per_core'],plan['larger_cores']-1))
    trials=[]
    for rows,rank in cohorts:
        # The first two rows occupy both slots and retain the original fill.
        # Probe three large consecutive row counts: a fixed per-row translation
        # of the clocks reproduces the same steady logical slot schedule.
        fill=_schedule(min(rows,2),widths,rank,p,d,scale)['core_ticks']
        wr=[_schedule(i,widths,rank,p,ds,scale)['core_ticks'] for i in (15,16,17)]
        if abs((wr[2]-wr[1])-(wr[1]-wr[0]))>1e-6:
            raise ValueError('Could not establish a steady period-one price on target')
        gap=wr[2]-wr[1]
        trials.append(dict(rows=rows,rank=rank,first_two_rows_ticks=fill,warm_ii_ticks=gap,
                           local_ticks=fill+max(0,rows-2)*gap))
    local=max(s['local_ticks'] for s in trials)
    pipe=v.projection['pipeline']
    floor=max(pipe['offchip_cold_bound_ticks'],pipe['l2_peak_bound_ticks'])
    us=max(local,floor)/(1000*c.frequency_ghz)
    return dict(sku=name,shape=shape,cohorts=trials,physical_floor_us=floor/(1000*c.frequency_ghz),
                public_baseline_us=v.latency_us,candidate_us=us,explicit_mte2=c.details.get('mte2_bandwidth_gbs'),
                status='extrapolated' if name==FUT else 'frozen diagnostic prediction')

def main():
    path=OUT/'hc-pr-freeze.json'
    if path.exists():raise FileExistsError(path)
    evidence=json.loads((OUT/'hc-dt-rows.json').read_bytes())
    pars=prices(evidence['lanes'])
    assert not (ROOT/'results/v0.31/post'/('vllm_hc_post-'+PR+'.json')).exists()
    # No PR trace or run.log is read by this script. The small-shape
    # public calibration was frozen before the large workload was selected.
    results=[prediction([1,1024,4,5120],chip,pars) for chip in (DT,PR,FUT)]
    path.write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),shape=[1,1024,4,5120],
        training='DT large trace only, rows 3–16 of ten preselected completed lane files. Two descriptor sizes for each payload.',
        eligible='Repeated rows with source tiling 4096+1024, two alternating UB slots, steady issue alignment confirmed. Current check uses at least three rows/AIV.',
        support='DT observations include contention and are local effective response costs; HBM hit and bus ownership are unobserved.',
        calibration_sha256=sha256((OUT/'hc-dt-rows.json').read_bytes()).hexdigest(),
        original_prices_sha256=sha256((ROOT/'configs/vllm-hc-post-arch35.json').read_bytes()).hexdigest(),
        candidate_script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),new_warm_read_cycles=pars,
        unchanged='initial two rows, write service, vector, clock, rank fanout, overall HBM/L2 floors',
        validation='Use a separate predeclared result on PR with full physical END, all output values and released text verification; compare same-shape elapsed time and DT/PR speedup against original public. Development size only, not default admission.',
        results=results),indent=2)+'\n')
    for r in results:print(r['sku'],r['public_baseline_us'],r['candidate_us'],r['cohorts'])
if __name__=='__main__':main()
