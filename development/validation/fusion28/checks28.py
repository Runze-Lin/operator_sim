"""Check state reachability and portable resource scaling independently of CA totals."""
from dataclasses import replace
from math import isclose
from statistics import median
from time import perf_counter
from landmarks import read,save,OUT
from schedule28 import estimate,load_chip
def main():
    checks=0;clocks=[]
    for job in read(OUT/'register.json')['jobs']:
        chip=load_chip(job['chip']);s=job['shape'];it=job['iters'];t=perf_counter();full=estimate(s,it,chip);clocks.append((perf_counter()-t)*1000)
        assert isclose(full['point_us'],job['point_us'],rel_tol=1e-12)
        active=job['conditional']['active']
        for lo,hi in ((0,0),(1,1),(0,active-1),(active-1,active),(active,chip.vector_cores),(0,chip.vector_cores)):
            ev=estimate(s,it,chip,lo,hi)
            expect={('first' if h==0 else 'resident' if h>=active else 'expand') for h in range(lo,hi+1)}
            assert set(ev['states'])==expect
            L,U=ev['range_us'];p=ev['point_us']
            assert L-1e-11<=p<=U+1e-11
            assert max(abs(p/L-1),abs(p/U-1))<=max(.25,ev['model_uncertainty_pct']/100)+1e-10
            if lo==hi:assert isclose(L,U)
            checks+=1
    base=load_chip()
    def case(chip):return estimate([2257,4],4,chip,0,0)['states']['first']
    ref=case(base);fast_vec=case(replace(base,vector_tops={**base.vector_tops,'32':base.vector_tops['32']*2}))
    slow_hbm=case(replace(base,hbm_bandwidth_gbs=4.));slow_dma=case(replace(base,details={**base.details,'mte2_bandwidth_gbs':1.}))
    high_clock=case(replace(base,frequency_ghz=base.frequency_ghz*1.2,vector_tops={k:(v*1.2 if v else v) for k,v in base.vector_tops.items()}))
    assert fast_vec['epochs_us']['steady_pair_period'] < ref['epochs_us']['steady_pair_period'] and fast_vec['us']<ref['us']
    assert slow_hbm['us']>slow_dma['us']>ref['us']>high_clock['us']
    for chip in [replace(base,details={**base.details,'ub_bytes':400}),replace(base,vector_tops={**base.vector_tops,'32':None})]:
        try:case(chip);raise AssertionError('unsupported target was priced')
        except ValueError:pass
    small=estimate([1,4],10,base)
    assert set(small['states'])=={'first','resident'}
    save(OUT/'checks.json',dict(frozen_values_recomputed=len(clocks),history_interval_checks=checks,
        no_expand_when_one_aiv=True,rejected_missing_fp32_and_insufficient_ub=True,
        isolated_resources_us={s:r['us']for s,r in [('dt_cold',ref),('2x_vec_fp32_only',fast_vec),('hbm_4_GBs',slow_hbm),('mte2_1_GBs',slow_dma),('1.2x_clock_same_ops_cycle',high_clock)]},
        online_equivalent_model_only_query_ms={'median':median(clocks),'max':max(clocks),'n':len(clocks),'scope':'in-process three-state solver after importing modules and loading each chip; original timings are host dependent'}))
if __name__=='__main__':main()
