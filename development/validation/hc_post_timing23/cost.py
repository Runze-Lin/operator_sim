"""Guard accelerated large repeated rows against the frozen scheduler; measure query cost."""
import json
from math import isclose
from statistics import median
from time import perf_counter_ns

from extract import OUT
from fast import accelerated
from model import ROOT,load,predict


def benchmark():
    shape_count=0;largest_delta=0.
    chips=('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT')
    for chip in chips:
        for d in (16,64,128,192,256,512,1024,1280,1536,3072,4080,4096,4112,7168,16384):
            for t in (1,56,57,64,65,129,1024,4097,16384):
                orig=predict(t,d,chip);replay=accelerated(t,d,chip)
                delta=abs(replay['core_ticks']-orig['core_ticks'])
                if not isclose(replay['device_us'],orig['device_us'],rel_tol=1e-9,abs_tol=1e-9):
                    raise ValueError((chip,t,d,orig,replay))
                assert {k:v for k,v in replay.items() if k not in ('device_us','core_ticks','skipped_steady_rows')}=={k:v for k,v in orig.items() if k not in ('device_us','core_ticks','skipped_steady_rows')}
                largest_delta=max(delta,largest_delta);shape_count+=1
    large=[]
    for chip in chips:
        cfg=load(ROOT/'configs/chips'/f'{chip}.json')
        for w in (1280,5120,7168):
            for s in (32768,131072,262144):
                t=s*8;times=[];base0=perf_counter_ns();oracle=predict(t,w,chip);slow=(perf_counter_ns()-base0)/1e6
                for _ in range(9):
                    start=perf_counter_ns();v=accelerated(t,w,chip);times.append((perf_counter_ns()-start)/1e6)
                assert isclose(v['device_us'],oracle['device_us'],rel_tol=1e-9,abs_tol=1e-9)
                gb=(v['logical_gm_read_bytes']+v['logical_gm_write_bytes'])/1e9
                large.append(dict(sku=chip,shape=[8,s,4,w],predicted_us=v['device_us'],extrapolation=True,ca_us=None,
                                  working_set_gb=gb,full_tensors_fit_card=gb<=cfg['hbm_capacity_gb'],
                                  query_median_ms=median(times),query_max_ms=max(times),frozen_oracle_single_ms=slow))
    return dict(configs=list(chips),regression_comparisons=shape_count,max_core_delta_ticks=largest_delta,
                timing='one Python process, preloaded chip and parameters, fresh schedule each call; 9 wall-clock samples. Host process initialization excluded.',
                large=large)


if __name__=='__main__':
    out=benchmark();(OUT/'query-cost.json').write_text(json.dumps(out,indent=2)+'\n')
    print('Checks',out['regression_comparisons'],'worst clocks',out['max_core_delta_ticks'],'query max ms',max(v['query_max_ms'] for v in out['large']))
