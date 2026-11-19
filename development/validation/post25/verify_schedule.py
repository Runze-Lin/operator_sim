"""Replay the closed-form schedule against a literal row-by-row max-plus issue recurrence."""
import math,random,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.post25.calibrate import predict,OUT
from development.validation.sku19.campaign import read,write
from operator_model import load_chip
def main():
    regs=read(OUT/'freeze.json');params=read(OUT/'prices.json');counts=[];worst=0;seen=0
    for r in regs['rows']:
        if r['plan']['route']!='post_token':continue
        chip=load_chip(r['chip']);f=chip.frequency_ghz*1000;cnt=r['plan']['rows_per_core'];q=predict(r['shape'],chip,r['prior_aiv'],params,trace_rows=range(cnt),previous=r['previous_shape'])['pipeline']
        assert abs(q['us']-r['us'])<1e-9 and abs(sum(q['stages_us'].values())-q['us'])<1e-9
        # Recompute unrolled with carried previous write, explicitly checking distance to all earlier work.
        t=q['timeline'];gap=q['gap_cycles'];p=params['by_state'][q['state']]['ports'];last=None
        ratio=(27/64)/(chip.vector_tops['32']/chip.vector_cores)*chip.frequency_ghz/1.65
        for i,row in enumerate(t):
            item='steady' if i else 'first'
            handoff=(p[item]['ready_to_write_fixed']+params['ready_to_write_per_group']*math.ceil(r['shape'][2]/64)*ratio)/f
            ready=max(x[1] for x in row['reads_us']);want=ready+handoff
            if last is not None:want=max(want,last+gap['write_gap_first' if i==1 else 'write_ii']/f)
            worst=max(worst,abs(want-row['write_us'][0]));assert abs(want-row['write_us'][0])<1e-10
            last=want;seen+=1
        counts.append(cnt)
    # Throughput check with the candidate's actual kernel contract only. These are API test inputs.
    cost=[]
    for sku in regs['measured']+regs['extrapolated']:
        for s in (32768,131072,262144):
            shape=[8*s,4,5120];chip=load_chip(sku);st=time.perf_counter_ns();q=predict(shape,chip,0,params);dt=(time.perf_counter_ns()-st)/1e6
            assert q['plan']['rows_per_core']==math.ceil(8*s/chip.vector_cores) and q['pipeline']['timeline']==[]
            cost.append(dict(sku=sku,tokens=8*s,width=5120,rows_per_aiv=q['plan']['rows_per_core'],call_time_ms=dt,us=q['us']))
    write(OUT/'schedule-check.json',dict(frozen_token_cases=len(counts),observed_rows=seen,maximum_unrolled_drift_us=worst,large_shape_cost=cost,interpretation='Timing cost for huge shape-unrolled counts, no CA. The built-in aclnnMhcPost need not be the upstream vLLM HcPost kernel. Absolute device numbers in this test are extrapolations; no end-to-end model measurement.'))
    print('ok',len(counts),seen,worst,max(v['call_time_ms'] for v in cost))
if __name__=='__main__':main()
