"""Frozen parity, device resource sensitivity and model query cost checks."""
import math
import statistics
import time
from dataclasses import replace
from evidence import OUT,DT,PR,FUT,read,save
from registration import REG
from model import forecast
from operator_model import load_chip
def run():
    values=[];err=[];chips={n:load_chip(n) for n in (DT,PR,FUT)}
    for j in read(REG)['jobs']:
        c=chips[j['chip']]
        for _ in range(100):
            start=time.perf_counter_ns();r=forecast(j['shape'],j['iters'],c,j['prior_h']);values.append((time.perf_counter_ns()-start)/1e6)
        err.append(abs(r['latency_us']-j['known']['latency_us']))
        assert r['vf_calls_per_core']==2*j['iters']+2 and r['active_aiv']==j['known']['active_aiv']
    assert max(err)<1e-12 and max(values)<100
    c=chips[DT]
    assert forecast([257,4],4,c,0)['register_passes']==1 and forecast([513,4],4,c,0)['register_passes']==2
    assert forecast([67,4],10,c,0)['latency_us']>forecast([67,4],10,c,64)['latency_us']
    assert forecast([67,4],10,replace(c,name='unseen'),0)==forecast([67,4],10,c,0)
    slower=replace(c,vector_tops={**dict(c.vector_tops),'32':c.vector_tops['32']/2})
    assert forecast([67,4],10,slower,0)['latency_us']>forecast([67,4],10,c,0)['latency_us']
    save(OUT/'checks.json',dict(frozen_rows=len(err),max_parity_error_us=max(err),queries=len(values),maximum_ms=max(values),
                                median_ms=statistics.median(values),p99_ms=sorted(values)[int(len(values)*.99)],mode='in-process after importing chip configs and loading resource parameters'))
if __name__=='__main__':run()
