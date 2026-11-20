"""Compare the public backend to the pre-acquisition forecasts and time distinct call contexts."""
import json
from math import ceil, isclose
from pathlib import Path
from statistics import median
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'results/v0.31/matmul/device'
SNIPPET='''from time import perf_counter_ns as now
from operator_model import estimate
import json,sys
soc=sys.argv[1]; shape=list(map(int,sys.argv[2:])); times=[]
for i in range(51):
    start=now(); result=estimate('matmul',shape,chip=soc,dtype='bf16'); times.append((now()-start)/1e6)
print(json.dumps(dict(us=result.latency_us,ms=times,status=result.status,active_aic=result.active_aic,active_aiv=result.active_aiv)))'''


def build():
    from hashlib import sha256
    freeze=(OUT/'freeze.json').read_bytes(); reg=json.loads(freeze); rows=[]
    for job in reg['rows']:
        for soc,p in job['predictions'].items():
            obs=json.loads(subprocess.check_output([sys.executable,'-c',SNIPPET,soc,*map(str,job['shape'])],cwd=ROOT));ms=obs['ms'];samples=sorted(ms[1:]);matched=isclose(p['us'],obs['us'],rel_tol=1e-12)
            if not matched:raise ValueError(f"Public price no longer matches frozen forecast for {job['shape']}, {soc}")
            rows.append(dict(shape=job['shape'],soc=soc,public_device_us=obs['us'],status=obs['status'],frozen_match=matched,public_roster=[obs['active_aic'],obs['active_aiv']],cold_ms=ms[0],warm_median_ms=median(samples),warm_p95_ms=samples[ceil(.95*len(samples))-1],warm_max_ms=max(samples)))
    (OUT/'query-cost.json').write_text(json.dumps(dict(scope='Source checkout public API. Fresh Python interpreter per shape and chip; module import excluded. Cold = first estimate after import, including lazy preparation; warm = 50 successive calls at fixed shape without clearing chip/parameter caches. Device microseconds differ from wall clock milliseconds. Main installed-wheel validation is separate.',freeze_sha256=sha256(freeze).hexdigest(),rows=rows),ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':build()
