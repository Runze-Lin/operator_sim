import json,sys,time,subprocess
from statistics import median
from freeze import OUT,ROOT,DT,PR,FUT
RUN="""from operator_model import estimate
from time import perf_counter_ns as tick
import json,sys
shape,sku=json.loads(sys.argv[1]);times=[]
for i in range(41):
 t=tick();r=estimate('matmul',shape,chip=sku,dtype='bf16');times.append((tick()-t)/1e6)
print(json.dumps([times,r.latency_us]))"""
rows=[]
for shape in ([1,2304,1536],[32768,3072,512],[1048576,128,512],[2097152,128,512],[2**32,128,512]):
 for sku in (DT,PR,FUT):
    context=json.dumps([shape,sku]);times,us=json.loads(subprocess.check_output([sys.executable,'-c',RUN,context],cwd=ROOT));rows.append(dict(shape=shape,sku=sku,first_ms=times[0],repeated_median_ms=median(times[1:]),repeated_max_ms=max(times[1:]),us=us))
(OUT/'cost.json').write_text(json.dumps(dict(protocol='New Python process and import operator_model before timer for each shape/SKU; first estimate includes lazy resource setup. Following 40 ordinary public queries; no simulator launches or trace reads. Last huge synthetic input tests algorithm cost only (tensor set exceeds HBM).',rows=rows),indent=2)+'\n')
