"""Direct API query overhead, before and after Python resource imports."""
from statistics import median, quantiles
from time import perf_counter_ns
from large_cast import OUT, CHIPS, estimate, read, write

jobs=read(OUT/'cast-v20-freeze.json')['jobs']+[read(OUT/'pro-freeze.json')['current']]
first=perf_counter_ns()
estimate('cast_bf16_fp32',jobs[0]['shape'],chip=next(iter(CHIPS.values())),dtype='bf16')
first_ms=(perf_counter_ns()-first)/1e6
values=[]
for j in jobs:
    for chip in CHIPS.values():
        samples=[]
        for _ in range(31):
            start=perf_counter_ns()
            result=estimate('cast_bf16_fp32',j['shape'],chip=chip,dtype='bf16')
            samples.append((perf_counter_ns()-start)/1e6)
        assert abs(result.latency_us-j['targets'][chip]['us'])<=1e-8
        values.append(dict(shape=j['shape'],chip=chip,n=len(samples),p50_ms=median(samples),
                           p95_ms=quantiles(samples,n=100,method='inclusive')[94],maximum_ms=max(samples)))
write(OUT/'query-wall.json',dict(first_api_ms=first_ms,rows=values,
      scope='Python process and operator_model public API imported, first query includes resource submodule imports. Repeated direct estimate() calls compute the resource schedule again; named chip loader may cache config. Wall times include host process scheduling.'))
