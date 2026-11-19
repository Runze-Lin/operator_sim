"""Public API wall time with new interpreter/modules/chip config for every cold query."""
import json
from pathlib import Path
import statistics
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.vector21.campaign import assert_resource_revision, OUT, read, write

PROGRAM="""import json,sys
from time import perf_counter_ns
from operator_model import estimate
op,shape,chip=json.loads(sys.argv[1])
rows=[]
for _ in range(32):
    t=perf_counter_ns(); out=estimate(op,shape,chip=chip); rows.append((perf_counter_ns()-t)/1e6)
print(json.dumps([out.latency_us,rows]))"""
def benchmark():
    assert_resource_revision()
    rows=[]
    for j in read(OUT/'freeze.json')['jobs']:
        for chip,frozen in j['targets'].items():
            us,ms=json.loads(subprocess.check_output([sys.executable,'-c',PROGRAM,
                json.dumps([j['op'],j['shape'],chip])],text=True,cwd=ROOT))
            assert abs(us-frozen['us'])<1e-8
            rows.append(dict(op=j['op'],shape=j['shape'],chip=chip,extrapolated_us=us,cold_ms=ms[0],
                             warm_median_ms=statistics.median(ms[1:]),warm_max_ms=max(ms[1:])))
    write(OUT/'query-wall.json',dict(queries=len(rows)*32,processes=len(rows),
        max_cold_ms=max(r['cold_ms'] for r in rows),max_warm_ms=max(r['warm_max_ms'] for r in rows),
        method='Cold query begins after API import in a new Python process; includes lazy resource and chip configuration loading. Followed by 31 individual calls.',rows=rows))
if __name__=='__main__':
    benchmark()
