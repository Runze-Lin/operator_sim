"""Verify an installed wheel in an external cwd, using only shipped price data."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter_ns

def main():
    folder, freeze_path, wheel = map(Path,sys.argv[1:])
    sys.path.insert(0,str(folder))
    os.chdir(folder.parent)
    start=perf_counter_ns()
    from operator_model import estimate
    import operator_model
    imported=(perf_counter_ns()-start)/1e6
    assert Path(operator_model.__file__).resolve().is_relative_to(folder.resolve())
    opts={'implementation':'fla_26_9_1_v2'}
    reg=json.loads(freeze_path.read_bytes())
    first=perf_counter_ns()
    estimate('kda_chunk',[1,2,1,1,128,128],opts)
    cold=(perf_counter_ns()-first)/1e6
    checks=[]
    for row in reg['predictions']:
        actual=estimate('kda_chunk',row['shape'],opts,chip=row['chip'])
        assert actual.status=='extrapolated' and actual.projection['qualification']=='experimental'
        assert actual.latency_us==row['serialized_device_us']
        assert all(x['device_us']==y['device_us'] and x['dma_bytes']==y['dma_bytes'] for x,y in zip(actual.projection['target_plan']['stages'],row['stages']))
        checks.append([row['chip'],row['shape'],actual.latency_us])
    latencies=[]
    for _ in range(50):
        start=perf_counter_ns()
        actual=estimate('kda_chunk',[8,262144,12,12,128,128],opts,chip='Ascend970DT')
        latencies.append((perf_counter_ns()-start)/1e6)
    assert actual.latency_us and max(latencies)<100
    assert estimate('kda_chunk',[1,2,1,1,128,128]).latency_us is None
    assert estimate('kda_chunk',[1,2,1,1,128,128],opts,timing_mode='calibrated').latency_us is None
    print(json.dumps(dict(python=sys.version.split()[0],wheel=wheel.name,wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
             verification='installed in isolated path, called outside checkout with Python -I; no CA/CANN files',
             import_ms=imported,first_uncached_query_ms=cold,qualifications_checked=checks,
             large_input=[8,262144,12,12,128,128],large_query_count=len(latencies),
             large_query_median_ms=statistics.median(latencies),large_query_max_ms=max(latencies),
             large_970_projection_us=actual.latency_us),indent=2))

if __name__=='__main__':main()
