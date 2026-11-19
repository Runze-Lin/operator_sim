"""Exercise large unchunked row counts online only, without confusing backend and vLLM kernels."""
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from operator_model import OperatorStream
from development.validation.post26.freeze import CHIPS,OUT
from development.validation.sku19.campaign import write

def main():
    rows=[]
    for d in (5120,7168):
        for s in (32*1024,128*1024,256*1024):
            for sku in CHIPS:
                stream=OperatorStream(sku,experimental_post_repeats=True);shape=[8*s,4,d]
                first=stream.advance('mhc_post',shape)
                start=time.perf_counter_ns();q=stream.advance('mhc_post',shape);cost=(time.perf_counter_ns()-start)/1e6
                diag=q.projection['timing_diagnostics'];rows.append(dict(shape=shape,batch=8,context=s,chip=sku,
                    rows_per_longest_aiv=diag['target_plan']['rows_per_core'],active=q.active_aiv,
                    channel_tiles=diag['channel_tiles'],first_us=first.latency_us,repeat_us=q.latency_us,
                    query_ms=cost,work=q.work))
    write(OUT/'large-api.json',dict(calibration_status='Workload-sized extrapolation on all three SKUs; no CA at these shapes. The actual upstream DeepSeek binding selects a separate HcPost implementation.',rows=rows))
if __name__=='__main__':main()
