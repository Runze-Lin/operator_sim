"""One-at-a-time configuration experiments; none of these rows have a CA label."""
import json
from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model import estimate, load_chip


def scaled(rates, factor):
    return {kind: None if rate is None else rate*factor for kind,rate in rates.items()}


def run():
    dt = load_chip()
    unknown = replace(dt,name='illustrative-reference-resources')
    configs = {
        'dt': dt,
        'unknown_same_resources': unknown,
        'cube_cores_x2_same_per_core': replace(unknown,cube_cores=2*dt.cube_cores,cube_tops=scaled(dt.cube_tops,2)),
        'vector_peaks_x0.5': replace(unknown,vector_tops=scaled(dt.vector_tops,.5)),
        'offchip_bandwidth_x0.5': replace(unknown,hbm_bandwidth_gbs=dt.hbm_bandwidth_gbs/2),
        'clock_and_peaks_x1.1': replace(unknown,frequency_ghz=dt.frequency_ghz*1.1,
                vector_tops=scaled(dt.vector_tops,1.1),cube_tops=scaled(dt.cube_tops,1.1)),
        'custom_120c_120v': load_chip(ROOT/'configs/chips/scenario-example.json'),
    }
    rows = []
    for op,shape in [('matmul',(512,512,256)),('fa',(9,32768,64,1,1)),('fa',(9,2048,64,32,4))]:
        ref = estimate(op,shape,chip=dt)
        for label,chip in configs.items():
            t = perf_counter_ns()
            answer = estimate(op,shape,chip=chip)
            elapsed = (perf_counter_ns()-t)/1e6
            if answer.latency_us is None or answer.latency_us<=0 or ref.work != answer.work or elapsed>=100:
                raise ValueError((label,shape,answer.reason,elapsed))
            if label=='unknown_same_resources' and abs(ref.latency_us-answer.latency_us)>1e-7:
                raise AssertionError(('expected resource-identity timing',op,shape,answer.latency_us,ref.latency_us))
            proj = answer.projection or {}
            rows.append(dict(case=label,op=op,shape=shape,latency_us=answer.latency_us,
                  aic=answer.active_aic,aiv=answer.active_aiv,query_ms=elapsed,
                  plan=proj.get('target_plan'),timing=proj.get('timing_diagnostics'),
                  inherited=chip.inherited_fields,ca_us=None))
    return rows


if __name__=='__main__':
    print(json.dumps(run(),indent=2,ensure_ascii=False))
