"""Run this file using an isolated interpreter with the built v0.23 wheel installed."""
from hashlib import sha256
from importlib.metadata import version
import json
from math import ceil, isclose
from pathlib import Path
from statistics import median
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/v0.23"
CHIPS = ("Ascend950DT_9582","Ascend950PR_957b","Ascend970DT")

def read(path):
    return json.loads(path.read_text())

def main():
    start=perf_counter_ns()
    from operator_model import __file__ as source, estimate, estimate_rms_history_band
    import_ms=(perf_counter_ns()-start)/1e6
    assert sys.flags.isolated and not Path(source).resolve().is_relative_to(ROOT)
    assert version("ascend-operator-model")=="0.23.0"
    seen={}; timings={}; bands=[]
    def call(op,shape,attrs,chip):
        key = (op,tuple(shape),json.dumps(attrs,sort_keys=True),chip)
        if key not in seen:
            start=perf_counter_ns();result=estimate(op,shape,attrs,chip=chip)
            elapsed=(perf_counter_ns()-start)/1e6
            assert result.status!="unsupported" and elapsed<100,(key,elapsed,result.reason)
            timings.setdefault(attrs.get("implementation",op),[]).append(elapsed)
            seen[key]=result
        return seen[key]
    for row in read(ROOT/"results/v0.22/evaluation.json")["rows"]:
        for chip,expected in ((row["chip"],row["us"]),(CHIPS[-1],row["future_us"])):
            actual=call(row["op"],row["shape"],row["attrs"],chip).latency_us
            assert actual==expected or (expected is not None and isclose(actual,expected,rel_tol=1e-10))
    for row in read(ROOT/"results/kda-resource23/freeze.json")["predictions"]:
        r=call("kda_chunk",row["shape"],{"implementation":"fla_26_9_1_v2"},row["chip"])
        assert isclose(r.latency_us,row["serialized_device_us"],rel_tol=1e-12)
    for chip in CHIPS:
        for seq in (32768,131072,262144):
            for op,shape,attrs in [("mhc_post",[1,8*seq,4,5120],{"implementation":"vllm_hc_post"}),
                                   ("kda_chunk",[8,seq,12,12,128,128],{"implementation":"fla_26_9_1_v2"}),
                                   ("rmsnorm",[8*seq,5120],{}),("quant",[8*seq,7168],{}),
                                   ("cast_bf16_fp32",[8*seq,5120],{}),("sigmoid_fp32",[8*seq,12],{})]:
                call(op,shape,attrs,chip)
        for shape in ((2,2944),(4,608),(128,1184),(563,4048),(8*32768,5120)):
            start=perf_counter_ns();r=estimate_rms_history_band(shape,chip=chip);elapsed=(perf_counter_ns()-start)/1e6
            assert elapsed<100 and 0<r.lower_us<=r.upper_us,(chip,shape,elapsed)
            bands.append(elapsed)
    def summary(times):
        return dict(n=len(times),median_ms=median(times),p95_ms=sorted(times)[ceil(.95*len(times))-1],max_ms=max(times))
    wheel,=(ROOT/"build/wheels/v023").glob("*.whl")
    data=dict(version=version("ascend-operator-model"),python=sys.version.split()[0],wheel_sha256=sha256(wheel.read_bytes()).hexdigest(),
              import_ms=import_ms,estimates=summary(sum(timings.values(),[])),per_route={k:summary(v) for k,v in timings.items()},history_bands=summary(bands))
    (OUT/"wheel-check.json").write_text(json.dumps(data,indent=2)+"\n");print(data["estimates"],data["history_bands"])

if __name__=="__main__":
    main()
