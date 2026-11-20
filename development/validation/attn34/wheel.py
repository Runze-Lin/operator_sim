"""Check installed operator_model in an isolated interpreter outside the source tree."""
from hashlib import sha256
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"results/v0.34/attn"


def run(prefix):
    if not sys.flags.isolated or Path.cwd().is_relative_to(ROOT):
        raise ValueError("run python -I outside source checkout")
    sys.path.insert(0,str(prefix))
    from operator_model import estimate
    import operator_model
    from importlib.metadata import distribution

    if not Path(operator_model.__file__).is_relative_to(prefix):
        raise ValueError("not installed package")
    dist=distribution("ascend-operator-model")
    frozen=OUT/"freeze.json"
    reg=json.loads(frozen.read_bytes())
    rows=[]
    for j in reg["jobs"]:
        for chip,want in j["prices"].items():
            r=estimate(j["op"],j["shape"],j["attrs"],chip=chip)
            assert r.latency_us==want["us"] and r.status==want["status"]
            assert r.projection["target_plan"]==want["plan"] and r.projection["stages_us"]==want["stages"]
            assert r.projection["timing_diagnostics"]["dma_read_bytes"]==want["dma_read_bytes"]
            assert r.work["logical_write_bytes"]==want["dma_write_bytes"]
            assert estimate(j["op"],j["shape"],j["attrs"],chip=chip,timing_mode="calibrated").latency_us is None
            rows.append(dict(shape=j["shape"],chip=chip,price_us=r.latency_us,absolute_drift_us=abs(r.latency_us-want["us"])))
    replay=estimate("attn_res_fwd",[1,7168,8,3],dict(addend=True,write_idx=3,materialize=True,output_norm=True),chip="Ascend950PR_957b")
    assert abs(replay.latency_us-12.799)<0.001 and replay.status!="unavailable"
    wheel,=(ROOT/"build/wheels/v034").glob("*.whl")
    result=dict(release=dist.version,wheel=wheel.name,wheel_sha256=sha256(wheel.read_bytes()).hexdigest(),
                freeze_sha256=sha256(frozen.read_bytes()).hexdigest(),isolated_interpreter=True,external_cwd=True,
                loaded_module=str(Path(operator_model.__file__).relative_to(prefix)),complete_frozen_forecasts=rows,
                replay_materialized_and_bank_write=dict(shape=[1,7168,8,3],chip="Ascend950PR_957b",price_us=replay.latency_us,
                                                        actual_legacy_observation_us=11.488,role="historical complete window, different output alias and bank writes"))
    (OUT/"wheel-check.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(dist.version,len(rows),"frozen prices and stage/work fields exactly match")
if __name__=="__main__":run(Path(sys.argv[1]).resolve())
