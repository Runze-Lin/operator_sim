"""Evaluate both frozen versions against the same fresh measurements."""
import json
import math
from pathlib import Path
import runpy
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model.ca import sha256
from operator_model.rms_v11 import predict
from operator_model.wide_v10 import predict as old

OUT = ROOT/"results/v0.11/rms"
inspect = runpy.run_path(str(ROOT/"scripts/wide-v10-trace.py"))["inspect"]
metrics = runpy.run_path(str(ROOT/"scripts/wide-v10-report.py"))["stats"]


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2)+"\n")


freeze = json.loads((OUT/"freeze.json").read_text())
for path, digest in freeze["sha256"].items():
    assert sha256(ROOT/path) == digest
dev = [r for r in json.loads((ROOT/"results/v0.10/wide/rows.json").read_text()) if r["op"]=="rmsnorm"]
fixed = statistics.median(r["reference_us"] for r in dev)
for r in dev:
    shape = r["shape"]
    context = {"resident_aiv":r["prior_aiv"]}
    # A previous independent dispatch becomes development once its timing informs v0.11.
    p = predict("rmsnorm", shape)
    r.update(role="development",previous_us=old("rmsnorm",shape)["predicted_us"],prediction=p,
        predicted_us=p["predicted_us"],sequence_us=predict("rmsnorm",shape,context)["predicted_us"],
        sequence_prediction=predict("rmsnorm",shape,context),fixed_us=fixed)
val = []
batches = []
for i, k in freeze["batches"]:
    batch = f"rmsnorm-{i}-{k}"
    path = ROOT/"runs/v11rms"/batch
    cases = json.loads((OUT/(batch+".json")).read_text())
    assert len(cases)==k-i
    for r in cases:
        assert sha256(ROOT/r["source"]) == r["db_sha256"]
        r.update(inspect(ROOT/r["source"]))
        r["sequence_us"] = r["sequence_prediction"]["predicted_us"]
        assert math.isclose(r["reference_us"],sum(r["phase_us"].values()),abs_tol=1e-7)
        assert math.isclose(r["previous_us"], old("rmsnorm",r["shape"])["predicted_us"])
        assert r["active_aiv"] == r["prediction"]["predicted_active_aiv"]
    val.extend(cases)
    batches.append(dict(name=batch,wall_s=cases[0]["batch_wall_s"],
        db_bytes=sum(p.stat().st_size for p in path.rglob("*.db")),
        raw_bytes=sum(p.stat().st_size for p in path.rglob("*") if p.is_file())))
rows = dev+val
assert len({tuple(r["shape"]) for r in rows})==len(rows)==43
for r in rows:
    wall = []
    for _ in range(100):
        start = time.perf_counter_ns()
        p = predict("rmsnorm",r["shape"],r.get("attrs",{}))
        wall.append((time.perf_counter_ns()-start)/1000)
    assert math.isclose(r["predicted_us"],p["predicted_us"],abs_tol=1e-10)
    r.update(query_median_us=statistics.median(wall),query_max_us=max(wall))
summary = {"rmsnorm":{}}
for role,subset in [("development",dev),("validation",val)]:
    groups = summary["rmsnorm"][role] = {}
    for label in ("all","few_wide","controls"):
        sub = [r for r in subset if label=="all" or r["group"]==label]
        groups[label]={k:metrics(sub,k) for k in ["predicted_us","previous_us","fixed_us","sequence_us"]}
save("rows.json",rows)
save("summary.json",summary)
save("costs.json", {"rmsnorm":dict(launches=10,distinct_holdouts=10,prior_traces_reused=len(dev),
    batches=batches,batch_wall_s=sum(b["wall_s"] for b in batches),
    db_bytes=sum(b["db_bytes"] for b in batches),raw_bytes=sum(b["raw_bytes"] for b in batches),
    query_median_us=statistics.median(r["query_median_us"] for r in val),
    query_max_us=max(r["query_max_us"] for r in val),
    benchmark="100 uncached shape computations per validation row in one loaded interpreter, profile cached. Batch time is sum across two processes.")})
