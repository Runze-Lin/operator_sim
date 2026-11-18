"""Fit only previously observed exclusive writeback stages; seal predictions before CA."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model.rms_v11 import predict, pressure
from operator_model.wide_v10 import predict as previous

OUT = ROOT / "results/v0.11/rms"
old = [r for r in json.loads((ROOT/"results/v0.10/wide/rows.json").read_text()) if r["op"] == "rmsnorm"]
assert len(old) == len({tuple(r["shape"]) for r in old}) == 33
if (OUT/"freeze.json").exists():
    raise FileExistsError("Predictions already frozen")
target = [r["phase_us"]["MTE3"] for r in old]
fits = []
for width in (32, 64, 128):
    features = []
    for r in old:
        t, d = r["shape"]
        features.append([1, pressure(r["active_aiv"], (t+63)//64, width), t*(d*2+4)/1024])
    (base, neighbour, service), norm = nnls(features, target)
    fits.append(dict(write_start_us=base, neighbour_us=neighbour, effective_us_per_kib=service,
                     window_bytes=width, residual_l2_us=norm))
profile = dict(min(fits, key=lambda fit: fit["residual_l2_us"]), candidates=fits,
               evidence="33 older RMSNorm critical-core disjoint MTE3 observations")
(OUT/"profile.json").write_text(json.dumps(profile,indent=2)+"\n")
shapes = [[3,5120],[2,5376],[4,4608],[1,6912],[33,1536],[38,1536],
          [64,4096],[128,4096],[256,4096],[512,4096]]
assert len({tuple(s) for s in shapes}) == 10 and not any(s == r["shape"] for s in shapes for r in old)
fixed = median(r["reference_us"] for r in old)
cases = []
for i, shape in enumerate(shapes):
    if i in (0,6):
        resident = 0
    p = predict("rmsnorm",shape)
    cases.append(dict(id=f"v11_rms_{i}",op="rmsnorm",shape=shape,attrs={},kind="NORM",groups=2,args=[*shape,0],
                      role="validation",prediction=p,previous_us=previous("rmsnorm",shape)["predicted_us"],
                      fixed_us=fixed,prior_aiv=resident,group="few_wide" if shape[0]<=4 and shape[1]>=2048 else "controls",
                      sequence_prediction=predict("rmsnorm",shape,{"resident_aiv":resident})))
    resident = max(resident,p["predicted_active_aiv"])
files = ["build/vector-v8/NORM","build/capture/libsim_soc_adapter.so","operator_model/rms_v11.py","operator_model/wide_v10.py",
         "operator_model/vector_v9.py","operator_model/vector_v8.py","results/v0.10/wide/profile.json","results/v0.11/rms/profile.json",
         "development/validation/rms11/freeze.py","development/validation/rms11/collect.py"]
digest = {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
(OUT/"freeze.json").write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),cases=cases,sha256=digest,
    per_batch_seconds=600,disk_budget_gib=1.6,batches=[[0,6],[6,10]],group_rule="few_wide T<=4 and D>=2048",
    budget="two batches; 10 launches; no re-fit",previous="v0.10 public shape-only estimate"),indent=2)+"\n")
