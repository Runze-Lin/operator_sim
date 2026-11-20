"""Verify the historical public-shape price on the installed baseline, outside the checkout."""
from hashlib import sha256
from importlib.metadata import distribution
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[3]
p=Path(sys.argv[1]).resolve()
if not sys.flags.isolated or Path.cwd().is_relative_to(ROOT):
    raise ValueError("run with -I outside the source tree")
sys.path.insert(0,str(p))
import operator_model
assert Path(operator_model.__file__).is_relative_to(p)
version=distribution("ascend-operator-model").version
assert version=="0.34.0"
arch=ROOT/"results/v0.35/post/replay.json"
rows=json.loads(arch.read_bytes())['replay']
diff=[]
for x in rows:
    r=operator_model.estimate("mhc_post",x["shape"],chip=x["chip"],dtype="bf16")
    assert r.latency_us and abs(r.latency_us-x["public_us"])<1e-9
    diff.append(abs(r.latency_us-x["public_us"]))
wheel,=(ROOT/"build/wheels/v034").glob("*.whl")
(ROOT/"results/v0.35/post/baseline-wheel.json").write_text(json.dumps(dict(
    baseline_version=version,wheel_sha256=sha256(wheel.read_bytes()).hexdigest(),count=len(diff),
    max_abs_drift_us=max(diff),input_scope="all 239 historical shape-only CA windows; actual public BF16 estimate from installed v0.34.0",
    caller_outside_repository=True,isolated=True),indent=2)+"\n")
print(version,len(diff),max(diff))
