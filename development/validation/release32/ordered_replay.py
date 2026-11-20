"""Validate all historical Post API timings through the isolated wheel, retaining original inputs."""
import json, sys, tempfile
from pathlib import Path
from hashlib import sha256
from math import isclose
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[3];prefix=Path(sys.argv[1]).resolve()
assert sys.flags.isolated;sys.path.insert(0,str(prefix))
import operator_model as installed
assert Path(installed.__file__).is_relative_to(prefix)
sys.path.append(str(ROOT))
from development.validation.post32 import evaluate as archived
obj={}
with tempfile.TemporaryDirectory(prefix='post-replay-before-') as f,patch.object(archived,'save',side_effect=lambda n,d:obj.update({n:d})):
 archived.score(archived.old_api(f))
target=ROOT/'results/v0.32/post/replay.json'
original=json.loads(target.read_bytes());expected={r['sha256']:r for r in original['rows']}
actual={r['sha256']:r for r in obj['replay.json']['rows']}
assert expected.keys()==actual.keys()
cols=('shape_us','stream_us','extrapolated_970_stream_us','extrapolated_970_shape_us','old_us','v031_shape_us','ordered_us')
for k,v in expected.items():
 assert v['scenario']==actual[k]['scenario'] and all(isclose(actual[k][c],v[c],rel_tol=1e-10,abs_tol=1e-7) for c in cols)
wheel,=(ROOT/'build/wheels/v032').glob('*.whl')
record=dict(historical_physical_windows=len(actual),compared_time_columns=cols,wheel_sha256=sha256(wheel.read_bytes()).hexdigest(),original_sha256=sha256(target.read_bytes()).hexdigest(),release=installed.__version__,source_hashes_verified=len(original['original_sources']),protocol='Original mixed/predecessor calls executed in order through separately installed v0.32/v0.31 wheels; no source model import, no history inferred from target shape, no original report overwritten.')
for path,sha in original['original_sources'].items():assert sha256((ROOT/path).read_bytes()).hexdigest()==sha
(ROOT/'results/v0.32/ordered-wheel-check.json').write_text(json.dumps(record,indent=2)+'\n');print(record['historical_physical_windows'])
