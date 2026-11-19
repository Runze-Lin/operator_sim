"""Write-once prediction registration. Does not schedule CA."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from model import DT, OUT, ROOT, predict, read

HERE=Path(__file__).resolve().parent
PROFILE=('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT')
# Different full shapes from the development register, spanning the subchunk seam.
HOLDOUT=((3,1),(5,2),(17,1),(33,2))
shapes=[[1,t,h,h,128,128] for t,h in HOLDOUT]
development=read(ROOT/'results/kda-v23/development-register.json')
assert not ({tuple(x) for x in shapes}&{tuple(x['shape']) for x in development['cases']})
files=[HERE/'model.py',HERE/'extract.py',Path(__file__).resolve(),OUT/'dt-primitive-services.json',
       ROOT/'results/v0.16/matmul.json',ROOT/'operator_model/chips.py',ROOT/'results/kda-v23/development-register.json']
files.extend(ROOT/'configs/chips'/f'{p}.json' for p in PROFILE)
def digest(p):return sha256(p.read_bytes()).hexdigest()
observed_raw=sorted(x.name for x in (ROOT/'results/kda-v23/raw').iterdir() if x.is_dir())
for t,h in HOLDOUT:
    assert not any(name.endswith(f'-t{t}-h{h}') for name in observed_raw)
result=dict(utc=datetime.now(timezone.utc).isoformat(),role='pre-observation candidate freeze; no CA authorized here',
            sha256={str(p.relative_to(ROOT)):digest(p) for p in files},development_registered_cases=development['cases'],
            observed_development_raw=observed_raw,matching_holdout_CA='none on development campaign; no CA has been requested by this script',
            predictions=[predict(s,p) for s in shapes for p in PROFILE])
with (OUT/'freeze.json').open('x') as f:
    json.dump(result,f,indent=2);f.write('\n')
