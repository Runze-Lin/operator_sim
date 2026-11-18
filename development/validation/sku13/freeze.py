"""Freeze transfer predictions and all installed parameter resources before opening new PR traces."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model import estimate

OUT = ROOT/'results/v0.13/sku'
budget_path = OUT/'holdout-budget.json'
budget = json.loads(budget_path.read_text())
if (ROOT/'runs/sku13-holdout').exists():
    raise ValueError('PR holdout already launched')
files = [budget_path, ROOT/budget['runner'], ROOT/'MANIFEST.in']
files.extend(ROOT/p.removeprefix('include ') for p in (ROOT/'MANIFEST.in').read_text().splitlines() if p.startswith('include '))
files.extend(sorted((ROOT/'operator_model').rglob('*.py')))
predictions = []
for case in budget['cases']:
    row = dict(op=case['op'], shape=case['shape'])
    for key, sku in [('dt','Ascend950DT_9582'),('pr',budget['sku'])]:
        r = estimate(case['op'],case['shape'],chip=sku)
        assert r.latency_us and r.status == ('estimated' if key=='dt' else 'extrapolated')
        row.update({key+'_us':r.latency_us, key+'_aic':r.active_aic, key+'_aiv':r.active_aiv})
    predictions.append(row)
freeze = dict(utc=datetime.now(timezone.utc).isoformat(),
              sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              predictions=predictions)
with (OUT/'holdout-freeze.json').open('x') as stream:
    stream.write(json.dumps(freeze,indent=2)+'\n')
print(predictions)
