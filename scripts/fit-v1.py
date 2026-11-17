#!/usr/bin/env python3
"""Fit calibration only, freeze every prediction before holdout is collected."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from operator_model.ca import sha256
from operator_model.model import Case, Predictor, fit


def main():
    out = ROOT/'results/v0.1'
    if (out/'frozen-predictions.json').exists():
        raise SystemExit('Already frozen: preserve this version and use a new version for revisions')
    spec = json.loads((ROOT/'configs/sampling-v1.json').read_text())
    data = json.loads((out/'dataset.json').read_text())
    if any(r['case']['split']=='holdout' for r in data['records']):
        raise SystemExit('Holdout has already been collected; use a new untouched holdout')
    records = [r for r in data['records'] if r['case']['split']=='calibration']
    model = fit(records,sha256(ROOT/'configs/sampling-v1.json'))
    model['fit_utc'] = datetime.now(timezone.utc).isoformat()
    model['training_keys'] = [r['cache_key'] for r in records]
    model['implementation_sha256'] = sha256(ROOT/'operator_model/model.py')
    (out/'model.json').write_text(json.dumps(model,indent=2)+'\n')
    predictor = Predictor(model)
    predictions = [{**case,'prediction':predictor.predict(Case(case['op'],tuple(case['shape'])),use_cache=False)} for case in spec['cases']]
    freeze = {'frozen_utc':datetime.now(timezone.utc).isoformat(),
              'model_sha256':sha256(out/'model.json'),
              'manifest_sha256':sha256(ROOT/'configs/sampling-v1.json'),
              'implementation_sha256':sha256(ROOT/'operator_model/model.py'),
              'predictions':predictions}
    (out/'frozen-predictions.json').write_text(json.dumps(freeze,indent=2)+'\n')
    for op,m in model['operators'].items():
        print(op,'correction',m['total_correction'],'mapping',m['mapping_checks'])
    print('Frozen predictions:',out/'frozen-predictions.json')


if __name__=='__main__':
    main()
