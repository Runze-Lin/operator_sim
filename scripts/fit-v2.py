#!/usr/bin/env python3
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.ca import sha256
from operator_model.model_v2 import Case,Predictor,fit


def main():
    out=ROOT/'results/v0.2';manifest=ROOT/'configs/sampling-v2.json'
    if (out/'frozen-predictions.json').exists():raise SystemExit('v0.2 is already frozen')
    spec=json.loads(manifest.read_text());records=json.loads((out/'dataset.json').read_text())['records']
    if any(r['case']['split']=='holdout' for r in records):raise SystemExit('New holdout already observed')
    model=fit(records,sha256(manifest))
    model['fit_utc']=datetime.now(timezone.utc).isoformat()
    model['implementation_sha256']=sha256(ROOT/'operator_model/model_v2.py')
    model['shared_implementation_sha256']=sha256(ROOT/'operator_model/model.py')
    (out/'model.json').write_text(json.dumps(model,indent=2)+'\n')
    predictor=Predictor(model)
    cases={r['case']['id']:r['case'] for r in records}
    cases.update({c['id']:c for c in spec['cases']})
    predictions=[{**c,'prediction':predictor.predict(Case(c['op'],c['shape']))} for c in cases.values()]
    freeze={'frozen_utc':datetime.now(timezone.utc).isoformat(),'model_sha256':sha256(out/'model.json'),
            'manifest_sha256':sha256(manifest),'implementation_sha256':model['implementation_sha256'],
            'shared_implementation_sha256':model['shared_implementation_sha256'],'predictions':predictions}
    (out/'frozen-predictions.json').write_text(json.dumps(freeze,indent=2)+'\n')
    for op,m in model['operators'].items():print(op,'weights',m['total_weights'],'regimes',m['branches'],'mapping',m['mapping_checks'])


if __name__=='__main__':main()
