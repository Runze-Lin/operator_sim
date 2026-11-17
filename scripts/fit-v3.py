#!/usr/bin/env python3
"""Fit using existing traces, diagnose baselines, then freeze before six new cases."""
from datetime import datetime,timezone
import json
from pathlib import Path
import runpy
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.ca import sha256
from operator_model.model_v3 import Case,Predictor,fit,fit_quant,quant_plan,quant_schedule
metrics=runpy.run_path(str(ROOT/'scripts/evaluate-v1.py'))['metrics']

def main():
    out=ROOT/'results/v0.3';manifest=ROOT/'configs/sampling-v3.json'
    if (out/'frozen-predictions.json').exists():raise SystemExit('v0.3 already frozen')
    spec=json.loads(manifest.read_text());records=json.loads((out/'dataset.json').read_text())['records']
    if any(r['case']['split']!='calibration' for r in records):raise SystemExit('New holdout already observed')
    if {(c['op'],tuple(c['shape'])) for c in spec['cases']} & {(r['case']['op'],tuple(r['case']['shape'])) for r in records}:
        raise SystemExit('Holdout shape already used')
    analysis=json.loads((out/'trace-analysis.json').read_text());base=json.loads((ROOT/'results/v0.2/model.json').read_text())
    model=fit(records,analysis,base);model['fit_utc']=datetime.now(timezone.utc).isoformat()
    predictor=Predictor(model)
    # Genuine leave-one-shape-out refitting of all quant phase parameters.
    cv=[]
    for r in records:
        if r['case']['op']!='quant':continue
        train=[t for t in records if t['case']['id']!=r['case']['id']]
        qmodel=fit_quant(train,analysis)
        p=quant_plan(Case('quant',r['case']['shape']));schedule=quant_schedule(p,qmodel['costs'])
        cv.append({'id':r['case']['id'],'shape':r['case']['shape'],'reference_us':r['measurement']['reference_us'],
                   'predicted_us':schedule['predicted_us'],
                   'constant_us':statistics.mean(t['measurement']['reference_us'] for t in train if t['case']['op']=='quant'),
                   'small_control':p['work_per_core']<=1024})
    diagnostics={'quant_leave_one_shape_out':cv,'quant_cv_metrics':{field:metrics(cv,field) for field in ('predicted_us','constant_us')},
                 'scope':'All prior inputs are development/calibration; CV is diagnostic, not a fresh independent evaluation.',
                 'baseline_explanation':{}}
    for op in ('quant','fa'):
        rs=[r for r in records if r['case']['op']==op];ys=[r['measurement']['reference_us'] for r in rs]
        diagnostics['baseline_explanation'][op]={'count':len(rs),'reference_min_us':min(ys),'reference_max_us':max(ys),
            'reference_cv':statistics.pstdev(ys)/statistics.mean(ys),
            'max_core_compute_busy_fraction_median':statistics.median(max(c['busy_fraction_by_pipe'].get('VECTOR_FAMILY' if op=='quant' else 'CUBE',0) for c in r['measurement']['cores']) for r in rs)}
    diagnostics['quant_pre_vector_fraction_median']=statistics.median(a['pre_vector_fraction'] for a in analysis['quant'])
    (out/'calibration-diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    model['manifest_sha256']=sha256(manifest)
    (out/'model.json').write_text(json.dumps(model,indent=2)+'\n')
    artifacts=['operator_model/model_v3.py','operator_model/model_v2.py','operator_model/model.py','operator_model/ca.py',
               'results/v0.3/trace-analysis.json','results/v0.3/imported-calibration.json','smoke/three_ops.cpp','build/smoke/three_ops']
    cases=[r['case'] for r in records]+spec['cases']
    freeze={'frozen_utc':datetime.now(timezone.utc).isoformat(),'model_sha256':sha256(out/'model.json'),
            'manifest_sha256':sha256(manifest),'implementation_sha256':sha256(ROOT/'operator_model/model_v3.py'),
            'shared_implementation_sha256':sha256(ROOT/'operator_model/model.py'),
            'artifact_sha256':{path:sha256(ROOT/path) for path in artifacts},
            'predictions':[{**c,'prediction':predictor.predict(Case(c['op'],c['shape']))} for c in cases]}
    (out/'frozen-predictions.json').write_text(json.dumps(freeze,indent=2)+'\n')
    print(json.dumps(diagnostics,indent=2))
    for c in spec['cases']:
        p=predictor.predict(Case(c['op'],c['shape']))
        print(c['id'],c['shape'],'predicted_us',p['predicted_us'],'instructions',p['estimated_max_core_selected_opcode_count'])
if __name__=='__main__':main()
