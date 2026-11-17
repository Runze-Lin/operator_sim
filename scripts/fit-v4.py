#!/usr/bin/env python3
"""Reuse old evidence, select small-control fallback, freeze all comparators before CA."""
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import runpy
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.ca import sha256
from operator_model.model import fit_nonnegative_line
from operator_model.model_v3 import Predictor as V3,fit_quant,quant_plan,quant_schedule
from operator_model.model_v4 import Case,Predictor,fit
metrics=runpy.run_path(str(ROOT/'scripts/evaluate-v1.py'))['metrics']

def main():
 out=ROOT/'results/v0.4';manifest=ROOT/'configs/sampling-v4.json'
 if (out/'frozen-predictions.json').exists():raise SystemExit('v0.4 already frozen')
 rs=json.loads((out/'imported-calibration.json').read_text())['records'];an=json.loads((out/'trace-analysis.json').read_text())
 spec=json.loads(manifest.read_text())
 if {(c['op'],tuple(c['shape'])) for c in spec['cases']} & {(r['case']['op'],tuple(r['case']['shape'])) for r in rs}:raise ValueError('Previously seen holdout')
 base=json.loads((ROOT/'results/v0.3/model.json').read_text());model=fit(rs,an,base)
 cv={};qs=[r for r in rs if r['case']['op']=='quant']
 for mode in ('v3_refit','constant_floor','saturated','shrink25','shrink50'):
  rows=[]
  for r in qs:
   train=[t for t in rs if t!=r];costs=fit_quant(train,an)['costs'];p=quant_plan(Case('quant',r['case']['shape']))
   small=[t for t in train if t['case']['op']=='quant' and quant_plan(Case('quant',t['case']['shape']))['work_per_core']<=4096]
   xs=[math.log2(1+quant_plan(Case('quant',t['case']['shape']))['active_aiv']) for t in small];ys=[t['measurement']['reference_us'] for t in small]
   if mode=='saturated':xs=[min(x,4) for x in xs]
   floor=fit_nonnegative_line(xs,ys)
   if mode in ('constant_floor','shrink25','shrink50'):
    floor['slope_us_per_unit']*={'constant_floor':0,'shrink25':.25,'shrink50':.5}[mode]
    floor['intercept_us']=statistics.mean(ys)-floor['slope_us_per_unit']*statistics.mean(xs)
   if mode=='saturated':floor={'intercept_us':floor['intercept_us']+floor['slope_us_per_unit']*min(4,math.log2(1+p['active_aiv'])),'slope_us_per_unit':0}
   if mode!='v3_refit':costs['control_floor']=floor
   rows.append({'id':r['case']['id'],'reference_us':r['measurement']['reference_us'],'predicted_us':quant_schedule(p,costs)['predicted_us'],
                'constant_us':statistics.mean(t['measurement']['reference_us'] for t in train if t['case']['op']=='quant'),'small':p['work_per_core']<=4096})
  cv[mode]={'rows':rows,'all':metrics(rows,'predicted_us'),'small':metrics([r for r in rows if r['small']],'predicted_us')}
 selected=min(cv,key=lambda k:cv[k]['small']['wape'])
 if selected!='saturated':raise ValueError('Fallback selection changed; review before freeze')
 (out/'calibration-diagnostics.json').write_text(json.dumps({'selected':selected,'candidate_cv':cv,'warning':'Development model selection on 13 historical quant shapes; not independent validation.'},indent=2)+'\n')
 model.update(fit_utc=datetime.now(timezone.utc).isoformat(),manifest_sha256=sha256(manifest))
 (out/'model.json').write_text(json.dumps(model,indent=2)+'\n');p=Predictor(model);old=V3(base)
 predictions=[]
 for c in spec['cases']:
  case=Case(c['op'],c['shape']);predictions.append({**c,'prediction':p.predict(case),'v03_prediction':old.predict(case),
      'v03_domain_extension':max(case.shape)>4096 or (case.op=='fa' and case.shape[3]!=1)})
 artifacts=['operator_model/model_v4.py','operator_model/model_v3.py','operator_model/model_v2.py','operator_model/model.py','operator_model/ca.py',
   'results/v0.3/model.json','results/v0.4/imported-calibration.json','results/v0.4/trace-analysis.json','results/v0.4/calibration-diagnostics.json',
   'scripts/fit-v4.py','scripts/collect-v1.py','smoke/three_ops.cpp','build/smoke/three_ops']
 frozen={'frozen_utc':datetime.now(timezone.utc).isoformat(),'model_sha256':sha256(out/'model.json'),'manifest_sha256':sha256(manifest),
  'implementation_sha256':sha256(ROOT/'operator_model/model_v4.py'),'shared_implementation_sha256':sha256(ROOT/'operator_model/model.py'),
  'artifact_sha256':{f:sha256(ROOT/f) for f in artifacts},'predictions':predictions}
 (out/'frozen-predictions.json').write_text(json.dumps(frozen,indent=2)+'\n')
 for x in predictions:print(x['id'],x['shape'],round(x['prediction']['predicted_us'],4),round(x['v03_prediction']['predicted_us'],4))
 print('CV selected',selected)
if __name__=='__main__':main()
