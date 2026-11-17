#!/usr/bin/env python3
"""Fit only an observed VF envelope residual on 19 old traces, then freeze."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.ca import sha256
from operator_model.model_v2 import nnls
from operator_model.model_v6 import Case,Predictor,applies
from operator_model.model_v5 import Predictor as V5


def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def main():
    out=ROOT/'results/v0.6'
    if (out/'frozen-predictions.json').exists():raise ValueError('Already frozen; no post-holdout refit')
    source=ROOT/'results/v0.4/dataset.json'
    records=[r for r in json.loads(source.read_text())['records'] if r['status']=='ok' and r['case']['op']=='quant']
    phases=json.loads((ROOT/'runs/model-v6/diagnostic-quant-phases.json').read_text())
    by_id={a['id']:a for a in phases};old=V5(ROOT/'results/v0.5/model.json');xs=[];ys=[];training=[]
    assert len(records)==len(by_id)==19
    assert len({tuple(r['case']['shape']) for r in records})==19
    for r in records:
        a=by_id[r['case']['id']];m=r['measurement'];p=old.predict(Case('quant',r['case']['shape']))['plan']
        path=Path(m['report_path'])/'simulator/trace.json'
        if sha256(path)!=a['trace_sha256'] or a['trace_sha256']!=m['trace_sha256']:raise ValueError('Stale phase reduction')
        c=next(c for c in m['cores'] if c['name']==a['core'])
        busy=c['busy_union_us_by_pipe'].get('VECTOR_FAMILY',0)
        envelope=sum(t['vf_span_us'] for t in a['iterations'])
        if len(a['iterations'])!=p['iterations'] or envelope<busy-1e-5:raise ValueError('Invalid phase evidence')
        x=[1,p['active_aiv']/64 if p['iterations']==1 else 0];y=max(0,envelope-busy)
        xs.append(x);ys.append(y)
        training.append({'id':r['case']['id'],'shape':r['case']['shape'],'role':'development_reclassified',
            'trace_sha256':a['trace_sha256'],'core':a['core'],'vf_envelope_us':envelope,'simd_union_us':busy,
            'gap_us':y,'features':x,'observed_aiv':m['active_aiv'],'predicted_aiv':p['active_aiv']})
    weights=nnls(xs,ys);rows=[]
    for i,r in enumerate(records):
        train=[j for j in range(19) if i!=j];loo=nnls([xs[j] for j in train],[ys[j] for j in train])
        pred=old.predict(Case('quant',r['case']['shape']));guard=applies(pred['plan'])
        current=pred['phase_schedule']['pipeline_us']+sum(a*b for a,b in zip(weights,xs[i])) if guard else pred['predicted_us']
        cv=pred['phase_schedule']['pipeline_us']+sum(a*b for a,b in zip(loo,xs[i])) if guard else pred['predicted_us']
        rows.append({'id':r['case']['id'],'shape':r['case']['shape'],'guard':guard,'reference_us':r['measurement']['reference_us'],
            'v05_us':pred['predicted_us'],'v06_us':current,'residual_loo_us':cv,'heldout_coefficients':loo})
    write(out/'trace-analysis.json',{'source_dataset_sha256':sha256(source),'training':training,'phase_reduction':phases})
    write(out/'calibration-diagnostics.json',{'coefficients':weights,'rows':rows,
        'scope':'LOO refits only new residual; frozen base already used some old points. Development selection diagnostic, NOT independent accuracy.',
        'selection':'One guarded candidate compared with unchanged v0.5; mixed gains/regressions retained; no new shape measured yet.'})
    model={'schema_version':6,'version':'0.6.0','soc_version':'Ascend950DT_9582',
        'base_v5_model':json.loads((ROOT/'results/v0.5/model.json').read_text()),
        'quant_vf_gap':{'coefficients':weights,'calibration_count':19,'guard':'single UB tile and cols_per_core>=1024',
            'meaning':'Observed VF window minus same-core SIMD interval union; scheduling residual, not ISA service or independent peak',
            'fit_features':['1','active_aiv/64 for singleUB, else0']},'fit_utc':datetime.now(timezone.utc).isoformat()}
    write(out/'model.json',model);new=Predictor(model);predictions=[]
    seen={tuple(r['case']['shape']) for r in records}
    for c in json.loads((ROOT/'configs/sampling-v6.json').read_text())['cases']:
        if tuple(c['shape']) in seen:raise ValueError('Seen holdout')
        case=Case(c['op'],c['shape']);predictions.append({**c,'prediction':new.predict(case),'v05_prediction':old.predict(case)})
    artifacts=['operator_model/model.py',*[f'operator_model/model_v{v}.py' for v in range(2,7)],
        'operator_model/ca.py','operator_model/npusim.py','scripts/fit-v6.py','scripts/collect-v6.py','configs/sampling-v6.json',
        'results/v0.5/model.json','results/v0.6/trace-analysis.json','results/v0.6/calibration-diagnostics.json',
        'smoke/three_ops.cpp','build/smoke/three_ops','scripts/cann-run.sh','build/capture/libsim_soc_adapter.so']
    write(out/'frozen-predictions.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'model_sha256':sha256(out/'model.json'),'artifact_sha256':{f:sha256(ROOT/f) for f in artifacts},'predictions':predictions})
    print('coefficients',weights)
    for p in predictions:print(p['id'],p['shape'],p['prediction']['predicted_us'],p['v05_prediction']['predicted_us'])
if __name__=='__main__':main()
