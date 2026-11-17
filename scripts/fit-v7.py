#!/usr/bin/env python3
"""Calibrate local routes and freeze all holdouts BEFORE collecting any holdout."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.ca import sha256
from operator_model.mechanism import predict_case
from operator_model.model_v7 import Case,Predictor,work,fit_nonnegative,contract_attrs,feature_value
OUT=ROOT/'results/v0.7'
MANIFESTS=sorted(ROOT.glob('configs/extension-*-v7.json'))


def records():
    result=[]
    for path in sorted((ROOT/'runs/model-v7').glob('*/result.json')):
        r=json.loads(path.read_text());r['record_path']=str(path.relative_to(ROOT));result.append(r)
    old=json.loads((ROOT/'docs/extensions/smoke-950dt.json').read_text())
    for r in old['records']:
        op=r['op'];shape=list(map(int,r['arguments'][1:3 if op=='rmsnorm' else 4]))
        attrs={} if op=='rmsnorm' else {'self_shape':[shape[1]],'alpha':1,'beta':1}
        m=r['measurement'];measurement={'reference_us':m['reference_us'],'reference_scope':m['reference_scope'],'kernels':[m],'kernel_count':1,'correctness':r['correctness']}
        result.append({'case':{'id':'reused_'+op,'op':op,'shape':shape,'attrs':attrs,'split':'calibration'},'status':'ok',
                       'measurement':measurement,'identity':r,'record_path':'docs/extensions/smoke-950dt.json','reused':True,'wall_seconds':0,'raw_bytes':0})
    return result


def observations(rs):
    output=[]
    for r in rs:
        if r['status']!='ok':continue
        c=r['case'];m=r['measurement']
        output.append({'case':c,'reference_us':m['reference_us'],'scope':m['reference_scope'],'parent_id':c['id'],'kernels':m['kernels'],'record_path':r['record_path']})
        if c['op']=='moe_ffn':
            if len(m['kernels'])!=2:raise ValueError('Cannot attribute MoE components without exactly two verified kernels')
            stages=m.get('stages',[])
            if [s['op'] for s in stages]!=c['components']:raise ValueError('Missing or unordered verified MoE stages')
            for stage in stages:
                op=stage['op'];matches=[k for k in m['kernels'] if k['db_sha256']==stage['db_sha256'] and int(k['helper_info']['launch_id'])==stage['launch_id']]
                if len(matches)!=1:raise ValueError('Ambiguous stage identity')
                k=matches[0]
                cc=dict(c,id=c['id']+'__'+op,op=op)
                output.append({'case':cc,'reference_us':k['reference_us'],'scope':k['reference_scope'],'parent_id':c['id'],'kernels':[k],'record_path':r['record_path']})
    return output


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'frozen-predictions.json').exists():raise ValueError('Already frozen; never refit an observed holdout')
    rs=records()
    if any(r['case']['split']=='holdout' for r in rs):raise ValueError('Holdout already observed')
    obs=observations(rs);specs=[json.loads(p.read_text()) for p in MANIFESTS]
    cases=[c for s in specs for c in s['cases']]
    model={'schema_version':7,'version':'v0.7','base_v6':json.loads((ROOT/'results/v0.6/model.json').read_text()),'routes':{},
           'scope':'DIAGNOSTIC BASELINES ONLY for new routes; base_v6 retains legacy mechanism models. General models use mechanism.py and explicit hardware profile.'}
    for op in sorted({o['case']['op'] for o in obs}-{'moe_ffn'}):
        rows=[o for o in obs if o['case']['op']==op and o['case']['split']=='calibration']
        if len(rows)<2:continue
        cs=[r['case'] for r in rows];attrs=contract_attrs(op,cs[0]['shape'],cs[0]['attrs'])
        if any(contract_attrs(op,c['shape'],c['attrs'])!=attrs for c in cs):raise ValueError('mixed route attributes '+op)
        feature='critical_work'
        if op in ('mhc_post','mhc_sinkhorn') and all(len(r['kernels'])==1 and r['kernels'][0]['active_aiv']==min(r['case']['shape'][0],64) for r in rows):
            feature='activation_log2'
        xs=[feature_value(op,c['shape'],c['attrs'],feature) for c in cs];ys=[r['reference_us'] for r in rows]
        allcases=cs+[c for c in cases if c['split'] in ('calibration','holdout') and (c['op']==op or(op in ('gmm2','grouped_matmul_swiglu_quant_v2') and c['op']=='moe_ffn'))]
        model['routes'][op]={'parameters':fit_nonnegative(xs,ys),'attrs':attrs,'feature':feature,'scope':rows[0]['scope'],
            'domain':[[min(c['shape'][i] for c in allcases),max(c['shape'][i] for c in allcases)] for i in range(len(cs[0]['shape']))],
            'calibration_ids':[c['id'] for c in cs],'calibration_work_min':min(xs),'calibration_work_max':max(xs)}
    (OUT/'model.json').write_text(json.dumps(model,indent=2)+'\n')
    predictor=Predictor(model)
    predictions=[{'id':c['id'],'case':c,'prediction':predictor.predict(Case(c['op'],tuple(c['shape'])),c['attrs'])} for c in cases if c['split']=='holdout']
    hardware=json.loads((ROOT/'configs/hardware-950dt-v7.json').read_text())
    mechanism_predictions=[{'id':c['id'],'prediction':predict_case(c['op'],c['shape'],c['attrs'],hardware)} for c in cases if c['split']=='holdout']
    for c in cases:
        if c['split']=='holdout' and c['op']=='moe_ffn':
            mechanism_predictions.extend({'id':c['id']+'__'+op,'prediction':predict_case(op,c['shape'],c['attrs'],hardware)} for op in c['components'])
    artifacts=['operator_model/mechanism.py','configs/hardware-950dt-v7.json','operator_model/extensions/__init__.py','operator_model/extensions/indexer_mhc.py','operator_model/extensions/kda_moe.py','operator_model/extensions/norm_addmm.py','operator_model/model_v7.py','operator_model/npusim_v7.py','scripts/collect-v7.py','scripts/fit-v7.py','results/v0.7/model.json',
               'operator_model/fast_query.py','operator_model/model_v6.py','operator_model/model_v5.py','operator_model/model_v4.py','operator_model/model_v3.py','operator_model/model_v2.py','operator_model/model.py','scripts/cann-run.sh','build/capture/libsim_soc_adapter.so']
    for path,spec in zip(MANIFESTS,specs):artifacts.extend([str(path.relative_to(ROOT)),spec['binary'],spec['source']])
    frozen={'frozen_utc':datetime.now(timezone.utc).isoformat(),'model_sha256':sha256(OUT/'model.json'),
            'artifact_sha256':{p:sha256(ROOT/p) for p in sorted(set(artifacts))},'predictions':predictions,'mechanism_predictions':mechanism_predictions,
            'budget':{'total_ca_seconds':3600,'per_case_bytes':768*2**20,'retained_bytes':4*2**30},
            'rule':'No holdout measurements read in fit; domains include prespecified holdout shapes, no labels.'}
    (OUT/'calibration.json').write_text(json.dumps(rs,indent=2)+'\n')
    (OUT/'frozen-predictions.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print('Frozen',len(model['routes']),'routes and',len(predictions),'holdouts')

if __name__=='__main__':main()
