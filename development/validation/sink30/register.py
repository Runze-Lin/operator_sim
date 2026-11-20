"""Seal current one-register schedule and already queued comparisons before their measurement."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/v0.30/sink';sys.path.insert(0,str(ROOT))
from operator_model import load_chip
from operator_model.sink_unknown import price
from operator_model.mhc_sink_resource import sink as old
from operator_model.sink_history import price as known
DT='Ascend950DT_9582';PR='Ascend950PR_957b';FUT='Ascend970DT'
def read(p):return json.loads(p.read_bytes())
def digest(p):return sha256(p.read_bytes()).hexdigest()
def predict(t,i,sku):
    c=load_chip(sku);a={'num_iters':i};x=price((t,4),c,a)
    try:legacy=old((t,4),c,a)[0]
    except ValueError:legacy=None
    return dict(us=x[0],prior_us=legacy,diag=x[3])
def freeze():
    out=OUT/'freeze.json'
    if out.exists():raise FileExistsError('frozen evidence cannot be overwritten')
    p=ROOT/'results/v0.26/sink/register.json'
    cast=[]
    for j in read(p)['jobs']:
        sh=j['shape'];sku=j['chip'];r=predict(sh[0],j['iters'],sku)
        if r['us'] is not None and r['diag']['target_plan'].get('register_passes')==1:
            # Use exact v0.26 blind data contract, including the reference clock and SKU.
            assert math.isclose(r['us'],j['unknown_us'],rel_tol=0,abs_tol=5e-13)
            cast.append(dict(sku=sku,shape=sh,iters=j['iters'],host_group=j['group'],index=j['index'],
                             forecast_us=r['us'],original_us=j['unknown_us'],delta_us=r['us']-j['unknown_us']))
    future=[];v28=ROOT/'results/v0.28/fusion';large=ROOT/'results/v0.29/large/fusion'
    seen={(j['shape'][0],j['iters'])for j in read(v28/'register.json')['jobs'] if j['group']=='first'}
    for j in read(v28/'register.json')['jobs']:
        if j['group']!='sequence' or j['chip'] not in (DT,PR):continue
        sku=j['chip'];label=j['group']+'-'+sku
        future.append(dict(cohort='v28-sequence',sku=sku,t=j['shape'][0],iters=j['iters'],h=j['prior_h'],index=j['index'],
          used_raw=str((v28/'raw'/label).relative_to(ROOT)),receipt=str((v28/'receipts'/(label+'.json')).relative_to(ROOT)),
          unique_new_shape=(j['shape'][0],j['iters'])not in seen))
    for j in read(v28/'eligibility.freeze.json')['jobs']:
        if j['chip'] not in (DT,PR):continue
        sku=j['chip'];label='eligibility-'+sku
        future.append(dict(cohort='v28-register',sku=sku,t=j['shape'][0],iters=j['iters'],h=j['prior_h'],index=j['index'],
             used_raw=str((v28/'raw'/label).relative_to(ROOT)),receipt=str((v28/('eligibility.receipt-'+sku+'.json')).relative_to(ROOT)),unique_new_shape=True))
    for j in read(large/'freeze.json')['jobs']:
        for o in j['outputs']:
            if o['op']!='mhc_sinkhorn':continue
            for sku in (DT,PR):
                label=j['id']+'-'+sku
                future.append(dict(cohort='v29-'+j['id'],sku=sku,t=o['shape'][0],iters=20,h=0 if j['op']=='mhc_sinkhorn' else None,index=None,
                    used_raw=str((large/'raw'/label).relative_to(ROOT)),receipt=str((large/(label+'.json')).relative_to(ROOT)),unique_new_shape=True))
    for item in future:
        item['had_started']=(ROOT/item['used_raw']).exists() or (ROOT/item['receipt']).exists()
        item['prediction']=predict(item['t'],item['iters'],item['sku'])
        item['future970']=predict(item['t'],item['iters'],FUT)
    sources=['operator_model/sink_unknown.py','operator_model/sink_history.py','operator_model/mhc_sink_resource.py',
             'results/v0.26/sink/prices.json','results/v0.26/sink/register.json','results/v0.28/fusion/register.json',
             'results/v0.28/fusion/eligibility.freeze.json','results/v0.29/large/fusion/freeze.json',
             'toolchains/cann/9.2.1/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/mhc_sinkhorn/op_kernel/arch35/mhc_sinkhorn_arch35.h',
             str(Path(__file__).resolve().relative_to(ROOT))]
    data=dict(utc=datetime.now(timezone.utc).isoformat(),price_source='v0.26 premeasurement unknown_us without any refit; single-register boundary from VF register iterations, former 10-shape cohort; G>1 has no new default qualification',source_sha256={s:digest(ROOT/s)for s in sources},
              original_ten_shape_price_parity=cast,future=future, additional_ca_started=0)
    with out.open('x')as f:json.dump(data,f,indent=2);f.write('\n')
if __name__=='__main__':freeze()
