"""Join qualified successor receipts to their exact frozen price and capture clock."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime
import json

DT,PR,FUT='Ascend950DT_9582','Ascend950PR_957b','Ascend970DT'
AT='attn_res_fwd · resident inference prefill'

def incoming(root,api):
    from development.validation.release32 import receipts as prev
    from math import isclose
    def read(p):return json.loads(p.read_bytes())
    def h(p):return sha256(p.read_bytes()).hexdigest()
    def make(label,s,chip,attrs,kernel,ids,ca,p,frozen,cat,why):
        return dict(op=label,shape=s,chip=chip,attrs=attrs,kernel=kernel,identity=[ids] if isinstance(ids,str) else ids,
                    ca_us=ca,source=str(p.relative_to(root)),forecast_before_ca_us=frozen,role=why,data_role=cat,classification=why)
    def accepted(path,registration,price_reg,c,shape):
        r=read(path);v=r.get('verdict',r)
        if v.get('qualified') is not True:return None
        actual_shape=r.get('shape')
        if actual_shape is None and r.get('id'):
            job,=[j for j in read(registration)['jobs'] if j['id']==r['id']];actual_shape=job['shape']
        assert (r.get('chip') or r.get('sku'))==c and actual_shape==shape and (r.get('freeze_sha256') or r.get('registration_sha256'))==h(registration),path
        cost=r['cost'];assert cost['exit_code']==0 and not cost['stop_reason'];cat,why=prev.relationship(read(price_reg)['utc'],cost.get('started_utc') or cost.get('begin'),cost.get('finished_utc') or cost.get('end') or cost.get('ended_utc'))
        return v,cat,why
    # Former reservations that may have completed since the previous score. The caller de-duplicates by physical identity.
    for label,op,shape,chip,attrs,k,src,ids,ca,old,expected,cat,why in prev.matmul_windows():
        yield make(label,shape,chip,attrs,k,ids,ca,root/src,expected,cat,why)
    for label,op,shape,chip,attrs,k,src,ids,ca,old,cat,why in (*prev.other_full_windows(),*prev.standalone_hcpost()):
        if label in ('lightning_indexer','attn_res_fwd'):continue # later target-specific registrations below
        yield make(label,shape,chip,attrs,k,ids,ca,root/src,old,cat,why)
    for path in (root/'results/v0.33/large').glob('rms-*-*.json'):
        v=read(path)
        if v.get('qualified') is not True:continue
        assert v['registration_sha256']==h(path.parent/'freeze.json') and len(v['trace_sha256'])==64
        yield dict(make('rmsnorm · msopprof',v['shape'],v['chip'],{},v['stage'],v['trace'],v['ca_us'],path,v['public_v032_us'],'replay','previously measured before offline requalification'),verified_trace_sha256=v['trace_sha256'])
    base=root/'results/v0.34';reg=base/'addmm/freeze.json';j=read(reg)
    for chip,tag in ((DT,'dt'),(PR,'pr')):
        p=reg.parent/(tag+'.json')
        if p.exists() and (a:=accepted(p,reg,reg,chip,j['shape'])):
            v,cat,why=a;assert v['api']=='aclnnAddmm' and v['output_checked']==j['expected_output_elements'] and v['bad']==0 and v['outputs_rejecting_bias_free_bf16_dot']>0
            yield make('addmm',j['shape'],chip,j['attrs'],v['kernel'],v['trace_sha256'],v['ca_us'],p,j['forecasts'][chip]['prediction_us'],cat,why)
    reg=base/'attn/freeze.json'
    for j in read(reg)['jobs']:
        for i,(chip,tag) in enumerate(((DT,'dt'),(PR,'pr'))):
            t,_,_,n=j['shape'];p=reg.parent/f'{t}-{n}-{tag}.json' if j['new'] else root/j['existing'][i]
            acquisition=reg if j['new'] else root/'results/v0.29/large/vector/freeze.json'
            if p.exists() and (a:=accepted(p,acquisition,reg,chip,j['shape'])):
                v,cat,why=a;assert '40021' in v['kernel'] and v['instruction_us']>0
                yield make(AT,j['shape'],chip,j['attrs'],v['kernel'],v['trace_sha256'],v['instruction_us'],p,j['prices'][chip]['us'],cat,why)
    reg=base/'indexer/freeze.json';ir=read(reg);older=root/'results/v0.29/large/fusion/freeze.json'
    for i,j in enumerate(ir['forecasts']):
        for chip in (DT,PR):
            if i==0:p=reg.parent/(chip+'.json');acquisition=reg
            else:
                run=next(x for x in read(older)['jobs'] if x['id']=='indexer' or any(y.get('op')=='lightning_indexer' for y in x['outputs']));p=older.parent/(run['id']+'-'+chip+'.json');acquisition=older
            if p.exists() and (a:=accepted(p,acquisition,reg,chip,j['shape'])):
                v,cat,why=a;s,=v['stages'];assert v['ca_us']['lightning_indexer']>0
                yield make('lightning_indexer',j['shape'],chip,{},s['kernel'],s['trace_sha256'],v['ca_us']['lightning_indexer'],p,j['prices'][chip]['us'],cat,why)

    reg=base/'recurrent/freeze.json'
    if reg.exists():
        f=read(reg);job=f['job'];assert job['shape'][0]==job['shape'][2]==job['shape'][3]==1
        for chip,tag in ((DT,'dt'),(PR,'pr')):
            path=reg.parent/(tag+'.json')
            if path.exists() and (a:=accepted(path,reg,reg,chip,job['shape'])):
                v,cat,why=a;st,=v['stages'];assert st['active_aiv']==1 and st['active_aic']==0 and v['numerical'] and len(v['released_text_sha256'])==64
                yield make('kda_recurrent',job['shape'],chip,job['attrs'],st['kernel'],st['trace_sha256'],v['ca_us'],path,f['forecasts'][chip]['us'],cat,why)
    reg=root/'results/v0.33/post/freeze.json';f=read(reg)
    for chip in (DT,PR):
        path=reg.parent/(chip+'.json')
        if not path.exists():continue
        cap=read(path);assert cap['freeze_sha256']==h(reg);cost=cap['cost']
        cat,why=prev.relationship(f['utc'],cost['started_utc'],cost.get('finished_utc'))
        for w in cap.get('rows',[]):
            if not w.get('timing_qualified'):continue
            idx=w['i'];job,=[x for x in f['rows'][chip] if x['i']==idx];assert cost['exit_code']==0 and not cost['stop_reason'] and w['shape']==job['shape']
            st=w['stage'];history=[x['shape'] for x in f['rows'][chip] if x['i']<=idx]
            args=w['shape'],chip,{},st['kernel'],st['trace_sha256'],st['ca_us'],path
            yield make('mhc_post · shape',*args,job['v032_shape_us'],cat,why)
            yield dict(make('mhc_post · stream',*args,None,'replay','prospective acquisition; default stream price without identical frozen public point'),launch_sequence=history)
    reg=root/'results/v0.32/post/freeze.json';pref=read(reg)['predictions']
    for target,ca,k,ids,src,future,cat,why in prev.post_windows(api):
        chip=target['chip'];sh=target['shape'];p=root/src;args=sh,chip,{},k,ids,ca,p
        calls=sorted((z for z in pref if z['chip']==chip and z['group']==target['group'] and z['ordinal']<=target['ordinal']),key=lambda z:z['ordinal'])
        yield make('mhc_post · shape',*args,target['shape_us'],cat,why)
        yield dict(make('mhc_post · stream',*args,target['ordered_us'],cat,why),launch_sequence=[x['shape'] for x in calls])
