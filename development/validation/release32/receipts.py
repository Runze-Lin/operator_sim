"""Only explicit registered endpoints. Do not glob partial/audit/diagnostic files."""
from pathlib import Path
from hashlib import sha256
import json, math, datetime
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/v0.32'; DT,PR='Ascend950DT_9582','Ascend950PR_957b'
def read(p):return json.loads(p.read_bytes())
def digest(p):return sha256(p.read_bytes()).hexdigest()
def stamp(s):return datetime.datetime.fromisoformat(s.replace('Z','+00:00'))
def close(a,b):return a is not None and b is not None and math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-7)
def qualified_stage(receipt,stage,ca):
 k=stage.get('kernel');h=stage.get('trace_sha256')
 if not (isinstance(k,str) and k and isinstance(h,str) and len(h)==64 and close(ca,stage.get('ca_us')) and ca>0):
  raise ValueError(f'Invalid complete stage: {receipt}')
 return k,h
def relationship(s,begin,end):
 if not begin or stamp(begin)<stamp(s):return 'replay',('blinded_after_start' if end and stamp(end)>stamp(s) else 'history_unconfirmed')
 return 'independent','before_acquisition'
def early_registered():
 sink=OUT/'large/freeze.json';sd=read(sink)
 base=ROOT/'results/v0.31/recurrent/freeze.json';bd=read(base)
 k16=OUT/'large/recurrent-freeze.json';kd=read(k16)
 for sku,tag in zip((DT,PR),('dt','pr')):
  fresh=OUT/'large'/('sink1024-'+sku+'.json')
  if fresh.is_file():
   rec=read(fresh);r=rec['verdict']
   if r.get('qualified') is True:
    assert rec['freeze_sha256']==digest(sink) and rec['chip']==sku
    assert r['shape']==[1024,4] and r['h']==0 and r['iters']==20 and r['all_output_elements']==16384
    p=next(x for x in sd['predictions'][sku] if x['shape']==r['shape'] and x['iterations']==r['iters'])
    s=r['stage'];ca=s['ca_us'];k,h=qualified_stage(fresh,s,ca)
    assert s['written_bytes']==65536 and rec['cost']['exit_code']==0 and not rec['cost']['stop_reason']
    cat,evidence=relationship(sd['utc'],rec['cost']['started_utc'],rec['cost']['ended_utc'])
    for name,attrs,old,pred in [('mhc_sinkhorn',dict(num_iters=20),p['baseline_public_us'],p['unknown_history_mean_us']),('mhc_sinkhorn · known history',dict(num_iters=20,resident_aiv=0),p['baseline_known_us'],p['explicit_history']['us'])]:
     yield name,'mhc_sinkhorn',r['shape'],sku,attrs,k,str(fresh.relative_to(ROOT)),h,ca,old,pred,cat,evidence
  for b,path,reg,prices in [
    (6,ROOT/f'results/v0.31/recurrent/control-{tag}.json',base,bd['cases']['6']['predictions']),
    (128,ROOT/f'results/v0.29/large/cube/recurrent-{tag}.json',base,bd['cases']['128']['predictions']),
    (16,OUT/f'large/recurrent16-{tag}.json',k16,kd['prices'])]:
   if not path.is_file():continue
   receipt=read(path);v=receipt.get('verdict',{});cost=receipt['cost']
   if v.get('qualified') is not True:continue
   if b==16 or b==6:assert receipt['freeze_sha256']==digest(reg)
   else:assert receipt['freeze_sha256']==digest(ROOT/'results/v0.29/large/cube/freeze.json')
   s,=v['stages'];k,h=qualified_stage(path,s,v['ca_us']);p=prices[sku]
   assert k.startswith('RecurrentKda_') and p['shape']==[b,1,12,12,128,128] and v['numerical'] and cost['exit_code']==0 and not cost['stop_reason']
   assert v['expected']=={'attention':b*12*128,'state':b*12*128*128} and all(z[1]==0 for z in v['actual'].values())
   cat,why=relationship(read(reg)['utc'],cost.get('begin'),cost.get('end'))
   yield 'kda_recurrent · dense multi-chain','kda_recurrent',p['shape'],sku,{},k,str(path.relative_to(ROOT)),h,v['ca_us'],p['latency_us'],p['latency_us'],cat,why
def matmul_windows():
 """Yield direct eligible registered Matmul receipts even if the asynchronous score lags."""
 f=OUT/'matmul/freeze.json';j=read(f);g=OUT/'matmul/kimi-head/freeze.json';q=read(g)
 pro_reg=OUT/'matmul/pro-large/freeze.json';pro=read(pro_reg)
 dec_reg=OUT/'matmul/decode/freeze.json';decode=read(dec_reg)
 jobs=[(f,j['utc'],r,r['candidate_receipts']) for r in j['rows'] if r['chip'] in (DT,PR)]
 def register(reg,utc,sku,shape,price,path,*,before=True,acquisition=None):
  r=dict(chip=sku,shape=shape,us=price['us'],old_us=price.get('old_us',price['us']),
         role='prediction before device acquisition' if before else 'running before freeze; result blinded',
         acquisition_register=acquisition or reg)
  jobs.append((reg,utc,r,[str(path.relative_to(ROOT))]))
 for sku,tag in zip((DT,PR),('dt','pr')):
  register(g,q['utc'],sku,q['shape'],q['predictions'][sku],g.parent/(tag+'.json'))
  register(pro_reg,pro['utc'],sku,pro['shape'],pro['predictions'][sku],pro_reg.parent/(tag+'.json'))
  full=pro['reuse_full'];acquisition=ROOT/full['source']
  register(pro_reg,pro['utc'],sku,full['shape'],full['predictions'][sku],acquisition.parent/('DeepSeek-V4-Pro-'+tag+'.json'),before=sku!=DT,acquisition=acquisition)
  for job in decode['jobs']:
   stem='-'.join(map(str,job['shape']))+'-'+tag
   register(dec_reg,decode['utc'],sku,job['shape'],job['predictions'][sku],dec_reg.parent/(stem+'.json'))
 for reg,utc,r,files in jobs:
  for name in files:
   path=ROOT/name
   if not path.is_file():continue
   d=read(path)
   if not d.get('qualified') or not d.get('ca_us') or len(d.get('stages',[]))!=1:continue
   s,=d['stages'];kernel,h=qualified_stage(path,s,d['ca_us']);meta=d
   if 'acquisition_register' in r:
    assert d.get('freeze_sha256')==digest(r['acquisition_register'])
   if d.get('acquisition_receipt'):
    parent=path.parent/(d.get('initial') or d['acquisition_receipt'])
    assert digest(parent)==d['acquisition_sha256'];meta=read(parent)
   sku=d.get('chip') or meta.get('chip');shape=d.get('shape') or meta.get('shape');cost=d.get('cost') or meta.get('cost',{})
   assert sku==r['chip'] and shape==r['shape']
   reason=r['role'];cat,timing=relationship(utc,cost.get('started_utc') or cost.get('begin'),cost.get('finished_utc') or cost.get('end'))
   if reason!='prediction before device acquisition':cat,timing='replay','in_flight_or_recovered_after_start'
   yield 'matmul','matmul',shape,sku,{},kernel,name,h,d['ca_us'],r['old_us'],r['us'],cat,timing
def post_windows(api):
 f=OUT/'post/freeze.json';j=read(f)
 for sku in (DT,PR):
  for group,path in [('isolated_large',ROOT/f'results/v0.31/post/builtin_mhc_post-{sku}.json'),('changed_split',ROOT/f'results/v0.29/post/split-{sku}.json')]:
   frozen=sorted((x for x in j['predictions'] if x['chip']==sku and x['group']==group),key=lambda x:x['ordinal'])
   if group=='isolated_large':
    p=next(x for x in read(ROOT/'results/v0.31/post/freeze.json')['jobs'] if x['op']=='mhc_post')
    path=path.with_name(p['id']+'-'+sku+'.json')
   if not path.is_file():continue
   rec=read(path);stream=api.OperatorStream(sku);future=api.OperatorStream('Ascend970DT')
   for p in frozen:
    sh=p['shape'];a=stream.advance('mhc_post',sh).latency_us;fu=future.advance('mhc_post',sh).latency_us
    if p['data_role']!='target':continue
    if group=='isolated_large':
     if not rec.get('qualified'):continue
     stages=rec.get('stages') or []
     if len(stages)!=1:raise ValueError(f'Expected one fully qualified Post stage in {path}')
     st,=stages
    else:
     row=next((z for z in rec.get('rows',[]) if z.get('i')==p['ordinal'] and z.get('shape')==sh and z.get('timing_qualified')),None)
     if not row:continue
     st=row['stage']
    ca=st['ca_us'];kernel,h=qualified_stage(path,st,ca)
    cat,why=relationship(j['utc'],rec.get('cost',{}).get('started_utc'),rec.get('cost',{}).get('finished_utc'))
    if not close(a,p['ordered_us']):raise ValueError('Installed stream differs from preregistered value')
    yield p,ca,kernel,h,str(path.relative_to(ROOT)),fu,cat,why

def other_full_windows():
 """Late qualified full instruction windows for previously registered large shapes.

 Wrapper-only timing is intentionally not combined with a device instruction interval.
 Each candidate is priced by the installed wheel, then classified according to whether
 the identical numerical forecast had actually been frozen for this operator.
 """
 base=ROOT/'results/v0.29/large'
 for family in ('fa','vector','cube','fusion'):
  reg=base/family/'freeze.json';f=read(reg)
  for j in f['jobs']:
   vector=family=='vector'
   if vector and j['protocol']!='instruction':continue
   if family=='cube' and j['id']=='recurrent':continue
   for c in ([j['chip']] if 'chip' in j else [DT,PR]):
    tag='dt' if c==DT else 'pr'
    slug=(j['case'] if vector else j['id'] if family=='fa' else j['id']+'-'+(tag if family=='cube' else c))
    p=reg.parent/(slug+'.json')
    if not p.is_file():continue
    rec=read(p);v=rec.get('verdict',rec)
    if v.get('qualified') is not True:continue
    assert (rec.get('freeze_sha256') or rec.get('registration_sha256'))==digest(reg)
    cost=rec['cost'];cat,why=relationship(f['utc'],cost.get('started_utc') or cost.get('begin'),cost.get('finished_utc') or cost.get('end'))
    if family=='fa':
     st,=v['stages'];k,h=qualified_stage(p,st,v['ca_us'])
     yield 'fa','fa',j['shape'],c,{},k,str(p.relative_to(ROOT)),h,v['ca_us'],j['predictions'][c]['us'],cat,why
    elif vector:
     ca=v['instruction_us'];k=v['kernel'];h=v['trace_sha256'];assert len(h)==64 and ca>0
     op=j['op'];label='rmsnorm · msopprof' if op=='rmsnorm' else op
     yield label,op,j['shape'],c,j['attrs'],k,str(p.relative_to(ROOT)),h,ca,j['prices'][c]['public_us'],cat,why
    elif family=='cube':
     for i,x in enumerate(rec['operators']):
      op=x['op'];label='fla_kda_v2' if op=='kda_chunk' else op
      stages=[v['stages'][i]] if j['id']=='moe' and i<2 else v['stages']
      keys=[qualified_stage(p,s,s['ca_us']) for s in stages];ca=x['ca_us'];assert close(sum(s['ca_us'] for s in stages),ca)
      yield label,op,j['shape'],c,j['attrs'],[e[0] for e in keys],str(p.relative_to(ROOT)),[e[1] for e in keys],ca,x['prediction_us'],cat,why
    else:
     for x in j['outputs']:
      raw=x['op'];op='mhc_post' if raw=='vllm_hc_post' else raw;label='mhc_post · shape' if raw=='mhc_post' else raw
      stages=[s for s in v['stages'] if s['op']==raw] if raw!='mhc' else v['stages']
      keys=[qualified_stage(p,s,s['ca_us']) for s in stages];ca=v['ca_us'][raw];assert close(sum(s['ca_us'] for s in stages),ca)
      yield label,op,x['shape'],c,x['attrs'],[e[0] for e in keys],str(p.relative_to(ROOT)),[e[1] for e in keys],ca,x['model'][c]['us'],cat,why

def standalone_hcpost():
 reg=ROOT/'results/v0.31/post/freeze.json';f=read(reg);j=next(v for v in f['jobs'] if v['op']=='vllm_hc_post')
 for c in (DT,PR):
  p=reg.parent/(j['id']+'-'+c+'.json')
  if not p.is_file():continue
  r=read(p)
  if r.get('qualified') is not True:continue
  assert r['chip']==c and r['shape']==j['shape'] and r['freeze_sha256']==digest(reg)
  st,=r['stages'];ca=r['ca_us']['vllm_hc_post'];k,h=qualified_stage(p,st,ca)
  cost=r['cost'];cat,why=relationship(f['utc'],cost['started_utc'],cost['finished_utc'])
  yield 'vllm_hc_post','mhc_post',j['shape'],c,j['attrs'],k,str(p.relative_to(ROOT)),h,ca,j['predictions'][c]['us'],cat,why
