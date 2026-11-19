"""Idempotent requalification of complete existing traces; join identical input contracts."""
from campaign_collect import *
from campaign_archive import dt_provenance
def key(op,shape,attrs):return (op,tuple(shape),json.dumps(attrs,sort_keys=True))
def accounted_cost(d):
 c=read(d/'cost.json');t=d/'cpu.txt'
 # GNU time closes its CPU accounting file when the complete host/profiler exits.
 # Use that timestamp so a paused scheduler is not charged as device collection.
 if t.is_file() and t.stat().st_size and c['exit_code']==0:
  start=datetime.fromisoformat(c['started_utc']).timestamp(); measured=t.stat().st_mtime-start
  if 0<measured<c['seconds']:
   c['scheduler_observed_seconds']=c['seconds'];c['seconds']=measured;c['elapsed_scope']='launch to profiler process exit (GNU time end file)'
 return c
def atomic_record(path,obj):
 temp=path.with_suffix('.pending');write(temp,obj);temp.replace(path)
def publish(final=False):
 allrecords=[];pairs=[];rejections=[];costs=[];dedup=set()
 for group in ('extension','core'):
  freeze=OUT/(group+'-freeze.json')
  if not freeze.exists():continue
  b=read(freeze)
  for j in b['jobs']:
   matched={}
   for chip in (DT,PR):
    orig=RAW/group/(j['id']+'_'+chip)
    for d in [orig,orig.parent/(orig.name+'-recovery')]:
     if not(d/'cost.json').exists():continue
     try:
      q=d/'strict.json'
      if not q.exists():write(q,qualify(j,chip,d))
      measured=read(q);cost=accounted_cost(d)
      log=(d/'run.log').read_text(errors='replace')
      actual_contract=re.findall(r'^case=[^\r\n]+',log,re.M)
      for r in measured:
       r['backend']='msopprof official full trace';r['cost']=cost
       r['input_contract_log']=actual_contract;r['runner_sha256']=j['runner_sha256']
       r['host_args']=j['args'];r['host_batch_row']=j['batch']
       r['run_log']=str((d/'run.log').relative_to(ROOT))
       matched[(r['op'],chip)]=r
       allrecords.append(r)
      costs.append(dict(group=group,case=j['id'],chip=chip,eligible=True,attempt=d.name,**cost))
     except (ValueError,OSError,KeyError,TypeError,IndexError) as e:
      cost=accounted_cost(d);bad=dict(group=group,case=j['id'],op=j['op'],chip=chip,eligible=False,reason=repr(e),attempt=d.name,**cost)
      if j['op']=='kda_chunk':
       bad['frozen_attrs']=j['outputs'][0]['attrs']
       bad['actual_host_contract']=dict(layout='BSND',dtype='bf16',beta_dtype='fp32',initial_state_dtype='fp32',use_gate_in_kernel=True,safe_gate=True,lower_bound=-0.03125,export_gk=True,output_final_state=True,chunk_size=64,outputs=['O','final_state','GK','QK','KK'],golden_checks=['output','aqk','akk','gk','state','input_state'])
       bad['contract_difference']='Frozen structural attrs omit safe_gate, lower_bound and export_gk. The host uses safe gated mode and requests GK/QK/KK. All timing fields are unavailable and no pair is produced.'
      rejections.append(bad);costs.append(bad)
   for o in j['outputs']:
    pr=matched.get((o['op'],PR));dt=matched.get((o['op'],DT))
    if not dt and o['dt']:
     dt=dt_provenance(o['dt'])
     if dt.get('active_aic')=={} or dt.get('active_aiv')=={}:
      dt['active_aic']=None;dt['active_aiv']=None
     allrecords.append(dt)
    if not pr or not dt or not (dt['ca_us'] and pr['ca_us']):continue
    ident=key(o['op'],o['shape'],o['attrs'])
    if ident in dedup:raise ValueError('duplicate contract '+str(ident))
    dedup.add(ident)
    def pred_speed(old=False):
     a=o.get('old_dt_us' if old else 'dt_us');c=o.get('old_pr_us'if old else 'pr_us')
     return a/c if a and c else None
    ratio=dt['ca_us']/pr['ca_us'];est=pred_speed()
    pairs.append(dict(op=o['op'],shape=o['shape'],attrs=o['attrs'],id=j['id'],dt=dt,pr=pr,
      dt_ca_us=dt['ca_us'],pr_ca_us=pr['ca_us'],**{k:v for k,v in o.items() if k.endswith(('_us','_aic','_aiv','_status'))},
      speedup_ca=ratio,speedup_pred=est,speedup_old=pred_speed(True),speedup_reuse_dt=1.,speedup_relative_error=est/ratio-1 if est else None,role='independent PR; archived DT or newly paired DT',backend_match=dt['backend']==pr['backend']))
 atomic_record(OUT/'records.json',allrecords);atomic_record(OUT/'pairs.json',pairs);atomic_record(OUT/'rejections.json',rejections)
 atomic_record(OUT/'all-costs.json',costs)
 if final:print(dict(counts=dict(Counter(p['op']for p in pairs)),attempted=len(costs),rejected=len(rejections)))
if __name__=='__main__':publish(True)
