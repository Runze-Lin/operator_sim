"""One frozen cold 1024x4 FP32, I20 pair. No reruns; reuse completed v0.26 lane."""
from pathlib import Path
import fcntl, hashlib, json, os, re, shutil, subprocess, sys, time
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'development/validation/sink26'),str(ROOT)]
from collect import capture,qualify
from registration import BINARY,VENDOR,IMPL
from development.validation.fusion28 import schedule28 as sched
from operator_model import estimate,load_chip
OUT=ROOT/'results/v0.32/large';RAW=ROOT/'runs/coverage32';OLD=ROOT/'results/v0.26/sink'
CHIPS=('Ascend950DT_9582','Ascend950PR_957b','Ascend970DT')
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_bytes())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(4<<20):h.update(b)
 return h.hexdigest()
def save(p,obj,new=False):
 if new:
  with p.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
 else:
  tmp=p.with_suffix('.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def rel(p):return str(p.relative_to(ROOT))
def pred(t,it,sku,h):
 target=load_chip(sku);t0=time.perf_counter();p=sched.phases([t,4],it,target,h);q=sched.estimate([t,4],it,target);a={'num_iters':it};base=estimate('mhc_sinkhorn',[t,4],a,chip=sku);known=estimate('mhc_sinkhorn',[t,4],{**a,'resident_aiv':h},chip=sku)
 return dict(shape=[t,4],iterations=it,resident_aiv=h,chip=sku,explicit_history=p,unknown_history_mean_us=q['mean_us'],history_sensitivity_us=q['range_us'],baseline_public_us=base.latency_us,baseline_public_status=base.status,baseline_known_us=known.latency_us,query_wall_ms=(time.perf_counter()-t0)*1000)
def freeze():
 assert not RAW.exists() and not (OUT/'freeze.json').exists()
 row={};jobs=[];old_reg=read(OLD/'register.json')
 for sku in CHIPS:
  j=dict(shape=[1024,4],iters=20,index=0,prior_h=0,chip=sku)
  jobs.append(j);seen=[pred(1024,20,sku,0)]
  for t,it in [(513,6),(4097,4)]:
   h=next(x['prior_h'] for x in old_reg['jobs'] if x['chip']==sku and x['shape']==[t,4] and x['iters']==it)
   seen.append(pred(t,it,sku,h))
  row[sku]=seen
 d26=ROOT/'development/validation/sink26';files=[BINARY,VENDOR,IMPL,d26/'collect.py',d26/'registration.py',ROOT/'smoke/extension_indexer_mhc.cpp',ROOT/'development/validation/fusion28/schedule28.py',ROOT/'results/v0.28/fusion/prices.json',ROOT/'operator_model/sink_history.py',ROOT/'operator_model/sink_unknown.py',ROOT/'operator_model/backend.py',ROOT/'results/v0.26/sink/prices.json',*[ROOT/'configs/chips'/(k+'.json') for k in CHIPS]]
 file=OUT/'freeze.json';save(file,dict(utc=now(),cases=jobs,predictions=row,role='1024,I20 paired prospective first call H0. 513,I6 and 4097,I4 are existing v0.26 records, replay only; future always extrapolation.', budget=dict(wall_s=1800,mib=4096),policy='one single-task full-device trace per 950 chip, sequential, nice 12. Reuse finished v0.26 standalone slot under its own lock. No modifications to queued workers. Entirely separate process and H=0 on each chip.',source_sha256={rel(p):sha(p) for p in files}),True)
def run():
 reg=read(OUT/'freeze.json');RAW.mkdir(exist_ok=True)
 os.nice(12);os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
 with (OLD/'raw/lock').open('r+') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert read(OLD/'state.json')['state']=='finished'
  for k in [BINARY,VENDOR,IMPL]:assert sha(k)==reg['source_sha256'][rel(k)]
  for j,sku in zip(reg['cases'],CHIPS[:2]):
   assert j['chip']==sku
   dest=OUT/f'sink1024-{sku}.json';folder=RAW/('sink1024-'+sku)
   if dest.exists():continue
   if folder.exists() and not (folder/'cost.json').exists():raise ValueError('interrupted run; no duplicate')
   if not folder.exists():
    assert shutil.disk_usage(ROOT).free>12*2**30
    save(OUT/'state.json',dict(utc=now(),pid=os.getpid(),phase='acquiring single independent cold window',chip=sku,machine_cpu_count=len(os.sched_getaffinity(0)),freeze_sha256=sha(OUT/'freeze.json')))
    cost=capture([j],folder,reg['budget'])
   else:cost=read(folder/'cost.json')
   save(OUT/'state.json',dict(utc=now(),pid=os.getpid(),phase='qualifying',chip=sku))
   try:
    if cost['started_utc']<=reg['utc']:raise ValueError('timing register after launch')
    r,=qualify([j],folder,cost)
   except (OSError,ValueError,AssertionError,KeyError,IndexError,StopIteration) as e:r=dict(qualified=False,reason=str(e))
   save(dest,dict(chip=sku,cost=cost,verdict=r,log_sha256=sha(folder/'run.log'),freeze_sha256=sha(OUT/'freeze.json'),actual_runner=rel(BINARY),installed=rel(VENDOR)),True)
   report()
  save(OUT/'state.json',dict(utc=now(),pid=os.getpid(),phase='terminal',endpoints=2))
def archives():
 """Rehash merged traces/binaries, exact matching host numerics, DRAM, terminal clocks."""
 from development.validation.rms_ordered_verify.verify import tables
 from collect import measured
 rows=[]
 for chip in CHIPS[:2]:
  path=OLD/f'receipts/wide-{chip}.json';a=read(path);folder=OLD/f'raw/wide-{chip}'
  logs=(folder/'run.log').read_text();sections=re.findall(r'^BATCH_BEGIN (\d+)\s*$\n(.*?)^BATCH_END \1\s*$',logs,re.S|re.M)
  assert len(sections)==4 and a['cost']['exit_code']==0 and a['cost']['stop_reason'] is None
  assert re.findall(r'^runtime_soc=(\S+)',logs,re.M)==[chip] and all(x in logs for x in ['SMOKE_PASS','All task success.','Profiling data parse finished.',('lpddr5_6400M_bailu.ini' if '950PR' in chip else 'hbm3_8000M.ini')])
  tabs=tables(logs)
  for old in a['rows'][-2:]:
   t=old['shape'][0];idx=old['index'];st=old['stage'];p=ROOT/st['trace'];vol=t*4*4
   assert (str(idx),) == (sections[idx][0],) and f'T={t} N=4 iters={old["iters"]} dtype=FP32' in sections[idx][1]
   assert re.findall(r'correctness name=output count=(\d+) bad=(\d+)',sections[idx][1])==[(str(vol),'0')]
   check=measured(p.parents[1],st['kernel'],tabs[idx],vol*4)
   assert old['qualified'] and all(check[k]==st[k] for k in ('ca_us','active_aiv','written_bytes','first_tick','last_tick','trace_sha256','binary_sha256'))
   rows.append(dict(chip=chip,shape=old['shape'],iterations=old['iters'],resident_aiv=old['h'],qualified=True,ca_us=st['ca_us'],aiv=st['active_aiv'],VF_calls_per_core=sorted(set(st['vf_calls_per_core'].values())),written_bytes=check['written_bytes'],all_output_elements=vol,kernel=st['kernel'],trace=st['trace'],trace_sha256=st['trace_sha256'],binary_sha256=st['binary_sha256'],old_receipt=rel(path),old_receipt_sha256=sha(path),cost_scope='Four-call whole process shared; not four independent timings',batch_ca_wall_s=a['cost']['seconds'],role='archived mechanism development replay; no new CA'))
 save(OUT/'archive.json',dict(rows=rows))
 report()
def report():
 reg=read(OUT/'freeze.json');arc=read(OUT/'archive.json')['rows'] if (OUT/'archive.json').exists() else []
 rows=[]
 for chip in CHIPS:
  for p in reg['predictions'][chip]:
   t=p['shape'][0];found=next((x for x in arc if x['chip']==chip and x['shape'][0]==t),{});cost=None
   if t==1024:
    target=OUT/f'sink1024-{chip}.json'
    if target.exists():
     q=read(target);cost=q['cost'];v=q['verdict'];found=dict(ca_us=(v.get('stage') or {}).get('ca_us'),reason=v.get('reason'),qualified=v['qualified'])
   ca=found.get('ca_us') if found.get('qualified') else None
   r=dict(chip=chip,shape=p['shape'],iterations=p['iterations'],H=p['resident_aiv'],known_us=p['explicit_history']['us'],unknown_us=p['unknown_history_mean_us'],baseline_us=p['baseline_public_us'],baseline_known_us=p['baseline_known_us'],active_aiv=p['explicit_history']['active'],rows_per_aiv=p['explicit_history']['rows'],passes=p['explicit_history']['groups'],measured_us=ca,qualified=ca is not None,reason=found.get('reason'),role='970DT extrapolation' if chip.endswith('970DT') else ('prospective' if t==1024 else 'archive replay'),wall_s=cost['seconds'] if cost else None)
   for field in ('known_us','unknown_us','baseline_known_us','baseline_us'):r[field+'_signed_pct']=100*(r[field]/ca-1) if ca is not None and r[field] is not None else None
   rows.append(r)
 save(OUT/'sink-score.json',dict(freeze_sha256=sha(OUT/'freeze.json'),rows=rows))
if __name__=='__main__':dict(freeze=freeze,run=run,archives=archives,report=report)[sys.argv[1]]()
