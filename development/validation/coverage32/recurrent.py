"""Post-Sinkhorn continuation, one fresh Kimi decode B16 S1 pair (192 states)."""
from pathlib import Path
from datetime import datetime,timezone
import fcntl,hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'development/validation/recurrent31'))
import check as old
RES=ROOT/'results/v0.32/large';RAW=ROOT/'runs/coverage32';REG=RES/'recurrent-freeze.json';SHAPE=[16,1,12,12,128,128]
def now():return datetime.now(timezone.utc).isoformat()
def write(p,d,new=False):
 if new:
  with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
 else:old.save(p,d)
def freeze():
 b,s,h,h2,k,v=SHAPE
 j=dict(id='recurrent',shape=SHAPE,host='recurrent',ops=['kda_recurrent'],prefix=['RecurrentKda'],args=[b,s,h],gib=8,hours=1,contract=f'case=kda_recurrent api=aclnnRecurrentKda B={b} S={s} Hq={h} Hv={h}',golden={'attention':b*s*h*v,'state':b*h*k*v})
 prices={}
 for sku in old.SKUS:
  t=time.perf_counter();p=old.estimate_recurrent_chains(SHAPE,chip=sku);prices[sku]=dict(**p,query_ms=(time.perf_counter()-t)*1000)
 assert all(not (RAW/f'recurrent16-{tag}').exists() for tag in ['dt','pr'])
 files=[old.MODEL,old.BIN/'recurrent',old.HDR,ROOT/'development/validation/large29/cube/recurrent.cpp',ROOT/'development/validation/large29/cube/verify.py',ROOT/'results/v0.31/recurrent/freeze.json',Path(__file__).resolve(),*[ROOT/'configs/chips'/(s+'.json') for s in old.SKUS]]
 write(REG,dict(utc=now(),job=j,prices=prices,budget='After Sinkhorn both attempts, reuse the same exclusive CA slot; at most 3600 s and 8 GiB/end, no retry. v0.31 B6 and large29 B128 are different shapes; their owners unchanged.',role='B16 dense decode H=96/TP8, not fused gate nor paged real serving chain',source_sha256={str(p.relative_to(ROOT)):old.digest(p) for p in files}),True)
def run():
 reg=old.read(REG);j=reg['job'];slotpath=ROOT/'results/v0.26/sink/raw/lock';os.nice(12);os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
 def phase(s,chip=None):write(RES/'recurrent-state.json',dict(utc=now(),phase=s,chip=chip,pid=os.getpid(),freeze_sha256=old.digest(REG)))
 phase('waiting on previously frozen Sinkhorn 1024 dual endpoints')
 while True:
  state=old.read(RES/'state.json');all_done=all((RES/('sink1024-'+sku+'.json')).exists() for sku in old.SKUS[:2])
  if state.get('phase')=='terminal' and all_done:break
  if not Path('/proc/'+str(state['pid'])).exists():phase('prior lane worker absent; inspect partial before admission');return
  time.sleep(20)
 with slotpath.open('r+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for sku,tag in zip(old.SKUS,('dt','pr')):
   folder=RAW/('recurrent16-'+tag);dest=RES/('recurrent16-'+tag+'.json')
   if dest.exists():continue
   assert old.digest(old.BIN/'recurrent')==reg['source_sha256'][str((old.BIN/'recurrent').relative_to(ROOT))]
   if not folder.exists():
    folder.mkdir();phase('acquiring Kimi B16 full instruction window',sku);cost=old.capture(j,sku,folder)
   else:
    if not (folder/'cost.json').exists():phase('unknown partial retained');return
    cost=old.read(folder/'cost.json')
   phase('qualifying numerical, all-core trace, ELF and state chain counts',sku)
   try:
    assert cost['begin']>reg['utc']
    result=old.qualify(j,sku,folder,cost,{},ROOT)
    if result['qualified']:
     st,=result['stages'];sim=Path(st['trace']).parent;obj=old.PACK/(st['kernel'].rsplit('_',1)[0]+'.o')
     elf=old.elf_text(obj)
     if old.elf_text(sim.parent/'dump/aicore_binary.o')!=elf:raise ValueError('installed .text mismatch')
     describe={};deltas={};writes=0
     for p in sim.glob('core*/trace.json'):
      starts=[];wb=0
      for e in old.events(p):
       if e['name']!='MOV_SRC_TO_DST_ALIGNv2' or e['ph']!='X':continue
       a=json.loads(e['args']['detail'])
       if a['instr_type']=='MTE2' and a['processed_bytes']==4:starts.append(round(e['ts']*1650))
       if a['instr_type']=='MTE3' and a['processed_bytes']==32768:wb+=a['processed_bytes']
      if wb!=len(starts)*65536:raise ValueError('state byte/chain count mismatch '+p.parent.name)
      writes+=wb;describe[len(starts)]=describe.get(len(starts),0)+1;xs=sorted(starts)
      deltas[p.parent.name]=[b-a for a,b in zip(xs,xs[1:])]
     if sum(k*v for k,v in describe.items())!=192:raise ValueError('incomplete chain coverage')
     result.update(state_writes= writes,chain_counts_to_cores=describe,beta_deltas_ticks=deltas,installed_text_sha256=hashlib.sha256(elf).hexdigest())
   except (AssertionError,TypeError,KeyError,IndexError,OSError,StopIteration,ValueError) as e:result=dict(qualified=False,reason=repr(e),ca_us=None)
   write(dest,dict(chip=sku,shape=SHAPE,cost=cost,verdict=result,freeze_sha256=old.digest(REG)),True)
   from subprocess import run as sub
   sub([sys.executable,str(Path(__file__).with_name('inventory.py'))],cwd=ROOT,check=False)
  phase('terminal')
if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
