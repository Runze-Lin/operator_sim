"""Capture exact SKU, all numeric checks and all launched kernel windows, bounded per attempt."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import defaultdict,Counter
import os,re,signal,subprocess,time,threading
from campaign import *
def nbytes(p):
 s=0
 for root,dirs,fs in os.walk(p):
  for f in fs:
   try:s+=os.stat(os.path.join(root,f)).st_size
   except FileNotFoundError:pass
 return s
def capture(j,chip,d,file):
 d.mkdir(parents=True,exist_ok=False);runner=ROOT/j['runner'];assert sha(runner)==j['runner_sha256']
 args=j['args']
 if j['batch']:
  (d/'batch.txt').write_text(' '.join(map(str,j['batch']))+'\n');args=[str(d/'batch.txt')]
 cmd=['/usr/bin/time','-f','%U %S','-o',str(d/'cpu.txt'),'bash',str(ROOT/'scripts/cann-run.sh'),'msopprof','simulator','--soc-version='+chip,'--dump=true','--launch-count=20','--timeout='+str(max(2,math.ceil(j['wall_s']/60))),'--output='+str(d/'profile'),str(runner),*args]
 t=time.monotonic();stamp=utc();reason=None; assert stamp>file['utc']
 with (d/'run.log').open('w')as log:
  p=subprocess.Popen(cmd,cwd=d,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  while p.poll() is None:
   if time.monotonic()-t>j['wall_s']:reason='wall cap'
   if nbytes(d)>j['raw_mib']*2**20:reason='raw cap'
   if j['op']=='kda_chunk' and re.search(r'\[ERROR\].*(?:execute_set_flag|execute_wait_flag|deadlock|invalid instruction)',(d/'run.log').read_text(errors='replace')):reason='CA synchronization/instruction failure'
   if reason:
    os.killpg(p.pid,signal.SIGKILL);p.wait();break
   time.sleep(2)
 cpu=(d/'cpu.txt').read_text().splitlines()
 cost=dict(seconds=time.monotonic()-t,raw_bytes=nbytes(d),cpu_seconds=sum(map(float,cpu[-1].split()))if cpu and re.fullmatch(r'[\d.]+ [\d.]+',cpu[-1]) else None,exit_code=p.returncode,stop_reason=reason,started_utc=stamp,finished_utc=utc(),command=cmd[5:])
 write(d/'cost.json',cost)
 return cost
def events(trace):
 dec=json.JSONDecoder().raw_decode
 with trace.open()as f:
  buf=f.read(2**20).split('"traceEvents":[',1)[1];i=0
  while True:
   if len(buf)-i<100000:buf=buf[i:]+f.read(2**20);i=0
   if buf[i]==']':return
   e,i=dec(buf,i);yield e
   if buf[i]==',':i+=1
def stages(j,chip,d):
 cost=read(d/'cost.json');log=(d/'run.log').read_text(errors='replace')
 if cost['exit_code']!=0 or cost['stop_reason']:raise ValueError('process/budget: '+str(cost['stop_reason']))
 if not all(s in log for s in ['SMOKE_PASS','All task success.','Profiling data parse finished.',('lpddr5_6400M_bailu.ini' if chip==PR else 'hbm3_8000M.ini')]) or 'SMOKE_FAIL' in log or re.findall(r'^runtime_soc=(\S+)',log,re.M)!=[chip]:raise ValueError('exact SKU/physical memory/parse/golden gate')
 if re.search(r'\[ERROR\].*(?:execute_set_flag|execute_wait_flag|deadlock|invalid instruction)',log):raise ValueError('CA sync failure')
 correct=re.findall(r'^correctness [^\r\n]+',log,re.M)
 if len(correct)!=j['nchecks'] or any(not re.search(r'\bbad=0\b',v)for v in correct):raise ValueError('numeric groups: '+str(len(correct)))
 names=re.findall(r'Start profiling on kernel: (\S+)',log)
 if len(names)!=j['kernels']:raise ValueError('launched kernel count: '+str(names))
 traces=sorted(d.glob('profile/OPPROF_*/simulator/trace.json'))
 if len(names)>1:
  ordinals=Counter();traces=[]
  for name in names:
   idx=ordinals[name];ordinals[name]+=1
   stage,=d.glob('profile/OPPROF_*/'+name+'/'+str(idx)+'/simulator/trace.json')
   traces.append(stage)
 if len(traces)!=len(names):raise ValueError('trace-count '+str(len(traces))+' kernels '+str(names))
 result=[]
 launch_logs=re.split(r'^core_name\s+duration_time\(us\)\s+running_time\(us\)\s*$',log,flags=re.M)[1:]
 if len(launch_logs)!=len(names):raise ValueError('expected one ordered vendor duration table per launched kernel')
 for name,tr,launch_log in zip(names,traces,launch_logs):
  wins={}; counts=Counter();qs=Counter()
  for e in events(tr):
   if e['ph']!='X':continue
   t=e['ts'];z=t+e['dur'];c=e['pid']
   if not(math.isfinite(t)and math.isfinite(z)and z>=t and re.fullmatch(r'core\d+\.(?:cube|vec)core\d+',c)):raise ValueError('malformed core event')
   l,h=wins.get(c,(t,z));wins[c]=(min(l,t),max(h,z));counts[e['name']]+=1;qs[e['tid']]+=1
  listed=re.findall(r'^(core\d+\.(?:cube|vec)core\d+)\s+[.\d]+\s+[.\d]+\s*$',launch_log,re.M)
  if set(listed)!=set(wins):raise ValueError('vendor duration table has wrong core set')
  for core,(l,h) in wins.items():
   vs=re.findall(r'^'+re.escape(core)+r'\s+([.\d]+)\s+[.\d]+\s*$',launch_log,re.M)
   if len(vs)!=1 or abs(float(vs[0])-(h-l))>max(.03,(h-l)*.02):raise ValueError('core window mismatches profiler')
  a=min(v[0]for v in wins.values());b=max(v[1]for v in wins.values())
  result.append(dict(kernel=name,ca_us=b-a,active_aic=sum('cubecore' in c for c in wins),active_aiv=sum('veccore'in c for c in wins),windows={c:[x-a,y-a]for c,(x,y)in wins.items()},trace=str(tr.relative_to(ROOT)),trace_sha256=sha(tr),binary_sha256=sha(tr.parents[1]/'dump/aicore_binary.o'),queue_events=dict(qs),mmad=counts['MMAD'],total_events=sum(qs.values())))
 return result,correct
def qualify(j,chip,d):
 ks,correct=stages(j,chip,d);results=[]
 for o in j['outputs']:
  sel=[ks[i]for i in o['indices']];rec=dict(op=o['op'],shape=o['shape'],attrs=o['attrs'],chip=chip,ca_us=sum(k['ca_us']for k in sel),active_aic=max(k['active_aic']for k in sel),active_aiv=max(k['active_aiv']for k in sel),kernel='+'.join(k['kernel']for k in sel),api=OPS[o['op']],correctness=correct,stages=sel,trace=[k['trace']for k in sel],trace_sha256=[k['trace_sha256']for k in sel],cost=read(d/'cost.json'),case_id=j['id'],role='frozen PR holdout' if chip==PR else 'new paired DT CA',scope='sum of serialized launches; excluding host gaps' if len(sel)>1 else 'one all-core window',frozen={k:v for k,v in o.items()if k.endswith(('_us','_status','_aic','_aiv'))})
  results.append(rec)
 return results
def main(group):
 b=read(OUT/(group+'-freeze.json'));folder=RAW/group;folder.mkdir(parents=True,exist_ok=True);start=time.monotonic()
 jobs=[]
 for j in b['jobs']:
  jobs.append((j,PR))
  if group=='core' and not j['outputs'][0]['dt']:jobs.append((j,DT))
 sizes={};outcomes=[];lock=threading.Lock()
 def one(pair):
  j,chip=pair;d=folder/(j['id']+'_'+chip)
  if time.monotonic()-start>=b['total_wall_s'] or sum(sizes.values())>=b['total_raw_gib']*2**30: return
  try:
   if not(d/'cost.json').exists():capture(j,chip,d,b)
   result=qualify(j,chip,d);write(d/'qualified.json',result); reason=None
  except Exception as exc:reason=repr(exc)
  cost=read(d/'cost.json')if(d/'cost.json').exists() else {}; sizes[d.name]=cost.get('raw_bytes',0)
  record=dict(case=j['id'],op=j['op'],chip=chip,eligible=reason is None,reason=reason,cost=cost)
  with lock:
   outcomes.append(record);write(OUT/(group+'-attempts.json'),outcomes);print(d.name,reason or 'QUALIFIED',len(outcomes),'/',len(jobs),flush=True)
 with ThreadPoolExecutor(b['workers'])as pool:list(pool.map(one,jobs))
 write(OUT/(group+'-cost.json'),dict(wall_s=time.monotonic()-start,attempts=len(outcomes),raw_bytes=sum(sizes.values()),simulator_cpu_s=sum((r['cost'].get('cpu_seconds')or 0) for r in outcomes)))
if __name__=='__main__': main(sys.argv[1])
