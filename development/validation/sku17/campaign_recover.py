"""Retry only incomplete, numerically unjudged infrastructure failures, once under reduced concurrency."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
from campaign_collect import *
def main(group):
 b=read(OUT/(group+'-freeze.json'));cands=[]
 for j in b['jobs']:
  if j['op']=='kda_chunk':continue
  for chip in (DT,PR):
   d=RAW/group/(j['id']+'_'+chip)
   if not (d/'cost.json').exists():continue
   cost=read(d/'cost.json');log=(d/'run.log').read_text(errors='replace')
   if cost['stop_reason']!='wall cap' or 'SMOKE_FAIL' in log:continue
   cands.append((j,chip,d))
 policy=OUT/(group+'-recovery-list.json')
 if not policy.exists():write(policy,dict(utc=utc(),rule='one identical shape/argv per original wall-cap without numeric failure; fixed 240s/440s retained. Original attempts retained separately.',ids=[d.name for _,_,d in cands]))
 def one(x):
  j,chip,d=x;dest=d.parent/(d.name+'-recovery')
  if not(dest/'cost.json').exists():capture(j,chip,dest,b)
  try:res=qualify(j,chip,dest);write(dest/'strict.json',res);print('RECOVER',dest.name,'ok',flush=True)
  except Exception as e:print('RECOVER',dest.name,repr(e),flush=True)
 with ThreadPoolExecutor(2)as pool:list(pool.map(one,cands))
if __name__=='__main__':
 group=sys.argv[1]
 with (RAW/('.'+group+'-recovery.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX); main(group)
