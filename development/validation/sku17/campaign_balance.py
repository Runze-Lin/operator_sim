"""Finish pending core jobs while core dispatcher is paused; cap global simulator concurrency."""
from campaign_collect import *
def main():
 b=read(OUT/'core-freeze.json');pending=[]
 # Pending controls are paused, but already-running children finish on their own.
 while any((d/'cpu.txt').exists() and not (d/'cost.json').exists() and (d/'cpu.txt').stat().st_size==0 for d in (RAW/'extension').glob('*_'+DT)):
  time.sleep(3)
 for j in b['jobs']:
  for s in ([PR,DT] if not j['outputs'][0]['dt']else[PR]):
   d=RAW/'core'/(j['id']+'_'+s)
   if not d.exists():pending.append((j,s,d))
 start=time.monotonic();x=0;mutex=threading.Lock()
 def task(t):
  nonlocal x
  j,s,d=t
  while True:
   with mutex:
    slots=5 if (OUT/'extension-cost.json').exists()else 2
    if x<slots:x+=1;break
   time.sleep(2)
  try:
   if d.exists():return
   capture(j,s,d,b)
   rows=qualify(j,s,d);write(d/'strict.json',rows)
   print('Balanced',d.name,'OK',flush=True)
  except Exception as ex: print('Balanced',d.name,repr(ex),flush=True)
  finally:
   with mutex:x-=1
 with ThreadPoolExecutor(5)as pool:list(pool.map(task,pending))
 write(OUT/'core-balanced-cost.json',dict(started_pending=len(pending),elapsed_s=time.monotonic()-start))
if __name__=='__main__':main()
