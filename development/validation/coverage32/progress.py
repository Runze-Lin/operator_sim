"""Read-only sampling: CPU appetite and progress of ONE actual CA, never changes affinity."""
from pathlib import Path
import json,os,time,datetime
R=Path(__file__).resolve().parents[3];O=R/'results/v0.32/large'
def childtree(pid):
 for v in (Path('/proc')/str(pid)/'task'/str(pid)/'children').read_text().split():
  yield int(v);yield from childtree(int(v))
def sample():
 pkg={}
 for i in os.sched_getaffinity(0):
  pkg[i]=(Path(f'/sys/devices/system/cpu/cpu{i}/topology/thread_siblings_list').read_text().strip())
 cpu={int(x[0][3:]):[sum(map(int,x[1:9])),int(x[4])+int(x[5])] for s in Path('/proc/stat').read_text().splitlines() for x in [s.split()] if x[0].startswith('cpu') and x[0][3:].isdigit()}
 pid=json.loads((O/'state.json').read_bytes())['pid'];data=[]
 try:
  for p in childtree(pid):
   d=Path('/proc')/str(p); cmd=(d/'comm').read_text().strip(); bits=(d/'stat').read_text().split(') ',1)[1].split();msk=os.sched_getaffinity(p)
   data.append((p,cmd,len(list((d/'task').iterdir())),int(bits[11])+int(bits[12]),len(msk)))
 except FileNotFoundError:pass
 return pkg,cpu,data
a,s,one=sample();start=time.monotonic();time.sleep(3);a,e,two=sample();dt=time.monotonic()-start
idle={}
for c,pk in a.items():
 t=e[c][0]-s[c][0];i=e[c][1]-s[c][1]
 if t:idle[pk]=max(idle.get(pk,0),i/t)
r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=dt,total_physical_groups=len(idle),physical_with_one_sibling_gt70pct_idle=sum(v>.7 for v in idle.values()),sampled_host_idle_equivalent_cpus=sum(e[c][1]-s[c][1] for c in e)/os.sysconf('SC_CLK_TCK')/dt,load=Path('/proc/loadavg').read_text().strip().split()[:3],children=[dict(pid=p,comm=n,threads=th,allowed_cpus=k,mean_cores=(ticks-i[3])/os.sysconf('SC_CLK_TCK')/dt) for p,n,th,ticks,k in two for i in one if i[0]==p])
(O/'host-sample.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
