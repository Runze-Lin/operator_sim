"""Read-only host process evidence. Don't report an unverified PID as active."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import json,os,subprocess,re
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/v0.32/large'
def rows():
 txt=subprocess.check_output(['ps','-u',str(os.getuid()),'-ww','-o','pid=,ppid=,stat=,etime=,comm=,args='],text=True)
 pids={};children=defaultdict(list)
 for line in txt.splitlines():
  a=line.split(None,5)
  if len(a)<6:continue
  p,par=int(a[0]),int(a[1]);pids[p]=a[2:];children[par].append(p)
 def desc(p):
  out=[]
  for c in children[p]:out.extend([c,*desc(c)])
  return out
 sources={'large MoE / addmm / KDA B128':('results/v0.29/large/cube/state.json','large29/cube/campaign.py'),'FA long':('results/v0.29/large/fa/state.json','large29/fa.py'),'RMS and TopK':('results/v0.29/large/vector/state.json','large29/vector/acquire.py'),'fusion':('results/v0.29/large/fusion/state.json','fusion_campaign.py'),'KDA B6':('results/v0.31/recurrent/state.json','recurrent31/check.py'),'new Sinkhorn B8 x 128':('results/v0.32/large/state.json','coverage32/sink.py'),'new KDA B16':('results/v0.32/large/recurrent-state.json','coverage32/recurrent.py'),'old B11 matmul lane':('results/v0.31/matmul/device/state.json','matmul31/device/campaign.py'),'old quant B4':('results/v0.27/quant-transfer/state.json','quant')};registered=[]
 for label,(s,marker) in sources.items():
  j=json.loads((ROOT/s).read_bytes());p=j.get('pid');ident=p in pids and marker in pids[p][-1];members=desc(p) if ident else []
  jobs=[]
  for n in members:
   stat,elapsed,comm,args=pids[n]
   if comm=='msopprof' and 'simulator' in args:jobs.append(dict(pid=n,elapsed=elapsed,soc=re.search(r'--soc-version=(\S+)',args)[1],profile=re.search(r'--output=(\S+)',args)[1]))
  tree=subprocess.check_output(['pstree','-p','-A',str(p)],text=True).splitlines()[:4] if ident else []
  registered.append(dict(label=label,state_file=s,state=j.get('phase') or j.get('state'),pid=p,pid_matches_worker=ident,actual_ca_children=jobs,process_tree=tree))
 return registered
d=dict(utc=datetime.now(timezone.utc).isoformat(),scan='host ps -u own_uid -ww and pstree -p <owned PID>, descendant real msopprof children; state and live observations separate.',state_sources=rows(),large29_log_last=(ROOT/'runs/large29/cube/moe-dt/run.log').read_bytes()[-1100:].decode(errors='replace').splitlines()[-3:],old_sink_lane=json.loads((ROOT/'results/v0.26/sink/state.json').read_bytes()),new_lane_lock='results/v0.26/sink/raw/lock')
(OUT/'host-audit.json').write_text(json.dumps(d,indent=2)+'\n')
