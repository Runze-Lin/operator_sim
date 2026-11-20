"""Read-only receipt watcher: re-evaluate only after a qualified identity changes."""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,sys,time
from hashlib import sha256
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=ROOT/'results/v0.32'
def read(p):return json.loads(p.read_bytes())
def signatures(prefix):
 if str(prefix) not in sys.path:sys.path.insert(0,str(prefix))
 if str(HERE) not in sys.path:sys.path.append(str(HERE))
 import operator_model as api
 import importlib,receipts
 importlib.reload(receipts)
 from receipts import early_registered, matmul_windows, post_windows, other_full_windows, standalone_hcpost
 from receipts import DT,PR
 sig=set()
 for x in (*early_registered(),*matmul_windows(),*other_full_windows(),*standalone_hcpost()):
  ident=x[7];hashes=tuple(ident) if isinstance(ident,(list,tuple)) else (ident,)
  sig.add((x[0],x[3],hashes))
  sig.add((x[0]+'#qualification:'+x[-2]+':'+x[-1],x[3],hashes))
 for r,_,_,h,_,__,___,____ in post_windows(api):sig.add(('post:'+r['group'],r['chip'],(h,)))
 for r in read(OUT/'matmul/score.json')['rows']:sig.add(('matmul',r['chip'],tuple(r['hash'])))
 for c in (DT,PR):
  p=OUT/'hcpost'/(c+'.json')
  if p.is_file():
   d=read(p)
   if d.get('physical_complete') is True:sig.add(('hcpost-physical',c,(d['trace_sha256'],)))
 digest=lambda x:sha256(json.dumps(sorted(x)).encode()).hexdigest()
 return digest(sig),digest([v for v in sig if v[0]=='matmul'])
def run(prefix):
 state=ROOT/'runs/release32';state.mkdir(parents=True,exist_ok=True);key=state/'last-qualified.sha256';prev=tuple(json.loads(key.read_bytes())) if key.exists() else None
 with (state/'evaluation.lock').open('a') as own:
  fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
  end=time.monotonic()+72*3600
  while time.monotonic()<end:
   try:
    sig=signatures(prefix)
    if sig!=prev:
     subprocess.run([sys.executable,'-I',str(HERE/'evaluate.py'),str(prefix)],cwd=ROOT,check=True)
     subprocess.run([sys.executable,'-I',str(HERE/'summary_table.py')],cwd=ROOT,check=True)
     if prev is None or sig[1]!=prev[1]:
      subprocess.run([sys.executable,'-I',str(ROOT/'development/validation/matmul32/large_scaling.py'),str(prefix)],cwd=ROOT,check=True)
     prev=sig;(state/'last-qualified.sha256').write_text(json.dumps(sig)+'\n')
   except (OSError,ValueError,AssertionError,KeyError,subprocess.SubprocessError) as e:print('refresh pending:',repr(e),flush=True)
   time.sleep(45)
if __name__=='__main__':run(Path(sys.argv[1]).resolve())
