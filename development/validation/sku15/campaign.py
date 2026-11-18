"""v0.15 PR-only independent holdout registration (DT historical pair optional)."""
from pathlib import Path
import hashlib,json
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/v0.15/sku';RUNS=ROOT/'runs/sku15'
PR='Ascend950PR_957b'; DT='Ascend950DT_9582'; RUNNER='build/smoke/three_ops'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def load(p):return json.loads(Path(p).read_text())
def write(p,d,exclusive=False):
 with p.open('x' if exclusive else 'w') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def inputs():
 hist=[]
 for root in ['runs/v8cube','runs/v8cube-isolated','results/v0.8/core','runs/model-v5']:
  for p in (ROOT/root).glob('*.json' if 'results/' in root else '*/result.json'):
   d=load(p)
   if not isinstance(d,dict) or d.get('status')!='ok':continue
   c=d.get('case',d)
   if c.get('op') not in ('fa','matmul'):continue
   m=d.get('measurement') or d['kernels'][0];ref=m['reference_us']
   db=Path(m['db_path']) if 'db_path' in m else next(p.parent.glob('profile/*/record/*.db'))
   if sha(db)!=m['db_sha256']: raise ValueError(p)
   hist.append((c['op'],c['shape'],dict(ca_us=ref,record=str(p.relative_to(ROOT)),db_sha256=sha(db),active_aic=m['active_aic'],active_aiv=m['active_aiv'],kernel=m['kernel'])))
 p=ROOT/'results/v0.5/trace-analysis.json';d=load(p);assert sha(d['db_path'])==d['db_sha256']
 hist.append(('fa',d['calibration_shape'],dict(ca_us=d['reference_us'],record=str(p.relative_to(ROOT)),db_sha256=d['db_sha256'],kernel=d['kernel'])))
 # No PR holdout result is read to choose shapes. Check only earlier identities below.
 shapes={'fa':[[9,32768,64,1,1],[9,16385,64,1,1],[9,8192,64,1,1],[9,2048,64,1,1],[32,512,64,1,1],[4,256,128,1,1],[17,384,64,1,2],[64,320,128,1,1],[1,64,64,1,1],[9,256,64,4,2],[8,128,128,1,1],[4,160,64,1,1]],'matmul':[[1,32,16],[31,64,257],[16,192,384],[128,192,256],[192,384,256],[255,512,256],[257,512,256],[384,512,256],[512,384,256],[511,512,256],[256,511,256],[128,768,256]]}
 old=set()
 for fn in ['v0.13/sku/pairs.json','v0.13/sku/holdout.json','v0.14/sku/attempts.json']:
  for r in load(ROOT/'results'/fn):
   s=r['shape']+([1,1] if r['op']=='fa' and len(r['shape'])==3 else []);old.add((r['op'],tuple(s)))
 result=[]
 for op,seq in shapes.items():
  for i,s in enumerate(seq):
   assert (op,tuple(s)) not in old;old.add((op,tuple(s)))
   backend='official_full_compacted_after_parse' if op=='fa' and s[1]>=2048 else 'msopprof'
   result.append(dict(id=op+'_'+'_'.join(map(str,s)),op=op,shape=s,cli_shape=s,dt=next((d for o,x,d in hist if o==op and x==s),None),coverage=('Q9 long-loop' if backend!='msopprof' else 'batch/head/tile path' if op=='fa' else 'M/N/K dispatch boundary'),role='independent_pr',priority='primary' if i<10 else 'reserve',runner=RUNNER,host_binary_sha256=sha(ROOT/RUNNER),backend=backend,raw_limit_mib=3072 if s[1]==32768 else 2048 if s[1]>=8192 else 768,wall_limit_s=900 if s[1]==32768 else 540 if s[1]>=8192 else 240))
 return result
if __name__=='__main__':
 b=load(ROOT/'results/v0.14/sku/budget.json')
 write(OUT/'budget.json',dict(utc=utc(),sku=PR,contracts={k:b['contracts'][k] for k in ('fa','matmul')},cases=inputs(),max_workers=4,max_total_seconds=1800,max_total_raw_mib=10240,mandatory_shape=[9,32768,64,1,1],reserve_rule='within-op sequential on failed qualification only; 32k mandatory; no error-based substitutions'),True)
