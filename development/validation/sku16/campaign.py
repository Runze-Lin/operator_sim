"""v0.16 resource-only independent holdout registration (DT historical pair optional)."""
from pathlib import Path
import hashlib,json
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/v0.16/sku';RUNS=ROOT/'runs/sku16'
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
# Deliberately select from shape geometry/previous identities, not PR durations.
SHAPES={
 'fa':[(9,32767,64,1,1),(9,12289,64,1,1),(9,2049,64,1,1),(1,224,64,1,1),(32,129,64,1,4),(17,257,128,1,1),(16,128,64,1,8),(8,192,64,1,1),(64,256,64,1,1),(9,384,128,2,2),(4,160,64,1,1),(8,128,128,1,1)],
 'matmul':[(4,80,48),(64,128,512),(2,32,32),(9,128,256),(224,512,256),(448,512,256),(513,384,256),(768,512,128),(1024,2048,32),(1536,2048,16),(320,384,256),(1024,2048,16)],
 'quant':[(1,128),(7,320),(17,1024),(36,512),(36,3072),(128,5120),(36,5120),(288,7168),(768,4096),(1024,4096),(17,512),(1,192)]}
def inputs():
 arch=ROOT/'results/v0.13/evaluation.json'; dt_archive=load(arch)['rows']
 old=set()
 for f in ['v0.13/sku/pairs.json','v0.13/sku/holdout.json','v0.14/sku/attempts.json','v0.15/sku/attempts.json']:
  for r in load(ROOT/'results'/f):
   s=r['shape']+([1,1] if r['op']=='fa' and len(r['shape'])==3 else [])
   old.add((r['op'],tuple(s)))
 cases=[]
 for op,seq in SHAPES.items():
  for i,shape in enumerate(seq):
   assert (op,shape) not in old and len(seq)==len(set(seq));old.add((op,shape))
   s=list(shape);prior=next((r for r in dt_archive if r['op']==op and r['shape']==s),None)
   dt=dict(record='results/v0.13/evaluation.json',id=prior['id'],ca_us=prior['dt_ca_us']) if prior and prior['dt_ca_us'] else None
   long=op=='fa' and s[1]>=2049; bigq=op=='quant' and s[0]*s[1]>=700000
   coverage=('Q9 KV-tail loop' if long else 'batch/head/short Q tiling' if op=='fa' else 'UB repeat or wide per-core block' if bigq else 'small, aligned or tail vector tile' if op=='quant' else 'high load/multi-wave candidate' if i>=7 else '24/28 AIC boundary candidate' if i>=4 else 'single/tail/K pages')
   cases.append(dict(id=op+'_'+'_'.join(map(str,s)),op=op,shape=s,cli_shape=s,role='independent_pr',priority='primary' if i<10 else 'reserve',coverage=coverage,backend='official_full_compacted_after_parse' if long else 'msopprof',runner=RUNNER,host_binary_sha256=sha(ROOT/RUNNER),dt=dt,raw_limit_mib=(2000 if bigq else 1800 if long else 640),wall_limit_s=(600 if long else 420 if bigq else 210)))
 return cases
if __name__=='__main__':
 write(OUT/'budget.json',dict(utc=utc(),sku=PR,contracts=load(ROOT/'results/v0.14/sku/budget.json')['contracts'],cases=inputs(),max_workers=4,max_total_seconds=1800,max_total_raw_mib=10240,mandatory_shape=[9,32767,64,1,1],reserve_rule='Only predeclared reserve after failed qualification; never replace mandatory PR 32767 with another shape; no replacements based on error'),True)
