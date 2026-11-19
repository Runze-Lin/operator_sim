"""Preregister exact host contracts and DT/PR/970 forecasts; freeze before a new PR run."""
from pathlib import Path
import hashlib,json,sys,math
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/v0.17/sku'; RAW=ROOT/'runs/sku17'
DT,PR='Ascend950DT_9582','Ascend950PR_957b'; FUT='Ascend970DT';sys.path.insert(0,str(ROOT))
from operator_model import estimate
def read(p):return json.loads(Path(p).read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
OPS={'quant':'aclnnDynamicMxQuant','fa':'aclnnFlashAttentionScoreV2','matmul':'aclnnMatmul','addmm':'aclnnAddmm','rmsnorm':'aclnnRmsNorm','lightning_indexer':'aclnnLightningIndexer','mhc_pre':'aclnnMhcPre','mhc_sinkhorn':'aclnnMhcSinkhorn','mhc_post':'aclnnMhcPost','mhc_pre_sinkhorn':'aclnnMhcPreSinkhorn','mhc':'aclnnMhcPre + aclnnMhcSinkhorn + aclnnMhcPost','kda_recurrent':'aclnnRecurrentKda','kda_chunk':'aclnnChunkKdaFwd','moe_ffn':'aclnnGroupedMatmulSwigluQuantV2 + aclnnGroupedMatmulV5','gmm2':'aclnnGroupedMatmulV5','grouped_matmul_swiglu_quant_v2':'aclnnGroupedMatmulSwigluQuantV2'}
CORE=('matmul','quant','fa')
def forecast(op,s,a):
 mode='resource' if op in CORE else 'auto'; results={}
 for k,chip in [('dt',DT),('pr',PR),('future',FUT)]:
  r=estimate(op,s,a,chip=chip,timing_mode=mode);results[k+'_us']=r.latency_us;results[k+'_status']=r.status;results[k+'_aic']=r.active_aic;results[k+'_aiv']=r.active_aiv
  if op!='kda_chunk' and not r.latency_us:raise ValueError((op,s,a,chip,r.reason))
 return results
def dt_ref(r):
 return dict(op=r['op'],shape=r['shape'],attrs=r.get('attrs',{}),chip=DT,ca_us=r['reference_us'],kernel=r['kernel'],api=OPS[r['op']],correctness=r['correctness'],active_aic=r.get('active_aic',r.get('observed_cores',{})),active_aiv=r.get('active_aiv',r.get('observed_cores',{})),trace=r['source'],trace_sha256=r.get('trace_or_db_sha256',r.get('db_sha256')),cost=r.get('cost'),role='reused DT qualified archive',calibration_source='results/v0.12/evaluation.json')
def outputs(r,archive):
 op=r['op'];s=r['shape'];out=[]
 for name,idx in ({'mhc':[('mhc',[0,1,2]),('mhc_pre',[0])],'moe_ffn':[('moe_ffn',[0,1]),('grouped_matmul_swiglu_quant_v2',[0]),('gmm2',[1])]}.get(op,[(op,[0])])):
  z=r if name==op else next(x for x in archive if x['op']==name and s==x['shape'] and (op!='moe_ffn' or x['attrs']==r['attrs']))
  out.append(dict(op=name,shape=s,attrs=z.get('attrs',{}),indices=idx,dt=dt_ref(z),**forecast(name,s,z.get('attrs',{}))))
 return out
def make_case(op,s,args,binary,rows,**extra):
 return dict(id=op+'_'+'_'.join(map(str,s)),op=op,shape=s,runner=binary,runner_sha256=sha(ROOT/binary),args=list(map(str,args)),outputs=rows,**extra)
def ext(archive):
 jobs=[]; ops=['mhc','moe_ffn','mhc_pre_sinkhorn','lightning_indexer','rmsnorm','addmm','mhc_post','mhc_sinkhorn','kda_recurrent']
 for op in ops:
  rows=[x for x in archive if x['op']==op];seen=set();n=0
  for r in rows:
   s=r['shape'];a=r.get('attrs',{});key=tuple(s)
   if key in seen or (op=='addmm' and a.get('self_shape') not in ([s[1]],s[:2])):continue
   if op=='mhc' and not any(x['op']=='mhc_pre' and x['shape']==s for x in archive):continue
   if op=='kda_recurrent' and s[-1]!=128:runner='build/vector-v8/KDA';args=['@list'];batch=['kda_recurrent',0,s[1],s[-1],0]
   else:
    runner='build/smoke/'+('extension_indexer_mhc' if op.startswith(('mhc','lightning')) else 'extension_kda' if op=='kda_recurrent' else 'extension_moe' if op=='moe_ffn' else 'extension_norm_addmm')
    args=([op,s[1],s[2],s[3]] if op=='lightning_indexer' else ['recurrent',s[1]] if op=='kda_recurrent' else [op,*s,a['num_iters']] if op=='mhc_sinkhorn' else [op,*s,'bias' if a['self_shape']==[s[1]] else 'matrix'] if op=='addmm' else [op,*s,','.join(map(str,a['expert_counts']))] if op=='moe_ffn' else [op,*s]);batch=None
   jobs.append(make_case(op,s,args,runner,outputs(r,archive),batch=batch,nchecks=len(r['correctness']),kernels={'mhc':3,'moe_ffn':2}.get(op,1),rank=n,wall_s=240,raw_mib=1200))
   n+=1;seen.add(key)
   if n==10:break
 chunk=[1,64,2,2,128,128];a=dict(dtype='bf16',beta_dtype='fp32',use_gate_in_kernel=True,output_final_state=True,chunk_size=64)
 jobs.append(make_case('kda_chunk',chunk,[64,2],'build/smoke/cube-v8-kda-safe',[dict(op='kda_chunk',shape=chunk,attrs=a,dt=None,indices=list(range(6)),**forecast('kda_chunk',chunk,a))],batch=None,nchecks=6,kernels=6,rank=0,wall_s=135,raw_mib=700))
 return sorted(jobs,key=lambda c:c['rank'])
def core(archive):
 old=set()
 for fn in ['v0.13/sku/pairs','v0.13/sku/holdout','v0.14/sku/attempts','v0.15/sku/attempts','v0.16/sku/attempts']:
  for r in read(ROOT/'results'/(fn+'.json')):
   old.add((r['op'],tuple(r['shape']+([1,1]if r['op']=='fa'and len(r['shape'])==3 else []))))
 cases={'matmul':[(3,64,64),(8,96,256),(33,128,129),(127,384,256),(225,512,256),(449,512,256),(1024,1792,32),(1536,1792,16),(1024,2304,16),(1280,2048,16)],'quant':[(1,256),(5,320),(19,1024),(30,768),(9,4096),(56,4096),(144,5120),(384,4096),(512,5120),(576,7168)],'fa':[(17,192,128,1,1),(4,160,64,1,1),(8,128,128,1,1),(3,192,64,1,2),(9,384,64,1,1),(17,512,128,1,1),(9,1023,64,1,1),(9,3072,64,1,1),(9,8191,64,1,1),(9,16384,64,1,1)]}
 jobs=[]
 for op,ss in cases.items():
  for i,shape in enumerate(ss):
   assert (op,shape) not in old
   s=list(shape);r=next((r for r in archive if r['op']==op and r['shape']==s),None)
   jobs.append(make_case(op,s,[op,*s],'build/smoke/three_ops',[dict(op=op,shape=s,attrs={},dt=dt_ref(r)if r else None,indices=[0],**forecast(op,s,{}))],batch=None,nchecks=1,kernels=1,rank=i,wall_s=440,raw_mib=3500 if op=='quant' else 1800))
 return sorted(jobs,key=lambda c:c['rank'])
def freeze(group):
 file=OUT/(group+'-freeze.json');assert not file.exists() and not (RAW/group).exists()
 dataset=ROOT/'results/v0.12/evaluation.json';a=read(dataset)['rows'];jobs=(ext if group=='extension'else core)(a)
 files=[ROOT/'MANIFEST.in',dataset];files+=list((ROOT/'operator_model').glob('*.py'));files.extend(ROOT/s.split(' ',1)[1] for s in (ROOT/'MANIFEST.in').read_text().splitlines()if s.startswith('include '))
 if group=='core':
  import subprocess
  code="""import json,sys;from operator_model import estimate,__version__
assert __version__=='0.16.0'
print(json.dumps([[estimate(j['op'],j['shape'],chip=c,timing_mode='resource').latency_us for c in ('Ascend950DT_9582','Ascend950PR_957b')] for j in json.load(sys.stdin)]))"""
  prior=json.loads(subprocess.check_output(['/tmp/operator-sim-v16-deploy/bin/python','-I','-c',code],input=json.dumps(jobs),text=True,cwd='/tmp'))
  for j,(dt,pr) in zip(jobs,prior):j['outputs'][0].update(old_dt_us=dt,old_pr_us=pr)
 write(file,dict(utc=utc(),group=group,jobs=jobs,references={str(p.relative_to(ROOT)):sha(p)for p in files},workers=5 if group=='extension'else 4,total_wall_s=7200 if group=='extension'else 4800,total_raw_gib=50 if group=='extension'else 32))
 print(group,len(jobs),sum(len(j['outputs'])for j in jobs))
if __name__=='__main__':freeze(sys.argv[1])
