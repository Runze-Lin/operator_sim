"""Resolve cached DT raw artifacts, launch position and stored qualification evidence."""
from pathlib import Path
from functools import lru_cache
from campaign import *
def smallcost(d):return {k:d[k]for k in ('wall_seconds','cpu_seconds','wall_s','raw_bytes','returncode')if k in d}
@lru_cache(None)
def proof(path):return sha(path)
def dt_provenance(dt):
 src=ROOT/dt['trace'];dt=dict(dt);dt['source_record']=dt['trace'];dt['source_record_sha256']=proof(src)
 for field in ('active_aic','active_aiv'):
  series=dt.get(field)
  if isinstance(series,list):
   dt[field+'_per_launch']=series
   dt[field]=max(series) if series else None
 digests=dt['trace_sha256'];digests=[digests]if isinstance(digests,str)else digests or []
 if src.suffix=='.db':
  seq=int(src.name.split('_')[0]);dt.update(backend='npusim compact',trace=[str(src.relative_to(ROOT))],trace_sha256=[proof(src)],launch_context='in application batch; launch '+str(seq),source_hash_match=(not digests or digests==[proof(src)]));return dt
 d=read(src)
 if 'records'in d:
  rows=d['records']
  def hit(r):
   m=r.get('measurement')or {};c=r.get('case',r)
   argv=r.get('arguments',[]);dims=[str(x) for x in dt['shape']]
   return c.get('op')==dt['op'] and (c.get('shape')==dt['shape'] or dims==argv[1:1+len(dims)] or (digests and m.get('trace_sha256') in digests))
  candidates=[r for r in rows if hit(r)]
  if candidates:d=candidates[0]
 m=d.get('measurement')or {}; sub=m.get('kernels')or d.get('kernels')or []
 if not digests and m.get('db_sha256'):digests=[m['db_sha256']]
 files=[]
 if m.get('trace_sha256') and m.get('report_path'):
  f=Path(m['report_path'])/'simulator/trace.json'
  if f.exists():files=[f]
 if not files and src.parent.name!='extensions' and 'result.json'==src.name:files=sorted(src.parent.glob('profile/npusim_*/record/*.db'))
 if not files:
  sources=[m]+sub+[d]
  files=[Path(x['db_path']) for x in sources if x.get('db_path') and Path(x['db_path']).is_file()]
 if digests and files:files=[p for p in files if proof(p) in digests]
 if files:
  dt['trace']=[str(p.relative_to(ROOT)) for p in files];actual=[proof(p)for p in files]
  dt['source_hash_match']=set(digests)==set(actual) if digests else None;dt['trace_sha256']=actual
  dt['backend']='npusim compact' if all(f.suffix=='.db' for f in files) else 'msopprof official full trace'
  dt['launch_context']='same serialized chain' if dt['op']in ('mhc','moe_ffn','gmm2','grouped_matmul_swiglu_quant_v2') else 'archived application'
 else:
  dt['backend']='archived CA, inspect referenced result';dt['source_hash_match']=None;dt['launch_context']='archived application'
 dt['cost']=smallcost(d)or dt.get('cost');return dt
