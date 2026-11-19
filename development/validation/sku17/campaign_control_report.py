"""Same-input official DT control vs the historical DT capture, not a model error."""
from campaign_collect import *
from campaign_archive import dt_provenance
def controls():
 b=read(OUT/'extension-freeze.json');out=[]
 for j in b['jobs']:
  if j['rank'] or j['op']=='kda_chunk':continue
  d=RAW/'extension'/(j['id']+'_'+DT)
  recovered=d.with_name(d.name+'-recovery')/'strict.json'
  q=recovered if recovered.exists() else d/'strict.json'
  if not q.exists():continue
  for x,o in zip(read(q),j['outputs']):
   if not o['dt']:continue
   h=dt_provenance(o['dt']);arch=h['ca_us'];new=x['ca_us']
   out.append(dict(op=o['op'],id=j['id'],shape=o['shape'],attrs=o['attrs'],dt_archived_us=arch,dt_msopprof_us=new,relative_vs_archive=new/arch-1,archive_backend=h['backend'],archive_context=h.get('launch_context'),archive_trace=h['trace'],official_trace=x['trace'],old_kernel=h['kernel'],new_kernel=x['kernel'],comparison_scope='single historical DT capture compared with fresh same-argv/msopprof DT; cannot transfer a correction factor to other shapes'))
 write(OUT/'dt-control-comparison.json',out);return out
if __name__=='__main__':print([(x['op'],round(x['relative_vs_archive']*100,2))for x in controls()])
