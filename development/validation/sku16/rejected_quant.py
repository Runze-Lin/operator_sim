"""Read existing native artifacts from budget stops; never qualify them for latency."""
import re
from campaign import *
def audit():
 out=[]
 for spec in load(OUT/'attempts.json'):
  if spec['qualified'] or spec['op']!='quant':continue
  folder=RUNS/spec['id'];log=(folder/'run.log').read_text()
  tr,=folder.glob('profile/OPPROF_*/simulator/trace.json')
  with tr.open('rb') as f:f.seek(-2,2); closed=f.read()==b']}'
  rows=re.findall(r'^(core\d+\.veccore\d+)\s+([\d.]+)\s+[\d.]+\s*$',log,re.M)
  # Per-core details for a documented representative core, not a critical latency.
  selected=tr.parent/'core0.veccore0/trace.json';counts=None
  if selected.is_file():
   try: raw=load(selected)['traceEvents']
   except ValueError: pass
   else:
    counts={}
    for e in raw:
     if e.get('ph')!='X' or e.get('name')!='MOV_SRC_TO_DST_ALIGNv2':continue
     detail=json.loads(e['args']['detail']); q=detail['instr_type']
     counts.setdefault(q,[]).append(dict(ts_us=e['ts'],duration_us=e['dur'],processed_bytes=detail.get('processed_bytes')))
  out.append(dict(id=spec['id'],shape=spec['shape'],eligible=False,reason=spec['reason'],SMOKE_PASS='SMOKE_PASS' in log,golden=re.findall(r'^correctness (.+)',log,re.M),pr_runtime=('runtime_soc='+PR)in log,pr_lpddr5='lpddr5_6400M_bailu.ini' in log,slice_cost_s=spec['seconds'],raw_bytes=spec['raw_bytes'],official_process_rc=spec['exit_code'],official_parse_finished='Profiling data parse finished.' in log,reported_core_count=len(rows),merged_trace_ends_with_closing_bytes=closed,representative_core='core0.veccore0',representative_movs=counts,per_core_json_sha256=sha(selected) if selected.is_file() else None))
 write(OUT/'multiub-incomplete.json',dict(qualification='Neither record is an independent CA time; not included in any WAPE or maximum.',entries=out));print([(x['id'],x['merged_trace_ends_with_closing_bytes'],{k:len(v) for k,v in (x['representative_movs'] or {}).items()})for x in out])
if __name__=='__main__':audit()
