"""Independent trace landmarks for worst-APE resource estimate per family; no refit."""
from campaign import *
import json

def critical(r):
    pr=r['pr']; c=max(pr['core_windows'],key=lambda x:x['end_us']);core=c['core']
    # Per-core detail time uses numeric pid=queue; use track metadata instead of a
    # hard-coded queue mapping. Anchor to full CA origin using the measured core start.
    path=(ROOT/pr['trace']).parent/core/'trace.json'; data=load(path)['traceEvents']
    tracks={e['pid']:e['args']['name'] for e in data if e['ph']=='M' and e['name']=='process_name'}
    threads={(e['pid'],e['tid']):e['args']['name'].rsplit('_',1)[0] for e in data if e['ph']=='M' and e['name']=='thread_name'}
    ev=[(e['name'],threads.get((e['pid'],e['tid']),tracks.get(e['pid'])),e['ts'],e['ts']+e['dur']) for e in data if e['ph']=='X']
    zero=min(x[2] for x in ev)-c['start_us'];groups={}
    for name,selection in {
      'mte2_data': lambda n,q:q=='MTE2' and n.startswith(('MOV_SRC_TO_DST_ALIGN','MOV_OUT_TO_L1')),
      'mte3_data': lambda n,q:q=='MTE3' and n.startswith('MOV_SRC_TO_DST_ALIGN'),
      'vector_exec':lambda n,q:q=='RVECEX','mmad':lambda n,q:n=='MMAD',
      'fix_write':lambda n,q:n=='FIX_L0C_TO_DST'}.items():
        es=[(s,t) for n,q,s,t in ev if selection(n,q)]
        if es:groups[name]=dict(count=len(es),first_issue_us=min(x[0] for x in es)-zero,last_finish_us=max(x[1] for x in es)-zero)
    return dict(id=r['id'],op=r['op'],shape=r['shape'],signed_error_percent=(r['pr_us']/r['pr_ca_us']-1)*100,active_match=(r['pr_aic'],r['pr_aiv'])==(r['active_aic'],r['active_aiv']),critical_core=core,full_core_window=c,predicted_stages_us=r['projection']['stages_us'],measured_landmarks=groups,trace_sha256=sha(path),measured_data_scope='critical-core detail timeline, includes only named opcode families; no addition of overlapping intervals',trace=str(path.relative_to(ROOT)))
if __name__=='__main__':
 rows=load(OUT/'holdout.json'); assert len(rows)==30
 res=[critical(max((r for r in rows if r['op']==op),key=lambda r:abs(r['pr_us']/r['pr_ca_us']-1))) for op in ['quant','matmul','fa']]
 write(OUT/'worst-landmarks.json',res)
