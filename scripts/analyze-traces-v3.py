#!/usr/bin/env python3
"""One-time offline trace reduction; no CA calls and no timing fit here."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from operator_model.ca import sha256,union_length

def reduce_quant(record):
    measure=record['measurement']; path=Path(measure['report_path'])/'simulator/trace.json'
    if sha256(path)!=measure['trace_sha256']:raise ValueError('Trace changed')
    chosen=max(measure['cores'],key=lambda c:c['start_us']+c['span_us'])
    trace=json.loads(path.read_text())['traceEvents']
    start=min(e['ts'] for e in trace if e['ph']=='X')
    es=[e for e in trace if e['ph']=='X' and e['pid']==chosen['name']]
    events=[]
    for e in es:
        if e['tid'] not in ('MTE2','MTE3','PUSHQ'):continue
        d=json.loads(e.get('args',{}).get('detail','{}'))
        if d.get('is_sync') or d.get('is_wait'):continue
        if e['name']=='VF' or 'processed_bytes' in d:
            events.append({'pipe':e['tid'],'name':e['name'],'start_us':e['ts']-start,
                           'end_us':e['ts']+e['dur']-start,'bytes':d.get('processed_bytes'),
                           'src_addr':d.get('src_addr'),'dst_addr':d.get('dst_addr')})
    events.sort(key=lambda e:e['start_us'])
    reads=[e for e in events if e['pipe']=='MTE2']
    # First MTE2 transaction is the 160-byte tiling/POD read, not input X.
    if not reads or reads[0]['bytes']!=160:raise ValueError('Unrecognized tiling load')
    payload=reads[1:];vfs=[e for e in events if e['name']=='VF'];writes=[e for e in events if e['pipe']=='MTE3']
    if len(vfs)!=3*len(payload) or len(writes)!=2*len(payload):raise ValueError('Unrecognized quant iteration')
    iterations=[]
    for i,read in enumerate(payload):
        group=vfs[i*3:i*3+3];out=writes[i*2:i*2+2]
        iterations.append({'read':read,'vf':group,'writes':out,
                           'vf_span_us':max(e['end_us'] for e in group)-group[0]['start_us'],
                           'write_drain_us':max(e['end_us'] for e in out)-min(e['start_us'] for e in out),
                           'vector_ready_gap_us':group[0]['start_us']-read['end_us']})
    return {'id':record['case']['id'],'shape':record['case']['shape'],'trace_sha256':measure['trace_sha256'],
            'core':chosen['name'],'reference_us':measure['reference_us'],'iterations':iterations,
            'prefix_us':payload[0]['start_us'],'suffix_us':measure['reference_us']-max(e['end_us'] for e in writes),
            'first_read_us':payload[0]['end_us']-payload[0]['start_us'],
            'pre_vector_fraction':vfs[0]['start_us']/measure['reference_us'],
            'flow_event_count':sum(e['ph'] in ('s','f') for e in trace),
            'causality_scope':'explicit phase order from source; trace overlap alone does not establish a flag dependency'}

def main():
    target=ROOT/'results/v0.3/trace-analysis.json'
    if target.exists():raise SystemExit('Reuse existing reduction; do not overwrite evidence')
    rows=[]
    for r in json.loads((ROOT/'results/v0.2/dataset.json').read_text())['records']:
        if r['case']['op']=='quant':
            row=reduce_quant(r);rows.append(row)
            print(row['id'],row['shape'],'prefix',round(row['prefix_us'],3),'read',round(row['first_read_us'],3),
                  'vf',[round(i['vf_span_us'],3) for i in row['iterations']],'suffix',round(row['suffix_us'],3),flush=True)
    target.write_text(json.dumps({'source_dataset_sha256':sha256(ROOT/'results/v0.2/dataset.json'),'quant':rows},indent=2)+'\n')
if __name__=='__main__':main()
