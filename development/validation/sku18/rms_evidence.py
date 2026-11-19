"""Recover six disjoint critical-core intervals from the v0.17 DT/PR archive."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate
PHASES=('SIMD','MTE3','MTE2','SCALAR')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda:f.read(4*1024*1024),b''):h.update(buf)
    return h.hexdigest()

def spans(path):
    if path.suffix=='.db':
        with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
            origin=db.execute('SELECT MIN(ExecInstrTickStart) FROM ExecutedInstructions').fetchone()[0]/1650
            cid,typ,start,end=db.execute('SELECT ExecInstrCoreId,CoreTypeId,MIN(ExecInstrTickStart),MAX(ExecInstrTickEnd) FROM ExecutedInstructions GROUP BY 1,2 ORDER BY 4 DESC,1,2 LIMIT 1').fetchone()
            rows=[(q,a/1650,b/1650,op) for q,a,b,op in db.execute("""SELECT json_extract(ExtendParams,'$.queue_type'),ExecInstrTickStart,ExecInstrTickEnd,ExecInstrName FROM ExecutedInstructions WHERE ExecInstrCoreId=? AND CoreTypeId=? AND ExecInstrIsDurationExcluded=0""",(cid,typ))]
        return origin,start/1650,end/1650,rows
    raw=json.loads(path.read_text())['traceEvents'];raw=[x for x in raw if x['ph']=='X' and 'dur' in x]
    last=max(raw,key=lambda x:x['ts']+x['dur']);origin=min(x['ts'] for x in raw)
    core=[x for x in raw if x['pid']==last['pid']]
    rows=[]
    for x in core:
        detail=x['args']['detail'];meta=json.loads(detail)
        if not meta.get('is_duration_excluded'):
            rows.append((meta['queue_type'],x['ts'],x['ts']+x['dur'],x['name']))
    return origin,min(x['ts'] for x in core),last['ts']+last['dur'],rows

def inspect(record):
    path=ROOT/record['trace'][0];hashed=digest(path);assert hashed==record['trace_sha256'][0]
    origin,start,stop,rows=spans(path)
    events=defaultdict(Counter);events[start];events[stop]
    for q,a,b,_ in rows:events[a][q]+=1;events[b][q]-=1
    prior=start;st=Counter(arrival=start-origin);active=Counter();union=masked=0.
    for t,changes in sorted(events.items()):
        dt=t-prior;phase=next((q for q in PHASES if active[q]),'other');st[phase]+=dt
        if active['MTE3']:
            union+=dt
            if active['SIMD']:masked+=dt
        active.update(changes);prior=t
    assert abs(sum(st.values())-record['ca_us'])<.003,(path,st,record['ca_us'])
    return dict(trace=record['trace'][0],sha256=hashed,stages_us=st,store_union_us=union,store_overlapped_by_simd_us=masked,opcode_counts=Counter(q+':'+op for q,_,_,op in rows if q in ('MTE2','MTE3')))

def main():
    out=[]
    for record in json.loads((ROOT/'results/v0.17/sku/pairs.json').read_text()):
        if record['op']!='rmsnorm':continue
        row=dict(shape=record['shape'],old_pr_us=record['pr_us'])
        for side in ('dt','pr'):
            rec=record[side]
            y=estimate('rmsnorm',row['shape'],chip=rec['chip'],timing_mode='resource')
            row[side]=dict(ca_us=rec['ca_us'],active_aiv=rec['active_aiv'],
                           launch_context=rec.get('launch_context') or 'one isolated process per PR collection',
                           predicted_us=y.latency_us,conditional_us=y.projection['timing_diagnostics']['conditional_us'],
                           predicted_stages_us=y.projection['stages_us'],**inspect(rec))
        out.append(row)
    (ROOT/'results/v0.18/rms-paired-phases.json').write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':main()
