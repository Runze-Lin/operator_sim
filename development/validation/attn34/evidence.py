"""Distinguish device-window qualification from agreement with the predicted per-core work."""
from pathlib import Path
from collections import Counter
import json,sys
R=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(R/'development/validation/coverage33'))
from rms_archive import moves, endpoint

def check_payloads(j,v):
    t,h,slots,n=j['shape'];sim=(R/v['trace']).parent
    roster={x.parent.name:x for x in sim.glob('core*/trace.json')}
    if set(roster)!=set(v['core_ticks']):raise ValueError('shard/core mismatch')
    reads=Counter();writes=Counter();origins=[];ends={}
    for name,p in roster.items():
        marks=endpoint(p);ends[name]=marks['END']['tick']
        origins.append(max(x['retire'] for x in marks.values())-v['core_ticks'][name][1])
        for e in moves(p):
            d=json.loads(e['args']['detail'])
            if d['processed_bytes']!=2*h or any(d.get(k) for k in ('is_sync','is_wait','is_duration_excluded')):continue
            if d['instr_type']=='MTE2':reads[name]+=1
            elif d['instr_type']=='MTE3':writes[name]+=1
    if max(origins)-min(origins)>3:raise ValueError('per-core terminal origins disagree')
    plan=j['prices'][j['chip']];pcount=len(writes)
    by_rows={k:cnt//2 for k,cnt in writes.items()}
    checks=dict(writes=sum(writes.values())==2*t and all(v%2==0 for v in writes.values()),
        actual_per_active_cached_weights=all(reads[k]==3+(2+n)*rows for k,rows in by_rows.items()),
        payload_aiv=pcount==plan['plan']['active_aiv'],launch_aiv=len(roster)==plan['plan']['launch_aiv'],
        per_busiest_rows=max(by_rows.values(),default=0)==plan['plan']['iterations'],
        all_read_bytes=sum(reads.values())*2*h==plan['dma_read_bytes'],all_write_bytes=sum(writes.values())*2*h==plan['dma_write_bytes'])
    return dict(physical_ends=ends,origin_offsets_ticks=[min(origins),max(origins)],row_read_counts=reads,row_write_counts=writes,
                actual_payload_aiv=pcount,actual_launch_aiv=len(roster),source_and_prediction_checks=checks,
                mechanisms_match=all(checks.values()),device_ca_us=v['instruction_us'],signed_pct=100*(plan['us']/v['instruction_us']-1))
