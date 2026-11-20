"""Measure stage boundaries and real four-output DMA windows across the archived channel repetitions."""
import json
from hashlib import sha256
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.32/post'

def read(path):
    return json.loads(path.read_bytes())

def main():
    measured=read(ROOT/'results/v0.24/post/evaluation.json')['rows']
    by_core=read(ROOT/'results/v0.24/post/topology.json')['rows']
    records=[]
    for row in measured:
        if not row.get('timing_qualified') or row['protocol']!='immediate repeat' or row['route']!='post_channel':
            continue
        sel=next(r for r in by_core if (r['chip'],r['group'],r['i'])==(row['chip'],row['group'],row['i']))
        core=sel['critical'][0]
        shard=(ROOT/row['trace']).parent/core/'trace.json'
        events=[r for r in read(shard)['traceEvents'] if r.get('ph')=='X']
        zero=min(e['ts'] for e in events if e['name']=='RV_VLD')
        writes=[];vf=[];axpy=0
        for e in events:
            detail=json.loads(e['args']['detail'])
            if detail['queue_type']=='MTE3' and detail.get('processed_bytes',0)>0:
                writes.append([e['ts']-zero,e['ts']-zero+e['dur'],detail['processed_bytes']])
            if e['name']=='VF':
                vf.append([e['ts']-zero,e['ts']-zero+e['dur']])
            if e['name']=='RV_VAXPY':
                axpy+=1
        writes.sort()
        assert len(writes)==4
        records.append(dict(chip=row['chip'],shape=row['shape'],fragment_width=row['fragment_width'],
                    original_case=row['group'],ordinal=row['i'],crit_core=core,cohort=sel['critical_cohort'],
                    known_repeat_ca_us=row['ca_us'], first_rv_load_zero_us=zero,
                    measured_stages_us=row['observed_phase_us'], modeled_repeat_stages_us=row['predicted_stages_us'],
                    writes_us_bytes=writes, vector_functions=len(vf), vector_axpy_trace_events=axpy,
                    ca_sha256=row['trace_sha'],shard=str(shard.relative_to(ROOT)),shard_sha256=sha256(shard.read_bytes()).hexdigest()))
    (OUT/'wide-repeats.json').write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
