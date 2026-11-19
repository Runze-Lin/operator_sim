"""Read the real KDA instruction shards even when whole-call timing is ineligible."""
from collections import Counter
from pathlib import Path
import json
import re
import sys

from probe import OUT, CAPTURE, ROOT, put, rel, digest
sys.path.insert(0, str(ROOT/'development/validation/rms_ordered_verify'))
from verify import events

def main():
    profile,=CAPTURE.glob('profile/OPPROF_*')
    binary=sorted(profile.glob('**/ChunkKdaFwd_*/0/dump/aicore_binary.o'))[0]
    directory=binary.parent.parent
    base=int((binary.parent/'pc_start_addr.txt').read_text(),16)
    errors=json.loads((OUT/'faults.json').read_text())
    flagged={int(i['offset'],16) for i in errors['instructions']}
    shards=sorted((directory/'simulator').glob('core*/trace.json'))
    rows={}
    examples={}
    for path in shards:
        counts=Counter();flags=[];entry=None;end=[];rows_seen=0;reason=None;mmad=[]
        try:
            for x in events(path):
                rows_seen+=1
                name=x['name'];counts[name]+=1
                tick=round(x['ts']*1650)
                pc=int(x['args']['pc_addr'],16)-base
                if entry is None or tick<entry:entry=tick
                if name=='END':end.append(tick)
                if name=='MMAD':mmad.append(tick)
                if name in ('SET_FLAG','WAIT_FLAG') and path.parent.name=='core0.cubecore0':
                    detail=json.loads(x['args']['detail'])
                    if detail.get('target_pipe')=='CUBE' and detail.get('consumer_pipe')=='MTE1':
                        flags.append((tick,name,detail['flag_id'],pc))
        except (ValueError,OSError,KeyError) as exc:
            reason=str(exc)
        c=path.parent.name
        rows[c]=dict(complete_json=reason is None,problem=reason,first_tick=entry,instruction_events=rows_seen,
                     mmads=len(mmad),first_mmad=min(mmad,default=None),physical_end_ticks=end,end_label_count=counts['END_LABEL'])
        if flags:
            last={}
            for tick,kind,flag,pc in sorted(flags):
                if kind=='SET_FLAG' and pc in flagged and flag in last and last[flag][1]=='SET_FLAG' and flag not in examples:
                    a=last[flag];examples[flag]=dict(previous_tick=a[0],previous_pc=hex(a[2]),duplicate_tick=tick,
                        duplicate_pc=hex(pc),first_mmad_tick=min(mmad,default=None),core=c)
                last[flag]=(tick,kind,pc)
    log=(CAPTURE/'run.log').read_text()
    begin=re.search('Start profiling on kernel: (ChunkKdaFwd_\\w+)',log)
    segment=log[begin.start():].split('Start profiling on kernel: ',2)[1]
    counts=dict(shards=len(rows),complete_json=sum(r['complete_json'] for r in rows.values()),
                physical_end=sum(bool(r['physical_end_ticks']) for r in rows.values()),
                end_label=sum(bool(r['end_label_count']) for r in rows.values()),
                cores_with_mmad=sum(bool(r['mmads']) for r in rows.values()),
                all_shard_mmad=sum(r['mmads'] for r in rows.values()))
    put(OUT/'partial-main.json',dict(kernel=begin[1],aggregate=rel(directory/'simulator/trace.json'),
        aggregate_sha256=digest(directory/'simulator/trace.json'),
        task_done_lines=[l for l in segment.splitlines() if '[TASK_DONE]' in l],delta_source='ticks from device simulator (1650 ticks/us)',
        observations=counts,first_duplicate_without_intervening_local_wait=examples,cores=rows,
        scope='diagnostic raw event counts only; no device-latency estimate or complete-host golden is derived from these shards'))
    print(counts,examples)

if __name__=='__main__':main()
