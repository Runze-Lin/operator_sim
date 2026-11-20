"""Widen only the running registered MoE simulator's inherited 16-CPU mask."""
from collections import Counter
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
RAW=ROOT/'runs/large29/cube/moe-dt'
OUT=ROOT/'results/v0.31/host'
ARGS=[str(RAW.parent/'bin/moe'),'moe_ffn','4096','7168','384','1','4096']


def counters():
    return {int(x[0][3:]):(sum(map(int,x[1:9])),int(x[4])+int(x[5])) for row in Path('/proc/stat').read_text().splitlines()
            if (x:=row.split())[0][3:].isdigit() and x[0].startswith('cpu')}


def observe():
    log=RAW/'run.log'
    with log.open('rb') as f:
        f.seek(max(0,log.stat().st_size-1024**2))
        cycles=[int(x) for x in re.findall(rb'Thread 32 is running. Cycle Count: (\d+)',f.read())]
    return dict(utc=datetime.now(timezone.utc).isoformat(),stage_local_last_cycle_marker=cycles[-1] if cycles else None,
                run_log_bytes=log.stat().st_size,acquisition_receipt=(RAW/'cost.json').exists())


def run(pid):
    proc=Path('/proc')/str(pid)
    def match():
        return (proc.stat().st_uid==os.getuid() and (proc/'cwd').resolve()==RAW and
                [v.decode() for v in (proc/'cmdline').read_bytes().split(b'\0') if v]==ARGS and
                os.getpriority(os.PRIO_PROCESS,pid)>=10 and not (RAW/'cost.json').exists())
    assert match()
    before=os.sched_getaffinity(pid)
    assert len(before)==16 and not (OUT/'moe-affinity.json').exists()
    allowed=os.sched_getaffinity(0)
    assert before<=allowed
    first=counters();time.sleep(2);second=counters()
    def load(i):
        t0,id0=first[i];t1,id1=second[i]
        return 1-(id1-id0)/max(1,t1-t0)
    chosen=set(before)
    def phys(i):
        return (Path('/sys/devices/system/cpu')/f'cpu{i}/topology/thread_siblings_list').read_text().strip()
    physical={phys(i) for i in chosen}
    for i in sorted(allowed,key=load):
        if phys(i) not in physical:
            chosen.add(i);physical.add(phys(i))
        if len(physical)>=80:break
    assert len(physical)>=64 and len(chosen)<len(allowed)
    prev=observe()
    counts=Counter()
    assert match()
    for p in (proc/'task').iterdir():
        tid=int(p.name)
        try:
            if os.sched_getaffinity(tid)==before:
                os.sched_setaffinity(tid,chosen)
                counts['widened']+=1
            else:
                counts['other_placement_preserved']+=1
        except ProcessLookupError:
            counts['already_exited']+=1
    end=dict(pid=pid,argv=ARGS,old_cpu_mask=sorted(before),new_cpu_mask=sorted(chosen),physical_cores=len(physical),
             threads=dict(counts),start=prev,source_freeze='results/v0.29/large/cube/freeze.json',
             scope='Host placement only, same real device acquisition, unchanged device providers and budgets. ')
    with (OUT/'moe-affinity.json').open('x') as fp:json.dump(end,fp,indent=2);fp.write('\n')
    print(dict(changed=counts,physical=len(physical),start=prev))


if __name__=='__main__':
    run(int(sys.argv[1]))
