"""Optionally give the existing small CA worker idle host cores; no device-contract changes."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import time
from datetime import datetime,timezone
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[3]
DST=ROOT/'results/v0.28/post';RAW=ROOT/'runs/post28'
HOST=ROOT/'build/vector-v8/MHC'
def now():return datetime.now(timezone.utc).isoformat()
def save(p,obj):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(p)
def jiffies():
    ans={}
    for line in Path('/proc/stat').read_text().splitlines():
        if not line.startswith('cpu') or not line[3:4].isdigit():continue
        cpu,*x=line.split();ans[int(cpu[3:])]=(sum(map(int,x[:8])),int(x[3])+int(x[4]))
    return ans
def workers():
    ans={}
    for p in Path('/proc').glob('[0-9]*'):
        try:
            exe=(p/'exe').resolve(strict=True);cwd=(p/'cwd').resolve(strict=True)
            # Account for other sim hosts and their collectors without reading cmdlines or environment.
            if not exe.is_relative_to(ROOT) or not (cwd.is_relative_to(ROOT/'runs') or cwd.is_relative_to(ROOT/'results')):continue
            for th in (p/'task').iterdir():
                vals=(th/'stat').read_text().rsplit(')',1)[-1].split()
                ans[(int(p.name),int(th.name))]=(int(vals[11])+int(vals[12]),int(vals[36]),cwd,exe)
        except (OSError,ValueError):continue
    return ans
def main(adaptive=False, incremental=False):
    reg=json.loads((DST/'registration.json').read_text());protected={};log=[]
    while True:
        s=json.loads((DST/'state.json').read_text())
        if s['phase']=='complete':break
        if s['phase']!='running':time.sleep(6);continue
        raw=RAW/s['case'];job=workers()
        candidate={pid for (pid,tid),(jiff,c,cwd,exe) in job.items() if cwd==raw and exe==HOST}
        if len(candidate)!=1:time.sleep(4);continue
        pid=candidate.pop()
        if pid in protected:time.sleep(6);continue
        if not (32<=len(os.sched_getaffinity(pid))<80 if incremental else len(os.sched_getaffinity(pid))==32):time.sleep(4);continue
        cpu_mask=os.sched_getaffinity(pid);all_cpus=os.sched_getaffinity(0)
        sib={cpu:(Path('/sys/devices/system/cpu')/f'cpu{cpu}/topology/thread_siblings_list').read_text().strip() for cpu in all_cpus}
        before=jiffies();p0=workers();busy=defaultdict(list);used=set();other=set()
        for _ in range(4):
            time.sleep(1);after=jiffies();p1=workers()
            for c in all_cpus:
                dt=after[c][0]-before[c][0];busy[c].append(1-(after[c][1]-before[c][1])/dt if dt else 1.)
            for key,(jiff,c,cwd,exe) in p1.items():
                if key[0]!=pid and key in p0 and jiff>p0[key][0]:
                    used.update([sib[c],sib[p0[key][1]]]);other.add(exe.name)
            before,p0=after,p1
        idle=[];unique=set()
        for c in sorted(all_cpus,key=lambda x:(max(busy[x]),sum(busy[x]),x)):
            if max(busy[c])<=(.35 if incremental else .25) and sib[c] not in used and sib[c] not in unique and c not in cpu_mask:
                idle.append(c);unique.add(sib[c])
        # Preserve low-utilization resources for independent CA jobs even when their
        # workers migrated over those cores briefly. Never assign such a core to Post.
        reserve=len(idle)
        if adaptive:
            reserve=len({sib[c] for c in all_cpus if c not in cpu_mask and max(busy[c])<=.25})
        keep=72;extra=min(80-len(cpu_mask),len(idle),max(0,reserve-keep))
        if incremental:
            # Workers that happened to migrate over a core are excluded from
            # extension, even if its average busy time is below the threshold.
            # Reserve >=40 continuously quiet distinct CPUs plus >=72 CPU
            # equivalents of sampled spare capacity outside the enlarged mask.
            outside={sib[c]:c for c in all_cpus if sib[c] not in {sib[x] for x in cpu_mask}}
            quiet=sum(max(busy[c])<=.35 for c in outside.values())
            spare=sum(1-sum(busy[c])/len(busy[c]) for c in outside.values())
            extra=min(80-len(cpu_mask),len(idle),max(0,quiet-40),max(0,int(spare)-72))
        if extra<(4 if incremental else 16):time.sleep(8);continue
        pth=Path('/proc')/str(pid)
        try:
            assert (pth/'exe').resolve(strict=True)==HOST and (pth/'cwd').resolve(strict=True)==raw
            if os.sched_getaffinity(pid)!=cpu_mask:continue
            mask=cpu_mask|set(idle[:extra])
            changed=0
            for th in (pth/'task').iterdir():
                try:os.sched_setaffinity(int(th.name),mask);changed+=1
                except ProcessLookupError:pass
            protected[pid]=not incremental or len(mask)>=80
            if not protected[pid]: del protected[pid]
            log.append(dict(utc=now(),case=s['case'],actual_worker_pid=pid,changed_threads=changed,
                            original_cpus=sorted(cpu_mask),extended_cpus=sorted(mask),other_active_executables=sorted(other),
                            physically_distinct_candidates=len(idle),retained_low_busy_physical=reserve-extra,untouched_extra_candidates_after=len(idle)-extra,
                            aggregate_host_load_1m=os.getloadavg()[0]))
            if incremental:
                log[-1].update(minimum_quiet_distinct_after=quiet-extra,minimum_idle_capacity_equivalents_after=spare-extra)
                log[-1].pop('retained_low_busy_physical',None)
            name='host-affinity-incremental.json' if incremental else ('host-affinity-v2.json' if adaptive else 'host-affinity.json')
            save(DST/name,dict(scope='Four 1s samples each <=35% busy and no observed foreign active worker on added physical cores, preserving at least 40 continuously quiet distinct cores and 72 available CPU-equivalents for others. Same live MHC only, no changed sampling budget.' if incremental else 'Same live MHC binary and registration. Four 1s samples each <=25% busy; distinct physical host cores and explicit exclusion of observed active CPU locations of other CA workers. Keep >=72 idle spare after extension. No new device launch or deadline extension.',rows=log))
        except (FileNotFoundError,ProcessLookupError):continue
if __name__=='__main__':
    if sys.argv[1:2] == ['start']:
        with (RAW/'host-admission.log').open('a') as out:
            p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*sys.argv[2:]],stdin=subprocess.DEVNULL,stdout=out,
                               stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,cwd=ROOT)
            print(p.pid,flush=True)
    else:
        main('adaptive' in sys.argv,'incremental' in sys.argv)
