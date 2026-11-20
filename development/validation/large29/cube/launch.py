"""Launch one durable, low-priority supervisor with a bounded host affinity."""
from pathlib import Path
import os
import subprocess
import time
import sys
import json

here=Path(__file__).resolve().parent;root=here.parents[3];out=root/'results/v0.29/large/cube';raw=root/'runs/large29/cube'
allowed=os.sched_getaffinity(0)
def stat():
    result={}
    for line in Path('/proc/stat').read_text().splitlines():
        p=line.split()
        if p[0].startswith('cpu') and p[0][3:].isdigit():
            a=list(map(int,p[1:9]));result[int(p[0][3:])]=(sum(a),a[3]+a[4])
    return result
a=stat();time.sleep(2);b=stat()
ranks=sorted(allowed,key=lambda c:-(b[c][1]-a[c][1])/max(1,b[c][0]-a[c][0]))
picked=[];physical=set()
for c in ranks:
    d=Path(f'/sys/devices/system/cpu/cpu{c}/topology')
    key=((d/'physical_package_id').read_text(),(d/'core_id').read_text())
    if key in physical:continue
    physical.add(key);picked.append(c)
    if len(picked)==16:break
with (raw/'supervisor.log').open('a') as f:
    p=subprocess.Popen([str(root/'toolchains/envs/cann-9.2.1/bin/python'),str(here/'campaign.py'),'run'],cwd=root,stdin=subprocess.DEVNULL,
                       stdout=f,stderr=f,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,picked))
    print(p.pid,picked)
    (out/'host.json').write_text(json.dumps(dict(pid=p.pid,logical_cpu_ids=picked,priority_nice=10,scope='lowest observed host utilization over two seconds, one logical thread per physical core, shared CA lock'),indent=2)+'\n')
