"""Read-only admission check for a single unbooked Pro B9×Q9 request."""
import json,os,re,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from freeze import ROOT,OUT
def sample():
    x={}
    for row in Path('/proc/stat').read_text().splitlines():
        c,*values=row.split()
        if c.startswith('cpu') and c[3:].isdigit():
            ns=list(map(int,values));x[int(c[3:])]=(sum(ns[:8]),ns[3]+ns[4])
    return x
def snap():
    a=sample();time.sleep(2);b=sample();values={}
    for c in os.sched_getaffinity(0):
        d=Path('/sys/devices/system/cpu')/f'cpu{c}'/'topology';key=((d/'physical_package_id').read_text().strip(),(d/'core_id').read_text().strip())
        values.setdefault(key,[]).append((b[c][1]-a[c][1])/max(1,b[c][0]-a[c][0]))
    return dict(utc=datetime.now(timezone.utc).isoformat(),physical=len(values),avg_idle=sum(sum(v)/len(v) for v in values.values())/len(values),idle80=sum(min(v)>=.8 for v in values.values()),idle70=sum(min(v)>=.7 for v in values.values()))
r=OUT/'pro-b9-mtp'/'host-check.json';register=json.loads((OUT/'decode/freeze.json').read_text());assert not any(j['shape']==[81,64,7168] for j in register['jobs'])
locks=subprocess.check_output(['lslocks','-o','PID,COMMAND,TYPE,MODE,PATH'],text=True)
locks=[x.split() for x in locks.splitlines() if str(ROOT) in x and any(s in x for s in ('matmul28/decode.lock','matmul28/prefill.lock','host30/lane.lock','matmul31/device/slot.lock','matmul/decode/raw/serial.lock'))]
times=[snap() for _ in range(3)]
record=dict(shape=[81,64,7168],workload='DeepSeek-V4-Pro indexer.weights_proj TP8, B9 Q9',sampling='Three successive 2s host /proc/stat samples; physical core available only when each SMT sibling is >=80% idle. No CA invocation.',observation=times,load_average=Path('/proc/loadavg').read_text().split()[:3],operator_sim_locks=locks,policy='Require 32 idle physical CPUs plus 8 reserve, matching the 32-thread 950 simulator. Do not create an acquisition or a waiter if condition is not met.',enough_quiet_cpus=all(x['idle80']>=40 for x in times),booked_as_new_device_work=False)
r.write_text(json.dumps(record,indent=2)+'\n');print(times)
