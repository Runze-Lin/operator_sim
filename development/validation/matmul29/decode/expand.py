"""Observe the exact live decode process; add genuinely quiet physical CPUs to an over-subscribed CA only."""
import datetime
import json
import os
from pathlib import Path
import re
import time

ROOT=Path(__file__).resolve().parents[4];D=ROOT/'results/v0.29/matmul/decode';BIN=ROOT/'build/smoke/matmul28'
LOG=D/'affinity-observations.json'
def read(p):return json.loads(p.read_text())
def log(msg):
    j=read(LOG)if LOG.exists()else {'policy':'One live decode host only. Start with 8–16 CPUs. After >=90s and ~1 distinct thread per CPU or worse, record live workers and only add physically distinct CPUs <=35% busy over three 1s samples. Keep >=8 idle physical CPUs outside and cap this lower-priority host at 64. No restarts or changes to budgets.','events':[]}
    j['events'].append(msg);t=LOG.with_suffix('.tmp');t.write_text(json.dumps(j,indent=2)+'\n');t.replace(LOG)
def hardware():
    return {int(v[0][3:]):(sum(map(int,v[1:9])),int(v[4])+int(v[5]))for x in Path('/proc/stat').read_text().splitlines()if re.match(r'cpu\d+ ',x)for v in [x.split()]}
def sample(allowed):
    last=hardware();worst={c:0. for c in allowed}
    for _ in range(3):
        time.sleep(1);cur=hardware()
        for c in allowed:worst[c]=max(worst[c],1-(cur[c][1]-last[c][1])/max(1,cur[c][0]-last[c][0]))
        last=cur
    return worst
def process(folder,shape):
    for pid in Path('/proc').glob('[0-9]*'):
        try:
            if (pid/'exe').resolve(strict=True)!=BIN or (pid/'cwd').resolve(strict=True)!=folder:continue
            args=(pid/'cmdline').read_bytes().strip(b'\x00').split(b'\x00');assert [int(v)for v in args[1:]]==shape
            return pid
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
def info(p):
    stat=(p/'stat').read_text().rsplit(')',1)[1].split()
    return (int(stat[11])+int(stat[12]))/os.sysconf('SC_CLK_TCK')
def main():
    cpus=os.sched_getaffinity(0);siblings={c:(Path('/sys/devices/system/cpu')/f'cpu{c}/topology/thread_siblings_list').read_text().strip()for c in cpus}
    last_logged={}
    while True:
        state=read(D/'state.json')
        if state['phase']=='finished':return
        case=state.get('case','');folder=D/'raw'/case
        if state['phase']!='acquiring' or not (folder/'run.log').exists():time.sleep(10);continue
        age=time.time()-datetime.datetime.fromisoformat(state['utc']).timestamp()
        if age<90:time.sleep(10);continue
        p=process(folder,[int(x)for x in case.split('-')[:3]])
        if not p:time.sleep(8);continue
        try:
            pid=int(p.name);old=os.sched_getaffinity(pid);th=list((p/'task').glob('[0-9]*'))
            if len(old)>=64 or len(th)<=len(old):time.sleep(12);continue
            cpu=info(p);util=sample(cpus);cpu2=info(p);taken={siblings[c]for c in old};free=[]
            for c in sorted(util,key=util.get):
                if util[c]<=.35 and siblings[c]not in taken:
                    taken.add(siblings[c]);free.append(c)
            count=max(0,min(16,len(free)-8,64-len(old)));addition=free[:count];changed=0
            if count>=4 and (p/'exe').resolve(strict=True)==BIN and (p/'cwd').resolve(strict=True)==folder and read(D/'state.json')['case']==case:
                new=old|set(addition)
                for t in th:
                    try:os.sched_setaffinity(int(t.name),new);changed+=1
                    except ProcessLookupError:pass
            if changed or time.monotonic()-last_logged.get(case,0)>75:
                record=(folder/'run.log').read_text(errors='replace')
                lastblock=re.findall(r'\[(?:info|INFO)\]\s+\[(\d+)\].*?\[(block_start|block_end)\]',record)
                log(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),case=case,pid=pid,host_age_s=age,original_threads=len(th),
                         old_mask=sorted(old),minute_cpu_s=cpu2,cores_consumed_during_sample=(cpu2-cpu)/3,
                         quiet_physical_outside_mask=len(free),latest_block_marker=lastblock[-1:] or None,
                         added=addition if changed else [],updated_threads=changed,unchosen_quiet=len(free)-(len(addition)if changed else 0)))
                last_logged[case]=time.monotonic()
        except (FileNotFoundError,ProcessLookupError):pass
        time.sleep(10)
if __name__=='__main__':main()
