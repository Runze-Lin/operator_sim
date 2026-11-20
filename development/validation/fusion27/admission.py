"""Read-only headroom accounting and low-weight scheduling for scoped fused processes."""
from pathlib import Path
import os,re,shutil,time
from observe import ROOT,OUT,ORIG,RAW,read,save,utc,tree,argv
ALL=sorted(os.sched_getaffinity(0));N=len(ALL)
POLICY=OUT/'admission.json'
def ticks():
    a={}
    for l in Path('/proc/stat').read_text().splitlines():
        name,*v=l.split()
        if name.startswith('cpu') and name[3:].isdigit() and int(name[3:]) in ALL:
            vals=list(map(int,v[:8]));a[int(name[3:])]=(sum(vals),vals[3])
    return a
def capacity(a,b,live=False):
    idle={c:max(0,min(1,(b[c][1]-v[1])/(b[c][0]-v[0]))) for c,v in a.items() if c in b and b[c][0]>v[0]}
    p=read(POLICY);mask=sorted(idle,key=lambda i:(-idle[i],i))[:p['affinity_cap']]
    avg=sum(idle[c] for c in mask)/len(mask) if mask else 0;load=os.getloadavg()
    pro=read(ROOT/'results/v0.23/pro-prefill/state.json');full=read(ROOT/'results/v0.23/full-hidden/v3/queue-status.json');fa=read(ROOT/'results/v0.25/coverage/supplement/state.json')
    ram=int(re.search(r'^MemAvailable:\s+(\d+) kB',Path('/proc/meminfo').read_text(),re.M)[1])*1024/2**30
    disk=shutil.disk_usage(ROOT).free/2**30
    reasons=[]
    if pro['state'] not in ('acquiring','scoring existing full trace','finished'):reasons.append('Pro waiting')
    if full['state']=='waiting-for-Kimi' and not any(Path(f'/proc/{n}').exists() for n in full.get('kimi_live_pids',[])):reasons.append('Flash ready to start')
    if full['state'] not in ('waiting-for-Kimi','flash-native-active','complete'):reasons.append('Flash admission')
    if fa['state'] not in ('acquiring','qualifying','finished','interrupted raw attempt; preserved without duplicate'):reasons.append('older FA awaiting CPU')
    if load[0]>(p['live_load1'] if live else p['new_load1'])*N:reasons.append('host load')
    if len(mask)<p['min_mask'] or sum(idle.values())<p['idle_reserve'] or avg<p['min_mask_idle']:reasons.append('measured idle cores')
    if disk<128 or ram<128:reasons.append('disk/memory reserve')
    return sorted(mask),dict(pro=pro['state'],fa=fa['state'],full_hidden=full['state'],load1=load[0],load5=load[1],idle_equivalent=round(sum(idle.values()),1),
                             mask_idle=round(avg,3),memory_gib=round(ram,1),free_disk_gib=round(disk,1),reasons=reasons)
def set_limits(tid,mask):
    os.setpriority(os.PRIO_PROCESS,tid,19)
    os.sched_setscheduler(tid,os.SCHED_IDLE,os.sched_param(0))
    os.sched_setaffinity(tid,mask)
def native(job):
    # Only allow the exact original host binary plus its frozen shape args, in the
    # registered raw working directory, within this collector's descendant tree.
    root=Path('/proc/2701028')
    if not root.exists() or not any('presink26/resume.py' in a for a in argv(root)):return None
    folder=RAW/job['id']
    for pid in tree(2701028):
        f=Path(f'/proc/{pid}')
        try:
            if argv(f)==[str(ROOT/job['runner']),*job['args']] and (f/'cwd').resolve()==folder.resolve():return pid
        except (FileNotFoundError,ProcessLookupError):pass
    return None
def widen(job,pid,mask):
    if native(job)!=pid or (RAW/job['id']/'cost.json').exists():return None
    f=Path(f'/proc/{pid}');start=(f/'stat').read_text().split(') ')[1].split()[19];old=sorted(os.sched_getaffinity(pid))
    if len(old)!=16:return None
    affected=[]
    for task in (f/'task').iterdir():
        tid=int(task.name)
        try:
            if (f/'stat').read_text().split(') ')[1].split()[19]!=start:raise RuntimeError('PID identity changed')
            if task.exists():
                set_limits(tid,mask);affected.append(tid)
        except (FileNotFoundError,ProcessLookupError):continue
    verified=[]
    for task in (f/'task').iterdir():
        try:
            tid=int(task.name);verified.append(dict(tid=tid,policy=os.sched_getscheduler(tid),nice=os.getpriority(os.PRIO_PROCESS,tid),affinity_count=len(os.sched_getaffinity(tid))))
        except (ProcessLookupError,FileNotFoundError):continue
    return dict(pid=pid,start=start,previous_affinity=old,new_affinity=mask,attempted=affected,verified_threads=verified,error=None)
def guard_live():
    jobs=read(ORIG/'freeze.json')['jobs'];p=read(POLICY)
    for j in jobs:
        if (OUT/(j['id']+'-widened.json')).exists():continue
        a=ticks();beg=None
        while True:
            # Never launch a CA from this process. If the original supervisor exits,
            # the independent continuation will handle only never-started inputs.
            if not Path('/proc/2701028').exists():return
            st=read(ORIG/'state.json')
            if (RAW/j['id']/'cost.json').exists() or (ORIG/(j['id']+'-result.json')).exists():break
            time.sleep(p['sample_s']);b=ticks()
            if st.get('case')!=j['id'] or st.get('state')!='acquiring':a=b;continue
            mask,info=capacity(a,b,True);a=b;pid=native(j)
            if pid is None or info['reasons']:beg=None
            elif beg is None:beg=time.monotonic()
            elapsed=0 if beg is None else time.monotonic()-beg
            record=dict(utc=utc(),id=j['id'],pid=pid,continuous_good_s=round(elapsed,1),**info)
            save(OUT/'live-state.json',record)
            if elapsed>=p['continuous_live_s']:
                altered=widen(j,pid,mask)
                if altered:
                    save(OUT/(j['id']+'-widened.json'),{**record,**altered});break
if __name__=='__main__':guard_live()
