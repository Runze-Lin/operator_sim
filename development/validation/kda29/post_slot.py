"""Move the final frozen Post case between exclusively locked host supervisors, only while parked."""
from datetime import datetime,timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.post28 import channel_cases as original

OUT=ROOT/'results/v0.29/post-slot'
POST=original.DST
RAW=original.RAW
TARGET='23x1984-Ascend950PR_957b'
EXPECTED_PID=3463104
FROZEN='30e06f6f9d5c05b7bfc31ee5f3f331f552715af50184342237d6de52fcdf49e8'
EXPECTED_STATE='queued for spare physical CPUs'
DT=POST/'receipts/23x1984-Ascend950DT_9582.json'


def read(p):return json.loads(p.read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return sha256(p.read_bytes()).hexdigest()
def save(p,x):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2)+'\n');temp.replace(p)


def stats():
    r={}
    for line in Path('/proc/stat').read_text().splitlines():
        if line.startswith('cpu') and line[3:4].isdigit():
            s,*v=line.split();r[int(s[3:])]=(sum(map(int,v[:8])),int(v[3])+int(v[4]))
    return r


def observed_workers(cpus, excluded):
    rs={}
    for p in Path('/proc').glob('[0-9]*'):
        try:
            x=p/'stat';v=x.read_text().rsplit(')',1)[1].split(); pid=int(p.name)
            if pid in excluded:continue
            exe=(p/'exe').resolve(strict=True);allowed=os.sched_getaffinity(pid)
            tasks=[]
            for th in (p/'task').iterdir():
                t=(th/'stat').read_text().rsplit(')',1)[1].split()
                tasks.append((int(th.name),int(t[11])+int(t[12]),int(t[36])))
            rs[(pid,int(v[19]))]=(exe.name,allowed,tasks)
        except (OSError,ValueError):continue
    return rs


def cpus():
    all_cpus=os.sched_getaffinity(0);skip={os.getpid(),EXPECTED_PID}
    topo={c:(Path('/sys/devices/system/cpu')/f'cpu{c}/topology/thread_siblings_list').read_text().strip() for c in all_cpus}
    busy={c:[] for c in all_cpus}
    pinned=set();foreign_spots=set();pinned_jobs={};a=stats();old=observed_workers(all_cpus,skip)
    for _ in range(3):
        time.sleep(1);b=stats();new=observed_workers(all_cpus,skip)
        for c in all_cpus:
            busy[c].append(1-(b[c][1]-a[c][1])/max(1,b[c][0]-a[c][0]))
        for key,(exe,mask,threads) in new.items():
            if key not in old:continue
            previous={t[0]:t for t in old[key][2]}
            active=[(s,previous[s[0]]) for s in threads if s[0] in previous and s[1]-previous[s[0]][1]>=4]
            if not active:continue
            if set(mask)<all_cpus or not set(mask).issuperset(all_cpus):
                pinned.update(topo[c] for c in mask&all_cpus)
                pinned_jobs[key[0]]=dict(program=exe,mask_size=len(mask),busy_threads=len(active))
            # Also avoid current locations of materially busy unpinned work, but
            # exclude only hot cores: the OS may migrate a task across many
            # mostly-idle CPUs between snapshots.
            for nt,ot in active:
                if nt[1]-ot[1]>=50:
                    foreign_spots.update(topo[c] for c in (nt[2],ot[2]) if c in all_cpus)
        old=new;a=b
    ranked=sorted(all_cpus,key=lambda c:(max(busy[c]),sum(busy[c]),c))
    choices=[];seen=set();idle_all=set()
    for c in ranked:
        if max(busy[c])>.35:continue
        idle_all.add(topo[c])
        if topo[c] in seen|pinned|foreign_spots:continue
        seen.add(topo[c]);choices.append(c)
    # Same-shape DT consumed 4,248 CPU-seconds in 152 s while using 58–80
    # physical cores. Admit 56 or more real idle cores AND retain headroom.
    n=min(80,len(choices), max(0,len(idle_all)-24))
    return dict(utc=now(),eligible=n>=56,selection=choices[:n],quiet_physical=len(idle_all),
                quiet_unreserved_physical=len(choices),busy_fixed_mask_physical=len(pinned),
                protected_heavy_core_locations=len(foreign_spots),reserved_for_other=len(idle_all)-n,
                foreign_bound_active_processes=pinned_jobs)


def original_owns_lock(pid):
    lock=RAW/'serial.lock';inode=lock.stat().st_ino
    return any(line.split()[1:4]==['FLOCK','ADVISORY','WRITE'] and line.split()[4]==str(pid)
               and line.split()[5].split(':')[-1]==str(inode) for line in Path('/proc/locks').read_text().splitlines())


def identity(pid):
    p=Path('/proc')/str(pid);cmd=(p/'cmdline').read_bytes().split(b'\0')
    s=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=int(s[19]),state=s[0],is_user=p.stat().st_uid==os.getuid(),
                cwd=str((p/'cwd').resolve(strict=True)),cmd=[x.decode() for x in cmd if x],
                children=[c for t in (p/'task').iterdir() for c in (t/'children').read_text().split()])


def parked():
    state=read(POST/'state.json')
    return (state.get('pid')==EXPECTED_PID and state.get('case')==TARGET and state.get('phase')==EXPECTED_STATE
            and not (RAW/TARGET).exists() and not (POST/'receipts'/(TARGET+'.json')).exists()
            and len(list((POST/'receipts').glob('*.json')))==5)


def run():
    reg=read(original.REG);assert sha(original.REG)==FROZEN
    assert all(sha(ROOT/p)==h for p,h in reg['hashes'].items())
    j,=[j for j in reg['jobs'] if j['id']==TARGET]
    prior=read(DT);assert prior['cost']['seconds']<155 and prior['cost']['cpu_seconds']>4200
    with (OUT/'guard.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            if not parked():
                state=read(POST/'state.json')
                if state.get('phase') in ('running','complete') and (state.get('case')==TARGET or state.get('phase')=='complete'):
                    save(OUT/'handoff.json',dict(utc=now(),phase='original supervisor retained slot',original_state=state));return
                time.sleep(10);continue
            head=cpus();save(OUT/'spare.json',head)
            if not head['eligible']:
                time.sleep(25);continue
            before=identity(EXPECTED_PID)
            assert before['is_user'] and before['cwd']==str(ROOT) and before['cmd'][-2:]==[str(Path(original.__file__).resolve()),'run']
            if before['children'] or not parked() or not original_owns_lock(EXPECTED_PID):
                time.sleep(5);continue
            os.kill(EXPECTED_PID,signal.SIGSTOP)
            try:
                for _ in range(70):
                    if identity(EXPECTED_PID)['state'] in ('T','t'):break
                    time.sleep(.04)
                check=identity(EXPECTED_PID)
                if not (check['start_ticks']==before['start_ticks'] and check['state'] in ('T','t')
                        and not check['children'] and parked() and original_owns_lock(EXPECTED_PID)):
                    time.sleep(.05);continue
                os.kill(EXPECTED_PID,signal.SIGTERM)
            finally:
                try:os.kill(EXPECTED_PID,signal.SIGCONT)
                except ProcessLookupError:pass
            break
        with (RAW/'serial.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not parked():raise RuntimeError('slot state changed at lock handoff')
            assert not original_owns_lock(EXPECTED_PID)
            accepted=dict(utc=now(),phase='exclusive lock acquired for last registered case',prior=before,host=head,
                          original_same_shape_dt=prior['cost'],freeze_sha256=FROZEN,
                          input_files_sha256={str(p.relative_to(ROOT)):sha(p) for p in (original.REG,DT)})
            save(OUT/'handoff.json',accepted)
            os.nice(14);os.sched_setaffinity(0,head['selection'])
            original.save(POST/'state.json',dict(utc=now(),phase='running',pid=os.getpid(),case=TARGET,cpus=head['selection'],
                                                 host=head,scheduler_appendix='results/v0.29/post-slot/handoff.json'))
            folder=RAW/TARGET
            res=dict(chip=j['sku'],cost=original.capture(j['sku'],folder,j['rows'],dict(wall_s=reg['seconds_cap_each'],mib=reg['miB_cap_each'])),
                     raw=str(folder.relative_to(ROOT)),scope='v0.24 original per-lane END and END_LABEL inclusive',freeze_sha256=FROZEN)
            try:res['rows']=original.data(res,j['rows'])
            except (AssertionError,ValueError,OSError,KeyError,IndexError) as exc:res['reason']=repr(exc)
            original.save(POST/'receipts'/(TARGET+'.json'),res)
            final=original.score(); assert final['complete']
            original.save(POST/'state.json',dict(utc=now(),phase='complete',pid=os.getpid(),passed_gate=final['qualifies_to_enable_on_ordered_channel_repeat']))
            save(OUT/'handoff.json',{**accepted,'phase':'finished single final case','finished_utc':now(),'receipt_sha256':sha(POST/'receipts'/(TARGET+'.json'))})


if __name__=='__main__':
    if sys.argv[1:] == ['snapshot']:
        result=cpus();save(OUT/'spare.json',result)
        print({k:v for k,v in result.items() if k not in ('selection','foreign_bound_active_processes')})
    elif sys.argv[1:] == ['start']:
        with (OUT/'worker.log').open('a') as log:
            p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run'],cwd=ROOT,stdin=subprocess.DEVNULL,
                               stdout=log,stderr=log,start_new_session=True);print(p.pid,flush=True)
    elif sys.argv[1:] == ['run']:
        run()
