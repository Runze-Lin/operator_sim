"""Separate live-placement handoff and the one approved DT retry; original receipts remain immutable."""
import datetime
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import campaign as base
import expand

HERE=Path(__file__).resolve().parent
D=base.D
LOG=D/'broad-host.json'
RETRY=D/'retry'
PREV='matmul29-kimi-spare'

def put(event):
    doc=base.read(LOG) if LOG.exists() else dict(registration_sha256=base.digest(base.REG),events=[])
    doc['events'].append(dict(utc=base.now(),**event))
    base.save(LOG,doc)

def broad(allowed):
    util=expand.sample(allowed)
    cpus=sorted(util,key=util.get)
    reserve=cpus[:8]
    # For admission require 64 physical cores each below 55% utilization in all
    # three snapshots, plus 8 protected cores. No time is charged while waiting.
    usable=[c for c in cpus if c not in reserve and util[c]<=.55]
    return usable[:80], dict(chosen_cpus=usable[:80],reserved_quiet=reserve,below55=len(usable),
        available_gib=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))/2**20,
        disk_gib=base.shutil.disk_usage(D).free/2**30)

def live():
    """One live PR change. No new CA. Affinity is permission to run, never a reservation."""
    state=base.read(D/'state.json');label=base.slug(9,base.PR)
    if state['phase']!='acquiring' or state.get('case')!=label:return
    folder=base.RAW/label
    p=expand.process(folder,[9,2304,1536])
    if p is None:return
    pid=int(p.name);before=os.sched_getaffinity(pid);util=expand.sample(os.sched_getaffinity(0))
    spare=sorted((c for c in util if c not in before),key=util.get)
    # Highest-idle 8 remain protected. Lower priority allows existing large
    # tasks to win contention even if their CPU becomes newly busy.
    reserve=spare[:8];candidates=[c for c in spare[8:] if util[c]<1]
    target=before|set(candidates[:max(0,80-len(before))])
    if len(target)<64:return
    if expand.process(folder,[9,2304,1536])!=p:return
    prior_nice=os.getpriority(os.PRIO_PROCESS,pid);updated=0
    for th in (p/'task').glob('[0-9]*'):
        try:
            os.setpriority(os.PRIO_PROCESS,int(th.name),19)
            os.sched_setaffinity(int(th.name),target);updated+=1
        except ProcessLookupError:pass
    put(dict(action='live PR diagnostic widening, no admission of another CA',case=label,pid=pid,age_s=time.time()-datetime.datetime.fromisoformat(state['utc']).timestamp(),old=sorted(before),new=sorted(target),protected=reserve,
             updated_threads=updated,nice_before=prior_nice,nice_after=19,per_allowed_idle_equivalents=sum(1-util[c]for c in target)))
    return True

def handoff():
    """Receipt is written before original worker's report and next 3s CPU sampling.
    Signal only the owning tmux pane immediately after receipt is durable, with
    no child and no following raw case. The profiling watchdog stays in charge.
    """
    pr=base.slug(9,base.PR);next_case=base.slug(16,base.DT);activated=False
    while True:
        st=base.read(D/'state.json')
        if st['phase']=='finished':return
        old=int(st['pid']);proc=Path('/proc')/str(old)
        if not proc.exists():return
        if not activated and st.get('case')==pr and st['phase']=='acquiring':
            activated=live()
        written=D/(pr+'.json')
        recent=written.exists() and 0<=time.time()-written.stat().st_mtime<.9 and st['phase'].startswith('verifying') and st.get('case')==pr
        idle=st['phase']=='waiting for physical CPU, RAM or storage' and time.time()-datetime.datetime.fromisoformat(st['utc']).timestamp()<3
        if recent or idle:
            pane=int(subprocess.check_output(['tmux','display-message','-p','-t',PREV,'#{pane_pid}'],text=True))
            cmd=(proc/'cmdline').read_bytes()
            children=(proc/'task'/str(old)/'children').read_text().strip()
            if pane==old and b'spare.base.host=spare.spare' in cmd and (proc/'cwd').resolve()==base.ROOT and not children and not (base.RAW/next_case).exists():
                subprocess.run(['tmux','send-keys','-t',PREV,'C-c'],check=True)
                for _ in range(100):
                    if not proc.exists():break
                    time.sleep(.05)
                else:raise RuntimeError('worker remains live; no take-over')
                put(dict(action='completed PR reaped; old scheduler retired before next host/routine, same flock',prior_pid=old,phase=st['phase'],children=children,pending_raw=False))
                break
        time.sleep(.12 if activated or written.exists() else 2)

def await_mask(job):
    allowed=os.sched_getaffinity(0)
    while True:
        cpus,detail=broad(allowed)
        if len(cpus)>=64 and detail['available_gib']>=48 and detail['disk_gib']>32:return cpus,detail
        base.state('waiting for 64 physical CPUs and 8-core headroom',case=job,host=detail)
        time.sleep(27)

def retry_publish(job,folder):
    cost=base.read(folder/'cost.json');label=base.slug(9,base.DT)
    try:
        if cost['exit_code'] or cost['stop_reason'] or cost['peak_bytes']>job['raw_mib']*2**20:raise ValueError('original wall/storage cap')
        data=base.ca.qualify(job,base.DT,folder)
    except (ValueError,AssertionError,OSError,KeyError,IndexError,StopIteration)as e:
        data=dict(qualified=False,ca_us=None,reason=repr(e))
    capture=dict(**data,case=label,chip=base.DT,shape=job['shape'],cost=cost)
    receipt=RETRY/'initial.json';base.save(receipt,capture)
    if not capture['qualified'] and not cost['stop_reason'] and not cost['exit_code']:
        try:extra=base.recover(job,capture,folder,acquisition=receipt)
        except (ValueError,AssertionError,OSError,KeyError,IndexError,StopIteration)as e:
            extra=None;capture['reason']+=repr(e)
        if extra:capture.update(extra,reason=None)
    if capture['qualified']:
        s,=capture['stages']
        assert s['binary_sha256']==base.read(base.REG)['provider_sha256'] and capture['correctness']==[f"correctness count={9*2304} bad=0"]
    capture.update(role='second acquisition after frozen M9 DT timeout, same shape/prior prices; separate attempt not a third independent geometry',
                   original_receipt_sha256=base.digest(D/(label+'.json')),registration_sha256=base.digest(base.REG),initial_sha256=base.digest(receipt))
    base.save(RETRY/'receipt.json',capture)

def run():
    reg=base.read(base.REG);allowed=os.sched_getaffinity(0)
    for path,digest in reg['acquisition_sha256'].items():assert base.digest(base.ROOT/path)==digest
    os.nice(15);os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    with (base.RAW/'serial.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        targets=[j for j in reg['targets'] if j['m']in reg['rows_to_capture']]
        jobs=[(targets[0],base.DT,True)]+[(j,sku,False)for j in targets for sku in base.MEASURED]
        for j,sku,retry in jobs:
            label=base.slug(j['m'],sku);folder=base.RAW/(label+'-broad-retry' if retry else label)
            dest=RETRY/'receipt.json' if retry else D/(label+'.json')
            if dest.exists():continue
            if folder.exists() and not (folder/'cost.json').exists():
                base.state('incomplete raw retained; no repeat',case=folder.name);return
            if not folder.exists():
                mask,detail=await_mask(folder.name)
                try:
                    os.sched_setaffinity(0,mask);folder.mkdir()
                    base.state('acquiring',case=folder.name,host=detail)
                    base.ca.measure(j,sku,folder)
                finally:os.sched_setaffinity(0,allowed)
            base.state('verifying full numerical output and terminal windows',case=folder.name)
            if retry:retry_publish(j,folder)
            else:base.publish(j,sku,folder)
        base.report();base.state('finished')
if __name__=='__main__':
    if sys.argv[1]=='handoff':handoff()
    run()
