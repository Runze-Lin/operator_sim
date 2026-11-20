"""Hand off the already registered Quant pair between idle supervisors, with the same budgets."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
from threading import Event, Thread
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.35/quant'
OLDER=ROOT/'results/v0.29/quant'
RAW=ROOT/'runs/quant29'
SHARED=ROOT/'runs/post25/queue.lock'
FIRST=ROOT/'results/v0.35/post'
CHIPS=('Ascend950PR_957b','Ascend950DT_9582')


def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def put(p,data):
    tmp=p.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n')
    tmp.replace(p)


def state(phase,**kw):
    put(OUT/'queue-state.json',dict(utc=now(),pid=os.getpid(),phase=phase,**kw))


def cpu_set(limit=32):
    def loads():
        result={}
        for x in Path('/proc/stat').read_text().splitlines():
            if re.match(r'cpu\d+ ',x):
                a=x.split()
                result[int(a[0][3:])]=(sum(map(int,a[1:9])),int(a[4])+int(a[5]))
        return result
    x=loads();time.sleep(1);y=loads()
    allowed=os.sched_getaffinity(0);seen=set();selected=[]
    for i in sorted(allowed,key=lambda i:-(y[i][1]-x[i][1])/max(1,y[i][0]-x[i][0])):
        sib=(Path(f'/sys/devices/system/cpu/cpu{i}/topology/thread_siblings_list')).read_text().strip()
        if sib not in seen:
            selected.append(i);seen.add(sib)
        if len(selected)==limit:break
    return selected


def retire_waiter(pid):
    """Only hand off an exact original Python sleep, before any endpoint exists."""
    from os.path import samefile
    owner=Path('/proc')/str(pid)
    handle=os.pidfd_open(pid)
    stopped=False
    try:
        assert samefile(owner/'cwd',ROOT)
        assert re.search(r'^Pid:\s*'+str(pid)+r'$',Path(f'/proc/self/fdinfo/{handle}').read_text(),re.M)
        args=(owner/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        assert args[1:]==[b'-u',b'development/validation/quant29/supervise.py']
        assert read(OLDER/'state.json')['state']=='waiting for 80 idle physical cores and storage'
        assert all(not (RAW/c).exists() and not (OLDER/(c+'.json')).exists() for c in CHIPS)
        signal.pidfd_send_signal(handle,signal.SIGSTOP);stopped=True
        for _ in range(150):
            if re.search(r'^State:\s+T', (owner/'status').read_text(),re.M):break
            time.sleep(.1)
        status=(owner/'status').read_text()
        assert re.search(r'^State:\s+T',status,re.M),re.search(r'^State:.*',status,re.M)[0]
        assert not any((t/'children').read_text().strip() for t in (owner/'task').iterdir())
        assert all(not (RAW/c).exists() for c in CHIPS)
        # Verify the exact same process is the current whole-pair lock owner.
        info=[p.read_text() for p in (owner/'fdinfo').iterdir()]
        lock_inode=(RAW/'lock').stat().st_ino
        assert any(re.search(r'lock:.*FLOCK.*WRITE\s+'+str(pid)+r'\s+\S+:'+str(lock_inode)+r'\s',s) for s in info)
        assert read(OUT/'queue-state.json')['phase']=='waiting for original pair owner'
        successor=read(OUT/'queue-state.json')['pid']
        assert successor!=pid and samefile(Path('/proc')/str(successor)/'cwd',ROOT)
        successor_args=(Path('/proc')/str(successor)/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        assert successor_args[1:]==[b'-u',b'development/validation/quant35/queue.py',b'run']
        signal.pidfd_send_signal(handle,signal.SIGTERM)
        put(OUT/'handoff.json',dict(utc=now(),released_waiter=pid,reason='standby only; no children or raw endpoint',previous_lock_inode=lock_inode))
    finally:
        if stopped:
            try:signal.pidfd_send_signal(handle,signal.SIGCONT)
            except ProcessLookupError:pass
        os.close(handle)


def check_registration():
    reg=read(OLDER/'freeze.json');guard=read(OUT/'freeze.json')
    assert sha(OLDER/'freeze.json')==guard['source_freeze_sha256']
    for name,expected in reg['hashes'].items():
        if sha(ROOT/name)==expected:continue
        assert name=='configs/chips/Ascend950PR_957b.json'
        historical=subprocess.check_output(['git','show',f'97e494f:{name}'],cwd=ROOT)
        assert hashlib.sha256(historical).hexdigest()==expected
        live=read(ROOT/name)
        # v0.30 added a Matmul-only throughput and its citation after Quant froze.
        del live['details']['matmul_initial_read_gbs'],live['sources']['details.matmul_initial_read_gbs']
        assert live==json.loads(historical)
    return reg,guard


def run():
    sys.path.insert(0,str(ROOT/'development/validation/quant29'));sys.path.insert(0,str(ROOT))
    from development.validation.quant29 import collect as old
    from study import digest,store
    from development.validation.quant29.report import publish
    reg,guard=check_registration()
    os.nice(17);os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    with (RAW/'replacement.lock').open('a') as replacement:
        fcntl.flock(replacement,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state('waiting for original pair owner')
        with (RAW/'lock').open('a') as pair,(ROOT/'runs/memory27-followup/remaining.lock').open('a') as mem:
            fcntl.flock(pair,fcntl.LOCK_EX);fcntl.flock(mem,fcntl.LOCK_EX)
            for chip in CHIPS:
                receipt=OLDER/(chip+'.json');folder=RAW/chip
                if receipt.exists():continue
                spec=dict(**reg['official'],active_aiv=reg['predictions']['pair'][chip]['active_aiv'])
                if not (folder/'cost.json').exists():
                    if folder.exists():state('partial directory retained',chip=chip);return
                    state('reserved after short mHC Post pair',chip=chip)
                    while not all((FIRST/('fresh-'+tag+'.json')).exists() for tag in ('dt','pr')):
                        time.sleep(30)
                    with SHARED.open('a') as lane:
                        fcntl.flock(lane,fcntl.LOCK_EX)
                        while True:
                            memory_kib=int(re.search(r'^MemAvailable:\s+(\d+)',Path('/proc/meminfo').read_text(),re.M)[1])
                            if shutil.disk_usage(ROOT).free >= 140*2**30 and memory_kib >= 48*2**20:break
                            state('waiting for host memory and original disk reserve',chip=chip);time.sleep(45)
                        affinity=os.sched_getaffinity(0);cpus=cpu_set()
                        stop=Event();thread=Thread(target=old.preserve,args=(folder,RAW/f'{chip}-controllers',stop),daemon=True)
                        state('acquiring original registration',chip=chip,cpus=cpus,shared=str(SHARED.relative_to(ROOT)))
                        old.status('collecting in shared slot; v0.35 supervisor',sku=chip)
                        thread.start()
                        try:
                            os.sched_setaffinity(0,cpus)
                            old.capture(dict(runner=str(old.RUNTIME.relative_to(ROOT)),args=['quant',*map(str,spec['shape'])],**spec['budget']),chip,folder)
                        finally:
                            stop.set();thread.join();os.sched_setaffinity(0,affinity)
                cost=read(folder/'cost.json')
                result=dict(sku=chip,shape=spec['shape'],qualified=False,instruction_us=None,wrapper_us=None,cost=cost,
                            freeze_sha256=sha(OLDER/'freeze.json'),controller_raw=str((RAW/f'{chip}-controllers').relative_to(ROOT)))
                try:
                    assert cost['started_utc']>guard['utc'] and cost['started_utc']>reg['utc']
                    result.update(old.measured(spec,chip,folder,cost))
                except (AssertionError,OSError,ValueError,TypeError,KeyError,IndexError) as exc:
                    result['reason']=repr(exc)
                store(receipt,result,exclusive=True);publish()
            old.status('finished');state('finished')


if __name__=='__main__':
    if sys.argv[1]=='run':run()
    elif sys.argv[1]=='handoff-idle':retire_waiter(int(sys.argv[2]))
    elif sys.argv[1]=='check':check_registration()
