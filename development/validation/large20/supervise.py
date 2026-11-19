"""Reattach the original CA timers, then finish the registered queues."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import signal
import time
import traceback
from large_cast import DT, PR, OUT, RAW, ROOT, qualify_one, read, write
from native import NATIVE,REG,qualify_vector
from pro import RECORD as PRO_REG, collect as pro_collect
from recover_cost import recover_cost,timer


def original(job,chip):
    out=OUT/(job['id']+'_'+chip+'.json');d=RAW/(job['id']+'_'+chip)
    if out.exists():return
    if (d/'run.log').exists() and not (d/'cost.json').exists():
        cost=recover_cost(d,job['wall_s'],job['raw_mib']*2**20)
        write(d/'cost.json',cost)
    qualify_one(job,chip)


def native():
    spec=read(REG)['jobs'][0];t,h=spec['shape'];d=NATIVE/f'cast_{t}_{h}'
    out=OUT/(d.name+'_native.json')
    if out.exists():return
    cost=recover_cost(d,spec['wall_s'],spec['raw_cap_gib']*2**30)
    result=qualify_vector(spec,d,cost,[t*h]);write(out,result)


def official():
    reg=read(OUT/'registration.json')
    def lane(c):
        for j in reg['jobs']:
            d=RAW/(j['id']+'_'+c)
            if (d/'paused-by-supervisor').exists():
                pid=timer(d)
                if pid:os.killpg(pid,signal.SIGCONT)
                (d/'paused-by-supervisor').unlink()
            original(j,c)
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(lane,c) for c in (DT,PR)]
        for task in tasks:task.result()


def pro():
    j=read(PRO_REG)['job']
    for c in (DT,PR):
        if (RAW/(j['id']+'_'+c)/'run.log').exists():original(j,c)
    pro_collect()


def serve():
    os.environ.update(OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0')
    with (RAW/'supervise.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (RAW/'supervisor.pid').write_text(str(os.getpid()))
        def guarded(fn):
            try:fn()
            except Exception:
                traceback.print_exc()
                raise
        with ThreadPoolExecutor(max_workers=3) as pool:
            tasks=[pool.submit(guarded,fn) for fn in (native,official,pro)]
            for t in tasks:t.result()


def start():
    with (RAW/'detached.log').open('a') as out:
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'serve'],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    print('supervisor PID',p.pid)


def handoff(pid):
    # Stop the old supervisor, leaving each independent CA child session alive.
    proc=Path('/proc')/str(pid)
    cmd=(proc/'cmdline').read_bytes().split(b'\0')
    if proc.stat().st_uid!=os.getuid() or not any(a.endswith(b'large20/supervise.py') for a in cmd) or b'serve' not in cmd:
        raise ValueError('unrecognized supervisor process')
    os.kill(pid,signal.SIGTERM)
    for _ in range(50):
        if not proc.exists():break
        time.sleep(.1)
    reg=read(OUT/'registration.json')
    for c in (DT,PR):
        running=[]
        for j in reg['jobs']:
            d=RAW/(j['id']+'_'+c)
            if not d.exists() or (OUT/(d.name+'.json')).exists():continue
            active=timer(d)
            if active:running.append((d,active))
        # Two original timers completed registration while the old supervisor
        # was being recovered. Resume later timers in their registered order.
        for d,active in running[1:]:
            (d/'paused-by-supervisor').touch()
            os.killpg(active,signal.SIGSTOP)
    start()


if __name__=='__main__':
    if sys.argv[1]=='handoff':handoff(int(sys.argv[2]))
    else:{'serve':serve,'start':start}[sys.argv[1]]()
