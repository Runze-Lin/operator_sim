"""Transfer only the idle, registered Kimi lane to the broad host admission."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.matmul29 import decode_recover

wide=decode_recover.wide_host
base=wide.base
OUT=ROOT/'results/v0.31/matmul/kimi'
PROPOSAL=ROOT/'results/v0.30/matmul-dispatch/kimi/host-amendment.json'


def mask(job):
    while True:
        eligible=os.sched_getaffinity(0)
        usage=wide.expand.sample(eligible)
        reserve=sorted(usage,key=usage.get)[:8]
        selected=sorted(eligible-set(reserve))
        ram=int(next(r.split()[1] for r in Path('/proc/meminfo').read_text().splitlines()
                     if r.startswith('MemAvailable:')))/2**20
        disk=shutil.disk_usage(ROOT).free/2**30
        info=dict(chosen_cpus=selected,reserved_quiet=reserve,available_gib=ram,disk_gib=disk,
                  source='v0.30 broad-host admission; no changes to device')
        if len(selected)>=64 and ram>=48 and disk>=32:
            return selected,info
        base.state('waiting for host capacity under revised registration',case=job,host=info)
        time.sleep(30)


def owner(server,session,program,wait_site):
    prefix=['tmux',*server]
    shown=subprocess.check_output([*prefix,'display-message','-p','-t',session,'#{pane_pid} #{pane_id}'],text=True)
    pid,tty=shown.split()
    proc=Path('/proc')/pid
    assert proc.stat().st_uid==os.getuid() and (proc/'cwd').resolve()==ROOT
    cmd=(proc/'cmdline').read_bytes().split(b'\0')
    assert any(Path(os.fsdecode(x)).name==program for x in cmd)
    assert (proc/'wchan').read_text()==wait_site
    assert not any(p.read_text().strip() for p in (proc/'task').glob('*/children'))
    pg,_,_,fg=(proc/'stat').read_text().rsplit(') ',1)[1].split()[2:6]
    assert int(pg)==int(pid)==int(fg)
    return proc,[*prefix,'send-keys','-t',tty,'C-c']


def handoff():
    assert all(base.digest(ROOT/p)==h for p,h in base.read(base.REG)['acquisition_sha256'].items())
    proposal=base.read(PROPOSAL)
    assert base.digest(base.REG)==proposal['reference'] and base.digest(wide.RETRY/'registration.json')==proposal['original_retry_registration']
    assert all(base.digest(ROOT/r['receipt'])==r['sha256'] for r in proposal['actual_broad_mask_controls'])
    decode_recover.main(check=True)
    attempts=[]
    try:
        def idle():
            state=base.read(base.D/'state.json')
            sec=(datetime.now(timezone.utc)-datetime.fromisoformat(state['utc'])).total_seconds()
            return (state['phase']=='waiting for 64 physical CPUs and 8-core headroom' and
                    state['case'].endswith('-broad-retry') and not (base.RAW/state['case']).exists() and
                    0<=sec<12)
        for group,session,program,site in ((['-L','operator-sim-matmul28'],'kimi-qualifier-29','decode_recover.py','locks_lock_inode_wait'),
                                           (['-L','default'], 'matmul29-kimi-broad','wide_host.py','hrtimer_nanosleep')):
            if program=='decode_recover.py' and (OUT/'standby-exit.json').exists():
                prior=base.read(OUT/'standby-exit.json')['host_parents']
                assert len(prior)==1 and prior[0]['host_program']==program
                assert not (Path('/proc')/str(prior[0]['pid'])).exists()
                assert subprocess.run(['tmux',*group,'has-session','-t',session],stderr=subprocess.DEVNULL).returncode==1
                attempts.extend(prior)
                continue
            for _ in range(100):
                if idle(): break
                time.sleep(.5)
            else: raise RuntimeError('No verified fresh host-wait checkpoint')
            proc,cmd=owner(group,session,program,site)
            if program=='wide_host.py':
                assert int(proc.name)==base.read(base.D/'state.json')['pid'] and idle()
            attempts.append(dict(host_program=program,pid=int(proc.name),wait_site=site,raw_started=False))
            subprocess.run(cmd,check=True)
            for _ in range(100):
                if not proc.exists():break
                time.sleep(.1)
            else:raise RuntimeError('Idle supervisor still live; another acquisition is not admitted')
        base.save(OUT/'handoff.json',dict(utc=base.now(),registration=base.digest(base.REG),proposal=base.digest(PROPOSAL),
                    host_parents=attempts,successor_pid=os.getpid(),status='both old waiters exited on their own foreground terminals'))
    except Exception as e:
        base.save(OUT/'handoff.json',dict(utc=base.now(),host_parents=attempts,applied=False,error=repr(e)))
        raise
    os.nice(4)
    wide.await_mask=mask
    wide.run()


if __name__=='__main__':
    handoff()
