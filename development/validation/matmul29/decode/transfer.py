"""Interrupt only the original worker during its known 27-second idle wait."""
import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

here=Path(__file__).resolve().parent
root=here.parents[3]; dest=root/'results/v0.29/matmul/decode';raw=dest/'raw'
reg=dest/'freeze.json'
worker=here/'campaign.py'
read=lambda p: json.loads(p.read_text())
name='matmul29-kimi-decode'
while True:
    s=read(dest/'state.json');pid=s['pid'];proc=Path('/proc')/str(pid)
    if s['phase']=='finished':sys.exit()
    if not proc.exists():break
    pane=int(subprocess.check_output(['tmux','display-message','-p','-t',name,'#{pane_pid}'],text=True))
    cmd=(proc/'cmdline').read_bytes().split(b'\0')
    assert pid==pane and proc.stat().st_uid==os.getuid() and (proc/'cwd').resolve()==root
    assert any((root/os.fsdecode(arg)).resolve()==worker for arg in cmd if arg) and b'run' in cmd
    elapsed=time.time()-datetime.datetime.fromisoformat(s['utc']).timestamp()
    # That exact state is published immediately before sleep(27); after a fresh
    # (<4 s) marker it cannot enter capture for over 23 seconds. Confirm the
    # kernel is actually sleeping and no host or raw case exists.
    w=(proc/'wchan').read_text().strip();children=(proc/'task'/str(pid)/'children').read_text().strip()
    if s['phase']=='waiting for physical CPU, RAM or storage' and 0<=elapsed<4 and 'sleep' in w and not children and not (raw/s['case']).exists():
        msg=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_pid=pid,new_pid=os.getpid(),
                 registration_sha256=sha256(reg.read_bytes()).hexdigest(),previous_state=s,kernel_wchan=w,
                 age_of_start_of_27_second_idle_wait_s=elapsed,child_processes=children,raw_case_exists=False,
                 action='Exact tmux pane Ctrl-C during idle scheduler sleep; no CA host present. New worker acquires the original single-slot flock.')
        subprocess.run(['tmux','send-keys','-t',name,'C-c'],check=True)
        for _ in range(100):
            if not proc.exists():break
            time.sleep(.08)
        else:raise RuntimeError('original worker did not exit; no duplicate started')
        (dest/'host-handoff.json').write_text(json.dumps(msg,indent=2)+'\n')
        break
    time.sleep(.4)
env=root/'toolchains/envs/cann-9.2.1/bin/python'
code="import sys;sys.path.insert(0,sys.argv[1]);import spare;spare.plan();spare.base.host=spare.spare;spare.base.run()"
os.execv(str(env),[str(env),'-c',code,str(here)])
