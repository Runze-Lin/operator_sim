"""Handoff only the identified sleeping supervisor, using a PID fd and its existing lock."""
import json
import os
from pathlib import Path
from select import poll,POLLIN
import signal
import subprocess
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/v0.26/cache-queue';CACHE=ROOT/'results/v0.25/cache';RAW=CACHE/'raw'
CANN=ROOT/'toolchains/envs/cann-9.2.1/bin/python'
OLD=ROOT/'development/validation/cache25/campaign.py'
EXPECTED=2528633
START=126334566

def now():return datetime.now(timezone.utc).isoformat()
def write(name,obj):
    p=OUT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(p)
def inspect(pid):
    p=Path('/proc')/str(pid)
    argv=[os.fsdecode(s) for s in (p/'cmdline').read_bytes().split(b'\0') if s]
    checks={}
    checks['command']=argv==[str(CANN),str(OLD),'run']
    checks['same_start_time']=int((p/'stat').read_text().split(') ')[1].split()[19])==START
    checks['sleeping_in_nanosleep']='nanosleep' in (p/'wchan').read_text()
    checks['no_children']=all(not f.read_text().strip() for f in (p/'task').glob('*/children'))
    lock=RAW/'queue.lock'
    fds=[f for f in (p/'fd').iterdir() if f.readlink()==lock]
    checks['original_exclusive_lock']=len(fds)==1 and ('WRITE '+str(pid)+' ') in (p/'fdinfo'/fds[0].name).read_text()
    checks['entire_queue_unstarted']=set(f.name for f in RAW.iterdir())=={'queue.lock'} and not list(CACHE.glob('*-result.json'))
    status=json.loads((CACHE/'state.json').read_text())
    checks['admission_wait']=status['state']=='awaiting existing large queues'
    pro=json.loads((ROOT/'results/v0.23/pro-prefill/state.json').read_text())
    full=json.loads((ROOT/'results/v0.23/full-hidden/v3/queue-status.json').read_text())
    # Each of these prevents the old predicate from releasing the first job.
    checks['both_original_prerequisites_pending']=pro['state']!='finished' and full['state']!='complete'
    return dict(utc=now(),pid=pid,original_argv=argv,checks=checks,cache_state=status['state'],pro=pro['state'],full_hidden=full['state'])

def main():
    env={**os.environ,'MPLCONFIGDIR':'/tmp/mpl-cache26'}
    subprocess.run([str(CANN),str(HERE/'guard.py'),'verify'],check=True,cwd=ROOT,env=env)
    fd=os.pidfd_open(EXPECTED)
    try:
        audit=inspect(EXPECTED);write('handover-preflight.json',audit)
        if sys.argv[1:]!=['handover'] or not all(audit['checks'].values()):
            if sys.argv[1:]==['inspect']:print(json.dumps(audit),flush=True);return
            raise RuntimeError('Existing supervisor preserved; guarded identity, lock, and pending conditions not all true')
        second=inspect(EXPECTED)
        if not all(second['checks'].values()):raise RuntimeError('Supervisor state changed; keep original')
        # The original supervisor is inside time.sleep, holds the intended lock,
        # has no running CA child, and both independent admission predicates fail.
        signal.pidfd_send_signal(fd,signal.SIGTERM)
        done=poll();done.register(fd,POLLIN)
        if not done.poll(5000):raise RuntimeError('Original supervisor did not exit; no replacement launch')
    finally:os.close(fd)
    if set(f.name for f in RAW.iterdir())!={'queue.lock'}:raise RuntimeError('Unexpected raw job; stop')
    write('handover.json',dict(utc=now(),old_pid=EXPECTED,old_proc_start=START,old_termination='SIGTERM to verified sleeping pidfd only',
                               new_pid=os.getpid(),old_acquisitions=0,shared_lock=str((RAW/'queue.lock').relative_to(ROOT)),
                               replacement='original cache25.run with an admission callback'))
    os.execve(CANN,[str(CANN),str(HERE/'guard.py'),'run'],env)
if __name__=='__main__':main()
