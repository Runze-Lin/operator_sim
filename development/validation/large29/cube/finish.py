"""Publish receipts as each original endpoint finishes and resume only from a saved cost."""
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from campaign import OUT,RAW,ROOT,read,run,save,stamp
from control import bind
from report import report


def finish():
    with (RAW/'reporter.lock').open('a') as single:
        fcntl.flock(single,fcntl.LOCK_EX|fcntl.LOCK_NB)
        bind();previous=None
        while True:
            state=read(OUT/'state.json')
            key=(state['utc'],sum(p.stat().st_mtime_ns for p in OUT.glob('*-[dp][tr].json')))
            if key!=previous:report();previous=key
            if state['phase']=='finished':return
            with (RAW/'owner.lock').open('a') as owner:
                try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:held=True
                else:held=False;fcntl.flock(owner,fcntl.LOCK_UN)
            if not held:
                # The ordinary large job has a durable cost before its qualifier
                # starts. Never restart an interrupted invocation without such a cost.
                unsafe=[p for p in RAW.glob('*/run.log') if not (p.parent/'cost.json').is_file()]
                if unsafe:
                    save(OUT/'resumption.json',dict(utc=stamp(),status='interrupted acquisition preserved',folders=[str(p.parent) for p in unsafe]));return
                save(OUT/'resumption.json',dict(utc=stamp(),status='resuming durable receipts; each shape runs at most once',pid=os.getpid()))
                run();report();return
            time.sleep(45)


if __name__=='__main__':
    if len(sys.argv)==2 and sys.argv[1]=='--detach':
        host=read(OUT/'host.json')
        with (RAW/'reporter.log').open('a') as f:
            w=subprocess.Popen([str(ROOT/'toolchains/envs/cann-9.2.1/bin/python'),str(Path(__file__).resolve())],
                               start_new_session=True,stdin=subprocess.DEVNULL,stdout=f,stderr=f,
                               preexec_fn=lambda:os.sched_setaffinity(0,set(host['logical_cpu_ids'])))
        print(w.pid)
    else:finish()
