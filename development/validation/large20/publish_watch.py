"""Keep reports and controller archives current after an interactive turn ends."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from large_cast import RAW, ROOT


def start():
    with (RAW/'report-watch.log').open('a') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'serve'],stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,close_fds=True,start_new_session=True,cwd=ROOT)
    print('report / counters supervisor PID',child.pid)


def serve():
    script=Path(__file__).resolve().parent
    # Reserve the CPU cores for CA. Neither chart rendering nor log parsing
    # requires BLAS worker threads.
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/operator-sim-mpl'}
    with (RAW/'report-watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=[[sys.executable,str(script/'dram.py')],[sys.executable,str(script/'analysis.py'),'--watch']]
        procs=[subprocess.Popen(j,cwd=ROOT,env=env) for j in jobs]
        failures=[p.wait() for p in procs]
        if any(failures):raise SystemExit(str(failures))


if __name__=='__main__':{'start':start,'serve':serve}[sys.argv[1]]()
