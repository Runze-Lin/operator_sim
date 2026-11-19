"""Refresh compact evidence and model comparisons after any new case completes."""
import os
import subprocess
import sys
import time

from series import OUT, ROOT, HERE, read
from mechanisms import update

if __name__=='__main__':
    previous=None
    while True:
        names=[tuple(sorted(p.name for p in OUT.glob(prefix+'-ascend950*.json'))) for prefix in ('holdout','development','supplement')]
        if names!=previous:
            subprocess.run([str(ROOT/'toolchains/envs/cann-9.2.1/bin/python'),str(HERE/'check_arrays.py')],check=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='1'))
            update()
            previous=names
        terminal=OUT/'precision-supplement-state.json'
        if terminal.exists() and read(terminal)['state']!='running':break
        time.sleep(15)
