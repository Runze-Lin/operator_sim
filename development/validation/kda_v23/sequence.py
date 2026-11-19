"""Complete the registered precision controls, then register independent targets before CA."""
import hashlib
from pathlib import Path
import subprocess
import sys

from series import ROOT, OUT, HERE, read, save,probe

LOCKED='7e278842c78dca05192219b2e8a36512b9e3b4180d277dc2cd1200081b0b61f1'
PINS=ROOT/'results/kda-resource23/freeze.json'


def checked_run(*arguments):
    subprocess.run([sys.executable,str(HERE/'precision_cases.py'),*arguments],check=True)


if __name__=='__main__':
    checked_run('run','development')
    state=read(OUT/'precision-development-state.json')
    if state['state']!='finished':
        save(OUT/'precision-holdout-state.json',dict(state='blocked by corrected-oracle development check',reason=state))
        sys.exit(1)
    if probe.digest(PINS)!=LOCKED:raise ValueError('model team freeze differs')
    freeze=read(PINS)
    for path,sha in freeze['sha256'].items():
        if probe.digest(ROOT/path)!=sha:raise ValueError('frozen resource differs: '+path)
    if not (OUT/'precision-holdout-register.json').exists():
        checked_run('register','holdout',str(PINS),*(str(ROOT/p) for p in freeze['sha256']))
    checked_run('run','holdout')
