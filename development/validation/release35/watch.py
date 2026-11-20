"""Update this evidence release on terminal receipts, without starting CA simulations."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.35'
OLDER=ROOT/'results/v0.29/quant'
WHEEL=Path(sys.argv[1]).resolve()
RAWDIR=ROOT/'runs/release35'
FINAL=[OUT/f'post/fresh-{tag}.json' for tag in ('dt','pr')]
FINAL.extend(OUT/f'gating/{tag}.json' for tag in ('dt','pr'))
QUANT=[OLDER/(chip+'.json') for chip in ('Ascend950DT_9582','Ascend950PR_957b')]
FINAL.extend(QUANT)
INPUTS=[*FINAL,OUT/'gating/score.json',ROOT/'results/v0.34/replay.json']


def fingerprint(paths):
    return tuple(p.stat().st_mtime_ns if p.exists() else None for p in paths)


def run(script):
    subprocess.run([sys.executable,'-I',str(ROOT/'development/validation'/script),str(WHEEL)],
                   cwd='/tmp',env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},check=True)


def main():
    RAWDIR.mkdir(parents=True,exist_ok=True)
    with (RAWDIR/'observer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Score once on launch and once per changed source.
        last=qprevious=None
        while True:
            now=fingerprint(INPUTS); qr=fingerprint(QUANT)
            if qr!=qprevious and any(qr):
                run('quant35/score.py')
            if now!=last:
                run('release35/score.py')
            last,qprevious=now,qr
            if all(p.exists() for p in FINAL):
                return
            time.sleep(60)


if __name__=='__main__':main()
