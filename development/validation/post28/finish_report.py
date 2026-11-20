"""Refresh the Post report when a frozen run writes a terminal receipt."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/v0.28/post'
REPORT = Path(__file__).with_name('report.py')


def watch():
    seen = None
    while True:
        score = json.loads((OUT/'channel-score.json').read_text())
        if seen != score['utc']:
            subprocess.run([sys.executable,str(REPORT)],cwd=ROOT,check=True)
            seen = score['utc']
        if score['complete']:
            break
        time.sleep(20)


if __name__ == '__main__':
    if 'start' in sys.argv[1:]:
        with (ROOT/'runs/post28/report-updates.log').open('a') as out:
            p = subprocess.Popen([sys.executable,str(Path(__file__).resolve())],cwd=ROOT,
                                 stdin=subprocess.DEVNULL,stdout=out,stderr=out,start_new_session=True)
            print(p.pid,flush=True)
    else:
        watch()
