"""Refresh the shadow report from existing independent processes without running a simulator."""
from pathlib import Path
import subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.post26.freeze import OUT as V26
from development.validation.sku19.campaign import read
def refresh():
    subprocess.run([sys.executable,str(Path(__file__).with_name('report.py'))],cwd=ROOT,check=True)
def run():
    last=None
    while True:
        current=(V26/'score.json').stat().st_mtime_ns
        if current!=last:refresh();last=current
        if read(V26/'status.json').get('state')=='complete':
            time.sleep(35);refresh();return
        time.sleep(22)
if __name__=='__main__':run()
