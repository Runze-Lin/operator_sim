"""Refresh the public API audit after each finished Post process on either queue."""
import subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from development.validation.post26.freeze import OUT,OLD
from development.validation.sku19.campaign import read
def main():
    last=None
    while True:
        k=tuple(sorted(str(p) for folder in (OUT,OLD) for p in (folder/'receipts').glob('*.json')))
        if k!=last:
            check=subprocess.run([sys.executable,str(Path(__file__).with_name('score.py'))],cwd=ROOT)
            if check.returncode:raise SystemExit(check.returncode)
            last=k
        if (OUT/'status.json').exists() and read(OUT/'status.json').get('state')=='complete':break
        time.sleep(25)
if __name__=='__main__':main()
