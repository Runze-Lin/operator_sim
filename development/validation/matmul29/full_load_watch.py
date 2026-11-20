"""Detach a receipt-only scorer on the job host. It never invokes a simulator."""
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[3]
with (root/'results/v0.29/matmul/full-load/watch.log').open('a') as file:
    worker=subprocess.Popen([sys.executable,str(Path(__file__).with_name('full_load_score.py')),'--watch'],
                            cwd=root,start_new_session=True,stdin=subprocess.DEVNULL,stdout=file,stderr=file)
    print(worker.pid)
