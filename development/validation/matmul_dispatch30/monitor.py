"""Refresh the Flash case study once per completed receipt; perform no simulation."""
import argparse
import fcntl
import time
import sys
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.matmul_dispatch30.qualify import qualify_ready
from development.validation.matmul_dispatch30.score import OUT,report as device_report
from development.validation.matmul_dispatch30.report import render


def main(watch):
    previous=None
    while True:
        files=[p for p in OUT.glob('*.json') if re.fullmatch(r'\d+-(dt|pr)\.json',p.name)]
        current=tuple(sorted((p.name,p.stat().st_size) for p in files))
        if current!=previous:
            qualify_ready();device_report();render();previous=current
        if not watch or len(files)>=16:return
        time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--watch',action='store_true')
    with (ROOT/'runs/matmul_dispatch30/device/report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main(parser.parse_args().watch)
