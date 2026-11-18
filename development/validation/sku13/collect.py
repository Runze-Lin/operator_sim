"""Bounded paired-SKU CA collection with a reused, byte-identical DT host binary."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import hashlib

OUT = ROOT / 'results/v0.13/sku'
holdout = sys.argv[1:] == ['--holdout']
if sys.argv[1:] and not holdout:
    raise SystemExit('Usage: collect.py [--holdout]')
budget = json.loads((OUT / ('holdout-budget.json' if holdout else 'budget.json')).read_text())
if holdout:
    freeze = json.loads((OUT/'holdout-freeze.json').read_text())
    for key, value in freeze['sha256'].items():
        if hashlib.sha256((ROOT/key).read_bytes()).hexdigest() != value:
            raise ValueError('Frozen input changed: ' + key)
for case in budget['cases'][:budget['max_attempts']]:
    binary = ROOT / budget['runner']
    if hashlib.sha256(binary.read_bytes()).hexdigest() != case['host_binary_sha256']:
        raise RuntimeError('Host runner identity mismatch')
    directory = ROOT / ('runs/sku13-holdout' if holdout else 'runs/sku13-profile') / case['op']
    directory.mkdir(parents=True, exist_ok=False)
    cmd = ['bash', str(ROOT/'scripts/cann-run.sh'), 'msopprof', 'simulator',
           '--soc-version=' + budget['sku'], '--dump=true', '--launch-count=10',
           '--timeout=2', '--output=' + str(directory/'profile'),
           str(binary), case['op'], *map(str, case['shape'])]
    start = time.monotonic()
    with (directory/'run.log').open('w') as log:
        proc = subprocess.Popen(cmd, cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        while proc.poll() is None:
            size = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
            if time.monotonic()-start > budget['case_timeout_seconds'] or size > budget['per_case_raw_cap_mib']*2**20:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                break
            time.sleep(1)
    cost = dict(seconds=time.monotonic()-start, exit_code=proc.returncode)
    (directory/'cost.json').write_text(json.dumps(cost))
    print(case['op'], cost, flush=True)
