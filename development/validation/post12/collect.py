"""Retry only unmeasured kernels, with one 64-core warm-up per isolated simulator."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'development/validation/post11'))
from recover import extract, sha256
from output_audit import verify

OUT = ROOT / 'results/v0.12/post'
RAW = ROOT / 'development/validation/post12/raw'

def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2) + '\n')


def collect(batch, contract):
    target = batch[-1]
    folder = RAW / ('batch-' + target['id'])
    folder.mkdir(parents=True, exist_ok=False)
    binary = folder / 'MHC'
    shutil.copy2(ROOT / 'build/vector-v8/MHC', binary)
    lst = folder / 'inputs.txt'
    lst.write_text(''.join('mhc_post '+str(i)+' '+' '.join(map(str,c['shape']))+'\n' for i,c in enumerate(batch)))
    cmd = ['bash', str(ROOT/'scripts/cann-run.sh'), 'env', 'LD_PRELOAD='+str(ROOT/'build/capture/libsim_soc_adapter.so'),
           'OPERATOR_SIM_TARGET_SOC=Ascend950DT_9582','npusim','record',str(binary),'-s','Ascend950DT',
           '-o',str(folder/'profile'),'-u',str(lst)]
    start = time.monotonic()
    with (folder/'run.log').open('w') as log:
        proc = subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=folder,start_new_session=True)
        while proc.poll() is None:
            if (time.monotonic()-start > contract['per_batch_seconds'] or
                    sum(p.stat().st_size for p in folder.rglob('*.db')) > contract['per_batch_gib']*2**30):
                os.killpg(proc.pid,signal.SIGTERM)
                try:
                    proc.wait(5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL)
                    proc.wait()
                raise RuntimeError('Frozen recovery budget reached at ' + str(folder))
            time.sleep(1)
    duration = time.monotonic()-start
    if proc.returncode != 0:
        raise RuntimeError('CA driver failed in ' + str(folder))
    good, failed = extract(folder, batch, duration)
    for row in good:
        row['output_audit'] = verify(row)
    save(folder.name+'.json', dict(completed=good,timing_failed=failed,wall_s=duration))
    return good, failed


if __name__ == '__main__':
    spec = json.loads((OUT / 'freeze.json').read_text())
    for path, digest in spec['sha256'].items():
        assert sha256(ROOT/path) == digest
    with ThreadPoolExecutor(max_workers=spec['workers']) as pool:
        for g,b in pool.map(lambda batch:collect(batch,spec),spec['batches']):
            print([(r['shape'],r['output_audit']['qualified']) for r in g],len(b),'helper failed',flush=True)
