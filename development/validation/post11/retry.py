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

from recover import ROOT, OUT, extract, sha256


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2) + '\n')


def freeze():
    old = json.loads((OUT / 'freeze.json').read_text())
    for path, digest in old['sha256'].items():
        assert sha256(ROOT / path) == digest
    original = json.loads((OUT / 'original-attempt.json').read_text())
    lost = [r['id'] for r in original['timing_failed']]
    assert len(lost) == 2 and len(original['completed']) == 8
    batches = []
    for c in old['cases']:
        if c['id'] not in lost:
            continue
        warm = dict(old['cases'][3], role='context_warmup', prior_aiv=0)
        batches.append([warm, c])
        assert c['prior_aiv'] == warm['prediction']['predicted_active_aiv'] == 64
    hashes = {path: sha256(ROOT / path) for path in
              ['development/validation/post11/retry.py','development/validation/post11/recover.py',
               'results/v0.11/post/freeze.json', 'results/v0.11/post/original-attempt.json']}
    if (OUT / 'retry-freeze.json').exists():
        raise RuntimeError('Recovery already frozen')
    save('retry-freeze.json', dict(utc=datetime.now(timezone.utc).isoformat(), sha256=hashes,
                                   isolated_batches=batches, per_batch_seconds=360, per_batch_gib=1.25,
                                   cause='helper finalization rejected negative final intervals, 0 instructions committed; numerical outputs passed'))


def collect(batch, contract):
    target = batch[-1]
    folder = ROOT / 'runs/v11post' / ('retry-' + target['id'])
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
    save(folder.name+'.json', dict(completed=good,timing_failed=failed,wall_s=duration))
    return good, failed


if __name__ == '__main__':
    freeze()
    manifest = json.loads((OUT / 'retry-freeze.json').read_text())
    for path,digest in manifest['sha256'].items():
        assert sha256(ROOT / path) == digest
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda b:collect(b,manifest), manifest['isolated_batches']))
    repaired = [r for good, failed in outcomes for r in good if r['role']=='validation']
    original = json.loads((OUT/'original-attempt.json').read_text())
    rows = original['completed'] + repaired
    print('qualified',len(rows),'of 10 distinct targets',flush=True)
    if len(rows)==10:
        cases = json.loads((OUT/'freeze.json').read_text())['cases']
        by_id = {r['id']:r for r in rows}
        save('mhc_post-0-10.json',[by_id[r['id']] for r in cases])
