"""Check the installed BF16 host branch without submitting a target Matmul kernel."""
from datetime import datetime, timezone
import hashlib
import json
from math import ceil
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
PARENT = ROOT / 'results/v0.28/matmul'
OUT = ROOT / 'results/v0.29/matmul'
RAW = ROOT / 'runs/matmul29-host'
BIN = ROOT / 'build/smoke/matmul29-host'
SOCS = {'Ascend950DT_9582': 32, 'Ascend950PR_957b': 28}
FROZEN = OUT / 'workspace-freeze.json'
CONTROL = [(1, 64, 7168), (4096, 32, 5120), (1, 2304, 1536)]
PROBES = [(m, 32, 5120) for m in (3584, 3585, 3840, 4095, 4097, 4352)] + [
    (17, 64, k) for k in (3840, 3841, 4095, 4096, 4097)] + [
    (64, n, 7168) for n in (3584, 3585, 4097)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rule(shape, cores):
    m, n, k = shape
    return ceil(m/256) * ceil(n/256) <= cores//2 and 256*ceil(k/256) >= max(8192, 256*cores)/2


def register():
    archived = json.loads((PARENT/'public-score.json').read_bytes())['rows']
    known = {tuple(r['shape']) for r in archived}
    for reg in PARENT.rglob('freeze.json'):
        known.update(tuple(r['shape']) for r in json.loads(reg.read_bytes())['rows'])
    if set(PROBES) & known:
        raise ValueError('A prospective shape overlaps an archived or in-flight experiment')
    rows = []
    for shape in CONTROL + PROBES:
        for soc, c in SOCS.items():
            mixed = rule(shape, c)
            rows.append(dict(shape=shape, soc=soc, control=shape in CONTROL, mixed_expected=mixed,
                             workspace_expected=512 + (262144*c if mixed else 0)))
    src = Path(__file__).parent
    fixed = dict(utc=datetime.now(timezone.utc).isoformat(), rows=rows,
                 sources={str(p.relative_to(ROOT)):sha(p) for p in (BIN, src/'host_query.cpp', src/'host_audit.py')},
                 hypothesis='Contiguous BF16: a 256x256 output grid with <= half as many jobs as AICs and K rounded up to 256 reaching the host threshold can choose split K. Tests separate installed release from the open source performance-price strategy.',
                 qualification='Only a matching runtime SKU, 3 replicated positive/negative controls, every successful workspace query, clean host completion, and no profiled target kernels can establish a host-branch signal. No CA time, tensor golden, MMAD geometry or chip timing inferred.',
                 budget='One descriptor batch per SKU; at most 180 seconds each. Sequential. No phase-two Matmul or workspace payload allocation. Inputs and output allocated to exact declared size.')
    with FROZEN.open('x') as f:
        json.dump(fixed, f, indent=2); f.write('\n')


def collect(soc):
    reg = json.loads(FROZEN.read_bytes())
    assert soc in SOCS and all(sha(ROOT/p) == s for p,s in reg['sources'].items())
    rows = [r.copy() for r in reg['rows'] if r['soc'] == soc]
    folder = RAW / soc; folder.mkdir(exist_ok=False)
    cases = folder/'cases.txt'; cases.write_text(''.join(' '.join(map(str,r['shape']))+'\n' for r in rows))
    cmd = ['timeout','-k','5','180','bash',str(ROOT/'scripts/cann-run.sh'),'msopprof','simulator',
           '--soc-version='+soc,'--dump=false','--timeout=3','--output='+str(folder/'profile'),str(BIN),str(cases)]
    stamp = time.monotonic(); start = datetime.now(timezone.utc).isoformat(); log = folder/'run.log'
    with log.open('w') as f: proc = subprocess.run(cmd,cwd=folder,stdout=f,stderr=subprocess.STDOUT)
    duration = time.monotonic()-stamp; lines=log.read_text(errors='replace');observed={}
    for m,n,k,ws in re.findall(r'^HOST_WS (\d+) (\d+) (\d+) (\d+)$', lines, re.M):
        observed[(int(m),int(n),int(k))]=int(ws)
    identity = re.search(r'^host_query_soc=(\S+)$',lines,re.M)
    for r in rows:
        ws=observed.get(tuple(r['shape'])); r.update(workspace_observed=ws,agreement=None if ws is None else ws==r['workspace_expected'])
    controls=all(r['agreement'] for r in rows if r['control'])
    kernel_names=re.findall(r'Start profiling on kernel:\s*(\S+)',lines)
    ok=bool(identity and identity[1]==soc and 'HOST_QUERY_DONE' in lines and controls and
            not kernel_names and len(observed)==len(rows))
    result=dict(soc=soc,started_utc=start,elapsed_s=duration,register_sha256=sha(FROZEN),wrapper_code=proc.returncode,
                host_completed='HOST_QUERY_DONE' in lines, observed_soc=identity[1] if identity else None,kernels=kernel_names,
                qualified_for_host_choice=ok, log_sha256=sha(log),rows=rows,ca_us=None)
    (OUT/(soc+'-workspace.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(soc,ok,duration,len(observed),flush=True)


if __name__ == '__main__':
    if sys.argv[1]=='freeze':register()
    else: collect(sys.argv[1])
