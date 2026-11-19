"""Register and run a bounded official CANN ChunkKdaFwd diagnostic on both SKUs."""
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / 'results/v0.21/chunk'
RUN = ROOT / 'runs/chunk21'
CHIPS = ('Ascend950DT_9582', 'Ascend950PR_957b')
FILES = ('build/smoke/chunk21', 'development/validation/chunk21/kimi.cpp',
         'smoke/extension_kda.cpp', 'development/validation/chunk21/probe.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)


def register(tokens, heads, tag):
    ART.mkdir(parents=True, exist_ok=True)
    path = ART / f'{tag}-registration.json'
    if path.exists():
        raise FileExistsError(path)
    save(path, dict(role='mechanism development, zero current latency estimates', tokens=tokens,
                    heads=heads, cube_shape=[1,tokens,heads,heads,128,128], chips=CHIPS, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    dtype='bf16', beta_dtype='fp32', gate_dtype='fp32', state_dtype='fp32',
                    layout='BSND', use_gate_in_kernel=True, lower_bound=-0.03125, safe_gate=True, state_v_first=True,
                    dt_bias=True, initial_state=True, cu_seqlens=[0,tokens], chunk_size=64,
                    chunk_indices=sum(([0,i] for i in range((tokens+63)//64)),[]),
                    outputs=['attention','final VK state','Aqk','Akk'], exports_gk=False,
                    budget_seconds=420, budget_bytes=1024**3,
                    hashes={file:sha(ROOT/file) for file in FILES}))


def collect(chip, conf, tag):
    dest = RUN / tag / chip
    dest.mkdir(parents=True, exist_ok=False)
    binary = ROOT / FILES[0]
    cmd = ['bash',str(ROOT/'scripts/cann-run.sh'),'msopprof','simulator',f'--soc-version={chip}',
           '--timeout=6','--launch-count=20','--dump=true',f'--output={dest}/profile',
           str(binary),str(conf['tokens']),str(conf['heads'])]
    start = time.monotonic()
    reason = None
    path = dest/'run.log'
    with path.open('w') as out:
        proc = subprocess.Popen(cmd, cwd=dest, stderr=subprocess.STDOUT, stdout=out, start_new_session=True)
        while proc.poll() is None:
            time.sleep(2)
            if time.monotonic()-start > conf['budget_seconds']:
                reason = 'wall budget'
            elif sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()) > conf['budget_bytes']:
                reason = 'storage budget'
            if reason:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
                break
    content=path.read_text(errors='replace')
    kernels=re.findall(r'Start profiling on kernel: ([^\s]+)',content)
    errors = [line[:420] for line in content.splitlines() if 'already has same set_flag' in line or 'SMOKE_FAIL' in line or '[ERROR]' in line]
    receipt=dict(chip=chip, command=cmd, seconds=time.monotonic()-start, exit=proc.returncode, stop=reason, kernels=kernels,
                 errors=errors[:16], error_count=len(errors), completion='SMOKE_PASS' in content, log_sha256=sha(path),
                 checks=[line for line in content.splitlines() if 'correctness name=' in line or 'input_vk_unchanged' in line],
                 raw_bytes=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),
                 potentially_eligible=proc.returncode==0 and 'SMOKE_PASS' in content and not errors)
    save(ART/f'{tag}-{chip}.json',receipt)


if __name__ == '__main__':
    tag=sys.argv[2]
    if sys.argv[1]=='register':
        register(int(sys.argv[3]),int(sys.argv[4]),tag)
    else:
        conf=json.loads((ART/f'{tag}-registration.json').read_text())
        if any(sha(ROOT/f)!=h for f,h in conf['hashes'].items()):
            raise ValueError('probe sources or binary changed since registration')
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda chip: collect(chip,conf,tag), conf['chips']))
