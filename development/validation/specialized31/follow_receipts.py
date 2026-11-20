"""Refresh the frozen wheel's reports after the remaining Matmul and Post endpoints finish."""
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'results/v0.31'


def read(p):return json.loads(p.read_bytes())


def inputs():
    kimi=ROOT/'results/v0.29/matmul/decode'
    files=[kimi/f'{m}-2304-1536-{sku}.json' for m in (16,81,144) for sku in ('dt','pr')]
    files += [OUT/'matmul/device'/f'11-1280-5120-{sku}.json' for sku in ('dt','pr')]
    files += [OUT/'matmul/kimi-prefill'/f'{sku}.json' for sku in ('dt','pr')]
    files += [OUT/'post'/f'{op}-{sku}.json' for op in ('vllm_hc_post','mhc_post')
              for sku in ('Ascend950DT_9582','Ascend950PR_957b')]
    markers=[(str(p.relative_to(ROOT)),sha256(p.read_bytes()).hexdigest() if p.is_file() else None) for p in files]
    long=read(ROOT/'results/v0.30/matmul/score.json')['rows']
    flash=read(ROOT/'results/v0.30/matmul-dispatch/device/score.json')['rows']
    # Published score files also update their clocks while nothing is measured.
    # Compare only the verdicts and source identities that change report contents.
    long_mark=[(r['shape'],r['chip'],r['status'],r['trace_sha256']) for r in long]
    flash_mark=[(r['chip'],r['status'],r.get('trace')) for r in flash if r['shape'][0]==524288]
    finished=(all(v for _,v in markers) and all(r['is_terminal'] for r in long)
              and len(flash_mark)==2 and all(v[1] in ('qualified','unqualified') for v in flash_mark))
    return [markers,long_mark,flash_mark],finished


def follow(prefix):
    raw=ROOT/'runs/specialized31'
    raw.mkdir(exist_ok=True)
    wheel=read(OUT/'wheel-check.json')['wheel_sha256']
    model=read(OUT/'model.json')['hashes']
    scripts=[HERE/n for n in ('release.py','matmul_report.py','matmul_scenarios.py','direction.py','kimi_pages.py')]
    code={p:sha256(p.read_bytes()).hexdigest() for p in scripts}
    with (raw/'report-follow.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=None
        while True:
            try:
                new,done=inputs()
            except (FileNotFoundError,json.JSONDecodeError):
                time.sleep(60)
                continue
            if new!=previous:
                actual_wheel,=(ROOT/'build/wheels/v031').glob('*.whl')
                if (sha256(actual_wheel.read_bytes()).hexdigest()!=wheel
                        or any(sha256(p.read_bytes()).hexdigest()!=h for p,h in code.items())
                        or any(sha256((ROOT/p).read_bytes()).hexdigest()!=h for p,h in model.items())):
                    print('The release inputs changed; preserving the last frozen snapshot.',flush=True)
                    return
                subprocess.run([sys.executable,'-I',HERE/'release.py',prefix],cwd=ROOT,check=True)
                subprocess.run([sys.executable,HERE/'matmul_report.py'],cwd=ROOT,check=True)
                subprocess.run([sys.executable,HERE/'kimi_pages.py'],cwd=ROOT,check=True)
                previous=new
            if done:return
            time.sleep(60)


if __name__=='__main__':
    follow(sys.argv[1])
