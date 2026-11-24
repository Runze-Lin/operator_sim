"""Acquire independent ACLNN primitive windows at two sizes on each 950 SKU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
from hashlib import sha256
import subprocess
import time


def acquire(chip, root, launcher, probe, groups=('mask', 'math', 'memory'), row_counts=(9,144), timeout=8):
    failed = []
    for rows in row_counts:
        for group in groups:
            dest = root/f'{chip}-{group}-{rows}'
            dest.mkdir(parents=True, exist_ok=False)
            command = ['bash', str(launcher), 'msopprof', 'simulator', '--soc-version='+chip,
                       '--launch-count=32', '--dump=false', f'--timeout={timeout}', '--output='+str(dest/'profile'),
                       str(probe), group, str(rows)]
            start = time.monotonic()
            with (dest/'run.log').open('w') as log:
                result = subprocess.run(command, cwd=dest, stdout=log, stderr=subprocess.STDOUT)
            log_text = (dest/'run.log').read_text(errors="replace")
            complete = (result.returncode == 0 and "SMOKE_PASS" in log_text and "All task success." in log_text
                        and "Profiling data parse finished." in log_text)
            if not complete:
                failed.append(dest.name)
            row = dict(complete=complete,sku=chip, group=group, rows=rows, exit_code=result.returncode,
                       seconds=time.monotonic()-start, probe_sha256=sha256(probe.read_bytes()).hexdigest(), command=command)
            (dest/'cost.json').write_text(json.dumps(row)+'\n')
            print(dest.name, "complete" if complete else "incomplete", flush=True)
    if failed:
        raise RuntimeError("Incomplete CA groups: "+", ".join(failed))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'launcher', 'probe'):
        p.add_argument(name, type=Path)
    p.add_argument("--groups", nargs="+", choices=("mask","math","memory"), default=("mask","math","memory"))
    p.add_argument("--rows", nargs="+", type=int, default=(9,144))
    p.add_argument("--timeout", type=int, default=8, help="Simulator budget per group in minutes")
    args = p.parse_args()
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(acquire, chip, args.root.resolve(), args.launcher.resolve(), args.probe.resolve(),args.groups,args.rows,args.timeout)
                   for chip in ('Ascend950DT_9582', 'Ascend950PR_957b')]
        for future in futures:
            future.result()
