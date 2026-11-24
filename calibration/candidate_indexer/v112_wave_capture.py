"""Capture a single frozen v1.12 native case and record off-line costs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
PROBE=Path(__file__).with_name('v112_wave_probe.py').resolve()
SOC={'dt':'Ascend950DT_9582','pr':'Ascend950PR_957b'}
BUDGET_MIN={'dt':38,'pr':25}


def deliver_signal(proc, sig):
    try:
        os.killpg(proc.pid,sig)
    except ProcessLookupError:
        pass


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('case',choices=SOC)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--python',type=Path,required=True)
    p.add_argument('--packages',type=Path,required=True)
    args=p.parse_args()
    root=args.raw.resolve()
    root.mkdir(parents=True,exist_ok=False)
    command=['bash',str(ROOT/'scripts/cann-run.sh'),'env','PYTHONPATH='+str(args.packages.resolve()),
             'CANNBOTDSL_NATIVE_BINARY_MODE=require','CANNBOTDSL_NATIVE_BINARY_REPORT='+str(root/'provider.json'),
             'OMP_NUM_THREADS=1','msopprof','simulator','--soc-version='+SOC[args.case],'--launch-count=1',
             '--aic-metrics=PipeUtilization','--dump=false','--timeout='+str(BUDGET_MIN[args.case]),
             '--output='+str(root/'profile'),str(args.python.absolute()),str(PROBE),args.case]
    log=root/'run.log'
    started=datetime.now(timezone.utc).isoformat()
    timer=time.monotonic()
    golden_s=None
    expired=False
    cursor=0
    carry=b''
    with log.open('wb') as output:
        proc=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            time.sleep(3)
            elapsed=time.monotonic()-timer
            if golden_s is None:
                with log.open('rb') as handle:
                    handle.seek(cursor)
                    chunk=handle.read()
                    cursor=handle.tell()
                if b'V112_WAVE_GOLDEN_PASS' in carry+chunk:
                    golden_s=elapsed
                carry=(carry+chunk)[-32:]
            # The measured device must finish within budget. When the output
            # is verified, let the already-recorded trace finish exporting.
            limit=BUDGET_MIN[args.case]*60 if golden_s is None else golden_s+12*60
            if elapsed>limit:
                expired=True
                deliver_signal(proc,signal.SIGTERM)
                try:
                    proc.wait(12)
                except subprocess.TimeoutExpired:
                    deliver_signal(proc,signal.SIGKILL)
                break
        rc=proc.wait()
    record=dict(case=args.case,sku=SOC[args.case],start_utc=started,
                wall_s=round(time.monotonic()-timer,3),golden_observed_s=golden_s,rc=rc,
                watchdog=expired,command=command,probe_sha256=hashlib.sha256(PROBE.read_bytes()).hexdigest())
    (root/'cost.json').write_text(json.dumps(record,indent=2)+'\n')
    raise SystemExit(124 if expired else rc)


if __name__=='__main__':
    main()
