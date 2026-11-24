"""Run one frozen 950DT control in its own simulator process and working directory."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time

from calibration.candidate_indexer.v112_wave_capture import deliver_signal

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
PROBE = SOURCE/'v113_control_probe.py'
SKU = 'Ascend950DT_9582'


def capture(arm, raw, packages, interpreter):
    raw = raw.resolve()
    raw.mkdir(parents=True,exist_ok=False)
    cmd = ['bash',str(ROOT/'scripts/cann-run.sh'),'env',f'PYTHONPATH={packages.resolve()}',
           'CANNBOTDSL_NATIVE_BINARY_MODE=require',f'CANNBOTDSL_NATIVE_BINARY_REPORT={raw}/provider.json',
           'OMP_NUM_THREADS=1','msopprof','simulator',f'--soc-version={SKU}','--launch-count=1',
           '--aic-metrics=PipeUtilization','--dump=false','--timeout=42',f'--output={raw}/profile',
           str(interpreter.absolute()),str(PROBE),arm]
    log = raw/'run.log'
    start = time.monotonic()
    ledger = dict(case=arm,sku=SKU,start_utc=datetime.now(timezone.utc).isoformat(),command=cmd,
                  probe_sha256=hashlib.sha256(PROBE.read_bytes()).hexdigest(),
                  cpu_control_sha256=hashlib.sha256((SOURCE/f'v113_control_{arm}_fixture.json').read_bytes()).hexdigest())
    with log.open('wb') as out, log.open('rb') as obs:
        proc = subprocess.Popen(cmd,cwd=raw,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        ledger['launcher_pid'] = proc.pid
        (raw/'start.json').write_text(json.dumps(ledger,indent=2)+'\n')
        expired = False
        golden = None
        tail = b''
        while proc.poll() is None:
            time.sleep(4)
            elapsed = time.monotonic()-start
            chunk = tail+obs.read()
            if golden is None and b'V113_CONTROL_GOLDEN_PASS' in chunk:
                golden = elapsed
                (raw/'golden-wall.json').write_text(json.dumps({'seconds':golden})+'\n')
            tail = chunk[-64:]
            # Golden releases the CA deadline; allow separate time to export.
            if elapsed > (42*60 if golden is None else golden+20*60):
                expired = True
                deliver_signal(proc, signal.SIGTERM)
                try:
                    proc.wait(15)
                except subprocess.TimeoutExpired:
                    deliver_signal(proc, signal.SIGKILL)
                break
        rc = proc.wait()
    ledger.update(wall_s=round(time.monotonic()-start,3),golden_observed_s=golden,rc=rc,watchdog=expired)
    (raw/'cost.json').write_text(json.dumps(ledger,indent=2)+'\n')
    return 124 if expired else rc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm',choices=('shared','moving','rotated'))
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--packages',type=Path,required=True)
    parser.add_argument('--python',type=Path,required=True)
    args = parser.parse_args()
    raise SystemExit(capture(args.arm,args.raw,args.packages,args.python))
