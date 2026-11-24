"""Acquire one development CA smoke, recording every native launch in the component graph."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

SOC = {"dt": "Ascend950DT_9582", "pr": "Ascend950PR_957b"}

def run(args):
    probe = args.probe.resolve(strict=True)
    raw = args.raw.resolve()
    raw.mkdir(parents=True, exist_ok=False)
    cmd = ["msopprof", "simulator", "--soc-version="+SOC[args.chip],
           "--launch-count=32", "--dump=false", "--timeout=9",
           "--output="+str(raw/"profile"), str(probe), args.mode,
           str(args.blocks), str(args.selected), str(args.heads)]
    if args.launcher:
        cmd = ["bash",str(args.launcher.resolve(strict=True)), *cmd]
    start = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    with (raw/"run.log").open("x") as log:
        proc = subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=raw,start_new_session=True)
        try:
            rc = proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                rc = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                rc = proc.wait()
    cost = {"role":"development-smoke","started_utc": start,
            "wall_seconds":time.monotonic()-clock,"exit_code":rc,"command":cmd,
            "probe_sha256":hashlib.sha256(probe.read_bytes()).hexdigest()}
    (raw/"cost.json").write_text(json.dumps(cost,indent=2)+"\n")
    return rc

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("compact", "dense"))
    p.add_argument("chip", choices=SOC)
    p.add_argument("probe", type=Path)
    p.add_argument("raw", type=Path)
    p.add_argument("--launcher", type=Path)
    p.add_argument("--blocks", type=int, default=100)
    p.add_argument("--selected", type=int, default=75)
    p.add_argument("--heads", type=int, default=32)
    raise SystemExit(run(p.parse_args()))
