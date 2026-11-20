"""Refresh the cross-operator views when a registered acquisition returns a new verdict."""
from hashlib import sha256
import fcntl
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"results/v0.29"
LARGE=OUT/"large"
ORIGINAL=ROOT/"results/v0.28/matmul"


def receipts():
    patterns={
        OUT/"matmul/decode": ["*-2304-1536-*.json"],
        OUT/"matmul/decode/retry": ["receipt.json"],
        OUT/"matmul": ["3585-32-5120-*.json"], ROOT/"results/v0.28/matmul":["*-dt*.json","*-pr*.json"],
        OUT/"quant":["Ascend950*.json"], ROOT/"results/v0.27/quant-transfer":["native.json"],
        LARGE/"vector":["instruction-*.json","wrapper-*.json"],
        LARGE/"fa":["fa-*.json"], LARGE/"cube":["*-dt.json","*-pr.json"],
        LARGE/"fusion":["*-Ascend950*.json","score.json"],
        ROOT/"results/v0.28/matmul/long-prefill":["*-dt.json","*-pr.json"],
        ROOT/"results/v0.28/matmul/wide-prefill":["*-dt.json","*-pr.json"],
        ROOT/"results/v0.28/matmul/multi":["*-dt.json","*-pr.json"],
        ROOT/"results/v0.28/matmul/narrow":["*-dt.json","*-pr.json"]}
    # Ignore heartbeats and host timestamps. A supplemental full-device drain receipt
    # is a distinct artifact even if the original run first returned unqualified.
    files={p for folder,globs in patterns.items() for glob in globs for p in folder.glob(glob)}
    return {str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files}


def finished():
    campaigns = [ORIGINAL/"prefill-status.json", OUT/"matmul/boundary-state.json",
                 OUT/"matmul/decode/state.json", OUT/"quant/state.json"]
    campaigns += [ORIGINAL/group/"state.json" for group in ("narrow","multi","long-prefill","wide-prefill")]
    campaigns += [LARGE/group/"state.json" for group in ("fa","cube","fusion","vector")]
    for path in campaigns:
        if not path.exists():
            return False
        state = json.loads(path.read_bytes())
        if state.get("phase",state.get("state")) != "finished":
            return False
    return (OUT/"matmul/decode/retry/receipt.json").exists()


def refresh():
    for name,args in [("matmul29/validate.py",["score"]),("matmul29/report.py",[]),
                      ("matmul28/workload_rows.py",[]),("large29/coverage.py",[]),
                      ("specialized29/snapshot.py",[])]:
        subprocess.run([sys.executable,str(ROOT/"development/validation"/name),*args],cwd=ROOT,check=True)


def main():
    raw=ROOT/"runs/large29";raw.mkdir(exist_ok=True)
    with (raw/"overview.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=None
        while True:
            try:
                state=receipts()
                done=finished()
                if state!=previous or done:
                    # The per-campaign scorers publish after their underlying receipts.
                    time.sleep(5)
                    refresh()
                    previous=state
                if done:
                    return
            except (OSError, ValueError, subprocess.CalledProcessError):
                logging.exception("Could not refresh. Will retry from the registered receipts.")
            time.sleep(55)


if __name__=="__main__":
    main()
