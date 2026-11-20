"""Update the v0.28 public evaluation when another registered CA endpoint finishes."""
import fcntl
import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from development.validation.matmul28.overview import OUT, read, receipt
from development.validation.matmul28.public_audit import run as score
from development.validation.matmul28.workload_rows import refresh as update_workloads
from development.validation.specialized28.snapshot import release as update_evaluation


def watch():
    jobs = read(OUT / "overview-prices.json")["rows"]
    sources = read(OUT.parent / "model.json")["checksum_sha256"]
    previous = None
    while True:
        status = [receipt(row, sku) for row in jobs for sku in ("dt", "pr")]
        # A first acquisition can be followed by a separately qualified drain.
        verdicts = [(r["status"], r.get("receipt"), r["ca_us"]) for r in status]
        if verdicts != previous:
            for path, expected in sources.items():
                if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"Frozen release changed at {path}. Reports left at their last verified state.")
            score()
            update_workloads()
            update_evaluation()
            previous = verdicts
            print(f"{sum(v[0] == 'qualified' for v in verdicts)}/{len(verdicts)} qualified", flush=True)
        terminal = [OUT / name / "state.json" for name in ("long-prefill","wide-prefill","narrow","multi")]
        if (all(v[0] != "pending" for v in verdicts) and
                all(p.exists() and read(p).get("phase") == "finished" for p in terminal)):
            return
        time.sleep(60)


if __name__ == "__main__":
    lock = ROOT / "runs/matmul28/refresh-public.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        watch()
