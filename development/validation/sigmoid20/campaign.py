"""Freeze input contracts and collect one native kernel per independent CA launch."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from development.validation.sku19.collect import capture
from development.validation.sku19.campaign import (
    DT,
    PR,
    FUTURE,
    digest,
    now,
    read,
    write,
)
from campaign_collect import stages

DEST = ROOT / "results/v0.20/newop"
RAW = ROOT / "runs/sigmoid20"
HOST = ROOT / "build/smoke/sigmoid20"
TRAIN = [(1, 12), (300, 12), (3072, 96)]
# One model-scale unchunked prefill for each requested length at TP8.
TEST = [
    (262144, 12),
    (72, 12),
    (1048576, 12),
    (2048, 12),
    (2097152, 12),
    (32768, 12),
    (9, 96),
    (4096, 24),
    (16384, 48),
    (131072, 12),
]
MODULE = ROOT / "operator_model/sigmoid_resource.py"


def register(stage):
    path = DEST / (stage + "-freeze.json")
    if path.exists():
        raise FileExistsError(path)
    chips = (DT,) if stage == "development" else (DT, PR)
    shapes = TRAIN if stage == "development" else TEST
    jobs = []
    for rows, heads in shapes:
        n = rows * heads
        j = dict(
            shape=[rows, heads],
            op="sigmoid_fp32",
            args=[str(rows), str(heads)],
            batch=False,
            runner=str(HOST.relative_to(ROOT)),
            kernels=1,
            nchecks=1,
            wall_s=10800 if n >= 2**20 else 1800,
            raw_mib=24576 if n >= 2**20 else 4096,
        )
        if stage == "holdout":
            from operator_model import estimate

            j["predictions"] = {}
            for chip in (*chips, FUTURE):
                e = estimate("sigmoid_fp32", j["shape"], dtype="fp32", chip=chip)
                assert e.latency_us
                j["predictions"][chip] = dict(
                    us=e.latency_us,
                    aiv=e.active_aiv,
                    stages=e.projection["stages_us"],
                    plan=e.projection["target_plan"],
                )
        jobs.append(j)
    write(
        path,
        dict(
            utc=now(),
            chips=chips,
            jobs=jobs,
            runner_sha256=digest(HOST),
            max_workers=2,
            policy="Full CANN official all-AIV window and full numerical reference. Longest inputs first; 3 hours and 24 GiB per large launch.",
            module_sha256=digest(MODULE) if stage == "holdout" else None,
        ),
    )


def collect(stage):
    f = read(DEST / (stage + "-freeze.json"))
    assert digest(HOST) == f["runner_sha256"]
    if stage == "holdout":
        assert digest(MODULE) == f["module_sha256"]
    out = DEST / (stage + ".json")
    rows = read(out) if out.exists() else []
    lock = Lock()
    os.environ.update(OMP_WAIT_POLICY="PASSIVE", GOMP_SPINCOUNT="0")

    def one(arg):
        j, chip = arg
        if any(r["shape"] == j["shape"] and r["sku"] == chip for r in rows):
            return
        case = "sigmoid_" + "_".join(map(str, j["shape"])) + "_" + chip
        folder = RAW / case
        if not (folder / "cost.json").exists():
            capture(j, chip, folder)
        row = dict(shape=j["shape"], sku=chip, cost=read(folder / "cost.json"))
        assert row["cost"]["started_utc"] > f["utc"]
        try:
            (k,), g = stages(j, chip, folder)
            assert (
                k["kernel"].startswith("Sigmoid_")
                and k["active_aic"] == 0
                and k["active_aiv"] > 0
            )
            row.update(qualified=True, kernel=k, correctness=g)
        except (ValueError, AssertionError, IndexError, OSError) as e:
            row.update(qualified=False, reason=repr(e))
        with lock:
            rows.append(row)
            write(out, rows)
            print(case, row["qualified"], row.get("reason"), flush=True)

    with ThreadPoolExecutor(f["max_workers"]) as pool:
        list(pool.map(one, ((j, c) for j in f["jobs"] for c in f["chips"])))


if __name__ == "__main__":
    {"register": register, "collect": collect}[sys.argv[1]](sys.argv[2])
