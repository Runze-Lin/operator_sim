"""Preserve transient CA memory-controller logs before msopprof removes native logs."""

from pathlib import Path
import re
import shutil
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from development.validation.sigmoid20.campaign import RAW, DEST, DT, read, write

PERIOD = re.compile(
    rb"Current time: (\d+), tDFI: ([\d.]+) ns, DMC Total bytes: (\d+), DDR Total bytes: (\d+)"
)
COMMANDS = re.compile(
    rb"^Read\s+: (\d+)\s+\| Total reads\s+: (\d+)\s+\| Write\s+: (\d+)\s+\| Total writes\s+: (\d+)",
    re.M,
)
ARCHIVE = RAW.with_name("sigmoid20-controllers")


def save():
    for d in RAW.glob("sigmoid*"):
        out = ARCHIVE / d.name
        for f in (d / "log").glob("lpddr_state*.log"):
            try:
                size = f.stat().st_size
                if not size:
                    continue
                target = out / f.name
                if target.exists() and target.stat().st_size >= size:
                    continue
                out.mkdir(exist_ok=True, parents=True)
                temp = target.with_suffix(".partial")
                shutil.copyfile(f, temp)
                if temp.stat().st_size >= size:
                    temp.replace(target)
            except FileNotFoundError:
                continue


def summarize():
    path = DEST / "holdout.json"
    rows = read(path) if path.exists() else []
    counters = []
    for r in rows:
        if not r.get("qualified"):
            continue
        t, h = r["shape"]
        d = ARCHIVE / f"sigmoid_{t}_{h}_{r['sku']}"
        files = list(d.glob("lpddr_state*.log"))
        if not files:
            continue
        dmc = ddr = reads = writes = 0
        end = set()
        intervals = set()
        period_count = 0
        for f in files:
            data = f.read_bytes()
            p = PERIOD.findall(data)
            c = COMMANDS.findall(data)
            if not p or len(p) != len(c):
                break
            period_count += len(p)
            end.add(int(p[-1][0]))
            intervals.update(float(x[1]) for x in p)
            rd, wr = int(c[-1][1]), int(c[-1][3])
            reads += rd
            writes += wr
            if sum(int(x[0]) for x in c) != rd or sum(int(x[2]) for x in c) != wr:
                break
            dmc += sum(int(x[2]) for x in p)
            ddr += sum(int(x[3]) for x in p)
        else:
            # Retained snapshots may end early; qualify coverage of a completed
            # component only with full expected channel count and matching report times.
            n = t * h
            expected = 128 if r["sku"] == DT else None
            counters.append(
                dict(
                    shape=[t, h],
                    sku=r["sku"],
                    files=len(files),
                    expected_channels=expected,
                    same_end_time=len(end) == 1,
                    end_time=sorted(end),
                    tdfi_ns=sorted(intervals),
                    sampled_intervals=period_count,
                    logical_read_bytes=4 * n,
                    logical_write_bytes=4 * n,
                    dmc_bytes=dmc,
                    ddr_bytes=ddr,
                    read_commands=reads,
                    write_commands=writes,
                    scope="Last complete captured interval by channel; full-kernel end alignment not independently established.",
                )
            )
    write(DEST / "controller-snapshots.json", counters)
    return len(rows)


if __name__ == "__main__":
    if sys.argv[-1] == "watch":
        last_holdout_change = None
        while True:
            save()
            holdout = DEST / "holdout.json"
            signature = holdout.stat().st_mtime_ns if holdout.exists() else 0
            if signature != last_holdout_change:
                if summarize() == 20:
                    break
                last_holdout_change = signature
            time.sleep(2)
    else:
        summarize()
