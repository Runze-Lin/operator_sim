"""Own the frozen CA queue in tmux and adopt subprocesses left by a former monitor."""

from datetime import datetime, timezone
from pathlib import Path
import fcntl
import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from development.validation.sigmoid20.campaign import DEST, RAW, ROOT, read, write

HERE = Path(__file__).resolve().parent
STATE = RAW.with_name("sigmoid20-supervisor")


def owned_processes():
    """All returned identities are constrained to this UID and sigmoid20 launch paths."""
    collector = []
    launch = {}
    for p in Path("/proc").glob("[0-9]*"):
        try:
            if p.stat().st_uid != os.getuid():
                continue
            cmd = (
                p.joinpath("cmdline").read_bytes().decode(errors="replace").split("\0")
            )
            if any(s.endswith("sigmoid20/campaign.py") for s in cmd) and cmd[-3:-1] == [
                "collect",
                "holdout",
            ]:
                collector.append(int(p.name))
            for arg in cmd:
                if arg.startswith("--output=" + str(RAW) + "/") and arg.endswith(
                    "/profile"
                ):
                    case = Path(arg.removeprefix("--output=")).parent.name
                    if case.startswith("sigmoid_"):
                        launch.setdefault(case, []).append(int(p.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return collector, launch


def birth(path):
    result = subprocess.check_output(["stat", "-c", "%W", str(path)], text=True)
    return int(result.strip())


def utc(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def raw_bytes(folder):
    total = 0
    for path in folder.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            pass
    return total


def adopt(j, chip, folders, live):
    """A finished child has /usr/bin/time, the host golden, and the final profiler marker."""
    case = "sigmoid_" + "_".join(str(v) for v in j["shape"]) + "_" + chip
    d = RAW / case
    log = d / "run.log"
    cpu = d / "cpu.txt"
    if not d.exists() or (d / "cost.json").exists():
        return False
    if not log.exists():
        return False
    started = birth(log)
    if case in live:
        budgets = (
            time.time() - started > j["wall_s"] or raw_bytes(d) > j["raw_mib"] * 2**20
        )
        if budgets:
            for pid in live[case]:
                try:
                    if os.getpgid(pid) == pid and Path(
                        f"/proc/{pid}/cmdline"
                    ).read_bytes().startswith(b"/usr/bin/time\0"):
                        os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        return True
    # Give a disappearing process time to finish its final file writes.
    if time.time() - log.stat().st_mtime < 15:
        return True
    txt = log.read_text(errors="replace")
    timer = cpu.read_text().splitlines() if cpu.exists() else []
    complete = (
        timer
        and len(timer) == 1
        and "SMOKE_PASS" in txt
        and "Op profiling finish" in txt
        and "All task success." in txt
    )
    if complete:
        stop = max(log.stat().st_mtime, cpu.stat().st_mtime)
        n = raw_bytes(d)
        write(
            d / "cost.json",
            dict(
                started_utc=utc(started),
                finished_utc=utc(stop),
                seconds=stop - started,
                raw_bytes=n,
                peak_bytes=n,
                peak_bytes_is_lower_bound=True,
                cpu_seconds=sum(map(float, timer[-1].split())),
                exit_code=0,
                stop_reason=None,
                command=[
                    "msopprof",
                    "simulator",
                    "--soc-version=" + chip,
                    "--dump=false",
                    j["runner"],
                    *j["args"],
                ],
                recovered="Observed full host and profiler completion plus clean /usr/bin/time. Wall derived from file birth and retirement; peak is a retained-data lower bound.",
            ),
        )
    else:
        # Preserve an interrupted native attempt. The fresh child can now use the original target folder.
        folders.mkdir(exist_ok=True, parents=True)
        d.rename(folders / f"{case}-{started}")
    return False


def main():
    STATE.mkdir(exist_ok=True, parents=True)
    lock = (STATE / "queue.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    f = read(DEST / "holdout-freeze.json")
    sidecars = {}
    for name in ("report", "controllers", "long_context", "long_figures"):
        out = (STATE / (name + ".log")).open("a")
        sidecars[name] = subprocess.Popen(
            [sys.executable, str(HERE / (name + ".py")), "watch"],
            cwd=ROOT,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    launches = None
    try:
        while True:
            done = (
                read(DEST / "holdout.json") if (DEST / "holdout.json").exists() else []
            )
            orig, live = owned_processes()
            states = {k: v.poll() for k, v in sidecars.items()}
            write(
                STATE / "status.json",
                dict(
                    at=utc(time.time()),
                    pid=os.getpid(),
                    collector=orig,
                    live_jobs=live,
                    completed=len(done),
                    sidecar_exit=states,
                ),
            )
            if len(done) == len(f["jobs"]) * len(f["chips"]):
                break
            if not orig:
                outstanding = [
                    adopt(j, c, STATE / "interrupted", live)
                    for j in f["jobs"]
                    for c in f["chips"]
                ]
                if not any(outstanding):
                    # All former processes retired before a fresh single collector takes the queue.
                    with (STATE / "collector.log").open("a") as output:
                        launches = subprocess.Popen(
                            [
                                sys.executable,
                                str(HERE / "campaign.py"),
                                "collect",
                                "holdout",
                            ],
                            cwd=ROOT,
                            stdout=output,
                            stderr=subprocess.STDOUT,
                        )
            if (
                launches is not None
                and launches.poll() is not None
                and launches.returncode != 0
            ):
                raise RuntimeError(
                    "Collector failed, inspect " + str(STATE / "collector.log")
                )
            for name, p in sidecars.items():
                if p.poll() is not None and p.returncode != 0:
                    raise RuntimeError(name + " failed, inspect supervisor log")
            time.sleep(8)
    finally:
        # Completed sidecars naturally retire after publishing their last report.
        for p in sidecars.values():
            if p.poll() is None:
                p.terminate()
                p.wait()
    for name in ("report", "controllers", "long_context", "long_figures"):
        subprocess.run(
            [sys.executable, str(HERE / (name + ".py"))], cwd=ROOT, check=True
        )
    write(
        STATE / "complete.json",
        dict(
            at=utc(time.time()),
            completed=len(done),
            qualified=sum(r["qualified"] for r in done),
        ),
    )


if __name__ == "__main__":
    main()
