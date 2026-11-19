"""Retain v2 controls and repeat them when its artificial 16-CPU mask limits the CA host."""

import os
from pathlib import Path
import signal
import sys
import time

import campaign
import resume

ROOT = campaign.ROOT
BASE = campaign.OUT
ACTIVE = BASE / "v3"
PREVIOUS = BASE / "v2"
RAW2 = campaign.RAW / "v2"


def register():
    if ACTIVE.exists():
        raise FileExistsError(ACTIVE)
    archived = campaign.read(PREVIOUS / "registration.json")
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 104:
        raise RuntimeError("Native process was observed with 102 threads; need at least 104 eligible CPUs.")
    ACTIVE.mkdir()
    inputs = dict(archived["sources"])
    inputs[str(Path(__file__).resolve().relative_to(ROOT))] = campaign.sha(Path(__file__).resolve())
    reg = dict(archived, sources=inputs, utc=campaign.now(), control_cpu_ids=allowed, control_wall_s=1800,
               control_scheme="Use the same full available CPU mask at nice 19 for both short controls, run serially, and trace file opens only. Keep v2 numerical, timing and output criteria; keep 24h full shape, 16/48 GiB and Kimi gate.",
               affinity_correction="v2 102-thread native process was pinned to 16 CPUs. After its original short control reaches its own wall limit, avoid repeating the same host bottleneck. Keep all v2 receipts. Cancel only the unfinished twin; then guarded takeover of the waiting Flash supervisor. Kimi is untouched. If the baseline passes, do not supersede v2.",
               source_registration_sha256=campaign.sha(PREVIOUS / "registration.json"))
    campaign.write(ACTIVE / "registration.json", reg)
    campaign.write(ACTIVE / "queue-status.json", dict(state="conditional-on-first-v2-control-wall-cap", utc=campaign.now()))


def argv(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, ProcessLookupError):
        return []


def wait_file(path, seconds=2100):
    end = time.monotonic() + seconds
    while time.monotonic() < end and not path.exists():
        time.sleep(1)
    if not path.exists():
        raise TimeoutError(path)
    return campaign.read(path)


def run():
    first = wait_file(PREVIOUS / "control-original.json")
    if first["qualified_core_envelope"] or first["cost"]["stop_reason"] != "wall cap":
        campaign.write(ACTIVE / "queue-status.json", dict(utc=campaign.now(), state="unused-keep-v2"))
        return
    raw = RAW2 / "control-redirected"
    if not (PREVIOUS / "control-redirected.json").exists():
        meta = wait_file(raw / "process.json")
        pid = meta["pid"]
        ids = argv(pid)
        if (os.getpgid(pid) != pid or bytes(str(raw), "utf-8") not in ids
                or ids[0] != b"/usr/bin/time" or b"strace" not in ids):
            raise RuntimeError("Refuse to interrupt unidentified recipient")
        os.killpg(pid, signal.SIGTERM)
        campaign.write(ACTIVE / "interrupted-v2-twin.json", dict(utc=campaign.now(), pid=pid,
                reason="First control hit its wall after pinning 102 CA host threads to 16 CPUs. Preserve the second attempt and avoid a second incomplete run with the same CPU barrier constraint."))
    wait_file(PREVIOUS / "control-comparison.json", 90)
    # Only supersede the sleep of the older, pending serial queue. Never signal a native process.
    status_path = PREVIOUS / "queue-status.json"
    for _ in range(30):
        if campaign.read(status_path).get("state") == "waiting-for-Kimi":
            break
        time.sleep(1)
    if (campaign.read(status_path).get("state") != "waiting-for-Kimi"
            or not campaign.kimi_processes() or campaign.KOUT.exists() or (RAW2 / "flash-retry").exists()):
        raise RuntimeError("No safe pending queue for handoff; v2 remains in charge")
    own = [int(p.name) for p in Path("/proc").glob("[0-9]*")
           if any(a.endswith(b"/full_hidden23/resume.py") for a in argv(p.name)) and b"run" in argv(p.name)]
    if len(own) != 1:
        raise RuntimeError(f"Could not verify exact prior queue identity: {own}")
    os.kill(own[0], signal.SIGINT)
    for _ in range(30):
        if not argv(own[0]):
            break
        time.sleep(.2)
    if argv(own[0]) or (RAW2 / "flash-retry").exists():
        raise RuntimeError("Old queue still exists")
    campaign.write(PREVIOUS / "handoff.json", dict(utc=campaign.now(), old_supervisor=own[0],
            state="superseded-during-Kimi-wait", replacement="results/v0.23/full-hidden/v3/queue-status.json"))
    campaign.write(BASE / "queue-status.json", dict(utc=campaign.now(), active="v3/queue-status.json",
            state="host-affinity-recovery", old="v2/handoff.json"))
    campaign.NATIVE = resume.ENTRY
    campaign.OUT = ACTIVE
    campaign.REG = ACTIVE / "registration.json"
    campaign.RAW = campaign.RAW / "v3"
    campaign.run()


if __name__ == "__main__":
    {"register": register, "run": run}[sys.argv[1]]()
