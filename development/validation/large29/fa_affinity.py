"""Give the 32-thread CA barriers enough host CPUs while retaining the worker's low priority."""
from pathlib import Path
import os
import time
from fa import ROOT, OUT, load, save, now


def cpu_usage():
    def ticks():
        return {int(s[0][3:]):tuple(map(int,s[1:9])) for l in Path("/proc/stat").read_text().splitlines()
                if (s:=l.split()) and s[0][3:].isdigit()}
    first=ticks();time.sleep(2);last=ticks()
    return {c:1-(last[c][3]+last[c][4]-v[3]-v[4])/max(1,sum(last[c])-sum(v))
            for c,v in first.items()}


def descendants(pid):
    yield pid
    for line in (Path("/proc")/str(pid)/"task"/str(pid)/"children").read_text().split():
        yield from descendants(int(line))


def expand():
    state=load(OUT/"state.json")
    if state["phase"]!="acquiring":
        return
    pid=state["pid"]
    proc=Path("/proc")/str(pid)
    assert proc.stat().st_uid==os.getuid()
    assert proc.joinpath("cmdline").read_bytes().split(b"\0")[1:3]==[b"development/validation/large29/fa.py",b"collect"]
    eligible=os.sched_getaffinity(0)
    usage=cpu_usage()
    before=os.sched_getaffinity(pid)
    # Allow OS placement across at least 32 physical CPUs, preserve original CPUs
    # and choose the least busy other cores. Priorities of all processes stay the same.
    chosen=set();physical=set()
    for core in sorted(eligible,key=lambda x:(x not in before,usage[x])):
        siblings=(Path("/sys/devices/system/cpu")/f"cpu{core}/topology/thread_siblings_list").read_text().strip()
        if siblings not in physical:
            chosen.add(core);physical.add(siblings)
        if len(chosen)>=48:break
    assert len(chosen)>=32 and before<=chosen
    entries=[]
    for child in descendants(pid):
        try:
            entry=Path("/proc")/str(child)
            assert entry.stat().st_uid==os.getuid()
            nice=os.getpriority(os.PRIO_PROCESS, child)
            if nice<12:
                raise RuntimeError(f"Unexpected priority of FA worker {child}: {nice}")
            tasks=list((entry/"task").iterdir())
            for t in tasks:
                try:os.sched_setaffinity(int(t.name),chosen)
                except ProcessLookupError:pass
            entries.append(dict(pid=child,comm=(entry/"comm").read_text().strip(),nice=nice,threads=len(tasks)))
        except FileNotFoundError:continue
    save(OUT/"host-affinity.json",dict(utc=now(),case=state["case"],before=sorted(before),after=sorted(chosen),
         processes=entries,reason="32 active CA host threads were oversubscribed on 12 CPUs. Simulator init "
         "logged 248 cycles/72.57s and 419/139.36s. Affinity only; frozen device and collection unchanged."))


if __name__=="__main__":
    expand()
