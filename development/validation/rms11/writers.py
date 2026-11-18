"""Reconstruct all critical-core stores and chip-wide rstd overlap from older DBs only."""
from collections import Counter
import json
from pathlib import Path
import sqlite3
from statistics import median

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/v0.11/rms"
rows = json.loads((ROOT/"results/v0.10/wide/rows.json").read_text())
evidence = []
for r in rows:
    if r["op"] != "rmsnorm":
        continue
    path = ROOT/r["db_source"]
    with sqlite3.connect(path.resolve().as_uri()+"?mode=ro",uri=True) as db:
        records = list(db.execute("""SELECT CoreTypeId, ExecInstrCoreId, ExecInstrTickStart, ExecInstrTickEnd, ExtendParams
            FROM ExecutedInstructions WHERE ExecInstrName='MOV_SRC_TO_DST_ALIGNv2'"""))
    per_core = {}
    for typ, core, start, end, metadata in records:
        attrs = json.loads(metadata)
        if attrs["queue_type"] == "MTE3":
            per_core.setdefault((typ,core), []).append(dict(start=start,end=end,bytes=attrs["processed_bytes"],
                                                           dst=attrs["dst_addr"]))
    # Adjacent output rstd words have fewer bytes per row than y, in every development trace.
    assert all(len(s)==2 for s in per_core.values())
    small = [min(s,key=lambda event:event["bytes"]) for s in per_core.values()]
    groups = Counter()
    all_events = Counter()
    for s in small:
        groups[s["dst"]//128] += 1
        all_events[s["start"]] += 1
        all_events[s["end"]] -= 1
    active = peak = 0
    for _,delta in sorted(all_events.items()):
        active += delta
        peak = max(peak,active)
    stores = sorted(per_core[(r["core_type"],r["core"])],key=lambda s:s["bytes"])
    first,second = stores
    overlap = max(0,min(first["end"],second["end"])-max(first["start"],second["start"]))
    latencies = [s["end"]-s["start"] for s in small]
    evidence.append(dict(shape=r["shape"],db=r["db_source"],core=r["core"],core_type=r["core_type"],
                         store_count=len(small)*2,peak_concurrent_rstd=peak,max_rstd_destinations_per_128B=max(groups.values()),
                         critical_stores=stores,critical_pair_overlap_ticks=overlap,
                         rstd_duration_ticks=dict(min=min(latencies),median=median(latencies),max=max(latencies)),
                         critical_exposed_MTE3_us=r["phase_us"]["MTE3"]))
(OUT/"writer-evidence.json").write_text(json.dumps(evidence,indent=2)+"\n")
