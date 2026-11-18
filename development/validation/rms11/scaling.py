"""Measure the critical write union and SIMD overlap along the frozen width ladder."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/"results/v0.11/rms"
rows = json.loads((OUT/"rows.json").read_text())
series = []
for r in rows:
    t, d = r["shape"]
    if r["role"] != "validation" or d != 4096:
        continue
    events = defaultdict(Counter)
    stores = []
    with sqlite3.connect((ROOT/r["db_source"]).resolve().as_uri()+"?mode=ro", uri=True) as db:
        cursor = db.execute("""SELECT ExecInstrTickStart,ExecInstrTickEnd,
            json_extract(ExtendParams,'$.queue_type'),json_extract(ExtendParams,'$.processed_bytes')
            FROM ExecutedInstructions WHERE ExecInstrCoreId=? AND CoreTypeId=? AND ExecInstrIsDurationExcluded=0""",
            (r["core"],r["core_type"]))
        for start,end,queue,size in cursor:
            if queue not in ("MTE3","SIMD"):
                continue
            events[start][queue] += 1
            events[end][queue] -= 1
            if queue == "MTE3" and size:
                stores.append(dict(bytes=size,duration_ticks=end-start))
    active = Counter()
    last = busy = overlap = 0
    for tick, changes in sorted(events.items()):
        if active["MTE3"]:
            busy += tick-last
            if active["SIMD"]:
                overlap += tick-last
        active.update(changes)
        last = tick
    assert abs((busy-overlap)/1650-r["phase_us"]["MTE3"]) < 1e-8
    series.append(dict(shape=r["shape"],active_aiv=r["active_aiv"],resident=r["prior_aiv"],output_kib=t*(2*d+4)/1024,
        ca_us=r["reference_us"],predicted_us=r["predicted_us"],v10_us=r["previous_us"],MTE3_union_ticks=busy,
        SIMD_overlap_ticks=overlap,MTE3_exposed_us=r["phase_us"]["MTE3"],MTE3_predicted_us=r["prediction"]["stages_us"]["MTE3"],
        last_core_writes=stores))
(OUT/"scaling.json").write_text(json.dumps(series,indent=2)+"\n")
