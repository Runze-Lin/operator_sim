"""Extract the serial first-fragment milestones from a development CA window."""

import argparse
import json
from pathlib import Path
from statistics import median
from calibration.mx_matmul.shards import trace_events
from calibration.msopprof import sha


def read(simulator):
    rows = []
    for path in sorted(simulator.glob("core*.cubecore*/trace.json")):
        es = [e for e in trace_events(path) if e.get("ph") == "X"]
        core_start = min(round(e["ts"] * 1650) for e in es)
        retire = max(round((e["ts"] + e["dur"]) * 1650) for e in es)
        reads = [e for e in es if "OUT_TO_L1" in e["name"]]
        matrix = [e for e in es if e["name"] == "MMAD_MX"]
        if not matrix:
            continue
        if len(matrix) != 1 or len(reads) != 4:
            raise ValueError(
                "This development extractor requires a single fragment and four scale/value descriptors"
            )
        dma = min(round(e["ts"] * 1650) for e in reads)
        ready = max(round((e["ts"] + e["dur"]) * 1650) for e in reads)
        start = round(matrix[0]["ts"] * 1650)
        end = round((matrix[0]["ts"] + matrix[0]["dur"]) * 1650)
        rows.append(
            dict(
                core=path.parent.name,
                core_start=core_start,
                setup=dma - core_start,
                first_dma=ready - dma,
                operand_ready=start - ready,
                mmad=end - start,
                drain=retire - end,
                trace_sha256=sha(path),
            )
        )
    z = min(r["core_start"] for r in rows)
    for r in rows:
        r["core_start"] -= z
    fields = ("core_start", "setup", "first_dma", "operand_ready", "mmad", "drain")
    return dict(
        unit="950 core cycles, 1650 cycles per microsecond",
        shape_MNK=(9, 256, 128),
        role="development: reanalysis of v2.4 DT validation, never scored independently here",
        summaries={
            k: {
                "min": min(r[k] for r in rows),
                "median": median(r[k] for r in rows),
                "max": max(r[k] for r in rows),
            }
            for k in fields
        },
        core_milestones=rows,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("simulator", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(read(a.simulator), indent=2) + "\n")
