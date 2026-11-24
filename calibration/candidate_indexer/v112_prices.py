"""Record a QSLI release's public prices for the old controls and two new CA geometries."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import operator_model
from operator_model import estimate

DT = "Ascend950DT_9582"
PR = "Ascend950PR_957b"
FUTURE = "Ascend970DT"
LONG = 1 << 20

# (label, queries, context, candidate cap, allocated PA columns, CA role)
CASES = (
    ("small", 1, 1024, 81, 8, "previously observed small table"),
    ("small_wide", 1, 1024, 81, 8201, "previously observed wide table"),
    ("split_row", 1, LONG, 2048, 8201, "DT and PR development trace"),
    ("shared_wave", 32, LONG, 2048, 8201, "DT previous holdout, PR new holdout"),
    ("varied_two_waves", 64, LONG, 2048, 8201, "DT new holdout with row-dependent pages"),
    ("last_chunk", 4096, LONG, 2048, 8201, "extrapolation only"),
)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record():
    package = Path(operator_model.__file__).parent
    sources = [package / "operators/attention/flash_sparse.py", package / "operators/attention/indexer.py",
               package / "operators/attention/candidate_indexer.py"]
    sources += list((package / "data/parameters").glob("flash_qsli*.json"))
    sources += [package / "data/chips" / f"{sku}.json" for sku in (DT, PR, FUTURE)]
    rows = []
    for label, queries, length, limit, columns, role in CASES:
        shape = (1, queries, 32, 128, length, 2048, 512)
        attrs = dict(implementation="a5_qsli_projection", table_pages=columns,
                     pa_block_size=128, candidate_limit=limit)
        for sku in (DT, PR, FUTURE):
            def quote(memory):
                return estimate("candidate_filtered_indexer", shape, attrs, chip=sku, dtype="mxfp4",
                                memory_policy=memory)
            model, cold = quote("ca_window"), quote("cold_io")
            rows.append(dict(case=label, sku=sku, role="extrapolation only" if sku == FUTURE else role,
                             shape=shape, attrs=attrs, latency_us=model.latency_us, stages=model.projection,
                             logical_time_factor=model.rates["logical_roofline"]["time_factor"],
                             cold_U_pct=100*cold.rates["roofline"]["utilization"]))
    return dict(version=operator_model.__version__, utc=datetime.now(timezone.utc).isoformat(),
                sha256={str(p.relative_to(package)): file_hash(p) for p in sources}, prices=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    # A frozen prediction file is immutable. Choose a new path for a later replay.
    with args.output.open("x") as stream:
        json.dump(record(), stream, indent=2, ensure_ascii=False)
        stream.write("\n")
