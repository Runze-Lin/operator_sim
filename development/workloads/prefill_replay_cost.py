"""Benchmark distinct public backend queries in the large-prefill contracts."""
import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter_ns

from operator_model import estimate
from .prefill_shapes import CHIPS


def benchmark(directory, repeats=3):
    keys = set()
    with gzip.open(directory / "contracts.jsonl.gz", "rt") as f:
        for line in f:
            case = json.loads(line)
            for r in case["contracts"]:
                if "standalone_call" in r:
                    keys.add(json.dumps(r["standalone_call"], sort_keys=True))
    rows = {}
    for key in sorted(keys):
        c = json.loads(key)
        for chip in CHIPS:
            for _ in range(repeats):
                before = perf_counter_ns()
                result = estimate(c["op"], c["shape"], c["attrs"], chip=chip,timing_mode="resource")
                ms = (perf_counter_ns()-before)/1e6
                group = (c["op"],"timed" if result.latency_us is not None else "work_only")
                rows.setdefault(group,[]).append((ms,chip,c["shape"]))
    summary=[]
    for (op,kind),values in sorted(rows.items()):
        values.sort()
        slowest,chip,shape = values[-1]
        summary.append(dict(op=op,kind=kind,queries=len(values),wall_p50_ms=values[len(values)//2][0],
            wall_max_ms=slowest,slowest_shape=shape,slowest_chip=chip))
    return dict(scope="Same process after package import, no exporter price cache; estimate resource per shape+attrs+SKU. Package parameter caches can warm after first query.",
        repeats=repeats,unique_op_shape_attrs=len(keys),chips=list(CHIPS),summary=summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    a=parser.parse_args()
    print(json.dumps(benchmark(a.directory),indent=2))
