"""Compare weighted device quotes for the same frozen rank-local calls and precision."""

import argparse
import csv
import gzip
import json
from collections import Counter
from math import isclose
from pathlib import Path
from statistics import median


def load_ledgers(directory):
    audit = json.loads((directory / "audit.json").read_text())
    if audit["sampled"] or audit["failures"] or audit["attempts"] != audit["priced"]:
        raise ValueError(f"Incomplete replay in {directory}")
    with gzip.open(directory / "quotes.json.gz", "rt") as stream:
        dataset = json.load(stream)
    with (directory / "operators.csv").open() as stream:
        by_operator = {r["raw_name"]: r for r in csv.DictReader(stream)}
    return audit, dataset, by_operator


def compare(before, after):
    old, lhs, before_ops = load_ledgers(before)
    new, rhs, after_ops = load_ledgers(after)
    for field in ("attempts", "mode", "input_hashes", "chip_sha256"):
        if old[field] != new[field]:
            raise ValueError(f"Different {field} invalidates paired comparison")
    if lhs["quotes"].keys() != rhs["quotes"].keys() or lhs["fields"] != rhs["fields"]:
        raise ValueError("Price schemas or input signatures differ")
    times = []
    changes = Counter()
    for key, row in rhs["quotes"].items():
        other = lhs["quotes"][key]
        op = row[0]
        if op != other[0] or row[4:8] != other[4:8]:
            raise ValueError(f"Call identity or logical work changed for {op} ({key})")
        if not isclose(other[1], row[1], rel_tol=1e-12, abs_tol=1e-12):
            changes[op] += 1
        times.append(row[-1])
    operators = []
    for op, count in sorted(changes.items()):
        base, current = before_ops[op], after_ops[op]
        old_us, new_us = float(base["serial_us"]), float(current["serial_us"])
        operators.append(
            dict(
                op=op,
                signatures_changed=count,
                signatures=int(current["signatures"]),
                nodes=int(current["nodes"]),
                old_serial_us=old_us,
                new_serial_us=new_us,
            )
        )
    times.sort()
    return dict(
        old_model=old["model_sha256"],
        new_model=new["model_sha256"],
        input_hashes=new["input_hashes"],
        chip_sha256=new["chip_sha256"],
        mode=new["mode"],
        attempts=new["attempts"],
        priced=new["priced"],
        changed=sum(changes.values()),
        nodes=sum(int(r["nodes"]) for r in after_ops.values()),
        changed_ops=operators,
        query_us=dict(
            median=median(times),
            p99=times[(len(times) - 1) * 99 // 100],
            maximum=times[-1],
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(compare(args.before, args.after), indent=2) + "\n"
    )
