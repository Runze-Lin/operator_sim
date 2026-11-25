"""Rebuild the v2.5 paired CA comparison from frozen, independently scored receipts."""

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from math import isclose

from operator_model import load_chip

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/v2.5"


def read(name):
    return json.loads((RESULTS / name).read_text())


def floor_us(op, shape, chip_name):
    c = load_chip(chip_name)
    if op == "MXFP8 Matmul":
        m, n, k = shape
        byte_count = (m + n) * (k + 2 * ((k + 63) // 64)) + 2 * m * n
        compute = 2 * m * n * k / (1e6 * c.cube_tops["8"])
    else:
        r, d = shape[:2]
        if op == "Indexer LayerNorm":
            ops, byte_count = 8 * r * d + 3 * r, 4 * r * d + 4 * d
        else:
            ops, byte_count = 5 * r * d + 3 * r, 8 * r * d + 2 * d + 4 * r
        compute = ops / (1e6 * c.vector_tops["32"])
    return max(compute, byte_count / (1000 * c.hbm_bandwidth_gbs))


def receipts():
    for r in read("add-rms-independent.json"):
        yield (
            "AddRmsNorm",
            r["shape"],
            r["chip"],
            r["v24"]["auto_us"],
            r["v25"]["auto_us"],
            r["measured_us"],
        )
    for r in read("layer-norm-independent.json"):
        yield (
            "Indexer LayerNorm",
            r["shape"],
            r["chip"],
            r["v24_auto_us"],
            r["release_model_us"]["auto"],
            r["ca_sum_kernel_windows_us"],
        )
    for r in read("mx8-validation.json")["rows"]:
        if r["qualified"]:
            yield (
                "MXFP8 Matmul",
                r["shape"],
                r["chip"],
                r["old_us"],
                r["model_us"],
                r["ca_us"],
            )


def accuracy(pairs):
    _, measured = zip(*pairs)
    return {
        "samples": len(pairs),
        "wape_pct": 100 * sum(abs(p - c) for p, c in pairs) / sum(measured),
        "max_abs_pct": 100 * max(abs(p / c - 1) for p, c in pairs),
        "median_abs_pct": 100 * median(abs(p / c - 1) for p, c in pairs),
    }


def build():
    timings = {}
    for version in ("v24", "v25"):
        bench = read(f"query-cost-{version}.json")
        expected_version = {"v24": "2.4.0", "v25": "2.5.0"}[version]
        if bench["version"] != expected_version:
            raise ValueError(f"Incorrect installed package for {version}")
        timings[version] = {
            (r["op"], r["chip"], tuple(r["shape"])): r for r in bench["rows"]
        }
    rows = []
    groups = defaultdict(list)
    for op, shape, chip, old, new, ca in receipts():
        key = tuple(shape)
        groups[op, chip].append((key, old, new, ca))
        for version, prediction in (("v24", old), ("v25", new)):
            installed = timings[version][op, chip, key]
            if not isclose(
                installed["model_us"], prediction, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(
                    f"{version} installed estimator differs from forecast for {(op, chip, key)}"
                )
        ideal = floor_us(op, shape, chip)
        rows.append(
            dict(
                op=op,
                shape="x".join(map(str, shape)),
                chip=chip,
                old_us=old,
                new_us=new,
                ca_us=ca,
                extrapolated_970dt_us=timings["v25"][op, chip, key][
                    "extrapolated_970dt_us"
                ],
                old_error_pct=100 * (old / ca - 1),
                new_error_pct=100 * (new / ca - 1),
                logical_roofline_us=ideal,
                cold_roofline_util_pct=100 * ideal / max(ideal, new),
            )
        )
    if len({(r["op"], r["shape"], r["chip"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate independent measurement")
    stats = {}
    for (op, chip), cases in sorted(groups.items()):
        values = [
            r["cold_roofline_util_pct"]
            for r in rows
            if (r["op"], r["chip"]) == (op, chip)
        ]
        stats.setdefault(op, {})[chip] = dict(
            v24=accuracy([(old, ca) for _, old, new, ca in cases]),
            v25=accuracy([(new, ca) for _, old, new, ca in cases]),
            cold_u_pct=[min(values), max(values)],
        )
    for op, summary in stats.items():
        dt = {
            key: (old, new, ca) for key, old, new, ca in groups[op, "Ascend950DT_9582"]
        }
        pr = {
            key: (old, new, ca) for key, old, new, ca in groups[op, "Ascend950PR_957b"]
        }
        pair = sorted(dt.keys() & pr.keys())
        if pair:
            speed = [[dt[k][i] / pr[k][i] for i in range(3)] for k in pair]
            summary["speedup_dt_over_pr"] = {
                v: dict(
                    **accuracy([(s[i], s[2]) for s in speed]),
                    direction_agrees=sum(
                        (s[i] > 1) - (s[i] < 1) == (s[2] > 1) - (s[2] < 1)
                        for s in speed
                    ),
                )
                for i, v in enumerate(["v24", "v25"])
            }
    if set(stats) != {"AddRmsNorm", "Indexer LayerNorm", "MXFP8 Matmul"}:
        raise ValueError("Missing independent evidence for one or more models")
    with (RESULTS / "paired-errors.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS / "accuracy.json").write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    build()
