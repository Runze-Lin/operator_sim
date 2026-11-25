"""Measure single in-process API queries and reproduce the stored holdout forecasts."""

import argparse
import json
from functools import partial
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from types import SimpleNamespace

from calibration.catalog_ca.layer_norm import layer_features
from calibration.system_vector_ca.report import tensor
from operator_model import __version__, estimate
from operator_model.system_routes import estimate_full_features

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/v2.5"


def queries():
    for row in json.loads((DATA / "layer-norm-freeze.json").read_text()):
        f = layer_features(row["shape"])
        yield (
            "Indexer LayerNorm",
            row,
            partial(estimate_full_features, f, chip=row["chip"], vector_model="auto"),
        )
    for row in json.loads((DATA / "add-rms-freeze.json").read_text()):
        r, d = row["shape"]
        x = tensor((r, d), "bfloat16")
        f = SimpleNamespace(
            raw_name="npu::npu_add_rms_norm",
            inputs=(x, x, tensor((d,), "bfloat16")),
            outputs=(x, tensor((r, 1), "float32"), x),
            attributes=(),
        )
        yield (
            "AddRmsNorm",
            row,
            partial(estimate_full_features, f, chip=row["chip"], vector_model="auto"),
        )
    for name in (
        "mx8-frozen.json",
        "mx8-prefill-frozen.json",
        "mx8-extension-frozen.json",
    ):
        for row in json.loads((DATA / name).read_text())["rows"]:
            if row["chip"] != "Ascend970DT":
                yield (
                    "MXFP8 Matmul",
                    row,
                    partial(
                        estimate,
                        "matmul",
                        tuple(row["shape"]),
                        {"weight_dtype": "mxfp8"},
                        dtype="mxfp8",
                        chip=row["chip"],
                    ),
                )


def benchmark(out, repeats):
    rows = []
    for op, case, query in queries():
        expected = query().latency_us
        if expected is None:
            raise ValueError(f"Unpriced query {op}, {case}")
        samples = []
        for _ in range(repeats):
            start = perf_counter_ns()
            result = query()
            samples.append((perf_counter_ns() - start) / 1000)
            if result.latency_us != expected:
                raise ValueError("Non-reproducible device estimate")
        rows.append(
            dict(
                op=op,
                shape=case["shape"],
                chip=case["chip"],
                model_us=expected,
                extrapolated_970dt_us=query(chip="Ascend970DT").latency_us,
                median_host_us=median(samples),
                maximum_host_us=max(samples),
            )
        )
    out.write_text(
        json.dumps(dict(version=__version__, repeats=repeats, rows=rows), indent=2)
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    benchmark(args.out, args.repeats)
