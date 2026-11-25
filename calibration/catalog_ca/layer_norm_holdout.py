"""Freeze v2.4 and current LayerNorm forecasts, collect trace windows and score them."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import statistics
from time import perf_counter_ns
from calibration.catalog_ca.layer_norm import ROOT, forecast, layer_features
from operator_model.system_routes import estimate_full_features
from calibration.catalog_ca.layer_norm_receipt import score
from calibration.mx_matmul.collect import capture, MEMORY_CONFIG
from operator_model.chips import load_chip
from operator_model.system_routes import vf
from operator_model.system_routes.vector import Tiles, _schedule

DATA = ROOT / "results/v2.5"
FREEZE = DATA / "layer-norm-freeze.json"
SHAPES = (
    (2, 128, 160),
    (4, 128, 128),
    (8, 192, 224),
    (16, 128, 160),
    (31, 256, 256),
    (48, 256, 320),
    (63, 384, 384),
    (128, 128, 160),
    (257, 256, 256),
    (1024, 128, 160),
)


def old(shape, chip):
    rows, width, _ = shape
    n = rows * width
    t = Tiles(
        "indexer_affine_layer_norm",
        n,
        width,
        2 * n + 4 * width,
        2 * n,
        8 * n + 3 * rows,
        class_visits=dict(
            load=3 * n, arithmetic=6 * n, reduce=2 * n, rsqrt=rows, store=n
        ),
    )
    with vf.selection("auto"):
        return _schedule(t, load_chip(chip)).latency_us


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("freeze", "collect", "score"))
    cli.add_argument("--raw", type=Path)
    cli.add_argument("--probe", type=Path)
    cli.add_argument("--opp", type=Path)
    cli.add_argument("--out", type=Path, default=DATA / "layer-norm-independent.json")
    cli.add_argument("--workers", type=int, default=2)
    a = cli.parse_args()
    if a.action == "freeze":
        DATA.mkdir(parents=True, exist_ok=True)
        rows = [
            dict(chip=c, shape=s, model_us=forecast(c, s), v24_auto_us=old(s, c))
            for s in SHAPES
            for c in MEMORY_CONFIG
        ]
        with FREEZE.open("x") as f:
            json.dump(rows, f, indent=2)
            f.write("\n")
        return
    rows = json.loads(FREEZE.read_text())

    def run(row):
        c, s = row["chip"], row["shape"]
        p = a.raw.resolve() / ("-".join(map(str, [c, *s])))
        current = forecast(c, s)
        published = {**row["model_us"], "auto": row["model_us"]["instruction_classes"]}
        if current != published:
            raise ValueError("Frozen forecast has changed")
        if a.action == "collect":
            if (p / "cost.json").exists():
                return
            capture(
                row,
                "layer_norm",
                sha256(FREEZE.read_bytes()).hexdigest(),
                p,
                a.probe,
                ROOT / "calibration/catalog_ca/layer_norm_launcher.sh",
                10,
            )
            return
        result = score(row, p, a.opp, freeze=FREEZE)
        features = layer_features(s)
        walls = []
        for _ in range(101):
            t = perf_counter_ns()
            estimate_full_features(features, chip=c)
            walls.append((perf_counter_ns() - t) / 1000)
        result["release_model_us"] = published
        result["auto_query_us_median"] = statistics.median(walls[1:])
        result["auto_query_us_p95"] = sorted(walls[1:])[94]
        return result

    with ThreadPoolExecutor(a.workers if a.action == "collect" else 1) as pool:
        data = list(pool.map(run, rows))
    if a.action == "score":
        a.out.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
