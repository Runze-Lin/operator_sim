"""Freeze rank-local LayerNorm forecasts and replay a strided production capture on two CA SKUs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

from calibration.mx_matmul.collect import capture, MEMORY_CONFIG
from calibration.system_vector_ca.report import tensor
from operator_model.system_routes import estimate_full_features

ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / "results/v2.4/layer-norm-freeze.json"
# GLM-5.3 strided views, one contiguous component control, one wider sensitivity test.
SHAPES = [
    (1, 128, 160),
    (9, 128, 160),
    (512, 128, 160),
    (512, 128, 128),
    (32, 256, 288),
]


def forecast(sku, dimensions):
    rows, width, stride = dimensions
    x = tensor((rows, width), "bfloat16")
    x.stride = (stride, 1)
    gamma = tensor((width,), "bfloat16")
    y = tensor(
        (
            rows,
            width,
        ),
        "bfloat16",
    )
    f = SimpleNamespace(
        raw_name="inference::indexer_layer_norm",
        inputs=(x, gamma, gamma),
        outputs=(y,),
        attributes=(("semantic", (("epsilon", 1e-6), ("normalized_shape", (width,)))),),
    )
    return {
        m: estimate_full_features(f, chip=sku, vector_model=m).latency_us
        for m in ("auto", "instruction_classes", "calibrated")
    }


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("action", choices=("freeze", "collect"))
    a.add_argument("--raw", type=Path)
    a.add_argument("--probe", type=Path)
    a.add_argument("--workers", type=int, default=3)
    a.add_argument(
        "--strided-multirow",
        action="store_true",
        help="Only collect multi-row input slices",
    )
    a.add_argument("--rows", type=int, nargs="+", help="Filter a frozen row count for retry")
    a.add_argument("--contiguous", action="store_true", help="Only replay the contiguous control")
    args = a.parse_args()
    if args.action == "freeze":
        rows = [
            dict(chip=sku, shape=d, model_us=forecast(sku, d))
            for d in SHAPES
            for sku in MEMORY_CONFIG
        ]
        with FREEZE.open("x") as out:
            json.dump(rows, out, indent=2)
            out.write("\n")
    else:

        def run(case):
            sku, d = case["chip"], case["shape"]
            if args.contiguous and d[1]!=d[2]: return
            if args.rows and d[0] not in args.rows:
                return
            if args.strided_multirow and (d[0] == 1 or d[1] == d[2]):
                return
            if forecast(sku, d) != case["model_us"]:
                raise ValueError("prediction changed")
            capture(
                case,
                "layer_norm",
                sha256(FREEZE.read_bytes()).hexdigest(),
                args.raw.resolve() / ("-".join(map(str, [sku, *d]))),
                args.probe,
                ROOT / "calibration/catalog_ca/layer_norm_launcher.sh",
                8,
            )

        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(run, json.loads(FREEZE.read_text())))
