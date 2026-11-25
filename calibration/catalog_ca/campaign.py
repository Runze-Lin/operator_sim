"""Freeze BF16 AddRmsNorm prices, collect both SKUs, and score qualified full windows."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

from calibration.mx_matmul.collect import MEMORY_CONFIG, capture
from calibration.system_vector_ca.receipt import receipt
from calibration.system_vector_ca.report import tensor
from operator_model.system_routes import estimate_full_features

DEV = ((1, 4096), (9, 512), (32, 2048))
HOLD = (
    (1, 1024),
    (4, 1536),
    (8, 4096),
    (15, 1024),
    (16, 2048),
    (33, 1024),
    (64, 512),
    (65, 256),
    (128, 1024),
    (257, 128),
)
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/v2.4"


def forecast(sku, shape):
    rows, width = shape
    x = tensor(shape, "bfloat16")
    f = SimpleNamespace(
        raw_name="npu::npu_add_rms_norm",
        inputs=(x, x, tensor((width,), "bfloat16")),
        outputs=(x, tensor((rows, 1), "float32"), x),
        attributes=(),
    )
    prices = {}
    for mode in ("auto", "calibrated", "instruction_classes"):
        e = estimate_full_features(f, chip=sku, vector_model=mode)
        if e.latency_us is None:
            raise ValueError(e.reason)
        prices[mode + "_us"] = e.latency_us
    return prices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "collect", "score"))
    parser.add_argument("--raw", type=Path)
    parser.add_argument(
        "--retry-raw",
        type=Path,
        help="separate directory for retrying completed failures",
    )
    parser.add_argument("--minutes", type=int, default=5)
    local_launcher = ROOT / "scripts/cann-run.sh"
    parser.add_argument("--launcher", type=Path, default=local_launcher if local_launcher.is_file() else None,
                        help="optional command wrapper sourcing CANN; otherwise use the active environment")
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--opp", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    path = RESULTS / "add-rms-freeze.json"
    if args.action == "freeze":
        rows = [
            dict(chip=c, shape=s, **forecast(c, s)) for c in MEMORY_CONFIG for s in HOLD
        ]
        with path.open("x") as f:
            json.dump(rows, f, indent=2)
            f.write("\n")
        return
    frozen = json.loads(path.read_text())
    if (
        not args.raw
        or (args.action == "collect" and not args.probe)
        or (args.action == "score" and not args.opp)
    ):
        parser.error("Expected --raw plus --probe for collect or --opp for score")

    def run(row):
        c, s = row["chip"], row["shape"]
        directory = args.raw.resolve() / f"{c}-{s[0]}-{s[1]}"
        if any(row[k] != v for k, v in forecast(c, s).items()):
            raise ValueError("Published forecasts changed after freeze")
        if args.retry_raw:
            retry = args.retry_raw.resolve() / directory.name
            if args.action == "collect":
                cost = directory / "cost.json"
                if not cost.exists() or retry.exists():
                    return
                previous = json.loads(cost.read_text())
                if (
                    previous["exit_code"] == 0
                    and "Profiling data parse finished."
                    in (directory / "run.log").read_text()
                ):
                    return
                directory = retry
            elif retry.exists():
                directory = retry
        if args.action == "collect":
            capture(
                row,
                "add_rms_norm",
                sha256(path.read_bytes()).hexdigest(),
                directory,
                args.probe,
                args.launcher,
                args.minutes,
            )
            return
        r = receipt(directory, args.opp)
        if (
            r["shape"] != s
            or r["chip"] != c
            or r["checked"] != 2 * s[0] * s[1] + s[0]
            or (directory / "freeze.sha256").read_text().strip()
            != sha256(path.read_bytes()).hexdigest()
        ):
            raise ValueError("Receipt differs from frozen case")
        return dict(row, **{k: v for k, v in r.items() if k not in row})

    with ThreadPoolExecutor(args.workers) as pool:
        verified = list(pool.map(run, frozen))
    if args.action == "score":
        (RESULTS / "add-rms-independent.json").write_text(
            json.dumps(verified, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
