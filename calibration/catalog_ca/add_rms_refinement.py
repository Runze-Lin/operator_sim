"""Collect or verify pre-frozen BF16 AddRmsNorm holdout cases (two simultaneous CA jobs)."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

from calibration.catalog_ca.campaign import ROOT, forecast
from calibration.mx_matmul.collect import MEMORY_CONFIG, capture
from calibration.system_vector_ca.receipt import receipt

DEST = ROOT / "results/v2.5"
FROZEN = DEST / "add-rms-freeze.json"
HOLDOUT = (
    (1, 7168),
    (2, 5120),
    (7, 1536),
    (12, 2560),
    (24, 1024),
    (48, 768),
    (63, 640),
    (96, 512),
    (129, 768),
    (192, 1536),
)


def prior_forecast(chip, shape):
    """Evaluate the shared v2.4 Tiles schedule that preceded the dedicated BF16 close."""
    from types import SimpleNamespace
    from operator_model import load_chip
    from operator_model.system_routes import vf
    from operator_model.system_routes.catalog_vector import _plan
    from operator_model.system_routes.vector import _schedule
    from calibration.system_vector_ca.report import tensor

    rows, width = shape
    x = tensor(shape, "bfloat16")
    f = SimpleNamespace(
        raw_name="npu::npu_add_rms_norm",
        inputs=(x, x, tensor((width,), "bfloat16")),
        outputs=(x, tensor((rows, 1), "float32"), x),
        attributes=(),
    )
    out = {}
    for mode in ("auto", "calibrated", "instruction_classes"):
        with vf.selection(mode):
            out[mode + "_us"] = _schedule(_plan(f, {}), load_chip(chip)).latency_us
    return out


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("freeze", "collect", "score"))
    cli.add_argument("raw", type=Path, nargs="?")
    cli.add_argument("--probe", type=Path)
    cli.add_argument("--launcher", type=Path)
    cli.add_argument("--opp", type=Path)
    a = cli.parse_args()
    if a.action == "freeze":
        rows = [
            dict(
                shape=shape,
                chip=sku,
                v24=prior_forecast(sku, shape),
                v25=forecast(sku, shape),
                **{"970dt_extrapolated_us": forecast("Ascend970DT", shape)["auto_us"]},
            )
            for shape in HOLDOUT
            for sku in MEMORY_CONFIG
        ]
        with FROZEN.open("x") as f:
            json.dump(rows, f, indent=2)
            f.write("\n")
        return
    if (
        not a.raw
        or a.action == "collect"
        and not a.probe
        or a.action == "score"
        and not a.opp
    ):
        cli.error("Specify RAW and either --probe (collect) or --opp (score)")
    rows = json.loads(FROZEN.read_text())
    digest = sha256(FROZEN.read_bytes()).hexdigest()

    def run(r):
        sku = r["chip"]
        s = r["shape"]
        folder = a.raw.resolve() / f"{sku}-{s[0]}-{s[1]}"
        if forecast(sku, s) != r["v25"] or prior_forecast(sku, s) != r["v24"]:
            raise ValueError("Frozen forecast changed")
        if a.action == "collect":
            if not folder.exists():
                capture(r, "add_rms_norm", digest, folder, a.probe, a.launcher, 4)
            return
        measured = receipt(folder, a.opp)
        if (
            measured["shape"] != s
            or measured["chip"] != sku
            or (folder / "freeze.sha256").read_text().strip() != digest
            or measured["checked"] != 2 * s[0] * s[1] + s[0]
        ):
            raise ValueError("Wrong shape, SKU, freeze or golden")
        return {**r, **measured}

    with ThreadPoolExecutor(2) as workers:
        results = list(workers.map(run, rows))
    if a.action == "score":
        (DEST / "add-rms-independent.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
