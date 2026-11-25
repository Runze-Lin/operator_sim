"""Freeze same-shape old/new MXFP8 prices and verify independent single-kernel CA receipts."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re

from operator_model import estimate
from operator_model.operators.matmul import mx8_native
from calibration.mx_matmul.collect import core_windows, MEMORY_CONFIG
from calibration.msopprof import CORE, sha
from calibration.mx_matmul.shards import trace_events

ROOT = Path(__file__).resolve().parents[2]
SHAPES = ((1, 128, 64), (16, 256, 1024))
SOCS = (*MEMORY_CONFIG, "Ascend970DT")
KERNEL = "QuantBatchMatmulV3_ND_ND_mxfp8e4m3_mxfp8e4m3_float8_e8m0_bf16_high_performance"
PREFILL = ((128, 128, 256), (256, 128, 512))
EXTENSION = ((1, 1024, 128), (9, 512, 1024), (11, 1024, 256),
             (128, 256, 512), (2176, 320, 64), (9, 896, 7168))


def freeze(shapes):
    rows = []
    for shape in shapes:
        for soc in SOCS:
            result = estimate(
                "matmul", shape, {"weight_dtype": "mxfp8"}, chip=soc, dtype="mxfp8"
            )
            rows.append(
                dict(
                    shape=shape,
                    chip=soc,
                    model_us=result.latency_us,
                    active=result.active_aic,
                    pages=result.projection["target_plan"]["k_pages"],
                )
            )
    return dict(
        model_sha256=sha(Path(mx8_native.__file__)),
        rows=rows,
    )


def read(folder, row, opp):
    log = (folder / "run.log").read_text()
    m, n, k = row["shape"]
    soc = row["chip"]
    golden = f"correctness count={m * n} bad=0" in log and "SMOKE_PASS" in log
    done = (
        golden
        and "All task success." in log
        and "timeout has reached" not in log
        and "Profiling data parse finished." in log
    )
    found = re.findall(r"Start profiling on kernel: (\S+)", log)
    record = dict(
        **row,
        log_sha256=sha(folder / "run.log"),
        golden=golden,
        qualified=False,
        kernel=found,
        api="aclnnQuantMatmulV5",
        dtype="mxfp8 × mxfp8, E8M0 K32, BF16 output",
        ca_us=None,
    )
    profiles = list((folder / "profile").glob("OPPROF_*"))
    if profiles:
        start = datetime.strptime(
            profiles[0].name.split("_")[1], "%Y%m%d%H%M%S"
        ).timestamp()
        record.update(
            wall_s=round((folder / "run.log").stat().st_mtime - start, 2),
            profiler_timeout="timeout has reached" in log,
        )
    if not done:
        return dict(record, reason="timeout or incomplete profile/golden")
    if not (
        f"route=a8w8 M={m} N={n} K={k}" in log
        and f"runtime_soc={soc}" in log
        and MEMORY_CONFIG[soc] in log
    ):
        raise ValueError(f"Captured geometry or simulator configuration differs in {folder}")
    if len(found) != 1 or not re.fullmatch(KERNEL + r"_\d+", found[0]):
        return dict(record, reason="captured ELF name differs from the frozen MXFP8 contract")
    (base,) = (folder / "profile").glob("OPPROF_*")
    elf = base / "dump/aicore_binary.o"
    installed = (
        opp / "built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_nn/quant_batch_matmul_v3"
        / (KERNEL + ".o")
    )
    if sha(elf) != sha(installed):
        raise ValueError(f"Dumped ELF differs from installed OPP in {folder}")
    roster = {n: float(t) for n, t in CORE.findall(log)}
    if not roster:
        raise ValueError(f"Missing profiler roster in {folder}")
    cores, hashes = core_windows(base / "simulator", roster)
    first = min(c["first_tick"] for c in cores.values())
    dt = (max(c["retire_tick"] for c in cores.values()) - first) / 1650
    rounds, issue, fragments = [], [], {}
    for path in sorted((base / "simulator").glob("core*.cubecore*/trace.json")):
        times = []
        for event in trace_events(path):
            if event.get("ph") != "X" or event.get("name") != "MMAD_MX":
                continue
            times.append(round(event["ts"] * 1650) - first)
            detail = json.loads(event["args"]["detail"])
            key = "x".join(str(detail[k]) for k in ("m", "n", "k"))
            fragments[key] = fragments.get(key, 0) + 1
        if not times:
            raise ValueError(f"Missing MX instruction on {path.parent.name}")
        rounds.append(len(times))
        issue.append(min(times))
    return dict(
        record,
        qualified=True,
        ca_us=dt,
        cores=len(cores),
        issued_mmad_mnk_histogram=fragments,
        mmads_per_core_range=[min(rounds), max(rounds)],
        first_mmad_tick_range=[min(issue), max(issue)],
        elf_sha256=sha(elf),
        trace_sha256=hashes,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("freeze", "report"))
    p.add_argument("output", type=Path)
    p.add_argument("--scratch", type=Path)
    p.add_argument("--freeze", type=Path)
    p.add_argument("--prefill", action="store_true")
    p.add_argument("--baseline", type=Path)
    p.add_argument(
        "--opp", type=Path,
        default=Path(os.environ.get("ASCEND_OPP_PATH") or ROOT / "toolchains/cann/9.2.1/cann-9.2.1/opp"),
        help="installed CANN 9.2.1 OPP root",
    )
    p.add_argument("--extension", action="store_true")
    p.add_argument("--extra-freeze", type=Path, action="append", default=[])
    p.add_argument("--extra-baseline", type=Path, action="append", default=[])
    a = p.parse_args()
    if a.action == "report" and not (a.freeze and a.baseline and a.scratch):
        p.error("report requires --freeze, --baseline and --scratch")
    if a.action == "freeze":
        obj = freeze(EXTENSION if a.extension else PREFILL if a.prefill else SHAPES)
        if a.extension:
            obj["selection"] = ("Six new geometries selected after the first four were scored; "
                                "same predictor SHA, forecasts sealed before this second campaign.")
    else:
        pairs = [(a.freeze, a.baseline)]
        if a.extra_freeze or a.extra_baseline:
            if len(a.extra_freeze) != len(a.extra_baseline):
                p.error("--extra-freeze and --extra-baseline must be supplied together")
            pairs.extend(zip(a.extra_freeze, a.extra_baseline))
        rows, extrapolations, seen = [], [], set()
        model_hashes = set()
        for current_path, old_path in pairs:
            frozen, previous = (json.loads(path.read_text()) for path in (current_path, old_path))
            model_hashes.add(frozen["model_sha256"])
            if len(model_hashes) > 1:
                raise ValueError("Independent campaigns used different predictor states")
            old_prices = {(r["chip"], tuple(r["shape"])): r["model_us"] for r in previous["rows"]}
            for row in frozen["rows"]:
                key = (row["chip"], tuple(row["shape"]))
                if key in seen:
                    raise ValueError(f"Repeated geometry in independent campaigns: {key}")
                seen.add(key)
                if row["chip"] == "Ascend970DT":
                    extrapolations.append(row)
                    continue
                folder = a.scratch / (row["chip"] + "-" + "x".join(map(str, row["shape"])))
                data = read(folder, row, a.opp)
                old_us = old_prices[key]
                ca = data["ca_us"]
                data.update(
                    old_us=old_us,
                    signed_error_pct=(row["model_us"] / ca - 1) * 100 if ca else None,
                    old_signed_error_pct=(old_us / ca - 1) * 100 if ca else None,
                )
                rows.append(data)
        obj = dict(
            freeze_sha256=[sha(x) for x, _ in pairs],
            old_freeze_sha256=[sha(x) for _, x in pairs],
            model_sha256=next(iter(model_hashes)),
            role="post-freeze independent geometry, device latency in microseconds",
            rows=rows, extrapolations_970dt=extrapolations,
        )
    with a.output.open("x") as out:
        json.dump(obj, out, indent=2)
        out.write("\n")
