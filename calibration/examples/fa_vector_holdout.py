"""Freeze and validate the initial FA cohort or the separately registered long-KV case."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/v2.2/fa-validation"
RAW = ROOT / "runs/fa-vector-v22"
SHAPES = [(60, 448, 64, 1, 1), (63, 640, 128, 1, 1), (64, 768, 64, 1, 1)]
SKUS = ("Ascend950DT_9582", "Ascend950PR_957b")
LONG = False
HOST = ROOT / "build/smoke/three_ops"
SECONDS, MIB = 360, 1100


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def predictions():
    from operator_model import estimate

    rows = []
    for chip in (*SKUS, "Ascend970DT"):
        for shape in SHAPES:
            row = dict(chip=chip, shape=shape)
            for mode in ("calibrated", "instruction_classes"):
                r = estimate(
                    "fa", shape, {"vector_model": mode}, dtype="bf16", chip=chip
                )
                diag = r.projection["timing_diagnostics"]
                (wave,) = diag["wave_types"]
                assert wave["waves"] == 1 and wave["occupied"] == 1
                page_count = diag["target_plan"]["iterations"]
                pages = [
                    (wave["first"], 1),
                    (wave["last"], 1),
                    (wave["repeat"], page_count - 2),
                ]
                assert page_count >= 3
                vf = sum((v["vec1"] + v["vec2"]) * num for v, num in pages)
                row[mode] = dict(
                    us=r.latency_us,
                    vf_cycles=vf,
                    mmad=diag["mmads_total"],
                    tile=diag["target_plan"]["tile"],
                    active_aic=r.active_aic,
                    active_aiv=r.active_aiv,
                    repeat_bound=wave["repeat"]["steady_bottleneck"],
                    repeat_demands=wave["repeat"]["demands_cycles"],
                )
            rows.append(row)
    return rows


def freeze(oldsrc):
    dest = OUT / "predictions.json"
    if dest.exists():
        raise FileExistsError(f"Existing freeze: {dest}. Select a new --output-dir to run a new study.")
    known = {
        tuple(r["shape"])
        for r in map(json.loads, (ROOT / "calibration/reference/windows.jsonl").open())
        if r["op"] == "fa"
    }
    assert not set(SHAPES) & known

    def prices(source):
        env = {**os.environ, "PYTHONPATH": str(source)}
        data = subprocess.check_output(
            [sys.executable, __file__, "prices", *(["--long"] if LONG else [])], env=env
        )
        return json.loads(data)

    old, new = prices(oldsrc), prices(ROOT / "src")
    for prior, row in zip(old, new):
        assert prior["shape"] == row["shape"] and prior["chip"] == row["chip"]
        assert prior["calibrated"] == row["calibrated"]
        row["v21_classes"] = prior["instruction_classes"]
    paths = [
        ROOT / "src/operator_model/operators/attention" / f
        for f in ("resources.py", "vector_classes.py")
    ]
    paths += list((ROOT / "src/operator_model/data/parameters").glob("fa_vector*.json"))
    dump(
        dest,
        dict(
            utc=datetime.now(timezone.utc).isoformat(),
            predictions=[r for r in new if r["chip"] in SKUS],
            extrapolated_970=[r for r in new if r["chip"] == "Ascend970DT"],
            hashes={str(p.relative_to(ROOT)): digest(p) for p in [*paths, HOST]},
            baseline="v2.1 651cd72",
            eligibility="v2.1 intermediate full pages Vector-limited, cold noncausal BF16 SBH",
            per_case_limits={"seconds": SECONDS, "MiB": MIB},
            clock_cycles_per_us=1650,
        ),
    )


def intervals(windows):
    end = -math.inf
    total = 0
    for a, b in sorted(windows):
        total += max(0, b - max(a, end))
        end = max(b, end)
    return total


def receipt(folder, row):
    log = (folder / "run.log").read_text()
    q, n, d, b, h = row["shape"]
    mem = "hbm3_8000M.ini" if row["chip"] == SKUS[0] else "lpddr5_6400M_bailu.ini"
    (gold,) = re.findall("^correctness .*$", log, re.M)
    (kernel,) = re.findall(r"Start profiling on kernel: (\S+)", log)
    assert all(
        x in log
        for x in (
            "All task success.",
            "Profiling data parse finished.",
            "SMOKE_PASS",
            mem,
        )
    )
    assert re.findall(r"^runtime_soc=(\S+)", log, re.M) == [row["chip"]]
    assert (
        f"elements={q * b * h * d} " in gold
        and "bad=0 " in gold
        and kernel.startswith("FlashAttentionScore_")
    )
    assert not re.search(
        r"\[ERROR\].*(?:execute_set_flag|execute_wait_flag|deadlock|invalid instruction)",
        log,
    )
    (trace,) = folder.glob("profile/OPPROF_*/simulator/trace.json")
    data = json.loads(trace.read_bytes())
    spans = defaultdict(list)
    counts = defaultdict(Counter)
    ends = Counter()
    timing = {}
    mmad = Counter()
    simd, vf = defaultdict(list), defaultdict(list)
    for e in data["traceEvents"]:
        if e.get("ph") != "X":
            continue
        core, name, t, dur = e["pid"], e["name"], e["ts"], e["dur"]
        assert math.isfinite(t + dur) and dur >= 0
        begin, end = timing.get(core, (t, t))
        timing[core] = (min(begin, t), max(end, t + dur))
        if name == "END":
            ends[core] += 1
        if name == "MMAD":
            meta = json.loads(e["args"]["detail"])
            mma_type = tuple(
                meta.get(k, "unreported")
                for k in ["matrix_a_dtype", "matrix_b_dtype", "matrix_c_dtype"]
            )
            mmad[",".join(mma_type)] += 1
        if dur > 0 and "veccore" in core and e["tid"].startswith("SIMD"):
            simd[core].append((t, t + dur))
        if dur > 0 and "veccore" in core and name == "VF":
            vf[core].append((t, t + dur))
        if dur > 0 and e["tid"].startswith("RVEC") and name.startswith("RV_"):
            spans[core].append((t, t + dur))
            counts[core][name] += 1
    assert set(timing) == set(ends) and set(ends.values()) == {1} and spans
    assert "bf16,bf16,f32" in mmad
    roster = dict(
        (k, float(v))
        for k, v in re.findall(
            r"^(core\d+\.(?:vec|cube)core\d+)\s+([\d.]+)\s+[\d.]+\s*$", log, re.M
        )
    )
    assert roster.keys() == timing.keys()
    assert all(
        abs(roster[k] - (z - a)) <= max(0.03, 0.02 * roster[k])
        for k, (a, z) in timing.items()
    )
    union = {lane: round(1650 * intervals(w)) for lane, w in spans.items()}
    return dict(
        ca_us=max(x[1] for x in timing.values()) - min(x[0] for x in timing.values()),
        busy_rv_cycles=max(union.values()),
        positive_rv_union_cycles_by_aiv=union,
        positive_rv_counts_by_aiv=counts,
        all_lanes=sorted(ends),
        mmad=mmad,
        simd_union_cycles={k: round(1650 * intervals(v)) for k, v in simd.items()},
        vf_descriptor_union_cycles={
            k: round(1650 * intervals(v)) for k, v in vf.items()
        },
        api="aclnnFlashAttentionScoreV2",
        provider="CANN 9.2.1 ops-transformer",
        kernel=kernel,
        binary_sha256=digest(trace.parents[1] / "dump/aicore_binary.o"),
        trace_sha256=digest(trace),
        gold=gold,
        seconds=json.loads((folder / "cost.json").read_text())["seconds"],
    )


def run_one(row):
    chip = row["chip"]
    shape = row["shape"]
    tag = chip + "-" + "x".join(map(str, shape[:3]))
    folder = RAW / tag
    if not (folder / "cost.json").exists():
        folder.mkdir(parents=True, exist_ok=False)
        wrapper = ROOT / "scripts/cann-run.sh"
        cmd = (["bash", str(wrapper)] if wrapper.is_file() else []) + [
            "msopprof",
            "simulator",
            "--soc-version=" + chip,
            "--dump=true",
            "--launch-count=20",
            "--timeout=6",
            "--output=" + str(folder / "profile"),
            str(HOST),
            "fa",
            *map(str, shape),
        ]
        start = time.monotonic()
        env = {**os.environ, "GOMP_SPINCOUNT": "0", "OMP_WAIT_POLICY": "PASSIVE"}
        with (folder / "run.log").open("w") as log:
            job = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=folder,
                env=env,
            )
            while job.poll() is None:
                size = 0
                for entry in folder.rglob("*"):
                    try:
                        size += entry.stat().st_size
                    except FileNotFoundError:
                        pass
                if time.monotonic() - start > SECONDS or size > MIB * 2**20:
                    os.killpg(job.pid, signal.SIGKILL)
                    job.wait()
                    break
                time.sleep(2)
        dump(
            folder / "cost.json",
            {"seconds": time.monotonic() - start, "returncode": job.returncode},
        )
    try:
        assert json.loads((folder / "cost.json").read_text())["returncode"] == 0
        data = receipt(folder, row)
        result = dict(status="qualified", chip=chip, shape=shape, **data)
    except (AssertionError, KeyError, ValueError, OSError) as exc:
        result = dict(status="unqualified", chip=chip, shape=shape, reason=repr(exc))
    dump(OUT / (tag + ".json"), result)
    print(tag, result["status"], flush=True)


def evaluate():
    manifest = json.loads((OUT / "predictions.json").read_text())
    records = []
    for row in manifest["predictions"]:
        case = row["chip"] + "-" + "x".join(map(str, row["shape"][:3]))
        receipt = json.loads((OUT / (case + ".json")).read_text())
        if receipt["status"] != "qualified":
            records.append(receipt)
            continue
        items = dict(
            chip=row["chip"],
            shape=row["shape"],
            status="qualified",
            ca_us=receipt["ca_us"],
            ca_busy_vf_fraction=receipt["busy_rv_cycles"] / (1650 * receipt["ca_us"]),
            busy_pushq_vf_descriptor_cycles=max(
                receipt["vf_descriptor_union_cycles"].values(), default=None
            ),
            busy_simd_cycles=max(receipt["simd_union_cycles"].values(), default=None),
            measured_aic=sum("cubecore" in lane for lane in receipt["all_lanes"]),
            measured_aiv=sum("veccore" in lane for lane in receipt["all_lanes"]),
            measured_mmad=sum(receipt["mmad"].values()),
            busy_rv_cycles=receipt["busy_rv_cycles"],
            wall_s=receipt["seconds"],
        )
        for model in ("calibrated", "v21_classes", "instruction_classes"):
            v = row[model]
            items[model] = dict(
                us=v["us"],
                vf_cycles=v["vf_cycles"],
                device_error_pct=100 * (v["us"] / receipt["ca_us"] - 1),
                vf_difference_pct=100
                * (v["vf_cycles"] / receipt["busy_rv_cycles"] - 1),
            )
        records.append(items)
    totals = {}
    for chip in SKUS:
        rows = [r for r in records if r["status"] == "qualified" and r["chip"] == chip]
        if not rows:
            continue
        totals[chip] = {}
        for arm in ("calibrated", "v21_classes", "instruction_classes"):
            sums = {}
            for name, m, t in [
                ("window", "us", "ca_us"),
                ("rv", "vf_cycles", "busy_rv_cycles"),
            ]:
                sums[name + "_wape_pct"] = (
                    100
                    * sum(abs(r[arm][m] - r[t]) for r in rows)
                    / sum(r[t] for r in rows)
                )
                sums[name + "_max_pct"] = 100 * max(
                    abs(r[arm][m] / r[t] - 1) for r in rows
                )
            totals[chip][arm] = sums
    dump(
        OUT / "comparison.json",
        dict(
            freeze_sha256=digest(OUT / "predictions.json"),
            rows=records,
            summary=totals,
            clock="1650 cycles/us",
            vf_scope="sum of per-page Vec1+Vec2 service on modeled busy AIV versus maximum union of timed RV_* execution intervals on any active AIV. Excludes zero-duration records and scalar/control gaps.",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prices", "freeze", "collect", "evaluate"])
    parser.add_argument("--v21-src", type=Path)
    parser.add_argument(
        "--long",
        action="store_true",
        help="Separate adaptive holdout with a longer steady phase",
    )
    parser.add_argument("--output-dir", type=Path, help="Separate directory for new forecasts and receipts")
    parser.add_argument("--raw-dir", type=Path, help="Directory for raw CA traces")
    args = parser.parse_args()
    if args.output_dir:
        OUT = args.output_dir.resolve()
    if args.raw_dir:
        RAW = args.raw_dir.resolve()
    if args.long:
        LONG = True
        SHAPES = [(64, 3072, 64, 1, 1)]
        OUT = OUT / "long"
    if args.action == "freeze":
        if args.v21_src is None:
            parser.error("freeze requires --v21-src")
        OUT.mkdir(exist_ok=True, parents=True)
    if args.action == "prices":
        print(json.dumps(predictions()))
    elif args.action == "freeze":
        freeze(args.v21_src)
    elif args.action == "evaluate":
        evaluate()
    else:
        manifest = json.loads((OUT / "predictions.json").read_text())
        assert all(digest(ROOT / p) == s for p, s in manifest["hashes"].items())
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(run_one, manifest["predictions"]))
