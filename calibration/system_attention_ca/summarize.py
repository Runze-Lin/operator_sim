"""Verify installed ELF, SKU, all core windows and probe golden before admitting a sample."""

import argparse
import json
from pathlib import Path
import re
from calibration.msopprof import CORE, sha
from calibration.candidate_indexer.receipt import ordinary_li_instructions
from calibration.mx_matmul.collect import MEMORY_CONFIG, core_windows, trace_events
from statistics import median


def mx_component(simulator, start):
    """Use the explicitly selected single-shard probe to identify issue/drain cycles."""
    mmads = []
    for path in sorted(simulator.glob("core*.cubecore*/trace.json")):
        for event in trace_events(path):
            if (
                event.get("ph") != "X"
                or event.get("name") != "MMAD_MX"
                or event["dur"] <= 0
            ):
                continue
            detail = json.loads(event["args"]["detail"])
            mmads.append(
                (
                    round(event["ts"] * 1650),
                    round(event["dur"] * 1650),
                    path.parent.name,
                    (detail["m"], detail["n"], detail["k"]),
                )
            )
    mmads.sort()
    if len(mmads) != 17 or any(
        m[2] != "core0.cubecore0" or m[3] != (64, 128, 128) for m in mmads
    ):
        raise ValueError(
            "MX single-shard issue geometry differs from the planned 17 pages"
        )
    gaps = [b[0] - a[0] for a, b in zip(mmads, mmads[1:])]
    return dict(
        issuing_aic=1,
        mmad_mnk=list(mmads[0][3]),
        mmad_mx_count=len(mmads),
        first_issue_cycles=mmads[0][0] - start,
        last_retire_cycles=mmads[-1][0] + mmads[-1][1] - start,
        median_issue_interval_cycles=median(gaps),
        issue_interval_cycles=gaps,
        matched_dtypes=["fp4_e2m1", "fp4_e2m1", "f32"],
    )


def receipt(raw, opp):
    run = json.loads((raw / "cost.json").read_text())
    log = (raw / "run.log").read_text()
    if (
        run["rc"]
        or "GOLDEN_PASS" not in log
        or "All task success." not in log
        or ("runtime_soc=" + run["soc"]) not in log
        or MEMORY_CONFIG[run["soc"]] not in log
    ):
        raise ValueError("incomplete qualification: " + str(raw))
    names = re.findall(r"Start profiling on kernel: (\S+)", log)
    blocks = re.split(
        r"^core_name\s+duration_time\(us\)\s+running_time\(us\)\s*$", log, flags=re.M
    )[1:]
    (base,) = (raw / "profile").glob("OPPROF_*")
    stages = []
    for name, block in zip(names, blocks, strict=True):
        path = base if len(names) == 1 else base / name / "0"
        binary = sha(path / "dump/aicore_binary.o")
        stem = re.sub(r"_\d+(?:_mix_ai[cv])?$", "", name)
        (obj,) = [f for f in opp.glob("*/*/" + stem + ".o") if sha(f) == binary]
        roster = {core: float(us) for core, us in CORE.findall(block)}
        cores, _ = core_windows(path / "simulator", roster)
        cycles = max(c["retire_tick"] for c in cores.values()) - min(
            c["first_tick"] for c in cores.values()
        )
        stages.append(
            dict(
                kernel=name,
                sku=run["soc"],
                elf_sha256=binary,
                object=str(obj.relative_to(opp)),
                cycles=cycles,
                active_aic=sum(".cubecore" in k for k in cores),
                active_aiv=sum(".veccore" in k for k in cores),
            )
        )
        if run["operator"] == "indexer":
            stages[-1]["mmad_shards"] = ordinary_li_instructions(path / "simulator")
        if run["operator"] == "mxsingle":
            stages[-1]["single_shard_schedule"] = mx_component(
                path / "simulator", min(c["first_tick"] for c in cores.values())
            )
    if not stages:
        raise ValueError("no AI Core execution")
    extent = run.get("extent", run["length"])
    if run["operator"] == "conv":
        # Legacy development receipts used the FIA default in length for a fixed-width probe.
        shape = [
            1,
            9,
            (4608 if extent == 512 and "extent" not in run else extent),
            4,
            11,
        ]
    elif run["operator"] == "fia":
        shape = [1, 9, 12, 2, 64, extent]
    elif run["operator"] == "indexer":
        shape = [1, 9, extent, 64, 128, 1024]
    else:
        shape = [1, 1, extent, 32, 128, 2048]
    return dict(
        operator=run["operator"],
        shape=shape,
        sku=run["soc"],
        wall_s=run["wall_s"],
        device_us=sum(s["cycles"] for s in stages) / 1650,
        stages=stages,
        log_sha256=sha(raw / "run.log"),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("opp", type=Path)
    p.add_argument("raw", type=Path, nargs="+")
    args = p.parse_args()
    print(json.dumps([receipt(r, args.opp) for r in args.raw], indent=2))
