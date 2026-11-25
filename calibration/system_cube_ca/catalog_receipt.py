"""Retain a small, provenance-checked diagnostic for the catalog's two CA probes."""

from collections import Counter
import argparse
import json
import re
from pathlib import Path

from calibration.msopprof import CORE, sha
from calibration.mx_matmul.shards import trace_events
from calibration.mx_matmul.collect import MEMORY_CONFIG, core_windows


def collect(root, soc, probe, frozen, opp):
    log = (root / "run.log").read_text()
    if MEMORY_CONFIG[soc] not in log or f"--soc-version" not in log:
        raise ValueError("Missing simulator memory or launch identity")
    api = "aclnnBatchMatMul" if probe == "bmm" else "aclnnQuantMatmulV5"
    checked = (
        "GOLDEN_PASS G=12 M=1 K=128 N=512 checked=6144" in log
        if probe == "bmm"
        else "correctness count=2304 bad=0" in log and "SMOKE_PASS" in log
    )
    timeout = "timeout has reached" in log
    parsed = "Profiling data parse finished." in log
    names = re.findall(r"Start profiling on kernel: (\S+)", log)
    clocks = re.findall(
        r"\[(\d+)\] \[(TASK_BEGIN|TASK_DONE)\] RTSQ_\d+ ACSQ_\d+ stream_id:(\d+), task_id:(\d+)",
        log,
    )
    result = dict(
        api=api,
        soc=soc,
        clock_ghz=1.65,
        simulator_memory=MEMORY_CONFIG[soc],
        model_us=frozen,
        input_shape_order="G,M,K,N" if probe == "bmm" else "M,N,K",
        shape=[12, 1, 128, 512] if probe == "bmm" else [9, 256, 128],
        golden_passed=checked,
        profiler_timeout=timeout,
        profiler_parsed=parsed,
        names=names,
        log_sha256=sha(root / "run.log"),
        component_windows=[],
        ca_us=None,
        error_pct=None,
        task_events=list(dict.fromkeys(clocks)),
    )
    tables = re.split(
        r"^core_name\s+duration_time\(us\)\s+running_time\(us\)[ \t]*$", log, flags=re.M
    )[1:]
    for name in names:
        stem = re.sub(r"_\d+(_mix_ai[cv])?$", "", name)
        installed = list(opp.glob("*/*/" + stem + ".o"))
        snapshots = list((root / "profile").rglob("aicore_binary.o"))
        verified = next(
            (
                (file, sha(file))
                for file in installed
                if any(sha(file) == sha(snap) for snap in snapshots)
            ),
            None,
        )
        window = dict(
            name=name,
            installed_elf=str(verified[0].relative_to(opp)) if verified else None,
            elf_sha256=verified[1] if verified else None,
            ca_us=None,
        )
        if (
            checked
            and parsed
            and not timeout
            and verified
            and len(names) == 1
            and len(tables) == 1
        ):
            (simulator,) = (root / "profile").glob("OPPROF_*/simulator")
            roster = {name: float(us) for name, us in CORE.findall(tables[0])}
            if not roster:
                raise ValueError("Expected complete core timing table")
            cores, hashes = core_windows(simulator, roster)
            duration = (
                max(c["retire_tick"] for c in cores.values())
                - min(c["first_tick"] for c in cores.values())
            ) / 1650
            window.update(ca_us=duration, launched=len(cores), trace_sha256=hashes)
            shapes = Counter()
            issues = []
            for path in simulator.glob("core*.*core*/trace.json"):
                for e in trace_events(path):
                    if e.get("ph") == "X" and e.get("name") == "MMAD_MX":
                        geom = json.loads(e["args"]["detail"])
                        shapes[(geom["m"], geom["n"], geom["k"])] += 1
                        issues.append(
                            (
                                path.parent.name,
                                round(e["ts"] * 1650),
                                round((e["ts"] + e["dur"]) * 1650),
                            )
                        )
            if issues:
                first = min(c["first_tick"] for c in cores.values())
                window.update(
                    productive_aic=len({c for c, _, _ in issues}),
                    first_mmad_cycles=min(a for _, a, _ in issues) - first,
                    last_mmad_end_cycles=max(b for _, _, b in issues) - first,
                    mmads_mnk_count=[
                        [*key, num] for key, num in sorted(shapes.items())
                    ],
                )

            result.update(ca_us=duration, error_pct=(frozen / duration - 1) * 100)
        result["component_windows"].append(window)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scratch", type=Path)
    p.add_argument("opp", type=Path, help="Installed ascend950 kernel directory")
    p.add_argument("output", type=Path)
    args = p.parse_args()
    results = []
    for probe, paths in (("bmm", ("ca-bmm", "bmm-passive")), ("mx8", ("mx8",))):
        inputs = json.loads((args.scratch / f"opsim24-{probe}-frozen.json").read_text())
        for variant in paths:
            for soc in MEMORY_CONFIG:
                results.append(
                    collect(
                        args.scratch / f"opsim24-{variant}-{soc}",
                        soc,
                        probe,
                        inputs[soc],
                        args.opp,
                    )
                )
    args.output.write_text(json.dumps(results, indent=2) + "\n")
