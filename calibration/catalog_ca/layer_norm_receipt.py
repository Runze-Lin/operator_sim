"""Verify ACLNN output, ELF and core retirements; keep each launch as a separate window."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from calibration.catalog_ca.layer_norm import FREEZE
from calibration.msopprof import CORE, sha
from calibration.mx_matmul.collect import MEMORY_CONFIG, core_windows


def score(case, directory, opp, *, freeze=FREEZE):
    cost = json.loads((directory / "cost.json").read_text())
    raw = (directory / "run.log").read_text()
    sku = case["chip"]
    rows, width, stride = case["shape"]
    if not (
        cost["exit_code"] == 0
        and "SMOKE_PASS" in raw
        and "Profiling data parse finished." in raw
        and f"runtime_soc={sku}" in raw
        and MEMORY_CONFIG[sku] in raw
        and f"correctness Y count={rows * width} bad=0" in raw
        and (directory / "freeze.sha256").read_text().strip() == sha(freeze)
    ):
        raise ValueError(f"Incomplete {directory}")
    kernels = re.findall(r"Start profiling on kernel: (\S+)", raw)
    expected = (
        ["Slice", "LayerNormV4"] if rows > 1 and stride != width else ["LayerNormV4"]
    )
    if [k.split("_")[0] for k in kernels] != expected:
        raise ValueError(f"Incomplete kernel set {kernels}")
    tables = re.split(
        r"^core_name\s+duration_time\(us\)\s+running_time\(us\)[ \t]*$", raw, flags=re.M
    )[1:]
    (base,) = (directory / "profile").glob("OPPROF_*")
    stages = []
    seen = Counter()
    for kernel, table in zip(kernels, tables, strict=True):
        path = base / kernel / str(seen[kernel])
        seen[kernel] += 1
        if len(kernels) == 1 and not path.exists():
            path = base
        digest = sha(path / "dump/aicore_binary.o")
        name = kernel.rsplit("_", 1)[0] + ".o"
        (obj,) = [p for p in opp.glob("*/*/" + name) if sha(p) == digest]
        roster = CORE.findall(table)
        clocks, hashes = core_windows(
            path / "simulator", {k: float(v) for k, v in roster}
        )
        if len(roster) != len(clocks):
            raise ValueError("duplicate core")
        ticks = max(x["retire_tick"] for x in clocks.values()) - min(
            x["first_tick"] for x in clocks.values()
        )
        stages.append(
            dict(
                kernel=kernel,
                object=str(obj.relative_to(opp)),
                elf_sha256=digest,
                aiv=sum(".veccore" in k for k in clocks),
                aic=sum(".cubecore" in k for k in clocks),
                cycles=ticks,
                window_us=ticks / 1650,
                trace_sha256=sha256(
                    json.dumps(hashes, sort_keys=True).encode()
                ).hexdigest(),
            )
        )
    return dict(
        **case,
        api="aclnnLayerNorm",
        dtype="bf16",
        auxiliary_outputs=False,
        checked_elements=rows * width,
        ca_sum_kernel_windows_us=sum(s["window_us"] for s in stages),
        stages=stages,
        wall_s=cost["wall_seconds"],
        probe_sha256=cost["probe_sha256"],
        log_sha256=sha(directory / "run.log"),
        freeze_sha256=sha(freeze),
    )


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--raw", nargs="+", type=Path, required=True)
    a.add_argument("--opp", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    args = a.parse_args()
    data = []
    for case in json.loads(FREEZE.read_text()):
        key = "-".join(map(str, [case["chip"], *case["shape"]]))
        paths = [p / key for p in args.raw if (p / key / "cost.json").exists()]
        if not paths:
            raise ValueError("No capture for " + key)
        data.append(score(case, paths[0], args.opp))
    args.out.write_text(json.dumps(data, indent=2) + "\n")
