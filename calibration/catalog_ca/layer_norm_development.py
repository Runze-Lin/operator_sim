"""Reduce the trace of the last core to the read, reduction, and write boundaries."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from calibration.catalog_ca.layer_norm import ROOT
from calibration.msopprof import sha
from calibration.mx_matmul.shards import trace_events


def digest_window(path):
    ev = list(trace_events(path))
    completed = [e for e in ev if e["ph"] == "X"]
    if sum(e["name"] == "END" for e in completed) != 1:
        raise ValueError("Missing physical END")
    start = min(round(e["ts"] * 1650) for e in completed)
    end = max(round((e["ts"] + e["dur"]) * 1650) for e in completed)
    pipes = {e["pid"]: e["args"]["name"] for e in ev if e.get("name") == "process_name"}

    def io(pipe):
        return sorted(
            (
                round(e["ts"] * 1650),
                round((e["ts"] + e["dur"]) * 1650),
                json.loads(e["args"]["detail"])["processed_bytes"],
            )
            for e in completed
            if pipes[e["pid"]] == pipe and e["name"].startswith("MOV_SRC_TO_DST")
        )

    return dict(
        core=path.parent.name, start=start, end=end, reads=io("MTE2"), writes=io("MTE3")
    )


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--raw", type=Path, nargs="+", required=True)
    a.add_argument("--out", type=Path, required=True)
    args = a.parse_args()
    result = []
    for receipt in json.loads(
        (ROOT / "results/v2.4/layer-norm-qualified.json").read_text()
    ):
        for stage in receipt["stages"]:
            matches = []
            for root in args.raw:
                case = root / ("-".join(map(str, [receipt["chip"], *receipt["shape"]])))
                if (
                    not (case / "run.log").exists()
                    or sha(case / "run.log") != receipt["log_sha256"]
                ):
                    continue
                for directory in case.glob("profile/OPPROF_*/**/simulator"):
                    if (
                        not (directory.parent / "dump/aicore_binary.o").exists()
                        or sha(directory.parent / "dump/aicore_binary.o")
                        != stage["elf_sha256"]
                    ):
                        continue
                    paths = sorted(directory.glob("core*.*core*/trace.json"))
                    hashes = {p.parent.name: sha(p) for p in paths}
                    if (
                        sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
                        == stage["trace_sha256"]
                    ):
                        matches.append(paths)
            (paths,) = matches
            windows = [digest_window(p) for p in paths]
            zero = min(w["start"] for w in windows)
            last = max(windows, key=lambda w: w["end"])
            row = dict(
                chip=receipt["chip"],
                shape=receipt["shape"],
                kernel=stage["kernel"],
                trace_sha256=stage["trace_sha256"],
                cores=len(paths),
                last_core=last["core"],
                retire_cycles=last["end"] - zero,
                first_read=last["reads"][0][0] - zero,
                last_read=last["reads"][-1][1] - zero,
                first_write=last["writes"][0][0] - zero,
                last_write=last["writes"][-1][1] - zero,
                write_bytes=[v[2] for v in last["writes"]],
            )
            if row["retire_cycles"] != stage["cycles"]:
                raise ValueError("Retirement differs from v2.4 receipt")
            result.append(row)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
