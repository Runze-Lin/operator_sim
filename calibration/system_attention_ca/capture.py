"""Capture one installed CANN kernel with msopprof in an isolated working directory."""

import argparse
import json
from pathlib import Path
import subprocess
from time import monotonic

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("launcher", type=Path)
parser.add_argument("probe", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("soc", choices=["Ascend950DT_9582", "Ascend950PR_957b"])
parser.add_argument(
    "operator", choices=["fia", "conv", "indexer", "mxindexer", "mxsingle"]
)
parser.add_argument(
    "length", type=int, help="KV length for attention/indexer, channel width for conv"
)
parser.add_argument("--timeout-minutes", type=int, default=6)
args = parser.parse_args()
if args.timeout_minutes < 1:
    parser.error("timeout must be positive")
root = args.output.resolve()
root.mkdir(exist_ok=False, parents=True)
cmd = [
    "bash",
    str(args.launcher.resolve()),
    "msopprof",
    "simulator",
    "--soc-version=" + args.soc,
    "--launch-count=8",
    "--dump=false",
    "--timeout=" + str(args.timeout_minutes),
    "--output=" + str(root / "profile"),
    str(args.probe.resolve()),
    args.operator,
]
cmd.append(str(args.length))
start = monotonic()
with (root / "run.log").open("w") as f:
    result = subprocess.run(
        cmd,
        cwd=root,
        stdout=f,
        stderr=subprocess.STDOUT,
        timeout=60 * (args.timeout_minutes + 2),
    )
log = (root / "run.log").read_text()
qualified = (
    result.returncode == 0
    and "GOLDEN_PASS" in log
    and "All task success." in log
    and "timeout has reached" not in log
)
(root / "cost.json").write_text(
    json.dumps(
        {
            "soc": args.soc,
            "operator": args.operator,
            "length": args.length,
            "extent": args.length,
            "rc": result.returncode,
            "qualified": qualified,
            "wall_s": monotonic() - start,
        },
        indent=2,
    )
)
raise SystemExit(0 if qualified else 1)
