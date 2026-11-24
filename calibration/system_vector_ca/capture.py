"""Collect selected native development or holdout windows on the two Ascend 950 SKUs."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse

from calibration.mx_matmul.collect import MEMORY_CONFIG, capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--launcher", required=True, type=Path)
    parser.add_argument("--suite", choices=("streams","fused","mhc","hash","hash-holdout"), required=True)
    parser.add_argument("--probe", type=Path, required=True,
                        help="Compiled fused.cpp, mhc.cpp, hash.cpp, or the elementwise fixture for streams")
    parser.add_argument("--cache", type=Path, help="Native FP8 CacheLoad probe, required for streams")
    parser.add_argument("--minutes", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--chip", choices=tuple(MEMORY_CONFIG), action="append")
    args = parser.parse_args()
    cases = {
        "streams": [("add",(144,7168)), ("mul",(144,4)), ("cache_load_fp8",(1,1024,512))],
        "fused": [("swiglu",(9,288)), ("rope",(9,512))],
        "mhc": [("mhc_pre_sinkhorn",(9,4,7168))],
        "hash": [("moe_gating_top_k_hash",(2,384,6))],
        "hash-holdout": [("moe_gating_top_k_hash",(9,384,6))],
    }[args.suite]
    if args.suite == "streams" and not args.cache:
        parser.error("--suite streams requires --cache")
    args.scratch.mkdir(exist_ok=True, parents=True)

    def run(job):
        (op,shape),chip=job
        directory=args.scratch/(args.suite+"-"+op+"-"+chip)
        capture(dict(chip=chip,shape=shape),op,args.suite,directory.resolve(),
                args.cache if op=="cache_load_fp8" else args.probe,args.launcher,args.minutes)

    jobs=[(case,chip) for case in cases for chip in (args.chip or MEMORY_CONFIG)]
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        list(executor.map(run,jobs))


if __name__=="__main__":
    main()
