"""Second freeze after the shared/rotated endpoints, before unblinding moving."""
import argparse
import json
from pathlib import Path
from v113_prices import HERE, record, digest


def second_freeze():
    frozen = record()
    frozen["initial_freeze_sha256"] = digest(HERE/"v113_frozen.json")
    frozen["input_sha256"]["v113b_prices.py"] = digest(Path(__file__))
    for arm in ("shared","rotated"):
        source = HERE/f"v113_control_{arm}.json"
        frozen["input_sha256"][source.name] = digest(source)
    frozen["protocol"].update(
        evidence="Shared and rotated inspected; team medians of second-row MMAD spans per interval were 1314.40 and 1320.89 cycles. A cold-row final-ring proxy of 2534 cycles was inappropriate for prefetched rows.",
        change="Retain the original v1.12 DT carried-row measurement of 1343.6129 cycles at all address alignments, subject to target resource floor. No prices fitted to the new arms.",
        independent="moving timing withheld until after this snapshot; one DT Q64 layout control for the selected model",
        budget="reuse the original three launches; zero additional CA launches")
    return frozen


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    frozen=second_freeze()
    with args.output.open("x") as out:
        json.dump(frozen,out,indent=2)
        out.write("\n")
