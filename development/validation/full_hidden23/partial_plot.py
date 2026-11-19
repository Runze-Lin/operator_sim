"""Recreate the large Cast duration comparison without treating partial CA as a label."""

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model import estimate

OUT = ROOT / "results/v0.23/full-hidden"


def main():
    partial = json.loads((OUT / "previous-partial.json").read_text())
    frozen = json.loads((OUT / "v2/registration.json").read_text())
    pred = estimate("cast_bf16_fp32", partial["shape"], dtype="bf16", chip=frozen["sku"])
    stages = pred.projection["stages_us"]
    assert abs(pred.latency_us - frozen["previous_frozen_us"]["v021_before_retry"]["dt"]) < 1e-8
    first = sum(stages[k] for k in ("arrival", "control_to_read", "read", "vector", "dependency_to_write", "write"))
    local = first + stages["steady_tiles"]
    full = pred.latency_us
    per_lane = {int(k): v for k, v in partial["logs"]["wrapper"].items()}
    zero = min(p["start"] for p in per_lane.values())
    observed = partial["partial_wrapper_lower_bound_us"]
    assert len(per_lane) == partial["completed_aiv"] and not partial["qualified"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": "#203541",
                         "axes.labelcolor": "#203541", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False,
                         "xtick.color": "#395261", "ytick.color": "#395261"})
    fig = plt.figure(figsize=(10.4, 7.6))
    fig.subplots_adjust(left=.21, right=.95, top=.81, bottom=.16, hspace=.43)
    top, bottom = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": (1, 2.4)})
    fig.suptitle("950DT Cast  |  262,144 × 5,120", x=.21, y=.956, ha="left",
                 fontweight="bold", fontsize=20)
    fig.text(.21, .9, "8 requests × 32k prefill  ·  BF16 → FP32  ·  64 AIVs  ·  7.5 GiB logical I/O", fontsize=12)
    top.set_title("Resource model   |   parallel duration bounds", loc="left", fontsize=12, fontweight="bold")
    top.barh([1, 0], [local, full], height=.4, color=["#197880", "#64519c"])
    top.set_yticks([1, 0], ["Critical AIV + UB tiles", "Shared-memory floor"])
    top.set_ylim(-.55, 1.55)
    for y, val in ((1, local), (0, full)):
        top.text(val - 30, y, f"{val:,.2f} µs", va="center", ha="right", color="white", weight="bold")
    top.axvline(full, color="#64519c", lw=1, ls="--")
    top.tick_params(axis="x", bottom=False)
    bottom.set_title("Native CA   |   recorded start and done of 31 completed AIVs", loc="left", fontsize=12, fontweight="bold")
    for i, lane in enumerate(sorted(per_lane)):
        data = per_lane[lane]
        start, done = (data["start"] - zero) / 1650, (data["done"] - zero) / 1650
        bottom.barh(i, done - start, left=start, color="#24787e", height=.54)
    indices = [0, 5, 10, 15, 20, 25, 30]
    bottom.set_yticks(indices, [f"AIV {sorted(per_lane)[i]}" for i in indices])
    bottom.set_ylim(len(per_lane) - .2, -2.7)
    bottom.axvline(observed, color="#192f41", lw=1, ls="--")
    bottom.text(observed - 10, -1.6, f"observed ≥ {observed:,.2f} µs", ha="right", weight="bold", fontsize=10)
    for ax in (top, bottom):
        ax.set_xlim(0, 2700)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#d5dfe5", lw=.7)
        ax.tick_params(axis="y", length=0)
    bottom.xaxis.set_major_locator(MultipleLocator(500))
    bottom.set_xlabel("Device time (µs)")
    fig.text(.21, .077, "Model bounds share the device window. Their maximum sets the prediction. CA zero is the earliest", fontsize=9)
    fig.text(.21, .054, "recorded start among completed AIVs. 33 AIVs have no wrapper end; full-window error is unavailable.", fontsize=9)
    fig.savefig(OUT / "cast-full-hidden-partial.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
