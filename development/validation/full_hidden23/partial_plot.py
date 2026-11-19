"""Recreate the large Cast duration comparison without treating partial CA as a label."""

import json
from hashlib import sha256
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


def snapshot():
    path = OUT / "stage-budgets.json"
    if path.exists():
        raise FileExistsError(path)
    reg = json.loads((OUT / "v2/registration.json").read_text())
    pred = estimate("cast_bf16_fp32", reg["shape"], dtype="bf16", chip=reg["sku"])
    assert abs(pred.latency_us - reg["previous_frozen_us"]["v021_before_retry"]["dt"]) < 1e-8
    source = ROOT / "operator_model/cast_resource.py"
    stages = dict(model_file=str(source.relative_to(ROOT)), model_sha256=sha256(source.read_bytes()).hexdigest(),
                  shape=reg["shape"], chip=reg["sku"], total_us=pred.latency_us,
                  stages_us=pred.projection["stages_us"], target_plan=pred.projection["target_plan"],
                  note="Current stage decomposition at the identical pre-retry frozen total; independent of full-run CA.")
    path.write_text(json.dumps(stages, indent=2) + "\n")


def main(receipt_path=None):
    partial = json.loads((OUT / "previous-partial.json").read_text())
    frozen = json.loads((OUT / "v2/registration.json").read_text())
    model = json.loads((OUT / "stage-budgets.json").read_text())
    stages = model["stages_us"]
    assert abs(model["total_us"] - frozen["previous_frozen_us"]["v021_before_retry"]["dt"]) < 1e-8
    first = sum(stages[k] for k in ("arrival", "control_to_read", "read", "vector", "dependency_to_write", "write"))
    local = first + stages["steady_tiles"]
    full = model["total_us"]
    if receipt_path is not None:
        end = json.loads(Path(receipt_path).read_text())
        assert end["qualified_core_envelope"] and end["shape"] == partial["shape"] and end["correct_elements"] == 1342177280 and end["active_aiv"] == 64
        logs = end["evidence"]
    else:
        logs = partial["logs"]
    per_lane = {int(k): v for k, v in logs["wrapper"].items()}
    zero = min(p["start"] for p in per_lane.values())
    observed = (max(x["done"] for x in per_lane.values()) - zero) / 1650
    if receipt_path is None:
        assert len(per_lane) == partial["completed_aiv"] and not partial["qualified"] and abs(observed-partial["partial_wrapper_lower_bound_us"]) < 1e-8
    else:
        assert len(per_lane) == 64 and abs(end["core_envelope_us"] - observed) < 1e-8

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
    bottom.set_title(f"Native CA   |   recorded start and done of {len(per_lane)} completed AIVs", loc="left", fontsize=12, fontweight="bold")
    for i, lane in enumerate(sorted(per_lane)):
        data = per_lane[lane]
        start, done = (data["start"] - zero) / 1650, (data["done"] - zero) / 1650
        bottom.barh(i, done - start, left=start, color="#24787e", height=.54)
    indices = [0, 5, 10, 15, 20, 25, 30] if receipt_path is None else [0, 10, 20, 30, 40, 50, 63]
    bottom.set_yticks(indices, [f"AIV {sorted(per_lane)[i]}" for i in indices])
    bottom.set_ylim(len(per_lane) - .2, -2.7)
    bottom.axvline(observed, color="#192f41", lw=1, ls="--")
    bottom.text(observed - 10, -1.6, f"observed {'≥ ' if receipt_path is None else ''}{observed:,.2f} µs", ha="right", weight="bold", fontsize=10)
    for ax in (top, bottom):
        ax.set_xlim(0, max(2700, observed*1.07))
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#d5dfe5", lw=.7)
        ax.tick_params(axis="y", length=0)
    bottom.xaxis.set_major_locator(MultipleLocator(500))
    bottom.set_xlabel("Device time (µs)")
    fig.text(.21, .077, "Model bounds share the device window. Their maximum sets the prediction. CA zero is the earliest", fontsize=9)
    note = ("recorded start among completed AIVs. 33 AIVs have no wrapper end; full-window error is unavailable."
            if receipt_path is None else "wrapper start across 64 AIVs; the last wrapper end closes the measured full-device window.")
    fig.text(.21, .054, note, fontsize=9)
    suffix = "partial" if receipt_path is None else "complete"
    fig.savefig(OUT / f"cast-full-hidden-{suffix}.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    if sys.argv[1:] == ["snapshot"]:
        snapshot()
    else:
        main(sys.argv[1] if len(sys.argv)>1 else None)
