"""Redraw the first-fragment example from its stored CA per-core deltas."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from operator_model import estimate

STAGES = ("core_start", "setup", "first_dma", "operand_ready", "mmad", "drain")
MODEL = ("core_start_spread", "scalar_setup", "first_dma", "operand_ready", "first_mmad", "drain")
LABELS = ("Core dispatch", "Scalar to first DMA", "Scale + value reads", "Operands ready", "MMAD_MX", "Retirement")
COLORS = ("#96a9ba", "#426989", "#198881", "#d1a13d", "#7759a2", "#bb5e35")


def plot(source, dest):
    evidence = json.loads(source.read_text())
    shape = evidence["shape_MNK"]
    sample = max(evidence["core_milestones"], key=lambda row: sum(row[k] for k in STAGES))
    predicted = estimate("matmul", shape, {"weight_dtype": "mxfp8"}, dtype="mxfp8", chip="Ascend950DT_9582")
    model = predicted.projection["target_plan"]["cycles"]
    fig, axes = plt.subplots(2, sharex=True, figsize=(8.7, 4.9), layout="constrained")
    for ax, label, stages in zip(
        axes,
        ["Resource model", f'CA · {sample["core"]} (last retirement)'],
        [[model[k] for k in MODEL], [sample[k] for k in STAGES]],
    ):
        start = 0
        total = sum(stages) / 1650
        for j, (length, color) in enumerate(zip(stages, COLORS)):
            ax.barh(j, length / 1650, left=start, color=color, height=.6)
            start += length / 1650
        ax.axvline(total, color="#31404c", linestyle="--", linewidth=1)
        ax.set_title(f"{label}   |   {total:.3f} µs", loc="left", fontweight="bold", fontsize=10)
        ax.set_yticks(range(6), LABELS, fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[-1].set(xlim=(0,4.65), xlabel="Device time from earliest core instruction (µs)")
    fig.suptitle("950DT native MXFP8 Matmul   |   9 × 256 × 128", fontsize=14, fontweight="bold")
    fig.savefig(dest, dpi=135)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.evidence, args.output)
