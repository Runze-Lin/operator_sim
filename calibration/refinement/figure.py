"""Plot paired model residuals for each previously unseen shape."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
OPS = ("AddRmsNorm", "Indexer LayerNorm", "MXFP8 Matmul")
SOCS = ("Ascend950DT_9582", "Ascend950PR_957b")


def main():
    with (ROOT / "results/v2.5/paired-errors.csv").open() as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 3, figsize=(9, 4.6), sharey=True)
    colors = ("#127973", "#745795")
    for op, axis in zip(OPS, axes):
        shapes = sorted(
            {r["shape"] for r in rows if r["op"] == op},
            key=lambda s: tuple(map(int, s.split("x"))),
        )
        for chip_index, chip in enumerate(SOCS):
            for r in (v for v in rows if v["op"] == op and v["chip"] == chip):
                x = shapes.index(r["shape"]) + (-0.12 if chip_index == 0 else 0.12)
                old, new = (
                    float(r[f"{version}_error_pct"]) for version in ("old", "new")
                )
                color = colors[chip_index]
                axis.plot([x, x], [old, new], color=color, alpha=0.6)
                axis.scatter(x, old, facecolor="white", edgecolor=color, s=25, zorder=3)
                axis.scatter(x, new, color=color, s=21, zorder=4)
        axis.axhline(0, color="0.25", lw=0.9)
        axis.grid(axis="y", color="0.9")
        axis.set_axisbelow(True)
        axis.set_title(op)
        axis.set_xticks(range(len(shapes)), shapes, rotation=70, fontsize=8, ha="right")
        axis.spines[["right", "top"]].set_visible(False)
    axes[0].set_ylabel("Signed duration error (%)")
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=7))
    legend = [
        Line2D([], [], color=c, label=sku.split("_")[0].replace("Ascend", ""))
        for sku, c in zip(SOCS, colors)
    ]
    for label, face in (("v2.4", "white"), ("v2.5", "0.2")):
        legend.append(
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                color="0.2",
                markerfacecolor=face,
                label=label,
            )
        )
    fig.legend(handles=legend, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(
        ROOT / "docs/figures/paired-errors-v25.png", dpi=190, bbox_inches="tight"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
