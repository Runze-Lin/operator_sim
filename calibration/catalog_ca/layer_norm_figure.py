"""Draw the indexed LayerNorm frozen forecasts and verified component windows."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def main():
    root = Path(__file__).resolve().parents[2]
    receipts = json.loads((root / "results/v2.4/layer-norm-qualified.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.05), sharex=True, sharey=True)
    colors = {"Slice": "#24778a", "LayerNormV4": "#74629b"}
    for axis, sku in zip(axes, sorted({row["chip"] for row in receipts})):
        rows = [row for row in receipts if row["chip"] == sku]
        for y, row in enumerate(rows):
            left = 0.0
            for stage in row["stages"]:
                kind = stage["kernel"].split("_")[0]
                width = stage["window_us"]
                axis.barh(y, width, left=left, height=0.54, color=colors[kind], zorder=2)
                left += width
            axis.plot(row["model_us"]["auto"], y, marker="|", ms=18, mew=2.4, color="#191919", zorder=3)
        axis.set_title(sku.split("_")[0].replace("Ascend", ""))
        axis.set_yticks(range(len(rows)), [f"{r} × {d}  (stride {s})" for r, d, s in (row["shape"] for row in rows)])
        axis.set_xlabel("Device time (µs)")
        axis.grid(axis="x", color="0.91", zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    legend = [Patch(color=c, label=k) for k, c in colors.items()]
    legend.append(Line2D([], [], linestyle="none", marker="|", ms=12, mew=2.4, color="#191919", label="Frozen model"))
    fig.legend(handles=legend, ncol=3, loc="upper center", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(root / "docs/figures/catalog-layer-norm-ca.png", dpi=185, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
