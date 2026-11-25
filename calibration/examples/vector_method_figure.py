"""Plot a small selection from the paired CA comparison; all contracts remain in the table."""

import json
from pathlib import Path

from matplotlib import pyplot as plt


def main():
    root = Path(__file__).resolve().parents[2]
    result = json.loads((root / "results/v2.3/native-summary.json").read_text())
    cases = (
        ("fa · bf16 · supplemental", "published supplemental", "FA · supplemental BF16"),
        ("fa", "replay", "FA · BF16 archive"),
        ("cache_load_mxfp4_antimx", "independent", "AntiMxQuant · MX4 → BF16"),
        ("matmul_mx_mixed", "independent", "Matmul · A8W4"),
        ("rmsnorm · visible stream", "replay", "RMSNorm · visible prefix"),
        ("mhc_sinkhorn · known history", "replay", "Sinkhorn · known history"),
        ("kda_recurrent", "replay", "KDA · recurrent"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharey=True, constrained_layout=True)
    for ax, sku, title in zip(axes, ("Ascend950DT_9582", "Ascend950PR_957b"), ("950DT", "950PR")):
        for j, (key, role, _) in enumerate(cases):
            s = result[key][role][sku]["paired_scores"]
            old, new = s["calibrated"][0], s["instruction_classes"][0]
            ax.plot((old, new), (j, j), color="#a8b7c3", lw=2, zorder=1)
            ax.scatter(old, j, c="#17587c", s=43, marker="o", label="Original" if j == 0 else None)
            ax.scatter(new, j, c="#b45823", s=45, marker="D", label="Instruction classes" if j == 0 else None)
        ax.grid(axis="x", alpha=0.22)
        ax.set(title=title, xlim=(0, 55), xlabel="Whole-window WAPE (%)")
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_yticks(range(len(cases)), [t[2] for t in cases])
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right", fontsize=8)
    fig.savefig(root / "docs/figures/vector-methods-v23.png", dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    main()
