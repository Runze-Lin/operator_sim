"""Plot paired CA observations and the frozen resident-branch estimates."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PAIRS = ROOT / 'results/v0.19/workloads/attn-res-extension-evaluation.json'


def main():
    data = json.loads(PAIRS.read_text())['pairs']
    rows = sorted(data, key=lambda r: (r['shape'][1], r['shape'][3], r['shape'][0]))
    labels = [f"{t} / {h:,} / {n}" for t, h, _, n in [r['shape'] for r in rows]]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.2), sharex=True, sharey=True, layout='constrained')
    for ax, sku, prefix in zip(axs, ['950DT', '950PR'], ['dt', 'pr']):
        pred = [r[prefix+'_us'] for r in rows]
        actual = [r[prefix+'_ca_us'] for r in rows]
        for i, (a, p) in enumerate(zip(actual, pred)):
            ax.plot([p, a], [i, i], color='#9ca4b1', linewidth=1.6)
        ax.scatter(actual, range(len(rows)), s=42, color='#0f766e', label='CA', zorder=3)
        ax.scatter(pred, range(len(rows)), marker='D', s=33, facecolor='none', edgecolor='#ab3f2d', linewidth=1.6,
                   label='Frozen model', zorder=3)
        ax.set_title(sku, loc='left', fontweight='semibold')
        ax.set_xlabel('Kernel time (µs)')
        ax.grid(axis='x', color='#d7dce0', linewidth=.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(0, 26)
    axs[0].set_yticks(range(len(labels)), labels)
    axs[0].set_ylabel('Tokens / Hidden / Active history rows')
    axs[1].legend(frameon=False, loc='lower right')
    fig.savefig(ROOT/'docs/examples/figures/attn_res_holdout.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
