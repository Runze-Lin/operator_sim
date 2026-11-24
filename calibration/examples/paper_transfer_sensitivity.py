"""Recompute the paper's one-input-at-a-time prospective 970DT scenarios."""
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from operator_model import estimate, load_chip


CASES = [
    ('Quant  8192 × 5120', 'quant', [8192, 5120], 'bf16', {}),
    ('Matmul  1280 × 2048 × 512', 'matmul', [1280, 2048, 512], 'bf16', {}),
    ('Matmul  262144 × 128 × 512', 'matmul', [262144, 128, 512], 'bf16', {}),
    ('FA  Q9, KV128Ki, D64', 'fa', [9, 131072, 64, 1, 1], 'bf16', {}),
    ('QSLI  Q4096, L1Mi, H32', 'candidate_filtered_indexer', [1, 4096, 32, 128, 1048576, 2048, 512], 'mxfp4',
     {'implementation': 'a5_qsli_projection', 'table_pages': 8201}),
]


def scenarios():
    chip = load_chip('Ascend970DT')
    def detail(**settings):
        return replace(chip, details={**chip.details, **settings})
    vector = {bits: value / 2 if value is not None else None for bits, value in chip.vector_tops.items()}
    return [chip, replace(chip, hbm_bandwidth_gbs=chip.hbm_bandwidth_gbs / 2),
            detail(l2_bandwidth_gbs=chip.details['l2_bandwidth_gbs'] / 2),
            replace(chip, vector_tops=vector), detail(ub_bytes=65536),
            detail(mte2_bandwidth_gbs=40 * chip.frequency_ghz)]


def main():
    labels, changes = [], []
    for label, op, shape, dtype, attrs in CASES:
        times = [estimate(op, shape, attrs, chip=chip, dtype=dtype, memory_policy='cold_io').latency_us
                 for chip in scenarios()]
        print(label, *(f'{t:.4f}' for t in times))
        labels.append(f'{label}\n{times[0]:,.2f} µs baseline')
        changes.append([100 * (time / times[0] - 1) for time in times[1:]])
    fig, ax = plt.subplots(figsize=(9.6, 4.6), layout='constrained')
    ax.imshow(changes, cmap='Blues', norm=Normalize(0, 90), aspect='auto')
    ax.set_yticks(range(len(labels)), labels, fontsize=10)
    ax.set_xticks(range(5), ['HBM BW\n×0.5', 'L2 BW\n×0.5', 'Vector peak\n×0.5', 'UB capacity\n64 KiB', 'MTE2 port\n40 B/cycle'])
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=6)
    for row, entries in enumerate(changes):
        for col, val in enumerate(entries):
            ax.text(col, row, f'{val:+.1f}%' if abs(val) >= 0.05 else '0.0%', ha='center', va='center',
                    color='white' if val > 48 else '#163348', fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    target = Path(__file__).resolve().parents[2] / 'docs/figures/paper-transfer-sensitivity.png'
    fig.savefig(target, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
