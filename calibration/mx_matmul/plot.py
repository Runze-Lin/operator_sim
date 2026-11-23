"""Render the compact dense MX case study. Requires matplotlib (offline only)."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

COLORS = {
    'Front end / control': '#899dab', 'MTE2 / read': '#167889',
    'AIV / expand': '#36a078', 'UB to L1': '#a467b0', 'MTE1 / load': '#486db6',
    'Cube / MMAD': '#744daf', 'FIXP / write': '#be5027', 'Steady / shared': '#77868d',
}
RESOURCE = {
    'launch': 'Front end / control', 'core_arrival_spread': 'Front end / control', 'read_fill': 'MTE2 / read', 'inline_unpack': 'AIV / expand',
    'ub_to_l1': 'UB to L1', 'l1_to_l0': 'MTE1 / load', 'cube_tail': 'Cube / MMAD',
    'write_tail': 'FIXP / write', 'page_intervals': 'Steady / shared',
    'additional_waves': 'Steady / shared', 'exposed_shared_traffic': 'Steady / shared',
}


def render(data, output):
    lanes = list(COLORS)
    panels = data['panels']
    fig, axes = plt.subplots(2, len(panels), figsize=(12, 7.2), sharex=True, sharey=True,
                             squeeze=False, layout='constrained')
    for col, panel in enumerate(panels):
        top, bottom = axes[:, col]
        cursor = 0
        for stage, elapsed in panel['model_stage_us'].items():
            lane = RESOURCE[stage]
            if elapsed > 0:
                top.barh(lanes.index(lane), elapsed, left=cursor, color=COLORS[lane], height=.37)
                cursor += elapsed
        top.axvline(panel['model_us'], color='#344651', ls='--', lw=1.2)
        mode = 'A4W4 native' if panel['dtype'] == 'mxfp4' else 'A8W4 inline expansion'
        top.set_title(f"{mode}  ·  Model budget {panel['model_us']:.3f} µs", loc='left', weight='bold')
        for lane, events in panel['ca_descriptor_us'].items():
            y = lanes.index(lane)
            for start, duration, _, _ in events:
                bottom.broken_barh([(start, duration)], (y-.18, .36), color=COLORS[lane], alpha=.64)
                bottom.plot(start, y, '|', color=COLORS[lane], markersize=6)
        bottom.axvline(panel['ca_us'], color='#344651', lw=1.2)
        bottom.set_title(f"CA descriptors  ·  All-core window {panel['ca_us']:.3f} µs", loc='left')
        bottom.set_xlabel('Device time from first instruction (µs)')
        for ax in (top, bottom):
            ax.set_yticks(range(len(lanes)), lanes)
            ax.set_ylim(len(lanes)-.5, -.5)
            ax.set_xlim(left=0)
            ax.grid(axis='x', color='#d4dee3', lw=.7)
            ax.set_axisbelow(True)
            ax.spines[['top','right','left']].set_visible(False)
            ax.tick_params(axis='y', length=0)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    m, n, k = panels[0]['shape']
    fig.suptitle(f'950DT dense MX Matmul   ·   M={m}, N={n}, K={k}', fontsize=16, weight='bold')
    fig.savefig(output, dpi=145)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('output', type=Path)
    args=parser.parse_args()
    render(json.loads(args.data.read_bytes()), args.output)
