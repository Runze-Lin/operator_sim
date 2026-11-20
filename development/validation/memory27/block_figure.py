"""Plot the same contiguous UB blocks in the frozen schedule and a qualified CA window."""
import argparse
from hashlib import sha256
import json
from math import ceil, floor
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[3]
PRO = ROOT / 'results/v0.23/pro-prefill'
OUT = ROOT / 'results/v0.27/memory'
STAGES = {'Read': ('#147988', 'MTE2 / GM → UB'), 'Vector': ('#7053a8', 'Vector / VCVT'),
          'Write': ('#b44a2b', 'MTE3 / UB → GM')}
INK = '#182e3b'


def load_blocks(sku):
    stem = f'pro-b8-256ki-{sku}'
    receipt_path = PRO / (stem + '.json')
    blocks_path = PRO / (stem + '-blocks.json')
    receipt = json.loads(receipt_path.read_bytes())
    blocks = json.loads(blocks_path.read_bytes())
    score = json.loads((ROOT/'results/v0.26/memory'/(stem+'-score.json')).read_bytes())
    if not (receipt['qualified'] and score['qualified']
            and sha256(receipt_path.read_bytes()).hexdigest() == score['receipt_sha256']
            and sha256(blocks_path.read_bytes()).hexdigest() == score['blocks_sha256']
            and blocks['ca_tile_elements'] == blocks['model_tile_elements']
            and blocks['model_us'] == receipt['model_us']
            and abs(blocks['ca_us'] - receipt['ca_us']) < 1e-7):
        raise ValueError('work-block identities or receipt mismatch')
    return blocks


def draw(blocks, first, count, path):
    lo, hi = first-1, first-1+count
    actual, model = blocks['ca_tiles'], blocks['model_tiles']
    if not (0 <= lo < hi <= len(actual)) or count > 12:
        raise ValueError('select 1–12 existing consecutive blocks')
    indices = range(lo, hi)
    endpoints = [x for side in (model, actual) for i in indices for win in side[i].values() for x in win]
    if hi == len(actual):
        endpoints.extend([blocks['model_us'], blocks['ca_us']])
    left = max(0, floor(min(endpoints)/2)*2-1)
    right = ceil(max(endpoints)/2)*2+1
    fig = plt.figure(figsize=(12, 8.8), facecolor='white')
    grid = fig.add_gridspec(3, 1, height_ratios=[1, 1, .35], top=.776, bottom=.11,
                            left=.17, right=.965, hspace=.66)
    top = fig.add_subplot(grid[0])
    bottom = fig.add_subplot(grid[1], sharex=top)
    overview = fig.add_subplot(grid[2])
    sku = blocks['chip'].split('_')[0].removeprefix('Ascend')
    fig.text(.17, .944, f'{sku} Cast  |  8 × 256Ki prefill', size=25, weight='bold', color=INK)
    fig.text(.17, .902, f"DeepSeek V4 Pro indexer weights  ·  [2,097,152, 64]  ·  {blocks['active_aiv']} AIVs  ·  blocks {first}–{hi} of {len(actual)}",
             size=13, color='#395566')
    handles = [Patch(color=color, label=label) for color, label in STAGES.values()]
    handles.append(Line2D([], [], marker='|', linestyle='', color=INK, label='CA descriptor issue'))
    fig.legend(handles=handles, frameon=False, loc='upper left', bbox_to_anchor=(.158,.882), ncol=4, fontsize=11)
    labels = [f"UB {i+1}  ·  slot {blocks['actual_tiles_with_addresses'][i]['slot']}" for i in indices]
    for ax, side, title, end in [(top, model, 'Resource model · block service', blocks['model_us']),
                                  (bottom, actual, 'CA trace · '+blocks['ca_core']+' (last to finish)', blocks['ca_us'])]:
        for row, i in enumerate(indices):
            for offset, (key, (color, _)) in zip((-.21, 0, .21), STAGES.items()):
                start, stop = side[i][key]
                ax.barh(row+offset, stop-start, left=start, height=.145, color=color, zorder=3)
                if ax is bottom and key != 'Vector':
                    ax.vlines(start, row+offset-.13, row+offset+.13, colors=color, lw=1, zorder=4)
        ax.set_yticks(range(count), labels)
        ax.set_ylim(count-.44, -.60)
        ax.set_xlim(left, right)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#d5e1e7')
        for row in range(count-1):
            ax.axhline(row+.5, color='#e2eaed', lw=.8)
        if left <= end <= right:
            ax.axvline(end, ls=(0,(3,3)), lw=1.1, color='#41596a')
        ax.set_title(title, loc='left', size=15, weight='bold', color=INK, pad=7)
        ax.set_title(f'full {end:.3f} µs', loc='right', size=15, weight='bold', color=INK, pad=7)
        ax.tick_params(length=0, pad=5, labelcolor='#344e60')
        ax.spines[:].set_visible(False)
    top.tick_params(labelbottom=False)
    bottom.set_xlabel('Time since first device instruction (µs)', color=INK)
    gap = [m['Write'][1]-c['Write'][1] for m,c in zip(model, actual, strict=True)]
    overview.axvspan(first-.45, hi+.45, color='#e7dfea', zorder=0)
    overview.axhline(0, color='#a8b6c0', lw=.8)
    overview.plot(range(1,len(gap)+1), gap, color='#465f73', lw=1.6)
    overview.set(xlim=(.5,len(gap)+.5), ylabel='Δ (µs)', xlabel='UB block · in execution order')
    overview.set_title('Write completion, model − CA  |  shaded blocks shown above', loc='left', fontsize=10, color=INK)
    overview.spines[['top','right']].set_visible(False)
    overview.tick_params(labelsize=8, length=2, color='#8496a1')
    fig.text(.17, .022, 'Model bars allocate shared service. CA bars include waiting through retirement. Both panels keep the full-kernel time origin.',
             color='#445c69', fontsize=9)
    fig.savefig(path, dpi=150, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sku', choices=('dt','pr'), default='dt')
    parser.add_argument('--first', type=int, default=62)
    parser.add_argument('--count', type=int, default=5)
    args = parser.parse_args()
    blocks = load_blocks(args.sku)
    name = f'pro-256ki-{args.sku}-blocks-{args.first}-{args.first+args.count-1}.png'
    draw(blocks, args.first, args.count, OUT / name)
