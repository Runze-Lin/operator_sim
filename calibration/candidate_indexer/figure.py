"""Draw the small candidate-indexer example using checked-in CA receipts and the public model."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from operator_model import estimate


HERE = Path(__file__).resolve().parent
COLORS = {'Conversions / gather': '#287f80', 'Dot / weighted scores': '#566bb1',
          'Membership': '#a07535', 'Selection / order': '#8056a1',
          'Descriptor / control': '#6a7986'}


def category(stage):
    if stage in ('aclnnGatherV2', 'aclnnCast'):
        return 'Conversions / gather'
    if stage in ('aclnnNeTensor', 'aclnnAll', 'aclnnInplaceMaskedFillScalar', 'candidate_setup'):
        return 'Membership'
    if stage in ('aclnnTopk', 'aclnnSort', 'histogram_and_store', 'position_order_and_map'):
        return 'Selection / order'
    if stage in ('first_mmad', 'sort_handoff', 'additional_shared_service'):
        return 'Descriptor / control'
    return 'Dot / weighted scores'


def main():
    plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'svg.hashsalt': 'operator-sim-indexer-example'})
    records = {route: {r['chip']: r for r in json.loads((HERE / (route+'.json')).read_text())}
               for route in ('dense', 'compact')}
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 5.8))
    fig.subplots_adjust(left=.22, right=.96, top=.78, bottom=.23, hspace=.42)
    for ax, sku in zip(axes, records['dense']):
        result = estimate('candidate_filtered_indexer', (1,1,32,128,800,75,512),
                          {'implementation': 'li_pipeline_projection', 'block_size': 8}, chip=sku)
        series = [list(result.projection['stages_us'].items())]
        for route in ('dense','compact'):
            series.append([(s['api'], s['window_us']) for s in records[route][sku]['stages']])
        for y, stages in enumerate(series):
            elapsed = 0.
            for stage, width in stages:
                if width > 0:
                    ax.barh(y, width, left=elapsed, height=.58, color=COLORS[category(stage)],
                            edgecolor='white', linewidth=.65)
                elapsed += width
            ax.text(elapsed+.38, y, f'{elapsed:.2f}', va='center', fontsize=9)
        ax.set_yticks([0,1,2], ['Fused model budget','CA full-score chain','CA candidate-first chain'])
        ax.invert_yaxis()
        ax.set_title(sku, loc='left', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['right','top','left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
    axes[-1].set_xlim(0, 41)
    axes[-1].xaxis.set_major_locator(MultipleLocator(5))
    axes[-1].set_xlabel('Cumulative device time (µs)')
    fig.suptitle('Flash candidate Indexer  |  800 keys, 600 eligible, 512 returned',
                 x=.03, y=.98, ha='left', fontsize=16, fontweight='bold')
    fig.legend([Patch(color=c) for c in COLORS.values()], COLORS.keys(), loc='upper left',
               bbox_to_anchor=(.02,.92), ncols=3, frameon=False, fontsize=9)
    fig.text(.03,.07,'Model: allocated critical-path costs for a proposed full-score fusion. Native fused CA: unavailable.\n'
             'CA: consecutive kernel windows, host gaps excluded. The ordinary LightningIndexer kernel in the compact chain includes top-k.',
             fontsize=9)
    path = HERE.parents[1] / 'docs/figures/candidate-indexer-ca.svg'
    fig.savefig(path, metadata={'Date': None})
    plt.close(fig)
    # The SVG backend leaves trailing spaces on multiline path coordinates.
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')


if __name__ == '__main__':
    main()
