"""Redraw an MX worked example; optional extraction uses both old DT/PR receipts."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from calibration.__main__ import read, require
from calibration.msopprof import sha
from .shards import trace_events


def collect(raw, mixed):
    from operator_model import estimate
    receipt = read(raw / 'receipt.json')
    phases = read(raw / 'phases.json')
    case = receipt['case']
    stage, = receipt['stages']
    prediction = estimate('matmul', case['shape'], case['attrs'], dtype=case['dtype'], chip=case['chip'])
    core = phases['last_mmad_core'].split('.')[0]
    events = defaultdict(list)
    first, last = float('inf'), 0
    directory, = (raw/'profile').glob('OPPROF_*/simulator')
    for name, expected in stage['physical_shard_sha256'].items():
        filename = directory/name/'trace.json'
        require(sha(filename)==expected, 'Physical trace identity changed')
        for event in trace_events(filename):
            if event['ph'] != 'X':
                continue
            start, end = event['ts'], event['ts']+event['dur']
            first, last = min(first, start), max(last, end)
            op = event['name']
            if mixed:
                if name.split('.')[0] != core:
                    continue
                if op=='MMAD_MX':
                    lane = 'MMAD'
                elif op=='MOV_UB_TO_L1':
                    lane = 'UB to L1 · AIV '+name[-1]
                else:
                    continue
            else:
                if op != 'MMAD_MX':
                    continue
                width = json.loads(event['args']['detail'])['n']
                lane = 'MMAD · N='+str(width)
            events[lane].append((start, end))
    require(abs(last-first-phases['full_window_us'])<.002, 'Window mismatch')
    compact = {}
    for lane, intervals in events.items():
        # Union across cores removes overlaps without filling idle gaps.
        union = []
        for start, end in sorted(intervals):
            if union and start <= union[-1][1]:
                union[-1][1] = max(end, union[-1][1])
            else:
                union.append([start, end])
        compact[lane] = [[round(a-first,6), round(b-first,6)] for a,b in union]
    return dict(chip=case['chip'],shape=case['shape'],kernel=stage['kernel'],ca=last-first,
                model=prediction.latency_us,stages=prediction.projection['stages_us'],events=compact,
                cluster=core if mixed else 'all AICs',trace_sha256=stage['trace_sha256'])


def render(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    palette = {'Front end':'#586979','Read':'#117883','Unpack / handoff':'#b06715',
               'Load / Cube':'#744b9a','Page intervals':'#3974a6','Later waves':'#489379',
               'Write / shared':'#b95039'}
    category = {'launch':'Front end','core_arrival_spread':'Front end','read_fill':'Read',
                'inline_unpack':'Unpack / handoff','ub_to_l1':'Unpack / handoff','l1_to_l0':'Load / Cube',
                'page_intervals':'Page intervals','additional_waves':'Later waves','cube_tail':'Load / Cube',
                'write_tail':'Write / shared','exposed_shared_traffic':'Write / shared'}
    fig, axes = plt.subplots(2,1,figsize=(10,6.1),sharex=True)
    fig.subplots_adjust(left=.19,right=.97,bottom=.17,top=.80,hspace=.52)
    extent = max(max(r['ca'], r['model']) for r in data['runs'])
    for ax,run in zip(axes,data['runs']):
        sku=run['chip'].removeprefix('Ascend').split('_')[0]
        left=0
        require(run['stages'].keys() <= category.keys(), 'Unknown critical-path stage')
        for stage in category:
            us = run['stages'].get(stage, 0)
            if us:
                color=palette[category[stage]]
                ax.barh(0,us,left=left,height=.48,color=color)
                left+=us
        labels=['Model budget']
        for lane,intervals in sorted(run['events'].items()):
            y=len(labels)
            color = ('#9a610f' if 'AIV 1' in lane else '#c8872b') if 'AIV' in lane else (
                '#714f99' if lane in ('MMAD','MMAD · N=256') else '#407e85')
            ax.broken_barh([(a,b-a) for a,b in intervals],(y-.2,.4),facecolors=color)
            labels.append(lane)
        ax.set_yticks(range(len(labels)),labels)
        ax.set_ylim(len(labels)-.45,-.6)
        ax.axvline(run['ca'],color='#263f4a',ls='--',lw=1)
        ax.set_title(f"{sku}   ·   {run['cluster']}      Model {run['model']:.3f} µs    |    CA {run['ca']:.3f} µs",
                     loc='left',size=11,weight='bold')
        ax.set_axisbelow(True); ax.grid(axis='x',color='#dce3e6')
        ax.spines[['right','top','left']].set_visible(False)
        ax.tick_params(axis='y',length=0)
    axes[-1].set_xlim(0,extent*1.04)
    axes[-1].set_xlabel('Device time from earliest physical-core instruction (µs)')
    fig.suptitle(data['title'],x=.05,y=.975,ha='left',size=17,weight='bold')
    fig.legend(handles=[Patch(color=c,label=n) for n,c in palette.items()],
               loc='upper center',bbox_to_anchor=(.50,.92),ncol=4,frameon=False,fontsize=8)
    fig.text(.05,.04,'Model row sums exposed critical-path costs. CA bars span instruction issue to retirement; dashed line marks all-core completion.',size=8)
    fig.savefig(output,dpi=140,facecolor='white')
    plt.close(fig)


def page_markers(run, retained):
    """Match alternating AIV descriptors and consecutive AIC fragments to K pages."""
    require(run['trace_sha256'] == retained['trace_sha256'], 'Different physical trace')
    require(abs(run['model'] - retained['model_window_us']) < 1e-6, 'Different model window')
    counts = retained['mmads_per_page']
    model = list(zip(retained['model_handoff_retire_us'], retained['model_first_mmad_issue_us']))
    n = len(counts)
    require(n == len(model) == len(retained['model_handoff_retire_us'])
            == len(retained['model_first_mmad_issue_us']), 'Incomplete model page events')
    events = run['events']
    lanes = [events['UB to L1 · AIV 0'], events['UB to L1 · AIV 1']]
    mmads = events['MMAD']
    require([len(lane) for lane in lanes] == [(n + 1) // 2, n // 2], 'Incomplete AIV descriptors')
    require(all(k >= 1 for k in counts) and sum(counts) == len(mmads), 'Incomplete AIC fragments')
    measured = []
    fragment = 0
    for i, k in enumerate(counts):
        measured.append((lanes[i % 2][i // 2][1], mmads[fragment][0]))
        fragment += k
    for points, finish in ((model, run['model']), (measured, run['ca'])):
        require(all(0 <= handoff <= issue <= finish for handoff, issue in points),
                'Page event falls outside the device or dependency window')
    return model, measured


def render_pages(data, retained, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from math import ceil

    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(12, 6.8))
    fig.subplots_adjust(left=.075, right=.98, top=.77, bottom=.15, hspace=.46, wspace=.26)
    teal, purple, ink = '#087983', '#73459b', '#263e4c'
    runs = data['runs']
    xmax = ceil(max(max(r['ca'], r['model']) for r in runs)) + .12
    for col, run in enumerate(runs):
        require(run['shape'] == retained['shape'], 'Different worked example')
        marker_rows = page_markers(run, retained['runs'][run['chip']])
        sku = run['chip'].split('_')[0].removeprefix('Ascend')
        fig.text(axes[0, col].get_position().x0, .817, sku, weight='bold', size=14, color=ink)
        for row, points in enumerate(marker_rows):
            ax = axes[row, col]
            for page, (handoff, issue) in enumerate(points, start=1):
                ax.hlines(page, handoff, issue, color='#92a8ad', linewidth=1.3, zorder=2)
                ax.scatter(handoff, page, marker='o', s=31, color=teal, zorder=3)
                ax.scatter(issue, page, marker='D', s=28, color=purple, zorder=3)
            end = run['ca'] if row else run['model']
            ax.axvline(end, color=ink, linestyle='--', linewidth=1)
            label = 'CA trace  ·  ' + run['cluster'] if row else 'Resource model'
            ax.set_title(label, loc='left', color=ink, size=11)
            ax.set_title(f'{end:.3f} µs', loc='right', color=ink, size=11, weight='bold')
            ax.set_yticks(range(1, len(points)+1), [f'K page {i+1}' for i in range(len(points))])
            ax.set_ylim(len(points)+.48, .52)
            ax.set_xlim(0, xmax)
            ax.set_xticks(range(int(xmax) + 1))
            ax.set_axisbelow(True)
            ax.grid(axis='x', color='#e0e7e9', linewidth=.8)
            ax.spines[['right', 'top', 'left']].set_visible(False)
            ax.tick_params(axis='y', length=0, labelsize=9)
            ax.tick_params(axis='x', labelsize=9)
    fig.suptitle('A8W4 weight handoff by K page   |   ' + ' × '.join(map(str, retained['shape'])),
                 x=.035, y=.98, ha='left', size=18, color=ink, weight='bold')
    fig.legend(handles=[
        Line2D([], [], marker='o', color=teal, linestyle='none', label='UB → L1 retirement'),
        Line2D([], [], marker='D', color=purple, linestyle='none', label='First MMAD issue'),
        Line2D([], [], linestyle='--', color=ink, label='Full device window')],
        loc='upper left', bbox_to_anchor=(.026,.925), ncol=3, frameon=False, fontsize=10)
    fig.supxlabel('Device time from earliest physical-core instruction (µs)', y=.075, size=11)
    fig.text(.035, .026, 'Gray segments show elapsed time between the two events on one page. '
             'CA pages use the AIC with the last MMAD and its paired AIVs.', size=9, color=ink)
    fig.savefig(output, dpi=150, facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('data',type=Path)
    cli.add_argument('output',type=Path)
    cli.add_argument('--extract',nargs=2,type=Path,metavar=('RAW_DT','RAW_PR'))
    cli.add_argument('--pages', type=Path, metavar='PAGE_MARKERS',
                     help='Draw the worked two-slot A8W4 page markers against compact CA intervals.')
    args=cli.parse_args()
    if args.extract:
        first,second=[read(raw/'receipt.json')['case'] for raw in args.extract]
        require([first['chip'],second['chip']]==['Ascend950DT_9582','Ascend950PR_957b'] and
                all(first[key]==second[key] for key in ('contract','shape','attrs','dtype')) and
                first['contract'] in ('matmul_mx_native','matmul_mx_mixed'), 'Expected matching DT/PR MX cases')
        mixed=first['contract']=='matmul_mx_mixed'
        runs=[collect(raw,mixed) for raw in args.extract]
        title=('A8W4 weight-page handoff' if mixed else 'A4W4 main and tail MMAD waves')+'    |    '+ ' × '.join(map(str,first['shape']))
        args.data.write_text(json.dumps(dict(title=title,runs=runs),indent=2)+'\n')
    if args.pages:
        render_pages(read(args.data), read(args.pages), args.output)
    else:
        render(read(args.data),args.output)
