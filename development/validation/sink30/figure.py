"""Align each physically reachable issue schedule to the complete real dispatch at T=241."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from register import ROOT, OUT, PR, digest, read
from operator_model import estimate, load_chip
from operator_model.sink_history import price


def model_events(epochs, iterations):
    records = [(0, epochs['tiling_begin'], epochs['tiling_end']),
               (1, epochs['read_begin'], epochs['read_end']),
               (2, epochs['setup_vf'], None), (3, epochs['softmax_vf'], None),
               (4, epochs['first_col_vf'], None)]
    for i in range(iterations - 1):
        row = epochs['first_row_vf'] + (epochs['first_pair_period'] +
              (i - 1) * epochs['steady_pair_period'] if i else 0)
        records.extend([(5, row, None), (6, row + epochs['steady_row_to_col' if i else 'first_row_to_col'], None)])
    return records + [(7, epochs['scatter_vf'], None), (8, epochs['write_begin'], epochs['write_end'])]


def render():
    prior_path = ROOT/'results/v0.28/fusion/eligibility.trace.json'
    ref = read(prior_path)
    assert digest(ROOT/ref['trace']) == ref['trace_sha256']
    receipt = read(ROOT/ref['receipt'])
    original = next(x for x in receipt['rows'] if x['stage']['trace_sha256'] == ref['aggregate_sha256'])
    assert original['qualified'] and original['stage']['kernel'].startswith('MhcSinkhorn_')
    assert all(n == 34 for n in original['stage']['vf_calls_per_core'].values())
    chip = load_chip(PR)
    t, iterations = original['shape'][0], original['iters']
    attrs = {'num_iters': iterations}
    p = estimate('mhc_sinkhorn', [t, 4], attrs, chip=chip)
    diag = p.projection['timing_diagnostics']
    colors = {'first':'#af5b2b','expand':'#4972a4','resident':'#157d76'}
    models = {}
    for name, h in [('first', 0), ('expand', 1), ('resident', p.active_aiv)]:
        time, _, _, cond = price([t, 4],chip,{**attrs,'resident_aiv':h})
        models[name] = {'end_us': time, 'representative_h': h, 'events': model_events(cond['issue_epochs_us'], iterations)}
        assert abs(time - diag['scenario_us'][name]) < 1e-12
    source = {k:ref[k] for k in ('origin','core','trace','trace_sha256','aggregate_sha256','receipt','labels','ca','ca_us')}
    size = original['stage']['written_bytes']
    doc = dict(**source,shape=[t,4,4],iterations=iterations,sku=PR,real_h=original['h'],histories=models,
               point_us=p.latency_us,active_aiv=p.active_aiv,output_bytes=size,
               source_figure_data=str(prior_path.relative_to(ROOT)),source_sha256=digest(prior_path),
               event_format='[lane, begin_us, end_us]; end=null for a modeled issue-only VF',
               interpretation='States are mutually exclusive. Average of their endpoints is the shape-only output.')
    (OUT/'unknown-t241-pr.json').write_text(json.dumps(doc,indent=2)+'\n')
    fig, axes = plt.subplots(2,1,figsize=(12,8.4),sharex=True)
    fig.subplots_adjust(left=.197,right=.955,top=.79,bottom=.14,hspace=.35)
    ink='#233946'
    fig.text(.197,.938,'950PR   |   MhcSinkhorn   ·   241 × 4 × 4',fontsize=23,weight='bold',color=ink)
    fig.text(.197,.895,'16 iterations   ·   49 AIVs   ·   up to 5 matrices per core   ·   one register pass',fontsize=12,color='#526875')
    handles=[Line2D([],[],color=c,linewidth=3,label=n.capitalize()+f"   {models[n]['end_us']:.3f} µs") for n,c in colors.items()]
    fig.legend(handles=handles,ncol=3,bbox_to_anchor=(.19,.866),loc='center left',frameon=False,fontsize=10)
    for ax in axes:
        ax.set_yticks(range(9),ref['labels'],fontsize=9.5)
        ax.set_ylim(8.68,-.68)
        ax.tick_params(axis='y',length=0)
        ax.set_axisbelow(True);ax.grid(axis='x',color='#d9e2e5',lw=.8)
        ax.spines[['left','right','top']].set_visible(False)
        ax.spines['bottom'].set_color('#9aafb7')
    top, bottom = axes
    top.set_title('Resource model  ·  reachable dispatch histories',loc='left',color=ink,weight='bold',fontsize=13)
    top.set_title(f'Unknown H    {p.latency_us:.3f} µs',loc='right',color=ink,weight='bold',fontsize=13)
    for (name,m),offset in zip(models.items(),(-.23,0,.23)):
        for lane,start,end in m['events']:
            if end is not None:top.barh(lane+offset,end-start,left=start,height=.13,color=colors[name],zorder=2)
            else:top.vlines(start,lane+offset-.09,lane+offset+.09,color=colors[name],lw=1.5,zorder=3)
        top.plot(m['end_us'],8+offset,marker='>',color=colors[name],ms=5)
    top.axvline(p.latency_us,color=ink,linestyle=(0,(4,3)),lw=1.4)
    bottom.set_title(f"CA trace  ·  {ref['core']}  ·  resident (H = {original['h']})",loc='left',color=ink,weight='bold',fontsize=13)
    bottom.set_title(f"Complete window    {ref['ca_us']:.3f} µs",loc='right',color=ink,weight='bold',fontsize=13)
    for lane,start,end in ref['ca']:
        col = '#157d76' if lane in (0,1) else '#ae5a31' if lane==8 else '#67569b'
        bottom.barh(lane,end-start,left=start,height=.37,color=col,alpha=.32 if 2<=lane<=7 else .9,zorder=2)
        if 2<=lane<=7:bottom.vlines(start,lane-.24,lane+.24,color=col,lw=1.6,zorder=3)
    bottom.axvline(ref['ca_us'],color=ink,linestyle=(0,(4,3)),lw=1.4)
    bottom.set_xlim(0,7)
    bottom.xaxis.set_major_locator(MultipleLocator(.5))
    bottom.set_xlabel('Device time since first instruction on any AIV (µs)',fontsize=12,labelpad=7)
    fig.text(.197,.061,'VF ticks mark issue. Pale CA bars include dependency waits. Each model row contains three alternative schedules.',fontsize=9.8,color='#425963')
    fig.text(.197,.036,'The top dashed line marks the mean of the three complete windows. The lower dashed line marks the last physical END.',fontsize=9.8,color='#425963')
    fig.savefig(OUT/'unknown-t241-pr.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    render()
