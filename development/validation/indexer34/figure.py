"""Reproduce the two-panel worked example from stored original-event summaries."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from operator_model import estimate
from operator_model.indexer_pipeline import _prices

shape = [1,2,3072,8,128]
chip = 'Ascend950PR_957b'
result = estimate('lightning_indexer',shape,chip=chip)
obs, = [r for r in json.loads((ROOT/'results/v0.34/indexer/physical.json').read_bytes())
        if r['shape']==shape and r['chip']==chip]
s = result.projection['stages_us']
ticks = 1650
model = result.latency_us
ca = obs['duration_ticks']/ticks
first = s['first_mmad']
last = first+s['page_intervals']+s['last_mmad']
hist_start = last+s['score_to_histogram']
p = _prices()
from operator_model import load_chip
target = load_chip(chip)
vec_rate = target.vector_tops['32']*1000/(target.frequency_ghz*target.vector_cores)
hist_span = (p['histogram_span'][0]+12*p['histogram_span'][1])*p['vector_ops_cycle']/vec_rate/ticks

fig, axes = plt.subplots(2,1,figsize=(11.4,6.4),sharex=True)
purple,teal,orange,gray = '#7654a5','#137a80','#bd5433','#dce6eb'
def bar(ax, y, start, stop, color):
    ax.broken_barh([(start,stop-start)],(y-.17,.34),facecolors=color)
def format_panel(ax, title, end):
    ax.set_title(title,loc='left',weight='bold',fontsize=13,pad=12)
    ax.set_title(f'{end:.3f} µs',loc='right',weight='bold',fontsize=13,pad=12)
    ax.axvline(end,color='#476477',linestyle=(0,(3,3)),lw=1)
    ax.grid(axis='x',color='#dae3e7',lw=.7)
    ax.set_axisbelow(True)
    ax.set_yticks([3,2,1,0],['Start to first MMAD','AIC · 24 KV pages','AIV · histogram','Final drain'])
    ax.set_ylim(-.5,3.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

a,b=axes
format_panel(a,'Resource model',model)
bar(a,3,0,first,gray)
for i in range(24):
    start=first+i*s['page_intervals']/23
    bar(a,2,start,start+s['last_mmad'],purple)
bar(a,1,hist_start,hist_start+hist_span,teal)
bar(a,0,hist_start+hist_span,model,orange)
format_panel(b,'CA trace · last score AIC and last histogram AIV',ca)
bar(b,3,0,obs['critical_mmad_ticks'][0][0]/ticks,gray)
for x,y in obs['critical_mmad_ticks']:
    bar(b,2,x/ticks,y/ticks,purple)
for x,y in obs['sort_ticks']:
    bar(b,1,x/ticks,y/ticks,teal)
bar(b,0,max(y for _,y in obs['sort_ticks'])/ticks,ca,orange)
b.set_xlabel('Device time from the first kernel instruction (µs)')
b.set_xlim(0,11.15)
fig.suptitle('950PR LightningIndexer  |  Q = 2, KV = 3072, G = 8, D = 128',
             x=.21,ha='left',fontsize=18,weight='bold',color='#19394a')
fig.text(.21,.906,'BF16 · 1 scoring AIC · 2 histogram AIVs · 128 keys per page',color='#456477')
fig.legend(handles=[Patch(color=purple,label='MMAD'),Patch(color=teal,label='Histogram'),Patch(color=orange,label='Output / retirement')],
           loc='upper left',bbox_to_anchor=(.20,.895),ncol=3,frameon=False)
fig.text(.21,.042,'Model page starts use the mean issue interval. CA bars show positive-duration instructions.\n'
         'Orange spans include all remaining work until retirement. A blank interval can contain other device activity.',
         fontsize=9,color='#456477')
fig.subplots_adjust(left=.21,right=.96,top=.76,bottom=.17,hspace=.6)
fig.savefig(ROOT/'docs/examples/figures/lightning-indexer-v34-pr.png',dpi=150,facecolor='white')
