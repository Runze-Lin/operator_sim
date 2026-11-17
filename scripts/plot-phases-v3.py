#!/usr/bin/env python3
"""Render existing quant phase evidence only; not a new profiler collection."""
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'toolchains/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
D=json.loads((ROOT/'results/v0.3/trace-analysis.json').read_text())['quant']
ids=['import_v1_q_cal_prefill','q2_test_ub_loop','q2_cal_ub_loop']
fig,axes=plt.subplots(3,1,figsize=(10,6),sharex=True,layout='constrained')
colors=['#238b8d','#9467bd']
for ax,id in zip(axes,ids):
 r=next(x for x in D if x['id']==id)
 ax.axvspan(0,r['prefix_us'],color='#e9edf2',zorder=0)
 for i,it in enumerate(r['iterations']):
  for y,events in [(2,[it['read']]),(1,it['vf']),(0,it['writes'])]:
   for e in events:ax.broken_barh([(e['start_us'],e['end_us']-e['start_us'])],(y-.3,.6),facecolor=colors[i%2],alpha=.75,edgecolor='white',linewidth=.5)
 ax.axvline(r['reference_us'],color='#e45756',ls='--',lw=1.2)
 ax.set_yticks([0,1,2],['Output DMA','VF commands','Input DMA'])
 ax.set_title(f"{r['shape']} — critical core {r['core']} — CA {r['reference_us']:.3f} us",loc='left',fontsize=10)
 ax.spines[['top','right','left']].set_visible(False)
 ax.grid(axis='x',alpha=.18)
axes[-1].set_xlabel('Time from global first instruction (us)')
axes[0].legend(handles=[Patch(color='#e9edf2',label='Prefix incl. tiling/control'),Patch(color=colors[0],label='UB iteration 1'),Patch(color=colors[1],label='UB iteration 2')],loc='upper left',bbox_to_anchor=(0,1.55),ncol=3,frameon=False)
fig.suptitle('Observed command intervals, including queueing — not exclusive execution service',fontsize=11)
fig.savefig(ROOT/'results/v0.3/quant-phase-evidence.png',dpi=160)
fig.savefig(ROOT/'results/v0.3/quant-phase-evidence.svg')
