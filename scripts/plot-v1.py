#!/usr/bin/env python3
"""Optional offline chart; matplotlib is not a predictor dependency."""
import json
from pathlib import Path
import argparse
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'toolchains/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--version',default='0.1',choices=['0.1','0.2'])
args=parser.parse_args()
out=root/'results'/('v'+args.version)
data=json.loads((out/'evaluation.json').read_text())
rows=[r for r in data['rows'] if r['split']=='holdout']
colors={'quant':'#1b9e77','matmul':'#377eb8','fa':'#984ea3'}
fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
limit=max(max(r['predicted_us'],r['reference_us']) for r in rows)*1.15
axes[0].fill_between([0,limit],[0,.9*limit],[0,1.1*limit],color='#e8edf2',label='+/-10%')
axes[0].plot([0,limit],[0,limit],color='#8895a7',lw=1)
for op,color in colors.items():
    selected=[r for r in rows if r['op']==op]
    axes[0].scatter([r['reference_us'] for r in selected],[r['predicted_us'] for r in selected],label=op,color=color,s=55)
axes[0].set(xlabel='CA instruction makespan (us)',ylabel='Frozen prediction (us)',title=f'{len(rows)} untouched holdout shapes',xlim=(0,limit),ylim=(0,limit))
axes[0].legend(frameon=False)
labels=[r['id'].replace('_test_',' / ') for r in rows]
axes[1].barh(labels,[100*r['ape'] for r in rows],color=[colors[r['op']] for r in rows])
axes[1].axvline(10,color='#8895a7',ls='--',lw=1)
axes[1].set(xlabel='Absolute percentage error (%)',title='Per-shape error (uniform weight)')
axes[1].invert_yaxis()
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x',alpha=.15)
fig.savefig(out/'holdout-error.png',dpi=180)
fig.savefig(out/'holdout-error.svg')
print(out/'holdout-error.png')
