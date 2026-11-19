"""Draw a two-tile detail from the qualified Pro B8×128ki acquisition."""
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/operator-sim-prefill24')
from matplotlib import use
use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'results/v0.23/pro-prefill/pro-b8-128ki-dt-blocks.json'
OUT = ROOT/'results/v0.24/pro-128ki-tail.png'


def draw():
    d = json.loads(SOURCE.read_text())
    assert d['model_tile_elements']==d['ca_tile_elements']
    assert len(d['ca_tiles'])==len(d['model_tiles'])==50
    colors = {'Read':'#167887','Vector':'#7954a5','Write':'#b34d2e'}
    labels = {'Read':'MTE2 · read','Vector':'Vector · convert','Write':'MTE3 · write'}
    fig,axes = plt.subplots(2,1,figsize=(10.7,6.9),sharex=True)
    left=.205
    fig.subplots_adjust(left=left,right=.96,top=.76,bottom=.16,hspace=.52)
    ink='#182e3b'
    fig.suptitle('950DT Cast  |  last two UB blocks',x=left,y=.977,ha='left',color=ink,fontsize=22,fontweight='bold')
    fig.text(left,.896,'DeepSeek V4 Pro  ·  B8 × 128k prefill  ·  [1,048,576, 64]  ·  64 AIVs  ·  BF16 → FP32',color='#435b6a',fontsize=11)
    fig.legend(handles=[Patch(color=colors[s],label=labels[s]) for s in colors],frameon=False,ncol=3,
               loc='upper left',bbox_to_anchor=(left-.009,.875))
    for ax,source,time,title in zip(axes,('model_tiles','ca_tiles'),('model_us','ca_us'),
            ('Resource model · steady-state service','CA trace · '+d['ca_core']+' (last to finish)')):
        rows = d[source][-2:]
        for i,block in enumerate(rows):
            for j,stage in enumerate(colors):
                a,b=block[stage]
                ax.barh(3*i+j,b-a,left=a,height=.43,color=colors[stage],zorder=3)
        ax.axhline(2.5,color='#bdcfd8',lw=.85)
        ax.set_yticks(range(6),[f'UB {i} · {label}' for i in (49,50) for label in ('read','convert','write')],fontsize=10)
        ax.set_ylim(5.55,-.55)
        ax.tick_params(axis='y',length=0,pad=8,colors=ink)
        ax.grid(axis='x',color='#dce4ea')
        ax.spines[['top','right','left']].set_visible(False)
        ax.spines['bottom'].set_color('#aabdc8')
        ax.axvline(d[time],color='#354f5e',ls=(0,(3,3)),lw=1)
        ax.set_title(title,loc='left',fontsize=13,fontweight='bold',color=ink)
        ax.set_title(f'{d[time]:.3f} µs',loc='right',fontsize=13,fontweight='bold',color=ink)
    xmin=min(d[s][-2]['Read'][0] for s in ('model_tiles','ca_tiles'))
    axes[-1].set_xlim(math.floor(xmin)-.15,math.ceil(max(d['ca_us'],d['model_us']))+.1)
    axes[-1].xaxis.set_major_locator(MultipleLocator(1))
    axes[-1].set_xlabel('Device time from the earliest instruction on any AIV (µs)')
    fig.text(left,.058,'UB 49 has 21,120 elements. UB 50 has 13,696. The crop preserves the full-kernel time origin.\nModel bars allocate shared service; CA bars show descriptor lifetimes and the conversion interval.',fontsize=9,color='#425967')
    fig.savefig(OUT,dpi=150,facecolor='white')
    plt.close(fig)

if __name__=='__main__':
    draw()
