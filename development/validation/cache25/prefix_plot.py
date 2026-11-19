"""Plot separable *marginal* memory budgets and measured last-core DMA, on one device axis."""
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v0.25/cache'
os.environ.setdefault('MPLCONFIGDIR','/tmp/operator-sim-cache25')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

e=json.loads((OUT/'flash-128ki-pr-tail.json').read_bytes())
p=json.loads((OUT/'prefix-replay.json').read_bytes())['physical_frontiers']['pressure']
m=('#19828a','#b54c28');vec='#7955a3';ink='#19333f'
fig,axs=plt.subplots(2,1,figsize=(11.8,7.3),sharex=True)
left=.19
fig.subplots_adjust(left=left,right=.96,top=.752,bottom=.17,hspace=.60)
fig.text(left,.951,'950PR Cast  |  Flash B8 × 128k prefill',fontsize=23,weight='bold',color=ink)
fig.text(left,.902,'BF16 → FP32   ·   1,048,576 × 32   ·   56 AIVs   ·   29 UB blocks on the last core',fontsize=12,color='#415f6c')
fig.legend(handles=[Patch(facecolor=m[0],label='Read'),Patch(facecolor=vec,label='Convert'),Patch(facecolor=m[1],label='Write'),
                    Patch(facecolor='#fff',edgecolor='#637780',hatch='///',label='Equivalent link service')],
           frameon=False,ncol=4,loc='upper left',bbox_to_anchor=(left-.008,.879),fontsize=10)
x_min=10**4
for i,r in enumerate(e['rows']):
    t=r['tile'];prev=p[t-2];now=p[t-1]
    # At this prefix both start and end are on the shared external resource.
    assert abs(prev['completion_us']-prev['mixed_memory_end_us'])<1e-7
    assert abs(now['completion_us']-now['mixed_memory_end_us'])<1e-7
    a,b=prev['completion_us'],now['completion_us']
    read=(now['prefix_fraction']-prev['prefix_fraction'])*64*2**20/1.4e6
    wr=(now['returned_external_write_MiB']-prev['returned_external_write_MiB'])*2**20/1.4e6
    assert abs((b-a)-read-wr)<1e-8
    # Stack the two equivalent byte budgets in one row. The split is accounting, not issue order.
    for begin,dur,col in [(a,read,m[0]),(a+read,wr,m[1])]:
        axs[0].barh(i,dur,left=begin,height=.22,color=col,edgecolor='white',hatch='///',lw=.5)
    axs[0].scatter(b,i,s=48,marker='|',color=ink,linewidth=1.5)
    for off,s,c in [(-.19,'Read',m[0]),(0,'Vector',vec),(.19,'Write',m[1])]:
        beg,end=r['ca_us'][s];axs[1].barh(i+off,end-beg,left=beg,height=.14,color=c)
    x_min=min(x_min,a,r['ca_us']['Read'][0])
for ax,title,end in zip(axs,['Pressure scenario  ·  added link service per block','CA trace  ·  '+e['actual_last_core']+' (last to finish)'],
                    [p[-1]['completion_us'],e['ca_us']]):
    ax.set_title(title,loc='left',color=ink,fontsize=13,weight='bold');ax.set_title(f'{end:.3f} µs',loc='right',color=ink,fontsize=14,weight='bold')
    ax.axvline(end,color=ink,ls=(0,(3,3)),lw=1);ax.grid(axis='x',color='#dce4e8');ax.set_axisbelow(True);ax.spines[:].set_visible(False)
    ax.set_ylim(2.53,-.53);ax.set_yticks(range(3),[f"UB {t['tile']}  ·  slot {t['slot']}" for t in e['rows']],fontsize=10);ax.tick_params(axis='y',length=0,pad=9)
    for q in [.5,1.5]:ax.axhline(q,color='#dce4e8',lw=.7)
axs[1].set_xlim(int(x_min)-.25,76.1);axs[1].set_xlabel('Device time from the first instruction on any AIV (µs)',labelpad=8)
fig.text(left,.085,'Top uses cumulative payload and L2 residency. The read/write split prices link bytes; only the block completion is a timing forecast.',fontsize=9,color='#496471')
fig.text(left,.052,'Bottom matches real GM/UB addresses and VCVT groups. The full-block payload is 21,120 elements; the measured last block has 7,832.',fontsize=9,color='#496471')
fig.savefig(OUT/'flash-128ki-pr-prefix.png',dpi=155);plt.close(fig)
