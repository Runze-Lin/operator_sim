"""Visualize observed physical writes and whole-kernel latency on legacy Cast data."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

OUT=Path(__file__).resolve().parents[3]/'results/cast-capacity'
MIB=2**20
LABELS={(1048576,32):'8 × 128ki / C32',(2097152,32):'8 × 256ki / C32',
        (262144,64):'8 × 32ki / C64',(8192,5120):'8192 × 5120',(4096,7168):'4096 × 7168',
        (262144,32):'8 × 32ki / C32'}


def render():
    data=json.loads((OUT/'controller-periods.json').read_text())
    rows=[r for r in json.loads((OUT/'replay.json').read_text())['rows'] if r['group']!='small']
    fig=plt.figure(figsize=(12,8),layout='constrained',facecolor='white')
    grid=fig.add_gridspec(nrows=2,ncols=2,height_ratios=[1.3,1])
    ax=fig.add_subplot(grid[0,:])
    colors=['#086a84','#af5018','#6b659c','#45805c','#bb5f77']
    for s,col in zip(data,colors):
        xs=[0];ys=[0]
        for pt in s['intervals']:
            xs.append(xs[-1]+pt['read_bytes']/MIB)
            ys.append(ys[-1]+pt['write_bytes']/MIB)
        ax.plot(xs,ys,linewidth=2,color=col,label=LABELS[tuple(s['shape'])])
        ax.scatter(xs[-1],ys[-1],s=27,color=col)
    xs=list(range(129));ax.plot(xs,[max(0,(3*x-128)*2/3) for x in xs],color='#333',ls='--',lw=1.5,label='Uniform 128-MiB cache, 2:1 write/read')
    ax.set(xlabel='Cumulative physical reads (MiB)',ylabel='Cumulative physical writes (MiB)',
        title='950DT • physical bus traffic in successive controller intervals')
    ax.legend(loc='upper left',ncol=2,fontsize=9,frameon=False);ax.grid(alpha=.18)
    for k,sku in enumerate(['Ascend950DT_9582','Ascend950PR_957b']):
        side=fig.add_subplot(grid[1,k]);pairs=[r for r in rows if r['chip']==sku]
        for i,r in enumerate(pairs):
            m=r['candidate_model_us'];label=i==0
            side.plot([i,i],[min(r['ca_us'],m['baseline']),max(r['ca_us'],m['baseline'])],color='#b0b8bc',lw=1)
            side.scatter(i-.17,r['ca_us'],s=45,color='#172e37',label='CA full window' if label else None)
            side.scatter(i,m['baseline'],s=48,marker='x',color='#ba621b',label='v0.21 frozen' if label else None)
            side.scatter(i+.14,m['two_slot'],s=32,marker='s',facecolors='none',edgecolors='#0b7f8a',label='Two-slot candidate' if label else None)
            if k:side.scatter(i+.27,m['pressure'],s=24,marker='^',color='#86549e',label='Retention sensitivity' if label else None)
        side.set_xticks(range(len(pairs)),[LABELS[tuple(r['shape'])] for r in pairs],rotation=25,ha='right',fontsize=8)
        side.set(ylabel='Kernel time (µs)',title=f'{sku.split("_")[0][6:]} • matched historical windows')
        side.set_ylim(bottom=0);side.grid(axis='y',alpha=.18);side.legend(loc='upper left',fontsize=8,frameon=False)
    fig.savefig(OUT/'controller-and-latency.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    render()
