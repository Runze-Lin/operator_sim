"""Two true launches of the identical [67,4] shape, with and without prior core coverage."""
import hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from observe import ROOT,OUT,read,save
COLORS=['#17788a','#17788a','#607985','#8659a4','#86639c','#345a92','#795799','#5f7185','#bc561d']

def make(sku):
    base=ROOT/'results/v0.26/sink';data=[]
    for group in ('first','grow'):
        src=base/f'independent-{sku}-67-{group}.json'
        if not src.exists():return
        d=read(src);data.append(d)
    output=OUT/(sku+'-same-input.png')
    if output.exists():return
    fig,axes=plt.subplots(2,2,figsize=(12,7.5),sharex=True,sharey=True)
    fig.subplots_adjust(left=.165,right=.974,bottom=.165,top=.79,hspace=.45,wspace=.20)
    fig.text(.165,.944,f'{sku.upper()}  |  Sinkhorn · same input and ten iterations',fontsize=19,fontweight='bold',color='#1f3945')
    fig.text(.165,.889,'67 FP32 matrices (4 × 4)  ·  34 active AIVs  ·  22 VF issues on each AIV',fontsize=12,color='#48606c')
    max_t=max(max(d['model_full_us'],d['ca_full_us']) for d in data)*1.035
    for ix,d in enumerate(data):
        for iy,key,tkey in [(0,'model','model_full_us'),(1,'ca','ca_full_us')]:
            ax=axes[iy,ix];duration=d[tkey]
            family='First launch' if ix==0 else 'Same-family reuse'
            label=f"{family} (H={d['prior_h']})" if iy==0 else f"CA · {d['critical_core']}"
            ax.set_title(label,loc='left',fontsize=11,pad=7,color='#213c49',fontweight='bold')
            ax.set_title(f'{duration:.3f} µs',loc='right',fontsize=10,pad=7,color='#213c49')
            for row in d[key]:
                lane,start,end=row[:3]
                if end is not None:ax.barh(lane,max(0,end-start),left=start,height=.32,alpha=1 if lane in (0,1,8) else .25,color=COLORS[lane],zorder=3)
                if lane not in (0,1,8):ax.plot([start,start],[lane-.24,lane+.24],color=COLORS[lane],lw=1.25,zorder=4)
            ax.set_ylim(8.65,-.65);ax.set_xlim(0,max_t);ax.set_yticks(range(9),['Tiling · MTE2','Input · MTE2','Setup · VF','Softmax · VF','First column · VF','Row · VF','Column · VF','Scatter · VF','Output · MTE3'],fontsize=9.5)
            ax.axvline(duration,color='#304c5a',ls=(0,(3,3)),lw=1);ax.grid(axis='x',color='#d5dfe3',lw=.6)
            ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_color('#afbec5');ax.tick_params(axis='y',length=0)
    fig.text(.165,.825,'RESOURCE MODEL',fontsize=10,color='#2e4853',fontweight='bold');fig.text(.165,.468,'MEASURED ALL-CORE WINDOW',fontsize=10,color='#2e4853',fontweight='bold')
    fig.text(.54,.104,'Device time from the first instruction on any core (µs)',ha='center',fontsize=11)
    fig.text(.165,.049,'Line = VF issue; pale band = measured descriptor duration.  Dashed line = end of the full window.  Both launches use the same clock and axes.',fontsize=9,color='#49606b')
    fig.savefig(output,dpi=165);plt.close(fig)
    save(OUT/(sku+'-same-input.json'),dict(source='two independently frozen executions; global-first instruction as zero for each launch',first=data[0],reuse=data[1]))
if __name__=='__main__':
    for sku in ('950dt','950pr'):make(sku)
