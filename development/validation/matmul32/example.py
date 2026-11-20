"""Draw the PR 128Ki wave budget against an archived critical core; no CA acquisition."""
from freeze import ROOT,OUT,PR,digest
from operator_model import estimate
import json
from statistics import mean
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def render():
    source=ROOT/'results/v0.31/matmul/long-pr/data.json';d=json.loads(source.read_bytes())
    e=estimate('matmul',d['shape'],chip=PR);diag=e.projection['timing_diagnostics']
    regs=diag['service_segments'];x=e.projection['stages_us']['first_mmad'];starts=[]
    for s in regs:
        for i in range(s['first'],s['end']):
            starts.append(x);x+=s['cycles_per_wave']/1650
    sets=[starts,[v/1650 for v in d['observed_first_mmad_cycles']]]
    assert len(sets[0])==len(sets[1])==147
    limits=[e.latency_us,d['ca_us']];q=[(0,36),(36,73),(73,110),(110,146),(146,147)]
    labs=['tiles 1–36','tiles 37–73','tiles 74–110','tiles 111–146','tile 147 / drain'];colors=['#167783','#1d8396','#366c98','#536fa6','#aa5d35']
    fig,axs=plt.subplots(2,1,figsize=(11,6.8),sharex=True);fig.subplots_adjust(left=.16,right=.94,top=.80,bottom=.16,hspace=.55)
    fig.text(.16,.94,'950PR MatMulV3   |   1,048,576 × 128 × 512',fontsize=20,weight='bold',color='#203741');fig.text(.16,.895,'B8 × 128Ki prefill · indexer.wk · 28 AICs · one B preload per AIC · 147 critical work units',fontsize=11,color='#365665')
    panels=['Resource model · wave service budget','CA trace · first MMAD of each tile on core7 (last to finish)'];rows=[]
    for ax,v,total,title in zip(axs,sets,limits,panels):
        v=v+[total];ax.barh(0,v[0],color='#bccbd0',height=.5)
        for row,(first,end) in enumerate(q):
            ax.barh(row,v[end]-v[first],left=v[first],color=colors[row],alpha=.65,height=.50)
            ax.vlines(v[first:end],row-.28,row+.28,colors=colors[row],lw=.6)
        ax.set_title(title,loc='left',fontsize=11,weight='bold');ax.set_title(f'{total:.3f} µs',loc='right',fontsize=12,weight='bold')
        ax.axvline(total,color='#344e5b',ls='--',lw=1);ax.set_yticks(range(5),labs,fontsize=9);ax.set_ylim(4.7,-.7);ax.spines[['left','top','right']].set_visible(False);ax.grid(axis='x',alpha=.25)
    axs[-1].set(xlim=(0,max(limits)*1.035),xlabel='Device time since first instruction across all cores (µs)')
    fig.text(.16,.065,'Ticks: wave budget boundaries above, observed per-tile MMAD issues below. The last interval ends at the full-device finish.',fontsize=9,color='#415966');fig.text(.16,.038,'Intervals can include data dependencies and overlapping engines. Bands are not exclusive HBM, MTE2 or Cube busy time.',fontsize=9,color='#415966')
    fig.savefig(OUT/'pr-128ki.png',dpi=140);plt.close(fig)
    for a,b in q[:-1]:rows.append(dict(first=a+1,last=b,predicted_cycles=mean((sets[0][i+1]-sets[0][i])*1650 for i in range(a,b)),observed_cycles=mean((sets[1][i+1]-sets[1][i])*1650 for i in range(a,b))))
    (OUT/'pr-128ki-figure.json').write_text(json.dumps(dict(shape=d['shape'],sku=PR,predicted_first_us=sets[0],observed_first_us=sets[1],full_windows_us=limits,quartiles=rows,source_sha=digest(source),critical_core=d['core'],time_zero='earliest real instruction across every active AIC'),indent=2)+'\n')
if __name__=='__main__':render()
