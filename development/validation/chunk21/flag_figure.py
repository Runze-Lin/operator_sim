"""Plot the paired SET records from the 950PR archived raw instruction window."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

root=Path(__file__).resolve().parents[3]
audit=json.loads((root/'results/v0.21/chunk/pr-flag-audit.json').read_text())
fig,ax=plt.subplots(figsize=(10,3.5))
fig.subplots_adjust(left=.15,right=.97,bottom=.26,top=.73)
for pair in audit['pairs']:
    a,b=pair['previous_same_pipe_pair']['cycle'],pair['flagged']['cycle']
    row=2-pair['flag_id']
    ax.plot([a,b],[row,row],ls=':',lw=1,color='#728497')
    for x,col in [(a,'#187d82'),(b,'#ad3d28')]:
        ax.scatter([x],[row],s=52,color=col,zorder=3)
        ax.annotate(f'{x:,}',(x,row),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
ax.set(yticks=[2,1,0],yticklabels=['event 0','event 1','event 2'],xlim=(14000,34000),ylim=(-.45,2.55),
       xlabel='CA core cycle (partial retirement trace)')
ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:,.0f}'))
ax.set_axisbelow(True)
ax.grid(axis='x',color='#dde5e8')
for side in ('left','right','top'): ax.spines[side].set_visible(False)
ax.tick_params(axis='y',length=0)
fig.suptitle('950PR ChunkKdaFwd   |   repeated Cube → MTE1 event',x=.12,y=.99,ha='left',fontsize=16,weight='bold')
handles=[Line2D([],[],ls='',marker='o',color=c,label=l) for c,l in [('#187d82','Earlier SET'),('#ad3d28','SET flagged by CA')]]
ax.legend(handles=handles,ncol=2,bbox_to_anchor=(0,1.17),loc='upper left',frameon=False,borderpad=0)
fig.text(.15,.02,'core0.cubecore0 · one full 64-token chunk · 2 heads · no matching WAIT retired between each pair',fontsize=9)
fig.savefig(root/'docs/examples/figures/kda-chunk-pr-event.png',dpi=155)
