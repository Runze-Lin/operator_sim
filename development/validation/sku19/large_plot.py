"""Plot the portable model over a token sweep; distinguish same-width complete CA observations."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from development.validation.sku19.campaign import CHIPS,OUT,read
from operator_model import estimate

def plot(pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained',sharex=True)
    rows=[2**n for n in range(3,20)]
    for axs,op in zip(axes,('rmsnorm','quant')):
        for side,color in zip(CHIPS,('#27759e','#b85318','#76539c')):
            chip=CHIPS[side]
            us=[];iters=[]
            for n in rows:
                m=estimate(op,(n,5120),chip=chip,timing_mode='resource')
                us.append(m.latency_us);iters.append(m.projection['target_plan']['iterations'])
            label='970DT projection' if side=='future' else '950'+side.upper()+' model'
            axs[0].plot(rows,us,'--' if side=='future' else '-',color=color,label=label)
            axs[1].plot(rows,iters,'--' if side=='future' else '-',color=color,label=label)
            if side!='future':
                g=[r for r in pairs if r['op']==op and r['shape'][1]==5120]
                for r in g:axs[0].scatter(r['shape'][0],r[side+'_ca_us'],marker='o' if side=='dt' else '^',s=50,color=color,edgecolor='black',zorder=3)
        for a in axs:
            a.set_xscale('log',base=2);a.set_yscale('log');a.grid(True,which='major',alpha=.25)
            a.axvline(262144,color='grey',lw=1,ls=':')
        axs[0].set_ylabel('Device time (µs)');axs[1].set_ylabel('UB iterations on busiest core')
        axs[0].set_title(op+' [tokens, 5120]')
    axes[0,0].scatter([],[],color='#27759e',edgecolor='black',label='950DT complete CA')
    axes[0,0].scatter([],[],marker='^',color='#b85318',edgecolor='black',label='950PR complete CA')
    axes[0,0].legend(fontsize=7);axes[1,0].set_xlabel('Tokens in one dispatch');axes[1,1].set_xlabel('Tokens in one dispatch')
    fig.savefig(OUT.parent/'prefill-scaling.png',dpi=145)
    plt.close(fig)
