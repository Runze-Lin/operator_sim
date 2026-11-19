"""Compare model and CA at the same per-AIV milestones, with held-process calibration."""
from collections import defaultdict
from math import ceil,isclose
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from operator_model import estimate, load_chip
from operator_model.rms_v12 import parameters
from development.validation.sku19.campaign import read,write
from development.validation.rms_dispatch_audit.extract import OUT
from development.validation.rms_dispatch_audit.schedule import fit, price, launch_class, useful

DATA=OUT/'expanding-sibling.json'
FIG=ROOT/'docs/examples/figures/rms-expanding-sibling.png'


def extract():
    archive=read(OUT/'core-windows.json')['rows']
    o=next(r for r in archive if r['shape']==[4,4608])
    group=Path(o['source']).parent
    blocked={tuple(r['shape']) for r in archive if Path(r['source']).parent==group}
    p=fit([r for r in archive if tuple(r['shape']) not in blocked and Path(r['source']).parent!=group])
    chip=load_chip('Ascend950DT_9582')
    f=estimate('rmsnorm',o['shape'],chip=chip)
    detail=f.projection['timing_diagnostics']
    target=price(o['shape'],chip,o['prior'],p,f)
    active=f.active_aiv;tokens,width=o['shape']
    assert tokens==active==4
    # All four stats fit in the same 128-byte line. Model releases have a
    # distribution within each state, so physical labels on resident lanes are
    # only order-statistic slots. The newly activated sibling has a unique slot.
    groups=defaultdict(list)
    ca=[]
    for b in o['blocks']:
        state,_,_=launch_class(b['block'],active,o['prior'])
        groups[state].append(b)
        e=b['stats'][0]
        ca.append(dict(block=b['block'],core=b['core'],lane=b['lane'],state=state,
                      arrive=b['begin'],issue=e['issue'],stat_done=e['done'],y_issue=e['y_issue'],
                      y_done=e['y_done'],retire=b['end']))
    scheduled=[]
    for state,blocks in groups.items():
        start=p['arrival'][state]
        cont=p['resident_issue_cycles_per_active']*(active-1) if state=='resident' else 0
        base=start+p['lead'][state]+cont+useful(width,1,detail['read_cycles_per_kib'])
        n=len(blocks)
        for rank,b in enumerate(blocks):
            issue=base+p['spread'].get(state,0)*((rank+1)/(n+1)-.5)
            off,rate=parameters()['normalize']
            y_start=issue+off+rate*ceil(width/128)*detail['vector_price_cycles_per_reference_cycle']
            off,rate=p['y_cycles']
            y_end=y_start+off+rate*2*width/1024
            scheduled.append(dict(block=b['block'],core=b['core'],lane=b['lane'],state=state,arrive=start,
                                  issue=issue,y_issue=y_start,y_done=y_end))
    tail=0
    for r in sorted(scheduled,key=lambda r:r['issue']):
        tail=max(r['issue']+p['solo'],tail+p['sector_interval'])
        r.update(stat_done=tail,retire=max(tail,r['y_done'])+p['drain'])
    assert isclose(max(r['retire'] for r in scheduled),target*1650,abs_tol=1e-6)
    assert max(r['retire'] for r in ca)==o['window']
    write(DATA,dict(shape=o['shape'],H=o['prior'],clock_cycles_per_us=1650,predicted_us=target,ca_us=o['window']/1650,
                    original_known_us=estimate('rmsnorm',o['shape'],{'resident_aiv':3},chip=chip).latency_us,
                    model=sorted(scheduled,key=lambda b:b['block']),ca=ca,source=o['source'],
                    role='held process and all its shapes removed from stage calibration; historical replay',
                    caveat='For the three resident AIVs, model rows show local release rank within that state; hardware may permute those ranks. The newly enabled lane is individually identified.'))


def draw():
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    r=read(DATA);freq=r['clock_cycles_per_us']
    colors=['#247e88','#6d54a2','#3e6eb1','#ba502c']
    titles=['First instruction → rstd issue','Normalize to y issue','rstd write','y write']
    fig,axes=plt.subplots(2,1,figsize=(11.2,6.3),sharex=True)
    fig.subplots_adjust(left=.23,right=.97,top=.77,bottom=.18,hspace=.55)
    fig.text(.23,.955,'950DT RMSNorm  |  4 × 4608',fontsize=21,fontweight='bold',color='#193342')
    fig.text(.23,.905,'3 warm AIVs + the second lane on core16   ·   4 × 4-byte statistics share one line',fontsize=10,color='#466172')
    fig.legend(handles=[Patch(color=c,label=t) for c,t in zip(colors,titles)],loc='upper left',bbox_to_anchor=(.22,.88),ncol=4,frameon=False,fontsize=9)
    for ax,group,title,window in zip(axes,('model','ca'),('Resource model   ·   launch and store schedule','CA trace   ·   same physical AIVs'),(r['predicted_us'],r['ca_us'])):
        for i,b in enumerate(r[group]):
            for y,(start,end),color in zip([i-.19,i+.02,i+.21,i+.21],
                    [('arrive','issue'),('issue','y_issue'),('issue','stat_done'),('y_issue','y_done')],colors):
                if start=='y_issue':y+=.14
                ax.barh(y,(b[end]-b[start])/freq,left=b[start]/freq,height=.13,color=color)
            ax.plot([b['issue']/freq]*2,[i+.10,i+.28],color=colors[2],lw=1)
        ax.axvline(window,color='#375461',ls='--',lw=1)
        ax.set_title(title,loc='left',fontweight='bold',color='#203743',fontsize=12,pad=8)
        ax.set_title(f'{window:.3f} µs',loc='right',fontweight='bold',fontsize=12,pad=8)
        ax.set_yticks(range(4),[f"block {b['block']} · core{b['core']}.v{b['lane']}" for b in r[group]],fontsize=9)
        ax.set_ylim(3.55,-.50);ax.grid(axis='x',alpha=.20);ax.set_axisbelow(True)
        for spine in ax.spines.values():spine.set_visible(False)
        ax.tick_params(axis='y',length=0)
    axes[-1].set_xlabel('Device time from first instruction on any core (µs)')
    fig.text(.23,.06,'Resident model rows order releases within the line; their physical order can vary.\nThe fourth block activates a sibling on a previously active physical core.',fontsize=9,color='#466172')
    fig.savefig(FIG,dpi=160,facecolor='white');plt.close(fig)


if __name__=='__main__':
    if '--replot' not in sys.argv:
        extract()
    draw()
