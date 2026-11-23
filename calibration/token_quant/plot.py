"""Render model time and the terminal CA shard on a common device clock for one held-out row."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from calibration.system_reference import archive
from operator_model import estimate

COLORS={'Front':'#ccd7dd','MTE2':'#167888','Vector + handoff':'#7657a9','MTE3':'#bb542e','Retire':'#737983'}

def plot(packed,output):
    bundle=archive(packed)
    sample,=[o for o in bundle['observations'] if o['receipt']['case']['id']=='token-fp8-9x1024-dt']
    phase=sample['terminal_phases']
    r=estimate('quant_token_fp8',[9,1024])
    start,rd,vec,wr= (r.projection['stages_us'][k] for k in ('first_payload_issue','read','vector_and_handoff','value_and_scale_writes'))
    fig,axes=plt.subplots(2,1,figsize=(9.5,5.7),sharex=True)
    for ax in axes:
        ax.set_yticks([2,1,0],['BF16 input','Row reduction / convert','Value + row scale'])
        ax.set_ylim(-.6,2.6)
        ax.grid(axis='x',color='#dce5e9');ax.set_axisbelow(True)
        for side in ('right','top','left'):ax.spines[side].set_visible(False)
    def bar(ax,y,a,b,color):
        ax.barh(y,b-a,left=a,height=.28,color=COLORS[color])
    top,bot=axes
    bar(top,2,0,start,'Front');bar(top,2,start,start+rd,'MTE2')
    bar(top,1,start+rd,start+rd+vec,'Vector + handoff')
    bar(top,0,start+rd+vec,start+rd+vec+wr,'MTE3')
    bar(top,0,start+rd+vec+wr,r.latency_us,'Retire')
    for d in phase['dma']:
        key=d['kind']; y=2 if key=='MTE2' else (.19 if d['bytes']>4 else -.19)
        bar(bot,y,d['issue']/1650,d['retire']/1650,key)
    bar(bot,1,phase['vector_first']/1650,phase['vector_last']/1650,'Vector + handoff')
    for ax,time,title in ((top,r.latency_us,'Resource model'),(bot,sample['measurement']['ca_us'],'CA  ·  '+phase['core'])):
        ax.axvline(time,ls='--',lw=1.3,color='#263e4a');ax.set_title(f'{title}     {time:.3f} µs',loc='left',fontweight='bold')
    axes[-1].set_xlabel('Device time from earliest instruction (µs)')
    fig.suptitle('950DT  ·  per-token FP8 quantization  ·  9 × 1024',fontsize=16,fontweight='bold')
    fig.legend(handles=[Patch(color=v,label=k) for k,v in COLORS.items()],loc='upper center',bbox_to_anchor=(.52,.934),ncol=5,frameon=False)
    fig.subplots_adjust(left=.22,right=.97,top=.77,bottom=.11,hspace=.57)
    fig.savefig(output,dpi=145,facecolor='white');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();plot(a.archive,a.output)
