"""Compare the predicted and measured first-row MTE2 descriptors for matched protocols."""
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR','/tmp/post-candidate-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from development.validation.post_candidate.evaluate import PR, OUT, read


def render():
    data = read(OUT/'issue-response.json')
    fig, axs = plt.subplots(2, 2, figsize=(12, 6.4), sharex=True, sharey=True)
    colors = ['#147a80', '#5376a6']
    lanes = ['256 B input','1,024 B residual','ready → SIMD']
    states = [('first', 'Standalone first call'), ('after_pre_sinkhorn', 'After Pre + Sinkhorn')]
    for col,(s,title) in enumerate(states):
        r = next(r for r in data['rows'] if r['chip']==PR and r['shape']==[1,4,128] and r['state']==s)
        for ax,field,src in [(axs[0,col],'predicted_read_window','Grouped prediction'),(axs[1,col],'ca_read_window','CA terminal core')]:
            q = r[field]
            for lane,(begin,dur) in enumerate(zip(q['issue_us'],q['response_us'])):
                ax.barh(lane,dur,left=begin,color=colors[lane],height=.32,zorder=3)
                ax.scatter(begin,lane,marker='|',s=100,color='#11283b',zorder=4)
            ready = max(a+b for a,b in zip(q['issue_us'],q['response_us']))
            ax.barh(2,q['handoff_us'],left=ready,color='#a7652d',height=.32,zorder=3)
            end = ready+q['handoff_us']
            ax.axvline(end,color='#172c42',lw=1,ls='--')
            ax.grid(axis='x',color='#dce4ea',zorder=0)
            ax.set_title(f'{src}  ·  {end:.3f} µs',loc='left',fontsize=11,pad=11)
            ax.set_yticks([0,1,2],lanes)
            for side in ['right','top']:
                ax.spines[side].set_visible(False)
        axs[0,col].text(0,1.31,title,transform=axs[0,col].transAxes,fontsize=14,fontweight='bold',color='#152d3c')
    axs[0,0].set_ylim(2.6,-.65)
    fig.suptitle('950PR mHC Post   |   BF16 [1, 4, 128]   |   MTE2 issue and response',
                 x=.035,ha='left',fontsize=17,weight='bold',y=.97)
    fig.supxlabel('Time from the first payload MTE2 issue (µs)',fontsize=12,y=.105)
    fig.text(.035,.035,'Ticks mark request issues. Teal and blue span issue to completion. Brown spans the final completion to the first SIMD operand load.',
             fontsize=9,color='#395366')
    fig.subplots_adjust(left=.15,right=.98,top=.74,bottom=.20,wspace=.19,hspace=.38)
    fig.savefig(OUT/'issue-response.png',dpi=165,facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    render()
