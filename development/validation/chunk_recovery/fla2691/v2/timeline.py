"""Plot measured resource occupancy of the three official kernels on one CA clock."""
import json
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from probe_v2 import OUT,probe

def merge(intervals):
    result=[]
    for a,b in sorted(intervals):
        if not result or a>result[-1][1]:result.append([a,b])
        else:result[-1][1]=max(b,result[-1][1])
    return result


def main():
    data=json.loads((OUT/'verified.json').read_bytes());stages=data['stages'];base=stages[0]['first_tick']
    lanes=['AIC/MTE2','AIC/MTE1','AIC/CUBE','AIC/FIXP','AIV/MTE2','AIV/VEC','AIV/MTE3']
    colors={'MTE2':'#237d84','MTE1':'#577db3','CUBE':'#7254a8','FIXP':'#b55d37','VEC':'#44916a','MTE3':'#c67b2d'}
    fig,ax=plt.subplots(figsize=(10.6,4.15)); fig.subplots_adjust(left=.14,top=.77,bottom=.19,right=.98)
    fig.text(.14,.94,'950DT · FLA Chunk KDA V2',fontsize=18,weight='bold',color='#1a3544')
    fig.text(.14,.87,'Measured CA trace   |   B = 1, T = 2, H = 1, K = V = 128, BF16',fontsize=11,color='#486170')
    for stage,label in zip(stages,['Prepare','FwdH','Finalize']):
        start=(stage['first_tick']-base)/1650;finish=(stage['last_tick']-base)/1650
        ax.axvspan(start,finish,color='#e5edf2',alpha=.62,zorder=-3)
        ax.text((start+finish)/2,1.02,f"{label}\n{stage['ca_us']:.3f} µs",transform=ax.get_xaxis_transform(),ha='center',fontsize=9)
        events=json.loads((probe.ROOT/stage['aggregate']).read_bytes())['traceEvents']
        busy=defaultdict(list);issues=[]
        for event in events:
            if event['ph']!='X':continue
            detail=json.loads(event['args']['detail'] or '{}')
            if detail.get('is_sync') or detail.get('is_wait') or detail.get('is_duration_excluded'):continue
            core='AIC' if 'cubecore' in event['pid'] else 'AIV'; lane=core+'/'+event['tid']
            if core=='AIV' and (event['tid'].startswith('RVEC') or event['tid'] in ('SIMD','Vector')):
                lane='AIV/VEC'
            if lane not in lanes:continue
            lo=(round(event['ts']*1650)-base)/1650;hi=(round((event['ts']+event['dur'])*1650)-base)/1650
            if event['name']=='MMAD':issues.append(lo)
            if hi>lo:busy[lane].append((lo,hi))
        for lane,spans in busy.items():
            ax.broken_barh([(a,b-a) for a,b in merge(spans)],(lanes.index(lane)-.26,.52),facecolors=colors[lane.split('/')[1]],linewidth=0)
        ax.vlines(issues,lanes.index('AIC/CUBE')-.38,lanes.index('AIC/CUBE')+.38,color='#31234e',linewidth=.75)
    ax.set_yticks(range(len(lanes)),lanes);ax.invert_yaxis();ax.set_xlim(0,data['device_elapsed_first_to_last_us']+.05)
    ax.spines[['right','left','top']].set_visible(False);ax.grid(axis='x',color='#c7d0d8',linewidth=.5,alpha=.65)
    ax.tick_params(axis='y',length=0);ax.set_xlabel('Device time from first instruction (µs)')
    fig.text(.14,.045,'Bars union non-sync instruction intervals across active cores in each lane. Vertical ticks mark MMAD issues.',fontsize=8,color='#486170')
    fig.savefig(OUT/'ca-phases.png',dpi=150,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':main()
