from math import ceil
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

COLORS={'Control':'#b7c7cd','MTE2':'#117889','MTE1':'#4d75b4','Cube':'#7653a5','Vector':'#2e8763','FIXP':'#b28322','MTE3':'#b44228'}
LEVEL={'Control':0,'MTE2':-.255,'MTE1':-.10,'Cube':.035,'Vector':.18,'FIXP':.18,'MTE3':.32}
TICKS=1650


def row_of(q):
    if q['phase'] in ('init','front'):return 0
    return 1+2*q['chunk']+(q['phase'] in ('S2','S3'))


def draw(data,path):
    fig,axes=plt.subplots(2,1,figsize=(13,9.35),sharex=True)
    fig.subplots_adjust(left=.231,right=.96,top=.804,bottom=.153,hspace=.3)
    fig.text(.231,.951,'950PR   Chunk KDA V2  /  recurrent state',size=22,weight='bold',color='#182e3b')
    fig.text(.232,.910,'129 tokens  ·  3 local heads  ·  128 × 128 state  ·  BF16 compute with FP32 final state',size=12,color='#385769')
    legend=[('MTE2','GM read'),('MTE1','L1 → L0'),('Cube','MMAD'),('Vector','Vector VF'),('FIXP','L0C → UB'),('MTE3','GM write')]
    fig.legend([Patch(color=COLORS[c]) for c,_ in legend],[t for _,t in legend],loc='upper left',bbox_to_anchor=(.222,.885),ncol=6,
               frameon=False,fontsize=10.5,handlelength=1.65,columnspacing=1.45)
    fig.patch.set_facecolor('white')
    labels=['Initial state · FP32 → BF16','Tokens 0–63  ·  W × H / value','0–63  ·  KᵀR / state','Tokens 64–127  ·  W × H / value',
            '64–127  ·  KᵀR / state','Token 128  ·  W × H / value','128  ·  KᵀR / final state']
    cap=ceil(max(data['model_stage_us'],data['ca_stage_us']))
    for ax,pack,sim,end,title in ((axes[0],data['prediction'],False,data['model_stage_us'],'Frozen resource model  ·  one head (three equivalent chains)'),
                                  (axes[1],data['observation'],True,data['ca_stage_us'],'CA trace  ·  head 2, workgroup 16 (last to finish)')):
        ax.set_xlim(0,cap);ax.set_ylim(6.62,-.60)
        ax.grid(axis='x',color='#dbe5e9',lw=.8);ax.set_axisbelow(True)
        ax.spines[:].set_visible(False)
        for y in (.57,2.57,4.57):ax.axhline(y,color='#cbd9df',lw=.8)
        ax.set_yticks(range(7),labels);ax.tick_params(axis='both',length=0,colors='#304e5e',labelsize=10.6,pad=7)
        ax.set_title(title,loc='left',pad=8,size=13.0,color='#233f4e',weight='bold')
        ax.set_title(f'{end:.3f} µs',loc='right',pad=8,size=14.5,color='#233f4e',weight='bold')
        # Each profile resource is one lane. Close parallel descriptors can share that lane;
        # numeric [issue,retire] and addresses remain separate in the companion JSON.
        segs=pack['initial']+[q for c in pack['chunks'] for q in c['resources']]
        if not sim:segs=[pack['front']]+segs
        for q in segs:
            a=q['start_tick']/TICKS;b=q['end_tick']/TICKS
            row=row_of(q);k=q['engine']; y=row+LEVEL[k]
            ax.barh(y,b-a,left=a,height=.108 if k!='Control' else .40,color=COLORS[k],lw=0,zorder=3,antialiased=True)
            if k=='Cube':ax.vlines(a,y-.10,y+.10,color='#4f307c',lw=.75,zorder=4)
        if sim:
            for dep in pack['state_handoffs']:
                j=pack['state_handoffs'].index(dep)
                y0=(0 if j==0 else 2*j)+LEVEL['MTE3']
                y1=1+2*j+LEVEL['MTE2']
                x0,x1=dep['producer_end_tick']/TICKS,dep['consumer_start_tick']/TICKS
                ax.plot([x0,x1],[y0,y1],color='#344b58',lw=.85,linestyle=(0,(2,2)),zorder=5)
                ax.plot(x0,y0,'.',color='#344b58',markersize=3.0,zorder=6)
                ax.plot(x1,y1,'.',color='#344b58',markersize=3.0,zorder=6)
        ax.axvline(end,color='#365463',lw=1.1,ls=(0,(3,3)),zorder=2)
        ax.xaxis.set_major_locator(MultipleLocator(1))
    axes[1].set_xlabel('Device time from the first FwdH instruction on any core (µs)',size=12,color='#293e4c',labelpad=11)
    fig.text(.231,.052,'Vertical dotted links connect each BF16 state write to the next S0 read at the same GM address. Purple ticks mark MMAD issue.',size=9.2,color='#375767')
    fig.text(.231,.026,'Upper bars are priced resource budgets. Lower bars are issued descriptors, vector functions and MMADs; scalar control and waits remain in the window.',size=9.2,color='#375767')
    fig.savefig(path,dpi=152,facecolor='white')
    plt.close(fig)
