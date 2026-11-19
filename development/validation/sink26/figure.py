"""Match exact VF issues and bulk DMA on the last actual core to the model's work."""
from evidence import DT,PR,OUT,read,save
from calibrate import annotated
from model import forecast
from operator_model import load_chip
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

NAMES=['Tiling read · MTE2','Payload read · MTE2','Setup · VF','Softmax · VF','First column · VF','Row norm · VF','Column norm · VF','Scatter · VF','Output write · MTE3']
C=['#147787','#147787','#627287','#7e55ac','#7145a0','#375a9e','#7750a3','#506580','#b04e27']

def plot(r,h,stem):
    t,n=r['shape'];count=r['iters'];chip=r['chip'];k=forecast(r['shape'],count,load_chip(chip),h);s=k['vf_setup_epochs']
    f=[v for v in r['events'] if v['name']=='VF'];d=[v for v in r['events'] if v['name'].startswith('MOV_SRC_TO_DST')]
    predicted=[(0,s['tiling_begin'],s['tiling_end']),(1,s['read_begin'],s['read_end']),(8,s['write_begin'],s['write_end']),
               (2,s['setup_vf'],None),(3,s['softmax_vf'],None),(4,s['first_col_vf'],None),(7,s['scatter_vf'],None)]
    for i in range(count-1):
        row=s['first_row_vf']+(s['first_pair_period']+(i-1)*s['steady_pair_period'] if i>0 else 0)
        predicted.extend([(5,row,None),(6,row+(s['steady_row_to_col'] if i else s['first_row_to_col']),None)])
    actual=[]
    def observed(lane,e):actual.append((lane,e['begin']/1650,e['end']/1650,e['pc']))
    for i,j in enumerate([0,1,8]):observed(j,d[i])
    for j,e in zip([2,3,4]+[n for i in range(count-1) for n in (5,6)]+[7],f):observed(j,e)
    target=OUT/(stem+'.json')
    save(target,dict(chip=chip,shape=r['shape'],iters=count,prior_h=h,critical_core=r['core'],row_labels=NAMES,
                     origin='earliest device instruction among all active cores; all timings in microseconds',
                     predicted_fields=['row','issue_or_descriptor_begin_us','descriptor_end_us_or_null'],
                     observed_fields=['row','issue_or_descriptor_begin_us','descriptor_end_us','actual_pc'],
                     model=predicted,ca=actual,model_full_us=k['latency_us'],ca_full_us=r['ca_us'],
                     measured_source=r['source'],shard_or_database_sha256=r['source_sha256'],aggregate_sha256=r['aggregate_sha256']))
    fig,axes=plt.subplots(2,1,sharex=True,figsize=(11.4,7.4))
    fig.subplots_adjust(left=.195,right=.965,top=.81,bottom=.19,hspace=.47)
    label=r['core']
    if label[0].isdigit():
        ci,kind=label.split('.');label=f'core{ci}.veccore{int(kind)-2}'
    fig.text(.195,.955,f"{chip[6:11]}  |  Sinkhorn  ·  T={t}  ·  4×4 FP32  ·  {count} iterations",fontsize=20,fontweight='bold',color='#1a3340')
    fig.text(.195,.91,f"{k['active_aiv']} AIVs  ·  up to {k['rows_per_core']} matrices per core  ·  previous same-family high-water {h} AIVs",fontsize=11.5,color='#445967')
    for ax,rows,duration,title in [(axes[0],predicted,k['latency_us'],f"Resource model   /   {k['selected']}"),(axes[1],actual,r['ca_us'],f"CA trace   /   {label} last to retire")]:
        ax.set_title(title,loc='left',pad=10,fontsize=12.5,fontweight='bold',color='#253e4b');ax.set_title(f'{duration:.3f} µs',loc='right',pad=10,fontweight='bold')
        for e in rows:
            j,b,z=e[:3]
            if z is not None:ax.barh(j,max(0,z-b),left=b,height=.35,color=C[j],alpha=1 if j in (0,1,8) else .28,zorder=3)
            if j not in (0,1,8):ax.plot([b,b],[j-.26,j+.26],color=C[j],lw=1.45,zorder=4)
        ax.set_ylim(8.6,-.65);ax.set_yticks(range(9),NAMES,fontsize=9.5);ax.tick_params(axis='y',length=0)
        ax.axvline(duration,color='#3c5260',ls=(0,(3,3)),lw=1.25);ax.grid(axis='x',color='#d1dce2',lw=.65)
        ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_color('#c0cbd1')
    axes[1].set_xlim(0,1.035*max(k['latency_us'],r['ca_us']));axes[1].set_xlabel('Device time since the first instruction on any core (µs)',fontsize=11,labelpad=8)
    fig.text(.195,.039,'VF tick = issue; translucent band = observed descriptor duration.  Model exposes VF issue times only.\nDMA bars include device-side wait.  Dashed vertical line = end of the complete all-core window.',fontsize=9.25,color='#475c69')
    fig.savefig(OUT/(stem+'.png'),dpi=165);plt.close(fig)

if __name__=='__main__':
    rows=read(OUT/'trace-evidence.json')
    for chip,t,name in [(DT,65,'dt-65-hot'),(DT,193,'dt-193-first'),(PR,193,'pr-193-first')]:
        row,h=next((r,h) for r,h in annotated([v for v in rows if v['chip']==chip]) if r['shape'][0]==t)
        plot(row,h,name)
