"""Export device-window and prefill scaling figures from frozen, paired data."""
import json
from pathlib import Path
import os
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/operator-sim-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from large_cast import DT, PR, ROOT, OUT, read

STAGE_GROUPS=[
    ('Arrival & control',('arrival','control_to_read'),'#94a3b8'),
    ('Tile 1 read',('read',),'#077b83'),
    ('Tile 1 SIMD',('vector',),'#7755a1'),
    ('Wait & first write',('dependency_to_write','write'),'#bc6738'),
    ('Pipelined tiles',('steady_tiles',),'#356aa2'),
    ('Shared service extension',('shared_io',),'#95504b')]


def model_bar(axis,diag,pred):
    x=0.
    for name,keys,color in STAGE_GROUPS:
        width=sum(diag['stages_us'][k] for k in keys)
        if width:axis.barh(0,width,left=x,height=.56,color=color)
        x+=width
    axis.axvline(pred,color='#283844',ls='--',lw=1.4)
    axis.set_yticks([])
    axis.set_ylim(-.7,.7)
    axis.set_title(f"Exposed model stages   ·   {diag['target_plan']['busy_core_waves']} UB tiles on the busiest AIV   ·   {pred:.3f} µs",loc='left',fontsize=11)


def merged(intervals):
    done=[]
    for start,stop in sorted(intervals):
        if done and start<=done[-1][1]+1e-5:done[-1][1]=max(done[-1][1],stop)
        else:done.append([start,stop])
    return [(a,b-a) for a,b in done]


def plot_window(row,native=False):
    t,h=row['shape'];f=OUT/(f'cast_{t}_{h}_native.json' if native else f'cast_{t}_{h}_{DT}.json')
    if not f.exists():return
    result=read(f)
    if not result.get('qualified_core_envelope' if native else 'qualified'):return
    diag=row['targets'][DT]['diagnostics'];pred=row['targets'][DT]['us']
    fig,(model,ca)=plt.subplots(2,1,sharex=True,figsize=(10,5.2),gridspec_kw={'height_ratios':[1,3]})
    fig.subplots_adjust(left=.13,right=.97,top=.78,bottom=.16,hspace=.65)
    fig.suptitle(f'950DT Cast   |   {t:,} × {h:,}   ·   BF16 → FP32',y=.98,fontsize=18,weight='bold')
    fig.legend(handles=[Patch(color=color,label=name) for name,keys,color in STAGE_GROUPS],
               loc='upper center',bbox_to_anchor=(.52,.92),ncol=3,frameon=False,fontsize=9)
    model_bar(model,diag,pred)
    if native:
        windows=sorted(result['core_ticks'].items(),key=lambda p:list(map(int,p[0].split('.'))))
        origin=min(pair[0] for _,pair in windows)
        for i,(_, (a,b)) in enumerate(windows):
            ca.plot([(a-origin)/1650,(b-origin)/1650],[i,i],lw=1.3,color=('#227a82' if i%2==0 else '#68a9ac'))
        ca.set_yticks([0,15,31,47,63],[1,16,32,48,64]);ca.set_ylabel('AIV')
        ref=result['core_envelope_us']
        ca.set_title(f'CA core wrapper   ·   {len(windows)} AIV start → done   ·   {ref:.3f} µs',loc='left',fontsize=11)
        ca.set_xlabel('Elapsed device time (µs) from each panel’s own origin')
        fig.text(.13,.02,'Each measured line spans one core wrapper, including its start-up wait. The model targets the instruction window.',fontsize=9)
    else:
        stage=result['stage'];ref=stage['ca_us'];core=max(stage['windows'],key=lambda c:stage['windows'][c][1])
        percore=(ROOT/stage['trace']).parent/core/'trace.json'
        evs=[e for e in read(percore)['traceEvents'] if e['ph']=='X']
        origin=min(e['ts'] for e in evs)-stage['windows'][core][0]
        queues=[('MTE2','#077b83'),('SIMD','#7755a1'),('MTE3','#bc6738'),('SCALAR','#526175')]
        work={q:[] for q,_ in queues}
        for e in evs:
            detail=json.loads(e['args']['detail'])
            q=detail.get('queue_type')
            if q in work and not detail.get('is_duration_excluded'):
                a=e['ts']-origin;work[q].append((a,a+e['dur']))
        for i,(q,color) in enumerate(queues):
            ca.broken_barh(merged(work[q]),(i-.28,.56),facecolors=color)
        ca.set_yticks(range(len(queues)),[q for q,c in queues]);ca.invert_yaxis()
        ca.set_title(f'CA last-to-finish AIV   ·   {core}   ·   all-core window {ref:.3f} µs',loc='left',fontsize=11)
        ca.set_xlabel('Elapsed device time (µs) from earliest core instruction')
        fig.text(.13,.02,'Overlapping events on each measured queue are merged. Model bars show exposed costs, rather than individual instruction starts.',fontsize=9)
    ca.axvline(ref,color='#142f38',lw=1.5);ca.axvline(pred,color='#283844',ls='--',lw=1.2)
    for ax in (model,ca):
        ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True);ax.spines[['right','top']].set_visible(False)
    ca.set_xlim(0,1.04*max(ref,pred));fig.savefig(OUT/('full-hidden.png' if native else 'indexer-timeline.png'),dpi=165);plt.close(fig)


def scale(data):
    rows=[r for r in data['cases'] if r['shape'][1]==32]
    contexts=[r['shape'][0]//(8*1024) for r in rows];x=np.arange(len(rows))
    fig,(lat,bw)=plt.subplots(1,2,figsize=(10,4.4))
    fig.subplots_adjust(left=.08,right=.97,top=.80,bottom=.17,wspace=.33)
    fig.suptitle('Flash indexer Cast   |   8 unchunked prefill requests',fontweight='bold',fontsize=16)
    for c,name,color in [(DT,'950DT','#136f78'),(PR,'950PR','#b85d32')]:
        pred=[r['current'][c]['us'] for r in rows]
        lat.plot(x,pred,'o--',color=color,alpha=.75,label=name+' model')
        qualified=[(i,r['devices'][c]) for i,r in enumerate(rows) if 'ca_us' in r['devices'][c]]
        if qualified:
            ix=[i for i,v in qualified]
            lat.plot(ix,[v['ca_us'] for i,v in qualified],'s-',color=color,label=name+' CA')
            bw.plot(ix,[v['logical_gbs']/1000 for i,v in qualified],'s-',color=color,label=name+' CA')
    lat.plot(x,[r['current']['Ascend970DT']['us'] for r in rows],'^:',color='#76559b',label='970DT extrapolated')
    lat.set_ylabel('Device latency (µs)');lat.set_yscale('log');lat.legend(frameon=False,fontsize=8,ncol=2)
    bw.set_ylabel('Logical bytes / CA duration (TB/s)');bw.legend(frameon=False,fontsize=9);bw.set_ylim(bottom=0)
    for ax in (lat,bw):
        ax.set_xticks(x,[f'{k}ki' for k in contexts]);ax.set_xlim(-.16,2.16)
        ax.set_xlabel('Context per request (tokens)');ax.grid(axis='y',alpha=.18);ax.spines[['right','top']].set_visible(False)
    fig.text(.08,.02,'Solid marks require completed full-precision checks and all-core CA timestamps. The rate includes all logical reads and writes.',fontsize=9)
    fig.savefig(OUT/'indexer-cast.png',dpi=165);plt.close(fig)


def draw(data):
    frozen=read(OUT/'cast-v20-freeze.json')['jobs']
    scale(data);plot_window(frozen[0]);plot_window(frozen[-1],native=True)
