"""Offline scoring for historical development windows and a fixed forward group."""
from math import ceil
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from campaign import OUT,REG,ROOT,OP,PR,DT,FUT,forecast,load,save
from operator_model import load_chip
from operator_model.mhc_pre_sink_resource import fused

def score(rows):
    return {k:{'WAPE_pct':100*sum(abs(r[k]-r['ca_us'])for r in rows)/sum(r['ca_us']for r in rows),
               'max_abs_pct':100*max(abs(r[k]/r['ca_us']-1)for r in rows)}for k in ('v025_us','candidate_us')}

def write():
    src=load(ROOT/'results/v0.25/coverage/audit.json');rows=[]
    for r in src['rows']:
        if r['op']!=OP:continue
        p=forecast(r['shape'],r['chip'])
        if abs(r['current_us']-p['v025_us'])>1e-7:raise ValueError('wrong baseline')
        rows.append(dict(shape=r['shape'],sku=r['chip'],**p,actual_aic=r['active_aic'],actual_aiv=r['active_aiv'],
                         ca_us=r['ca_us'],trace_sha256=r['trace_sha256'],sources=r['origins'],projection_970dt_us=forecast(r['shape'],FUT)['candidate_us']))
    groups=[]
    for chip in [PR,DT]:
        for label,test in [('all',lambda t:True),('T<=16',lambda t:t<=16),('16<T<=64',lambda t:16<t<=64),('T>64',lambda t:t>64)]:
            subset=[r for r in rows if r['sku']==chip and test(r['shape'][0])]
            groups.append(dict(chip=chip,group=label,n=len(subset),**score(subset)))
    time=[];target=load_chip(PR)
    for _ in range(201):
        a=perf_counter_ns();fused([80,4,4096],target,{'num_iters':20,'need_backward':False});time.append((perf_counter_ns()-a)/1e6)
    save(OUT/'replay.json',dict(groups=groups,rows=rows,role='Development replay, multiple windows of a shape kept only with a different original digest',
        candidate_query_ms={'n':len(time),'median':median(time),'p95':sorted(time)[int(.95*len(time))],'max':max(time)}))

def draw():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    r=next(r for r in load(OUT/'archive.json') if r['sku']==PR and r['shape'][0]==80)
    p=forecast(r['shape'],PR);s=p['stages_us'];c=load(OUT/'prices.json')['prices']
    f=s['dispatch_and_handoff']*c['first_vf_cycles']/(c['first_vf_cycles']+c['rendezvous_cycles'])
    model=[]
    for row in [('front',f),('Stage 1 / AIV',s['stage1_cast']),('Stage 1 / AIC',s['stage1_cube']),('handoff',s['dispatch_and_handoff']-f),('Stage 2 / AIV',s['stage2_rows'])]:
        a=model[-1][2] if model else 0;model.append((row[0],a,a+row[1]))
    ve=r['vec']['milestones'];cube=r['cube']['milestones'];vv=ve['VF'];pp=next(e['us'][0]for e in vv if e['pc'].endswith('14cb0'));lo=lambda e:e['us']
    actual={'Stage 1 / AIV':[lo(e)for e in vv if lo(e)[0]<pp],'Stage 1 / AIC':[lo(e)for e in cube['MMAD']],
            'FIXP descriptor':[lo(e)for e in cube['FIX_L0C_TO_DST']],'Stage 2 / AIV':[lo(e)for e in vv if lo(e)[0]>=pp]}
    save(OUT/'pr-t80-figure.json',dict(archive=r['trace_sha256'],model_stages=model,actual_us=actual,ca_us=r['ca_us'],**p,
             note='For the plot only, the fixed launch and rendezvous budget is split in the ratio of the two actual DT milestone medians. Candidate latency and phase work prices do not use PR offsets.'))
    ink='#19323f';colors={'front':'#c1cfd4','Stage 1 / AIV':'#137e86','Stage 1 / AIC':'#7454a4','handoff':'#b4c2c9','FIXP descriptor':'#b94f2e','Stage 2 / AIV':'#1c6f9c'}
    labels=['Stage 1 / AIV','Stage 1 / AIC','FIXP descriptor','Stage 2 / AIV']
    fig,axs=plt.subplots(2,1,figsize=(11.6,7.3),sharex=True)
    left=.195;fig.subplots_adjust(left=left,right=.955,top=.755,bottom=.15,hspace=.55)
    fig.text(left,.951,'950PR fused PreSinkhorn  |  80 × 4 × 4,096',fontsize=22,color=ink,weight='bold')
    fig.text(left,.906,'22 AICs  ·  44 AIVs  ·  eight MMADs on a full K768 split  ·  two post rows per busy AIV',fontsize=11.5,color='#466371')
    fig.legend(handles=[Patch(color=colors[n],label=l) for n,l in [('Stage 1 / AIV','Activation / VF'),('Stage 1 / AIC','Cube / MMAD'),('FIXP descriptor','FIXP window'),('Stage 2 / AIV','Post / VF'),('handoff','Control & handoff')]],
         loc='upper left',bbox_to_anchor=(left-.008,.879),ncol=5,frameon=False,fontsize=9)
    for name,a,b in model:
        y=0 if name=='front' else (2 if name=='handoff' else labels.index(name))
        axs[0].barh(y,b-a,left=a,color=colors[name],height=.25)
    for name,windows in actual.items():
        for a,b in windows:
            y=labels.index(name)
            axs[1].barh(y,max(.011,b-a),left=a,color=colors[name],height=.25,alpha=.48 if name=='FIXP descriptor' else 1)
            if name=='Stage 1 / AIC':axs[1].vlines(a,y-.19,y+.19,color='#49316e',lw=.8)
    for ax,title,end in zip(axs,('Resource candidate  ·  work and control budgets','CA trace  ·  '+r['vec']['core']+' / '+r['cube']['core']),[p['candidate_us'],r['ca_us']]):
        ax.axvline(end,color=ink,ls=(0,(3,3)),lw=1);ax.set_title(title,loc='left',weight='bold',color=ink,fontsize=13);ax.set_title(f'{end:.3f} µs',loc='right',weight='bold',color=ink,fontsize=14)
        ax.set_ylim(3.5,-.5);ax.set_yticks(range(4),['Activation / cast','K projection','Handoff / FIXP','Post / Sinkhorn'],fontsize=10);ax.tick_params(axis='y',length=0)
        ax.spines[:].set_visible(False);ax.grid(axis='x',color='#dde5e9');ax.set_axisbelow(True)
    axs[1].set_xlim(-.05,15);axs[1].set_xlabel('Device time from the first core instruction (µs)')
    fig.text(left,.060,'Top separates work from fixed control; exact barrier arrival is unmodeled. Bottom shows real VF and MMAD intervals on the last AIV and AIC.',fontsize=8.7,color='#466371')
    fig.text(left,.036,'The FIXP descriptor opens before the first MMAD and includes waits. The 22.5k-cycle kernel envelope covers all 66 active cores.',fontsize=8.7,color='#466371')
    fig.savefig(OUT/'pr-t80.png',dpi=155);plt.close(fig)
if __name__=='__main__':write();draw()
