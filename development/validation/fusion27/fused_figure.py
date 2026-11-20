"""Draw the highest-retirement AIV and AIC against independent pre-CA phase budgets."""
from pathlib import Path
import hashlib
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from observe import ROOT,OUT,ORIG,read,save

NAMES=('Launch / control','Stage 1 · AIV cast','Stage 1 · AIC MMAD','Handoff / FIXP','Stage 2 · AIV Sinkhorn')
COLORS=('#9fb4bd','#147985','#7851a6','#b9522a','#3873a1')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def make(case):
    receipt=OUT/'receipts'/(case+'.json')
    if not receipt.exists():receipt=ORIG/(case+'.json')
    r=read(receipt)
    if not r['qualified']:raise ValueError('requires a golden and full retirement receipt')
    p=r['prediction'];shape=r['shape'];s=r['stage'];phase=p['stages_us'];b=read(ORIG/'prices.json')['prices']
    front=phase['dispatch_and_handoff']*b['first_vf_cycles']/(b['first_vf_cycles']+b['rendezvous_cycles'])
    widths=(front,phase['stage1_cast'],phase['stage1_cube'],phase['dispatch_and_handoff']-front,phase['stage2_rows'])
    assert phase['exposed_hbm_floor']==0
    model=[];pos=0.
    for lane,width in enumerate(widths):
        model.append(dict(lane=lane,begin_us=pos,end_us=pos+width));pos+=width
    if abs(pos-p['candidate_us'])>1e-10:raise ValueError('incomplete modeled window')
    sources={};actual=[];vec_core=cube_core=None
    for family in ('vec','cube'):
        core=max((c for c in s['windows'] if family in c),key=lambda c:s['windows'][c][1])
        path=(ROOT/s['trace']).parent/core/'trace.json'
        ev=sorted((e for e in read(path)['traceEvents'] if e['ph']=='X'),key=lambda e:e['ts'])
        start=ev[0]['ts']-s['windows'][core][0];stage2=None
        if family=='vec':
            stage2=next(e['ts'] for e in ev if e['name']=='VF' and e['args']['pc_addr'].endswith('14cb0'));vec_core=core
        else:cube_core=core
        for e in ev:
            name=e['name']
            if family=='vec' and name=='VF':lane=1 if e['ts']<stage2 else 4
            elif family=='cube' and name in ('FIX_L0C_TO_DST','MMAD'):lane=3 if name=='FIX_L0C_TO_DST' else 2
            else:continue
            t=e['ts']-start;detail=json.loads(e['args'].get('detail','{}'))
            actual.append(dict(lane=lane,name=name,core=core,pc=e['args']['pc_addr'],begin_us=t,end_us=t+e['dur'],
                               tile_detail={k:detail[k] for k in ('m','k','n','matrix_a_dtype') if k in detail}))
        sources[family]=dict(core=core,window_us=s['windows'][core],origin_correction_us=start,path=str(path.relative_to(ROOT)),sha256=sha(path))
    mm=sum(e['name']=='MMAD' for e in actual)
    fig,axs=plt.subplots(2,1,figsize=(11.8,7.5),sharex=True)
    left=.215;fig.subplots_adjust(left=left,right=.966,top=.765,bottom=.17,hspace=.52)
    chip=r['chip'][6:11];t,n,d=shape;ink='#213d49'
    fig.text(left,.95,f'{chip}  |  Fused PreSinkhorn  ·  {t} × {n} × {d:,}',fontsize=21,weight='bold',color=ink)
    fig.text(left,.905,f'{s["active_aic"]} AICs and {s["active_aiv"]} AIVs  ·  largest local K = {p["target_plan"]["largest_k"]}  ·  up to {p["target_plan"]["rows_per_aiv"]} post rows per AIV',fontsize=12,color='#45606b')
    fig.legend(handles=[Patch(color=COLORS[i],label=v) for i,v in enumerate(('Control budget','Activation VF','Cube / MMAD','FIXP / handoff','Post / VF'))],
               loc='upper left',bbox_to_anchor=(left-.008,.88),ncol=5,frameon=False,fontsize=9.3)
    for ax,entries,title,end in ((axs[0],model,'Frozen resource estimate',p['candidate_us']),
            (axs[1],actual,f'CA trace  ·  {vec_core} and {cube_core}',r['ca_us'])):
        for e in entries:
            j=e['lane'];actual_event='name' in e;begin=e['begin_us'];duration=e['end_us']-begin
            ax.barh(j,duration,left=begin,height=.30,color=COLORS[j],alpha=.35 if actual_event and j==3 else .88)
            if actual_event and j in (1,2,4):ax.vlines(begin,j-.21,j+.21,color=COLORS[j],lw=.65)
        ax.set_ylim(4.5,-.52);ax.set_yticks(range(5),NAMES,fontsize=10)
        ax.axvline(end,color=ink,ls=(0,(3,3)),lw=1.3);ax.tick_params(axis='y',length=0)
        ax.grid(axis='x',color='#d6e2e7',lw=.65);ax.set_axisbelow(True);ax.spines[['top','right','left']].set_visible(False)
        ax.spines['bottom'].set_color('#b9c9d1');ax.set_title(title,loc='left',color=ink,fontsize=12,weight='bold')
        ax.set_title(f'{end:.3f} µs',loc='right',color=ink,fontsize=14,weight='bold')
    axs[1].set_xlim(0,max(r['ca_us'],p['candidate_us'])*1.035);axs[1].set_xlabel('Device time since the first instruction on any core (µs)',labelpad=8)
    fig.text(left,.063,f'CA ticks show VF and MMAD issues ({mm} MMADs on this AIC). Pale FIXP spans include dependency waits. Dashed ends cover all cores.',fontsize=9,color='#48616c')
    fig.text(left,.035,'The upper control and handoff spans split a joint budget by the 950DT marker medians. The model has no individual barrier-arrival estimate.',fontsize=9,color='#48616c')
    fig.savefig(OUT/(case+'-timeline.png'),dpi=165);plt.close(fig)
    save(OUT/(case+'-timeline.json'),dict(source=str(receipt.relative_to(ROOT)),source_sha256=sha(receipt),frozen_case=case,
        lanes=NAMES,shape=shape,chip=r['chip'],model_us=p['candidate_us'],ca_us=r['ca_us'],baseline_us=p['v025_us'],
        origin='Global first complete instruction on any active core = 0 us on both axes. CA shards aligned to official kernel envelope.',
        model=model,model_control_split='Only for plotting, split joint fixed cost in the first-VF/rendezvous 950DT training median ratio.',
        budgets=phase,ca=actual,sources=sources,aggregate_sha256=s['trace_sha256'],model_mmad_for_largest_split=p['target_plan']['critical_aic_mmad'],
        actual_mmad_on_plotted_aic=mm,model_plan=p['target_plan']))
if __name__=='__main__':make(sys.argv[1])
